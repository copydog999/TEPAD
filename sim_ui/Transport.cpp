// Transport.cpp : serial / TCP data source implementation.
#include "framework.h"
#include "Transport.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace {

const int kMaxQueuedLines = 4096;   ///< queue limit; when the UI stalls the oldest data is dropped to stay real-time
const DWORD kReadTimeoutMs = 100;   ///< per-read blocking limit so shutdown stays responsive
const DWORD kThreadWaitMs = 3000;   ///< upper bound for waiting on the receiver thread at shutdown
const std::size_t kReadChunk = 8192;

CString LastErrorText(LPCTSTR prefix) {
    const DWORD code = ::GetLastError();
    LPTSTR buffer = nullptr;
    const DWORD length = ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                             FORMAT_MESSAGE_FROM_SYSTEM |
                                             FORMAT_MESSAGE_IGNORE_INSERTS,
                                         nullptr, code, 0, reinterpret_cast<LPTSTR>(&buffer), 0,
                                         nullptr);
    CString message;
    if (length > 0 && buffer != nullptr) {
        message.Format(_T("%s failed (error %lu): %s"), prefix, code, CString(buffer, length));
        ::LocalFree(buffer);
    } else {
        message.Format(_T("%s failed (error %lu)"), prefix, code);
    }
    return message;
}

}  // namespace

CSimTransport::CSimTransport()
    : kind_(KindNone),
      handle_(INVALID_HANDLE_VALUE),
      socket_(static_cast<unsigned long long>(INVALID_SOCKET)),
      running_(false),
      stopRequested_(false),
      worker_(nullptr),
      threadHandle_(nullptr),
      splitter_(kReadChunk),
      receivedBytes_(0),
      droppedLines_(0) {}

CSimTransport::~CSimTransport() { Close(); }

void CSimTransport::EnumerateSerialPorts(CStringArray& ports) {
    ports.RemoveAll();
    for (int index = 1; index <= 32; ++index) {
        CString name;
        name.Format(_T("\\\\.\\COM%d"), index);
        const HANDLE handle = ::CreateFile(name, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                           OPEN_EXISTING, 0, nullptr);
        if (handle != INVALID_HANDLE_VALUE) {
            ::CloseHandle(handle);
            ports.Add(CString(_T("COM")) + CString(std::to_wstring(index).c_str()));
        }
    }
}

bool CSimTransport::OpenSerial(LPCTSTR port, DWORD baud, CString* error) {
    Close();
    CString normalized = CString(port);
    normalized.Trim();
    normalized.MakeUpper();
    if (normalized.Find(_T("COM")) != 0) normalized = _T("COM") + normalized;
    if (normalized.GetLength() < 4) {
        if (error != nullptr) *error = _T("invalid serial port name, expected e.g. COM4");
        return false;
    }

    CString device;
    device.Format(_T("\\\\.\\%s"), static_cast<LPCTSTR>(normalized));
    const HANDLE handle = ::CreateFile(device, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                       OPEN_EXISTING, 0, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        if (error != nullptr) *error = LastErrorText(_T("open serial port"));
        return false;
    }

    DCB dcb;
    memset(&dcb, 0, sizeof(dcb));
    dcb.DCBlength = sizeof(dcb);
    if (!::GetCommState(handle, &dcb)) {
        if (error != nullptr) *error = LastErrorText(_T("GetCommState"));
        ::CloseHandle(handle);
        return false;
    }
    dcb.BaudRate = baud == CBR_9600 ? CBR_9600 : baud;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = FALSE;
    dcb.fOutxCtsFlow = FALSE;
    dcb.fOutxDsrFlow = FALSE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    dcb.fOutX = FALSE;
    dcb.fInX = FALSE;
    if (!::SetCommState(handle, &dcb)) {
        if (error != nullptr) {
            *error = LastErrorText(_T("SetCommState (baud ") + CString(std::to_wstring(baud).c_str()) + _T(" possibly unsupported)"));
        }
        ::CloseHandle(handle);
        return false;
    }

    COMMTIMEOUTS timeouts;
    timeouts.ReadIntervalTimeout = 20;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.ReadTotalTimeoutConstant = kReadTimeoutMs;
    timeouts.WriteTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant = kReadTimeoutMs;
    ::SetCommTimeouts(handle, &timeouts);
    ::SetupComm(handle, 16384, 4096);
    ::PurgeComm(handle, PURGE_RXCLEAR | PURGE_TXCLEAR);

    handle_ = handle;
    kind_ = KindSerial;
    endpoint_.Format(_T("%s @ %lu 8N1"), static_cast<LPCTSTR>(normalized), baud);
    receivedBytes_ = 0;
    droppedLines_ = 0;
    return StartWorker(error);
}

bool CSimTransport::OpenTcp(LPCTSTR host, UINT port, CString* error) {
    Close();
    static bool winsockReady = false;
    if (!winsockReady) {
        WSADATA data;
        if (::WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            if (error != nullptr) *error = _T("WSAStartup failed");
            return false;
        }
        winsockReady = true;
    }

    const SOCKET socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket == INVALID_SOCKET) {
        if (error != nullptr) *error = _T("socket creation failed");
        return false;
    }
    sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<u_short>(port));
    CString hostText = CString(host);
    hostText.Trim();
    if (::InetPton(AF_INET, hostText, &address.sin_addr) != 1) {
        // gethostbyname is deprecated by Winsock (C4996), so use GetAddrInfoW;
        // only the first IPv4 stream socket result is taken, which is enough for 127.0.0.1 / localhost / LAN host names.
        CString service;
        service.Format(_T("%u"), port);
        addrinfoW hints;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        addrinfoW* result = nullptr;
        if (::GetAddrInfoW(hostText, service, &hints, &result) != 0 || result == nullptr) {
            ::closesocket(socket);
            if (error != nullptr) *error = _T("cannot resolve host: ") + hostText;
            return false;
        }
        memcpy(&address.sin_addr, &reinterpret_cast<const sockaddr_in*>(result->ai_addr)->sin_addr,
               sizeof(address.sin_addr));
        ::FreeAddrInfoW(result);
    }
    if (::connect(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR) {
        const int code = ::WSAGetLastError();
        ::closesocket(socket);
        if (error != nullptr) {
            error->Format(_T("connect to %s:%u failed (Winsock error %d); start serial_bridge.py first"),
                          static_cast<LPCTSTR>(hostText), port, code);
        }
        return false;
    }

    DWORD timeout = kReadTimeoutMs;
    ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout),
                 sizeof(timeout));

    socket_ = static_cast<unsigned long long>(socket);
    kind_ = KindTcp;
    endpoint_.Format(_T("TCP %s:%u"), static_cast<LPCTSTR>(hostText), port);
    receivedBytes_ = 0;
    droppedLines_ = 0;
    return StartWorker(error);
}

bool CSimTransport::StartWorker(CString* error) {
    splitter_.Reset();
    stopRequested_ = false;
    running_ = true;
    // Created suspended: disable auto-delete first and then resume, so Close() cannot double-free a thread that already exited.
    worker_ = AfxBeginThread(&CSimTransport::WorkerThreadEntry, this, THREAD_PRIORITY_NORMAL,
                             0, CREATE_SUSPENDED, nullptr);
    if (worker_ == nullptr) {
        running_ = false;
        kind_ = KindNone;
        if (error != nullptr) *error = _T("failed to start receiver thread");
        return false;
    }
    worker_->m_bAutoDelete = false;
    ::DuplicateHandle(::GetCurrentProcess(), worker_->m_hThread, ::GetCurrentProcess(),
                      &threadHandle_, 0, FALSE, DUPLICATE_SAME_ACCESS);
    ::ResumeThread(worker_->m_hThread);
    return true;
}

void CSimTransport::Close() {
    stopRequested_ = true;
    if (handle_ != INVALID_HANDLE_VALUE) {
        // Wake up the blocked ReadFile, then close the handle.
        ::CancelIoEx(handle_, nullptr);
    }
    if (threadHandle_ != nullptr) {
        const DWORD waitResult = ::WaitForSingleObject(threadHandle_, kThreadWaitMs);
        if (waitResult == WAIT_OBJECT_0) {
            delete worker_;
            worker_ = nullptr;
        } else {
            // The thread is still stuck in a driver call; killing it is riskier, so hand auto-delete back and reclaim on exit.
            if (worker_ != nullptr) worker_->m_bAutoDelete = true;
            worker_ = nullptr;
        }
        ::CloseHandle(threadHandle_);
        threadHandle_ = nullptr;
    }
    if (handle_ != INVALID_HANDLE_VALUE) {
        ::CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
    }
    if (socket_ != static_cast<unsigned long long>(INVALID_SOCKET)) {
        const SOCKET socket = static_cast<SOCKET>(socket_);
        ::shutdown(socket, SD_BOTH);
        ::closesocket(socket);
        socket_ = static_cast<unsigned long long>(INVALID_SOCKET);
    }
    {
        CSingleLock guard(&lock_, TRUE);
        lines_.clear();
    }
    splitter_.Reset();
    running_ = false;
    kind_ = KindNone;
    endpoint_.Empty();
}

UINT CSimTransport::WorkerThreadEntry(LPVOID param) {
    static_cast<CSimTransport*>(param)->WorkerLoop();
    return 0;
}

void CSimTransport::WorkerLoop() {
    std::vector<char> chunk(kReadChunk);
    while (!stopRequested_) {
        int available = -1;
        if (kind_ == KindSerial) {
            DWORD read = 0;
            const BOOL ok = ::ReadFile(handle_, chunk.data(), static_cast<DWORD>(chunk.size()),
                                       &read, nullptr);
            available = ok ? static_cast<int>(read) : -1;
            if (!ok) {
                const DWORD code = ::GetLastError();
                if (code == ERROR_OPERATION_ABORTED || code == ERROR_INVALID_HANDLE) break;
                // Brief back-off after a read error, to avoid spinning in a bad state.
                ::Sleep(50);
            }
        } else if (kind_ == KindTcp) {
            available = ::recv(static_cast<SOCKET>(socket_), chunk.data(),
                               static_cast<int>(chunk.size()), 0);
            if (available == SOCKET_ERROR) {
                const int code = ::WSAGetLastError();
                if (code == WSAETIMEDOUT || code == WSAEINPROGRESS) {
                    available = 0;
                } else {
                    break;
                }
            } else if (available == 0) {
                break;  // peer closed
            }
        } else {
            break;
        }
        if (stopRequested_) break;
        if (available <= 0) continue;

        receivedBytes_ += static_cast<unsigned long long>(available);
        const std::vector<std::string> lines = splitter_.Feed(chunk.data(),
                                                              static_cast<std::size_t>(available));
        if (!lines.empty()) PushLines(lines);
    }
    SetConnectedState(false);
}

void CSimTransport::PushLines(const std::vector<std::string>& lines) {
    CSingleLock guard(&lock_, TRUE);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].empty()) continue;
        lines_.push_back(CString(lines[i].c_str()));
        while (static_cast<int>(lines_.size()) > kMaxQueuedLines) {
            lines_.pop_front();
            ++droppedLines_;
        }
    }
}

void CSimTransport::SetConnectedState(bool connected) {
    // Currently only used as the stop marker; later this can PostMessage a disconnect to the main window.
    running_ = connected && !stopRequested_;
}

int CSimTransport::Drain(std::vector<CString>& outLines, int maxCount) {
    CSingleLock guard(&lock_, TRUE);
    int taken = 0;
    while (!lines_.empty() && taken < maxCount) {
        outLines.push_back(lines_.front());
        lines_.pop_front();
        ++taken;
    }
    return taken;
}

bool CSimTransport::SendLine(LPCTSTR text, CString* error) {
    if (!running_ || text == nullptr) {
        if (error != nullptr) *error = _T("transport not connected");
        return false;
    }
    const CW2A ascii(text, CP_UTF8);  // protocol frames are ASCII CSV
    std::string payload(ascii.m_psz ? ascii.m_psz : "");
    if (payload.empty() || payload[payload.size() - 1] != '\n') payload += '\n';
    const char* data = payload.c_str();
    const int length = static_cast<int>(payload.size());

    if (kind_ == KindSerial) {
        DWORD written = 0;
        if (!::WriteFile(handle_, const_cast<char*>(data), length, &written, nullptr) ||
            static_cast<int>(written) != length) {
            if (error != nullptr) *error = LastErrorText(_T("serial send failed"));
            return false;
        }
        ::FlushFileBuffers(handle_);
        return true;
    }
    if (kind_ == KindTcp) {
        const int sent = ::send(static_cast<SOCKET>(socket_), data, length, 0);
        if (sent == SOCKET_ERROR) {
            if (error != nullptr) {
                error->Format(_T("TCP send failed (Winsock error %d)"), ::WSAGetLastError());
            }
            return false;
        }
        return true;
    }
    if (error != nullptr) *error = _T("unknown transport type");
    return false;
}
