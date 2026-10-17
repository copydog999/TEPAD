// Transport.h : data source layer - serial (Win32 COMM) and TCP client transports.
//
// A background worker thread keeps reading the byte stream, splits it into lines and pushes them into a thread-safe queue;
// the UI thread drains it with a timer through Drain(). The send direction is SendLine() (the UI "send box").
#pragma once

#include <deque>
#include <string>
#include <vector>

#include "core/FrameStream.h"

class CSimTransport {
public:
    enum Kind { KindNone = 0, KindSerial, KindTcp };

    CSimTransport();
    ~CSimTransport();

    /// port looks like "COM4" (the prefix may be omitted, "4" also works); common baud rates are 9600/115200.
    bool OpenSerial(LPCTSTR port, DWORD baud, CString* error);
    /// host looks like "127.0.0.1"; port is the TCP port (scripts/serial_bridge.py --tcp).
    bool OpenTcp(LPCTSTR host, UINT port, CString* error);
    void Close();

    bool IsOpen() const { return running_ ? true : false; }
    Kind kind() const { return kind_; }
    CString Endpoint() const { return endpoint_; }

    /// Detect the COM ports present on the system (tries CreateFile, scans up to COM32).
    static void EnumerateSerialPorts(CStringArray& ports);

    /// Called by the UI thread: take at most maxCount raw text lines.
    int Drain(std::vector<CString>& outLines, int maxCount);
    /// Called by the UI thread: send one line to the data source (\n is appended automatically).
    bool SendLine(LPCTSTR text, CString* error);

    /// Cumulative statistics for the status bar.
    unsigned long long ReceivedBytes() const { return receivedBytes_; }
    unsigned long long DroppedLines() const { return droppedLines_; }

    /// Worker thread entry point (internal use).
    static UINT WorkerThreadEntry(LPVOID param);

private:
    void WorkerLoop();
    void PushLines(const std::vector<std::string>& lines);
    void SetConnectedState(bool connected);
    /// Start the receiver thread after the handle/socket was opened successfully.
    bool StartWorker(CString* error);

    Kind kind_;
    HANDLE handle_;          ///< serial handle; INVALID_HANDLE_VALUE in TCP mode
    unsigned long long socket_;  ///< TCP socket (SOCKET handle value)
    volatile bool running_;
    volatile bool stopRequested_;
    CWinThread* worker_;
    HANDLE threadHandle_;    ///< duplicated worker thread handle, used for waits with a timeout
    CCriticalSection lock_;
    std::deque<CString> lines_;
    tepad::LineSplitter splitter_;
    CString endpoint_;
    volatile unsigned long long receivedBytes_;
    volatile unsigned long long droppedLines_;
};
