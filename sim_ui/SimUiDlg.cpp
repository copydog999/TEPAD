// SimUiDlg.cpp : main dialog - serial/TCP frame reception, dual-window detection, curves/log/send/calibration.
#include "framework.h"
#include "SimUiDlg.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace {

// Layout constants (pixels)
const int kMargin = 8;
const int kGap = 6;
const int kRowHeight = 24;
const int kInfoRowHeight = 20;

const COLORREF kColorG = RGB(30, 110, 210);
const COLORREF kColorP = RGB(30, 160, 90);
const COLORREF kColorScore = RGB(215, 90, 40);
const COLORREF kColorOk = RGB(20, 130, 60);
const COLORREF kColorWarn = RGB(190, 60, 40);
const COLORREF kColorText = RGB(50, 56, 66);

// Playback speed presets (ms/tick): the combo box and the SPEED command sent to the data bridge share this table,
// so even the slowest choice never blocks the serial/TCP send loop (the bridge changes its pacing as soon as it gets SPEED).
const int kSpeedMsChoices[] = {10, 25, 50, 100, 250, 500, 1000};
const int kSpeedChoiceCount =
    static_cast<int>(sizeof(kSpeedMsChoices) / sizeof(kSpeedMsChoices[0]));

// Curve rolling window presets (tick), one-to-one with the items of the curve span combo box.
const int kTickspanChoices[] = {120, 300, 720, 1440};
const int kTickspanChoiceCount =
    static_cast<int>(sizeof(kTickspanChoices) / sizeof(kTickspanChoices[0]));

CString ToCString(const std::string& text) {
    if (text.empty()) return CString();
    const int length = ::MultiByteToWideChar(CP_UTF8, 0, text.c_str(),
                                             static_cast<int>(text.size()), nullptr, 0);
    if (length <= 0) {
        return CString(text.c_str());  // invalid UTF-8: fall back to the ANSI interpretation
    }
    std::vector<wchar_t> buffer(static_cast<std::size_t>(length) + 1, 0);
    ::MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()),
                          buffer.data(), length);
    return CString(buffer.data(), length);
}

std::string ToUtf8(const CString& text) {
    if (text.IsEmpty()) return std::string();
    const CW2A utf8(text, CP_UTF8);
    return std::string(utf8.m_psz ? utf8.m_psz : "");
}

CString FromDouble(double value, int precision) {
    CString text;
    text.Format(_T("%.*f"), precision, value);
    return text;
}

CString FromLongLong(long long value) {
    CString text;
    text.Format(_T("%lld"), value);
    return text;
}

// Measure the text width (pixels) with the UI font. Chinese and English differ a lot in length ("Load calibration" vs "Load"),
// so every control width is measured; that keeps the re-layout after a language switch from overlapping.
int UiTextWidth(CWnd* owner, CFont* font, const CString& text) {
    if (owner == nullptr || text.IsEmpty()) return 0;
    CClientDC dc(owner);
    CFont* old = font != nullptr ? dc.SelectObject(font) : nullptr;
    const CSize size = dc.GetTextExtent(text);
    if (old != nullptr) dc.SelectObject(old);
    return size.cx;
}

// Recommended button/label width: text + left/right padding.
int CaptionWidth(CWnd* owner, CFont* font, const CString& caption, int minWidth) {
    return std::max(minWidth, UiTextWidth(owner, font, caption) + 16);
}

// Recommended combo width: the widest item, plus room for the drop-down arrow.
int ComboWidth(CWnd* owner, CFont* font, const CComboBox& combo, int minWidth) {
    int best = 0;
    for (int i = 0; i < combo.GetCount(); ++i) {
        CString item;
        combo.GetLBText(i, item);
        best = std::max(best, UiTextWidth(owner, font, item));
    }
    return std::max(minWidth, best + 26);
}

}  // namespace

CSimUiDlg::CSimUiDlg(CWnd* pParent)
    : CDialogEx(IDD_SIM_UI_DIALOG, pParent),
      m_frameCount(0),
      m_alarmCount(0),
      m_badLineCount(0),
      m_lastTick(-1),
      m_spikeFrames(0),
      m_shadeFrames(0),
      m_injectSigmaMultiple(8.0),
      m_injectElapsedFrames(0),
      m_injectAwaitingAlarm(false),
      m_injectKind(0),
      m_injectVerdict(0),
      m_lastG(0.0),
      m_lastP(0.0),
      m_lastPointZ(0.0),
      m_lastShort(0.0),
      m_lastLong(0.0),
      m_lastScore(0.0),
      m_lastAlarm(false),
      m_lastWindowsReady(false),
      m_linkPhase(0),
      m_calibrationOk(false),
      m_calibrationState(0),
      m_displayPoints(720),
      m_speedMs(kDefaultSpeedMs),
      m_lang(Lang::Zh),
      m_paused(false),
      m_detectorReady(false),
      m_uiReady(false),
      m_autoConnectTcp(false) {
    m_linkStateColor = kColorText;
    m_calibrationInfoColor = kColorText;
    m_injectInfoColor = kColorText;
}

void CSimUiDlg::DoDataExchange(CDataExchange* pDX) { CDialogEx::DoDataExchange(pDX); }

BEGIN_MESSAGE_MAP(CSimUiDlg, CDialogEx)
    ON_WM_TIMER()
    ON_WM_SIZE()
    ON_WM_CTLCOLOR()
    ON_BN_CLICKED(IDC_BTN_CONNECT, &CSimUiDlg::OnConnectClicked)
    ON_BN_CLICKED(IDC_BTN_LOAD_CALIBRATION, &CSimUiDlg::OnLoadCalibrationClicked)
    ON_BN_CLICKED(IDC_BTN_BROWSE_CALIBRATION, &CSimUiDlg::OnBrowseCalibrationClicked)
    ON_BN_CLICKED(IDC_BTN_SEND, &CSimUiDlg::OnSendClicked)
    ON_BN_CLICKED(IDC_BTN_INJECT_SPIKE, &CSimUiDlg::OnInjectSpikeClicked)
    ON_BN_CLICKED(IDC_BTN_INJECT_SHADE, &CSimUiDlg::OnInjectShadeClicked)
    ON_BN_CLICKED(IDC_BTN_CLEAR, &CSimUiDlg::OnClearClicked)
    ON_BN_CLICKED(IDC_BTN_EXPORT_LOG, &CSimUiDlg::OnExportLogClicked)
    ON_BN_CLICKED(IDC_CHECK_PAUSE, &CSimUiDlg::OnPauseToggled)
    ON_BN_CLICKED(IDC_BTN_LANGUAGE, &CSimUiDlg::OnLanguageClicked)
    ON_CBN_SELCHANGE(IDC_COMBO_TICKSPAN, &CSimUiDlg::OnTickspanChange)
    ON_CBN_SELCHANGE(IDC_COMBO_SPEED, &CSimUiDlg::OnSpeedChange)
    ON_CBN_SELCHANGE(IDC_COMBO_TRANSPORT, &CSimUiDlg::OnTransportChange)
END_MESSAGE_MAP()

// ---------------------------------------------------------------------------
// Initialization and layout
// ---------------------------------------------------------------------------

BOOL CSimUiDlg::OnInitDialog() {
    CDialogEx::OnInitDialog();
    // The title carries only the system name: algorithm details live in the README and curve titles, not the title bar.
    SetWindowText(Bi(_T("TEPAD Simulation Host"), _T("TEPAD Simulator")));

    ApplyUiFont();
    // Static text uses the dialog background brush; text colors are assigned per control ID in OnCtlColor.
    m_staticBrush.CreateSolidBrush(::GetSysColor(COLOR_3DFACE));
    CreateChildControls();
    // Re-layout is only allowed once the controls exist: the Relayout below is the first one that takes effect.
    m_uiReady = true;
    Relayout();

    m_calibrationPathEdit.SetWindowText(DefaultCalibrationPath());
    RefreshLinkText();
    SetStatusText(Bi(_T("waiting for data…"), _T("waiting for data…")));

    // Try to load the default calibration at startup; failure is only reported in the info row and does not block the UI.
    OnLoadCalibrationClicked();

    SetTimer(kPollTimer, static_cast<UINT>(kPollIntervalMs), nullptr);

    // Started with --auto (as the one-click script does): connect straight to the local data bridge over TCP, saving two clicks.
    if (m_autoConnectTcp) {
        m_transportCombo.SetCurSel(1);
        OnTransportChange();
        OnConnectClicked();
    }
    return TRUE;
}

void CSimUiDlg::ApplyUiFont() {
    if (m_uiFont.m_hObject != nullptr) return;
    LOGFONT lf;
    memset(&lf, 0, sizeof(lf));
    lf.lfHeight = -12;
    lf.lfCharSet = DEFAULT_CHARSET;
    lf.lfQuality = PROOF_QUALITY;
    lf.lfPitchAndFamily = VARIABLE_PITCH | FF_SWISS;
    _tcscpy_s(lf.lfFaceName, LF_FACESIZE, _T("Microsoft YaHei UI"));
    m_uiFont.CreateFontIndirect(&lf);
    SetFont(&m_uiFont, FALSE);
}

void CSimUiDlg::CreateChildControls() {
    const DWORD editStyle = WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_BORDER | ES_AUTOHSCROLL;
    const DWORD buttonStyle = WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_PUSHBUTTON;
    const DWORD staticStyle = WS_CHILD | WS_VISIBLE | SS_LEFTNOWORDWRAP;
    const DWORD listComboStyle = WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWNLIST | WS_VSCROLL;

    // All captions are applied by ApplyLanguage() at the end: nothing is set during creation, so Chinese and English have only
    // one source, and no control can be left with a stale caption after a language switch.
    m_transportCombo.Create(listComboStyle, CRect(0, 0, 100, kRowHeight), this,
                            IDC_COMBO_TRANSPORT);

    m_endpointCombo.Create(WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWN | WS_VSCROLL,
                           CRect(0, 0, 150, 220), this, IDC_EDIT_ENDPOINT);
    CStringArray serialPorts;
    CSimTransport::EnumerateSerialPorts(serialPorts);
    if (serialPorts.IsEmpty()) {
        // Without a USB serial adapter the common ports must still be selectable; the user can also type one.
        m_endpointCombo.AddString(_T("COM1"));
        m_endpointCombo.AddString(_T("COM3"));
        m_endpointCombo.AddString(_T("COM4"));
        m_endpointCombo.AddString(_T("COM5"));
    } else {
        for (int i = 0; i < serialPorts.GetCount(); ++i) m_endpointCombo.AddString(serialPorts[i]);
    }
    m_endpointCombo.SetWindowText(_T("COM4"));

    m_baudCombo.Create(WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWN | WS_VSCROLL,
                       CRect(0, 0, 90, 120), this, IDC_COMBO_BAUD);
    const LPCTSTR rates[] = {_T("9600"), _T("19200"), _T("38400"), _T("57600"), _T("115200"),
                             _T("230400"), _T("460800"), _T("921600")};
    for (std::size_t i = 0; i < sizeof(rates) / sizeof(rates[0]); ++i) {
        m_baudCombo.AddString(rates[i]);
    }
    m_baudCombo.SetCurSel(4);  // 115200; in TCP mode this box holds the port number

    m_connectButton.Create(_T(""), buttonStyle, CRect(0, 0, 72, kRowHeight), this,
                           IDC_BTN_CONNECT);
    m_linkState.Create(_T(""), staticStyle, CRect(0, 0, 240, kInfoRowHeight), this,
                       IDC_STATIC_LINK_STATE);

    m_calibrationPathEdit.Create(editStyle, CRect(0, 0, 300, kRowHeight), this,
                                 IDC_EDIT_CALIBRATION_PATH);
    m_browseCalibrationButton.Create(_T(""), buttonStyle, CRect(0, 0, 60, kRowHeight), this,
                                     IDC_BTN_BROWSE_CALIBRATION);
    m_loadCalibrationButton.Create(_T(""), buttonStyle, CRect(0, 0, 72, kRowHeight), this,
                                   IDC_BTN_LOAD_CALIBRATION);
    m_calibrationInfo.Create(_T(""), staticStyle,
                             CRect(0, 0, 480, kInfoRowHeight), this, IDC_STATIC_CALIBRATION_INFO);

    m_tickspanLabel.Create(_T(""), staticStyle, CRect(0, 0, 60, kInfoRowHeight), this,
                           IDC_STATIC_TICKSPAN_LABEL);
    m_tickspanCombo.Create(listComboStyle, CRect(0, 0, 96, 200), this, IDC_COMBO_TICKSPAN);
    for (int i = 0; i < kTickspanChoiceCount; ++i) {
        CString item;
        item.Format(_T("%d tick"), kTickspanChoices[i]);
        m_tickspanCombo.AddString(item);
    }
    m_tickspanCombo.SetCurSel(2);  // 720 tick

    // Playback speed: the selected millisecond value is sent to the data bridge with a SPEED command, which changes the frame interval immediately.
    m_speedLabel.Create(_T(""), staticStyle, CRect(0, 0, 60, kInfoRowHeight), this,
                        IDC_STATIC_SPEED_LABEL);
    m_speedCombo.Create(listComboStyle, CRect(0, 0, 96, 200), this, IDC_COMBO_SPEED);
    for (int i = 0; i < kSpeedChoiceCount; ++i) {
        CString item;
        item.Format(_T("%d ms/tick"), kSpeedMsChoices[i]);
        m_speedCombo.AddString(item);
        if (kSpeedMsChoices[i] == m_speedMs) m_speedCombo.SetCurSel(i);
    }
    if (m_speedCombo.GetCurSel() < 0) {
        m_speedCombo.SetCurSel(3);   // 100 ms/tick
        m_speedMs = kSpeedMsChoices[3];
    }
    m_languageButton.Create(_T(""), buttonStyle, CRect(0, 0, 72, kRowHeight), this,
                            IDC_BTN_LANGUAGE);

    m_pauseButton.Create(_T(""), WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX,
                         CRect(0, 0, 84, kRowHeight), this, IDC_CHECK_PAUSE);
    m_clearButton.Create(_T(""), buttonStyle, CRect(0, 0, 110, kRowHeight), this, IDC_BTN_CLEAR);
    m_exportButton.Create(_T(""), buttonStyle, CRect(0, 0, 110, kRowHeight), this,
                          IDC_BTN_EXPORT_LOG);

    // The two curve controls register their own window class; the parent only positions them.
    m_measureCurve.Create(this, IDC_STATIC_CURVE_MEASURE, CRect(0, 0, 320, 140));
    m_measureCurve.SetRange(0.0, 1.05);
    m_scoreCurve.Create(this, IDC_STATIC_CURVE_SCORE, CRect(0, 0, 320, 140));
    m_scoreCurve.SetRange(0.0, 0.0);  // vertical auto scaling

    m_alarmList.Create(WS_CHILD | WS_VISIBLE | WS_TABSTOP | LVS_REPORT | LVS_SHOWSELALWAYS |
                           WS_BORDER | LVS_NOSORTHEADER,
                       CRect(0, 0, 400, 120), this, IDC_LIST_ALARMS);
    m_alarmList.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
    const int widths[] = {126, 54, 58, 58, 62, 66, 66, 62, 168};
    for (int i = 0; i < static_cast<int>(sizeof(widths) / sizeof(widths[0])); ++i) {
        m_alarmList.InsertColumn(i, _T(""), LVCFMT_LEFT, widths[i]);
    }

    m_sendEdit.Create(editStyle, CRect(0, 0, 420, kRowHeight), this, IDC_EDIT_SEND);
    m_sendEdit.SetWindowText(_T("PING"));
    m_sendButton.Create(_T(""), buttonStyle, CRect(0, 0, 72, kRowHeight), this, IDC_BTN_SEND);

    // Manual injection row: a spike changes P for one frame (+N x sigma), shading applies P x 0.30 for N frames.
    m_injectLabel.Create(_T(""), staticStyle, CRect(0, 0, 60, kInfoRowHeight), this,
                         IDC_STATIC_INJECT_LABEL);
    m_injectSpikeSigmaEdit.Create(editStyle, CRect(0, 0, 48, kRowHeight), this,
                                  IDC_EDIT_INJECT_SPIKE_SIGMA);
    m_injectSpikeSigmaEdit.SetWindowText(_T("8"));
    m_injectSpikeButton.Create(_T(""), buttonStyle, CRect(0, 0, 108, kRowHeight), this,
                               IDC_BTN_INJECT_SPIKE);
    m_injectShadeTicksEdit.Create(editStyle, CRect(0, 0, 48, kRowHeight), this,
                                  IDC_EDIT_INJECT_SHADE_TICKS);
    m_injectShadeTicksEdit.SetWindowText(_T("60"));
    m_injectShadeButton.Create(_T(""), buttonStyle, CRect(0, 0, 124, kRowHeight), this,
                               IDC_BTN_INJECT_SHADE);
    m_injectInfo.Create(_T(""), staticStyle, CRect(0, 0, 420, kInfoRowHeight), this,
                        IDC_STATIC_INJECT_INFO);
    m_statsText.Create(_T(""), staticStyle, CRect(0, 0, 620, kInfoRowHeight), this,
                       IDC_STATIC_STATS);

    ApplyLanguage();
}

// Pick one caption set for the current language. Every human-readable string goes through here; technical symbols
// such as tick/score/G/P are not sentences and stay identical in both sets, so they are not in the translation table.
CString CSimUiDlg::Bi(LPCWSTR zh, LPCWSTR en) const {
    return CString(m_lang == Lang::Zh ? zh : en);
}

CString CSimUiDlg::AlarmHeader(int column) const {
    switch (column) {
    case 0: return Bi(_T("time"), _T("time"));
    case 1: return Bi(_T("tick"), _T("tick"));
    case 2: return Bi(_T("G"), _T("G"));
    case 3: return Bi(_T("P"), _T("P"));
    case 4: return Bi(_T("point_z"), _T("point_z"));
    case 5: return Bi(_T("ms(L10)"), _T("ms(L10)"));
    case 6: return Bi(_T("ml(L60)"), _T("ml(L60)"));
    case 7: return Bi(_T("score"), _T("score"));
    default: return Bi(_T("channels"), _T("channels"));
    }
}

// Language switch: only text and control widths change; no state is touched (data, alarms and injection readings are kept).
void CSimUiDlg::ApplyLanguage() {
    SetWindowText(Bi(_T("TEPAD Simulation Host"), _T("TEPAD Simulator")));

    const int transportSel = m_transportCombo.GetCurSel();
    m_transportCombo.ResetContent();
    m_transportCombo.AddString(Bi(_T("Serial COM"), _T("Serial COM")));
    m_transportCombo.AddString(Bi(_T("TCP (bridge)"), _T("TCP (bridge)")));
    m_transportCombo.SetCurSel(transportSel < 0 ? 0 : transportSel);

    m_connectButton.SetWindowText(m_transport.IsOpen()
                                      ? Bi(_T("Disconnect"), _T("Disconnect"))
                                      : Bi(_T("Connect"), _T("Connect")));
    m_browseCalibrationButton.SetWindowText(Bi(_T("Browse…"), _T("Browse…")));
    m_loadCalibrationButton.SetWindowText(Bi(_T("Load calibration"), _T("Load calibration")));
    m_tickspanLabel.SetWindowText(Bi(_T("Curve span"), _T("Curve span")));
    m_speedLabel.SetWindowText(Bi(_T("Playback"), _T("Playback")));
    // The button shows the language you will switch to, so it is obvious what a click will do.
    m_languageButton.SetWindowText(Bi(_T("English"), _T("Chinese")));
    m_pauseButton.SetWindowText(Bi(_T("Pause"), _T("Pause")));
    m_clearButton.SetWindowText(Bi(_T("Clear curves/log"), _T("Clear curves/log")));
    m_exportButton.SetWindowText(Bi(_T("Export alarm log"), _T("Export alarm log")));
    m_sendButton.SetWindowText(Bi(_T("Send"), _T("Send")));
    m_injectLabel.SetWindowText(Bi(_T("Inject"), _T("Inject")));
    m_injectSpikeButton.SetWindowText(Bi(_T("Inject spike (×σ)"), _T("Inject spike (×σ)")));
    m_injectShadeButton.SetWindowText(Bi(_T("Inject shade (×0.3)"), _T("Inject shade (×0.3)")));

    const int columns = m_alarmList.GetHeaderCtrl() == nullptr
                            ? 0
                            : m_alarmList.GetHeaderCtrl()->GetItemCount();
    for (int i = 0; i < columns; ++i) {
        CString header = AlarmHeader(i);
        LVCOLUMN lvc;
        memset(&lvc, 0, sizeof(lvc));
        lvc.mask = LVCF_TEXT;
        lvc.pszText = const_cast<LPTSTR>(static_cast<LPCTSTR>(header));
        m_alarmList.SetColumn(i, &lvc);
    }

    m_measureCurve.SetTitle(Bi(_T("Live curves: G irradiance / P power"),
                               _T("Live curves: G irradiance / P power")));
    m_scoreCurve.SetTitle(Bi(_T("Detection score: max(z/τz, ms/τ10, ml/τ60)"),
                             _T("Detection score: max(z/τz, ms/τ10, ml/τ60)")));
    m_scoreCurve.SetThreshold(1.0, Bi(_T("alarm threshold score=1"), _T("alarm threshold score=1")));
    RefreshCurves();

    RefreshLinkText();
    RefreshCalibrationText();
    RefreshInjectText();
    RefreshStatsText();
}


void CSimUiDlg::Relayout() {
    // WM_SIZE arrives while the window is still being created, before the child controls exist; re-laying out early would
    // send messages to controls without an HWND (Debug MFC asserts), so wait until the UI is ready.
    if (!m_uiReady) return;
    CRect client;
    GetClientRect(&client);
    if (client.Width() < 360 || client.Height() < 300) return;

    CFont* font = m_uiFont.m_hObject != nullptr ? &m_uiFont : nullptr;
    const int rowWidth = client.Width() - 2 * kMargin;
    int x = kMargin;
    int y = kMargin;

    CString caption;

    // Row 1: data source + connect + state (combo height includes the expanded list)
    const int transportWidth = ComboWidth(this, font, m_transportCombo, 92);
    m_transportCombo.MoveWindow(x, y, transportWidth, 200);
    x += transportWidth + kGap;
    m_endpointCombo.MoveWindow(x, y, 150, 220);
    x += 150 + kGap;
    m_baudCombo.MoveWindow(x, y, 92, 220);
    x += 92 + kGap;
    m_connectButton.GetWindowText(caption);
    const int connectWidth = CaptionWidth(this, font, caption, 60);
    m_connectButton.MoveWindow(x, y, connectWidth, kRowHeight);
    x += connectWidth + kGap;
    m_linkState.MoveWindow(x, y + 3, std::max(80, rowWidth - (x - kMargin)), kInfoRowHeight);

    // Row 2: calibration path + browse + load; row 3: calibration summary
    y += kRowHeight + kGap;
    x = kMargin;
    m_loadCalibrationButton.GetWindowText(caption);
    const int loadWidth = CaptionWidth(this, font, caption, 60);
    m_browseCalibrationButton.GetWindowText(caption);
    const int browseWidth = CaptionWidth(this, font, caption, 54);
    const int pathWidth = std::max(120, rowWidth - loadWidth - browseWidth - 2 * kGap);
    m_calibrationPathEdit.MoveWindow(x, y, pathWidth, kRowHeight - 2);
    x += pathWidth + kGap;
    m_browseCalibrationButton.MoveWindow(x, y, browseWidth, kRowHeight);
    x += browseWidth + kGap;
    m_loadCalibrationButton.MoveWindow(x, y, loadWidth, kRowHeight);
    y += kRowHeight + 2;
    m_calibrationInfo.MoveWindow(kMargin, y, rowWidth, kInfoRowHeight);

    // Row 4: curve span + playback speed + pause + clear + export
    y += kInfoRowHeight + kGap;
    x = kMargin;
    m_tickspanLabel.GetWindowText(caption);
    const int tickspanLabelWidth = UiTextWidth(this, font, caption) + 4;
    m_tickspanLabel.MoveWindow(x, y + 3, tickspanLabelWidth, kInfoRowHeight);
    x += tickspanLabelWidth + kGap;
    const int tickspanWidth = ComboWidth(this, font, m_tickspanCombo, 76);
    m_tickspanCombo.MoveWindow(x, y, tickspanWidth, 200);
    x += tickspanWidth + kGap;
    m_speedLabel.GetWindowText(caption);
    const int speedLabelWidth = UiTextWidth(this, font, caption) + 4;
    m_speedLabel.MoveWindow(x, y + 3, speedLabelWidth, kInfoRowHeight);
    x += speedLabelWidth + kGap;
    const int speedWidth = ComboWidth(this, font, m_speedCombo, 76);
    m_speedCombo.MoveWindow(x, y, speedWidth, 200);
    x += speedWidth + kGap;
    m_pauseButton.GetWindowText(caption);
    const int pauseWidth = CaptionWidth(this, font, caption, 60);
    m_pauseButton.MoveWindow(x, y, pauseWidth, kRowHeight);
    x += pauseWidth + kGap;
    m_clearButton.GetWindowText(caption);
    const int clearWidth = CaptionWidth(this, font, caption, 54);
    m_clearButton.MoveWindow(x, y, clearWidth, kRowHeight);
    x += clearWidth + kGap;
    m_exportButton.GetWindowText(caption);
    const int exportWidth = CaptionWidth(this, font, caption, 60);
    m_exportButton.MoveWindow(x, y, exportWidth, kRowHeight);

    // Row 5: manual fault injection (spike multiple / shading frames) + alarm latency reading on the right
    y += kRowHeight + kGap;
    x = kMargin;
    m_injectLabel.GetWindowText(caption);
    const int injectLabelWidth = UiTextWidth(this, font, caption) + 4;
    m_injectLabel.MoveWindow(x, y + 3, injectLabelWidth, kInfoRowHeight);
    x += injectLabelWidth + kGap;
    m_injectSpikeSigmaEdit.MoveWindow(x, y, 48, kRowHeight - 2);
    x += 48 + kGap;
    m_injectSpikeButton.GetWindowText(caption);
    const int spikeWidth = CaptionWidth(this, font, caption, 80);
    m_injectSpikeButton.MoveWindow(x, y, spikeWidth, kRowHeight);
    x += spikeWidth + kGap;
    m_injectShadeTicksEdit.MoveWindow(x, y, 48, kRowHeight - 2);
    x += 48 + kGap;
    m_injectShadeButton.GetWindowText(caption);
    const int shadeWidth = CaptionWidth(this, font, caption, 92);
    m_injectShadeButton.MoveWindow(x, y, shadeWidth, kRowHeight);
    x += shadeWidth + kGap;
    m_injectInfo.MoveWindow(x, y + 3,
                            std::max(140, static_cast<int>(client.right - kMargin - x)),
                            kInfoRowHeight);

    // Bottom: send row (lowest, with the language button on the right) + status row + alarm log header
    const int sendTop = client.bottom - kMargin - kRowHeight;
    m_languageButton.GetWindowText(caption);
    const int languageWidth = CaptionWidth(this, font, caption, 56);
    m_sendButton.GetWindowText(caption);
    const int sendWidth = CaptionWidth(this, font, caption, 56);
    m_languageButton.MoveWindow(client.right - kMargin - languageWidth, sendTop, languageWidth,
                                kRowHeight);
    m_sendButton.MoveWindow(client.right - kMargin - languageWidth - kGap - sendWidth, sendTop,
                            sendWidth, kRowHeight);
    const int editWidth = rowWidth - sendWidth - languageWidth - 2 * kGap;
    m_sendEdit.MoveWindow(kMargin, sendTop + 1, std::max(120, editWidth), kRowHeight - 2);
    const int statsTop = sendTop - kGap - kInfoRowHeight;
    m_statsText.MoveWindow(kMargin, statsTop, rowWidth, kInfoRowHeight);
    const int logTop = statsTop - kGap - kInfoRowHeight;

    // Middle: the alarm log takes a fixed height at the bottom, everything above it goes to the two curves
    const int clientMidTop = y + kRowHeight + kGap;
    int logHeight = logTop - clientMidTop;
    logHeight = std::min(std::max(logHeight, 90), 260);
    const int logAreaTop = logTop - logHeight;
    const int curvesTop = clientMidTop;
    const int curvesHeight = std::max(120, logAreaTop - curvesTop - 2);
    const int halfWidth = std::max(200, (rowWidth - kGap) / 2);
    m_measureCurve.MoveWindow(kMargin, curvesTop, halfWidth, curvesHeight);
    m_scoreCurve.MoveWindow(kMargin + halfWidth + kGap, curvesTop,
                            std::max(200, rowWidth - halfWidth - kGap), curvesHeight);
    m_alarmList.MoveWindow(kMargin, logAreaTop, rowWidth, logHeight);

    m_measureCurve.Refresh();
    m_scoreCurve.Refresh();
}

void CSimUiDlg::SetStatusText(const CString& text) {
    if (m_statsText.GetSafeHwnd()) m_statsText.SetWindowText(text);
}

void CSimUiDlg::SetLinkStateText(const CString& text, COLORREF color) {
    if (!m_linkState.GetSafeHwnd()) return;
    m_linkStateColor = color;
    m_linkState.SetWindowText(text);
    m_linkState.Invalidate();
}

void CSimUiDlg::SetCalibrationInfoText(const CString& text, COLORREF color) {
    if (!m_calibrationInfo.GetSafeHwnd()) return;
    m_calibrationInfoColor = color;
    m_calibrationInfo.SetWindowText(text);
    m_calibrationInfo.Invalidate();
}

// ---------------------------------------------------------------------------
// Dynamic text: the state stores only numbers/raw values and the sentence is built for the current language.
// A single click on the language button therefore re-renders existing readings without injecting again.
// ---------------------------------------------------------------------------

void CSimUiDlg::RefreshLinkText() {
    CString text;
    COLORREF color = kColorText;
    switch (m_linkPhase) {
    case 1:
        text = Bi(_T("connected "), _T("connected ")) + m_linkDetail;
        color = kColorOk;
        break;
    case 2:
        text = Bi(_T("connect failed: "), _T("connect failed: ")) + m_linkDetail;
        color = kColorWarn;
        break;
    case 3:
        text = Bi(_T("disconnected"), _T("disconnected"));
        color = kColorWarn;
        break;
    case 4:
        text = Bi(_T("link closed (peer shut down or reader exited)"),
                  _T("link closed (peer shut down or reader exited)"));
        color = kColorWarn;
        break;
    case 5:
        text = Bi(_T("disconnect first, then switch source"),
                  _T("disconnect first, then switch source"));
        color = kColorWarn;
        break;
    default:
        text = Bi(_T("offline"), _T("offline"));
        color = kColorWarn;
        break;
    }
    SetLinkStateText(text, color);
}

void CSimUiDlg::RefreshCalibrationText() {
    CString text;
    COLORREF color = kColorText;
    switch (m_calibrationState) {
    case 1:
        text = Bi(_T("fill in a calibration JSON path first"), _T("fill in a calibration JSON path first"));
        color = kColorWarn;
        break;
    case 2:
        text = Bi(_T("calibration load failed: "), _T("calibration load failed: ")) + m_calibrationDetail;
        color = kColorWarn;
        break;
    case 3: {
        // Both the path and the summary are language-neutral raw text; only the "loaded" prefix is localized.
        CString path;
        if (m_calibrationPathEdit.GetSafeHwnd()) m_calibrationPathEdit.GetWindowText(path);
        text = Bi(_T("loaded "), _T("loaded ")) + path + Bi(_T(" | "), _T(" | ")) +
               m_calibrationDetail;
        color = kColorOk;
        break;
    }
    default:
        text = Bi(_T("calibration not loaded"), _T("calibration not loaded"));
        break;
    }
    SetCalibrationInfoText(text, color);
}

void CSimUiDlg::RefreshInjectText() {
    CString text;
    COLORREF color = kColorText;
    if (m_injectKind == 0) {
        text = Bi(_T("no fault injected; use the buttons above to see how fast it alarms"),
                  _T("no fault injected; use the buttons above to see how fast it alarms"));
    } else {
        const CString kind = m_injectKind == 1 ? Bi(_T("spike"), _T("spike"))
                                               : Bi(_T("shading"), _T("shading"));
        const CString frames = FromLongLong(m_injectElapsedFrames);
        switch (m_injectVerdict) {
        case 2:
            text.Format(Bi(_T("%s injected -> alarm after %s frames (~%s min), see alarm log"),
                           _T("%s injected -> alarm after %s frames (~%s min), see alarm log")),
                        static_cast<LPCTSTR>(kind), static_cast<LPCTSTR>(frames),
                        static_cast<LPCTSTR>(frames));
            color = kColorWarn;
            break;
        case 3:
            text.Format(Bi(_T("%s injected -> %s frames passed, still no alarm (deadline hit)"),
                           _T("%s injected -> %s frames passed, still no alarm (deadline hit)")),
                        static_cast<LPCTSTR>(kind), static_cast<LPCTSTR>(frames));
            break;
        default:
            // The in-progress text must not contain "no alarm"/"alarm after", otherwise the automation script
            // would misread "still waiting" as a final verdict (see selftest\ui_smoke.ps1).
            text.Format(Bi(_T("%s injected -> waited %s frames, watching for the first alarm"),
                           _T("%s injected -> waited %s frames, watching for the first alarm")),
                        static_cast<LPCTSTR>(kind), static_cast<LPCTSTR>(frames));
            color = kColorWarn;
            break;
        }
    }
    SetInjectInfoText(text, color);
}

void CSimUiDlg::RefreshStatsText() {
    CString stats;
    if (m_lang == Lang::Zh) {
        stats.Format(_T("frames %s | alarms %s | bad lines %s | tick %s | G %s | P %s"),
                     static_cast<LPCTSTR>(FromLongLong(m_frameCount)),
                     static_cast<LPCTSTR>(FromLongLong(m_alarmCount)),
                     static_cast<LPCTSTR>(FromLongLong(m_badLineCount)),
                     static_cast<LPCTSTR>(FromLongLong(m_lastTick)),
                     static_cast<LPCTSTR>(FromDouble(m_lastG, 4)),
                     static_cast<LPCTSTR>(FromDouble(m_lastP, 4)));
    } else {
        stats.Format(_T("frames %s | alarms %s | bad lines %s | tick %s | G %s | P %s"),
                     static_cast<LPCTSTR>(FromLongLong(m_frameCount)),
                     static_cast<LPCTSTR>(FromLongLong(m_alarmCount)),
                     static_cast<LPCTSTR>(FromLongLong(m_badLineCount)),
                     static_cast<LPCTSTR>(FromLongLong(m_lastTick)),
                     static_cast<LPCTSTR>(FromDouble(m_lastG, 4)),
                     static_cast<LPCTSTR>(FromDouble(m_lastP, 4)));
    }
    if (m_detector && m_detector->Usable()) {
        CString tail;
        const CString mark = m_lastAlarm
                                 ? Bi(_T("  ★ALARM"), _T("  ★ALARM"))
                                 : (m_lastWindowsReady ? CString(_T(""))
                                                      : Bi(_T("  (windows warming up)"),
                                                           _T("  (windows warming up)")));
        if (m_lang == Lang::Zh) {
            tail.Format(_T(" | z %s | ms %s | ml %s | score %s%s"),
                        static_cast<LPCTSTR>(FromDouble(m_lastPointZ, 3)),
                        static_cast<LPCTSTR>(FromDouble(m_lastShort, 3)),
                        static_cast<LPCTSTR>(FromDouble(m_lastLong, 3)),
                        static_cast<LPCTSTR>(FromDouble(m_lastScore, 3)),
                        static_cast<LPCTSTR>(mark));
        } else {
            tail.Format(_T(" | z %s | ms %s | ml %s | score %s%s"),
                        static_cast<LPCTSTR>(FromDouble(m_lastPointZ, 3)),
                        static_cast<LPCTSTR>(FromDouble(m_lastShort, 3)),
                        static_cast<LPCTSTR>(FromDouble(m_lastLong, 3)),
                        static_cast<LPCTSTR>(FromDouble(m_lastScore, 3)),
                        static_cast<LPCTSTR>(mark));
        }
        stats += tail;
    }
    SetStatusText(stats);
}

// CStatic has no SetTextColor, so colors are applied per control ID in WM_CTLCOLORSTATIC.
HBRUSH CSimUiDlg::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor) {
    HBRUSH brush = CDialogEx::OnCtlColor(pDC, pWnd, nCtlColor);
    if (nCtlColor == CTLCOLOR_STATIC && pWnd != nullptr) {
        pDC->SetBkMode(TRANSPARENT);
        const UINT id = static_cast<UINT>(pWnd->GetDlgCtrlID());
        if (id == IDC_STATIC_LINK_STATE) pDC->SetTextColor(m_linkStateColor);
        else if (id == IDC_STATIC_CALIBRATION_INFO) pDC->SetTextColor(m_calibrationInfoColor);
        else if (id == IDC_STATIC_INJECT_INFO) pDC->SetTextColor(m_injectInfoColor);
        else pDC->SetTextColor(kColorText);
        if (m_staticBrush.GetSafeHandle() != nullptr) brush = static_cast<HBRUSH>(m_staticBrush.GetSafeHandle());
    }
    return brush;
}

CString CSimUiDlg::DefaultCalibrationPath() const {
    // Walk upward from the directory of the exe looking for calibration\dual_window_calibration.json.
    // The upward walk is required: in development the binary lives in sim_ui\build\x64\Release\, three levels below
    // sim_ui\calibration\, so a hard-coded two-level candidate list would miss every time.
    TCHAR modulePath[MAX_PATH] = {0};
    ::GetModuleFileName(nullptr, modulePath, MAX_PATH);
    CString directory = modulePath;
    const int slash = directory.ReverseFind(_T('\\'));
    if (slash >= 0) directory = directory.Left(slash + 1);

    CString probe = directory;
    for (int level = 0; level < 6; ++level) {
        const CString candidate = probe + _T("calibration\\dual_window_calibration.json");
        if (::GetFileAttributes(candidate) != INVALID_FILE_ATTRIBUTES) return candidate;
        probe.TrimRight(_T("\\/"));
        const int cut = probe.ReverseFind(_T('\\'));
        if (cut < 0) break;   // reached the root
        probe = probe.Left(cut + 1);
    }
    // If all probes fail, fall back to "calibration next to the exe" (PostBuild copies one there),
    // and the user can still edit the path box or pick another file with "Browse...".
    return directory + _T("calibration\\dual_window_calibration.json");
}

// ---------------------------------------------------------------------------
// Calibration loading
// ---------------------------------------------------------------------------

void CSimUiDlg::RestartDetector() {
    if (!m_calibration.Usable()) {
        m_detector.reset();
        m_detectorReady = false;
        return;
    }
    m_detector.reset(new tepad::DualWindowDetector(m_calibration));
    m_detectorReady = true;
}

void CSimUiDlg::OnLoadCalibrationClicked() {
    CString path;
    if (m_calibrationPathEdit.GetSafeHwnd()) m_calibrationPathEdit.GetWindowText(path);
    path.Trim();
    if (path.IsEmpty()) {
        m_calibration = tepad::Calibration();
        RestartDetector();
        m_calibrationOk = false;
        m_calibrationState = 1;
        m_calibrationDetail.Empty();
        RefreshCalibrationText();
        return;
    }

    std::string error;
    tepad::Calibration loaded;
    if (!tepad::LoadCalibrationFile(ToUtf8(path), &loaded, &error)) {
        m_calibration = tepad::Calibration();
        RestartDetector();
        m_calibrationOk = false;
        m_calibrationState = 2;
        // The core-layer error text (shared with selftest) is not translated; it is shown with the localized prefix.
        m_calibrationDetail = ToCString(error);
        RefreshCalibrationText();
        SetStatusText(Bi(_T("no calibration: curves only, no score or alarm"),
                         _T("no calibration: curves only, no score or alarm")));
        return;
    }

    m_calibration = loaded;
    RestartDetector();
    m_calibrationOk = m_calibration.Usable();
    m_calibrationState = 3;
    m_calibrationDetail = ToCString(m_calibration.summary);
    RefreshCalibrationText();
    RefreshCurves();
}

void CSimUiDlg::OnBrowseCalibrationClicked() {
    const CString filter =
        Bi(_T("TEPAD calibration (*.json)|*.json|All files (*.*)|*.*||"),
           _T("TEPAD calibration (*.json)|*.json|All files (*.*)|*.*||"));
    CFileDialog dialog(FALSE, _T("json"), nullptr,
                       OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_PATHMUSTEXIST,
                       filter, this);
    if (dialog.DoModal() != IDOK) return;
    m_calibrationPathEdit.SetWindowText(dialog.GetPathName());
    OnLoadCalibrationClicked();
}

// ---------------------------------------------------------------------------
// Connection control
// ---------------------------------------------------------------------------

void CSimUiDlg::OnConnectClicked() {
    if (m_transport.IsOpen()) {
        m_transport.Close();
        m_connectButton.SetWindowText(Bi(_T("Connect"), _T("Connect")));
        m_linkPhase = 3;
        m_linkDetail.Empty();
        RefreshLinkText();
        return;
    }

    CString endpoint;
    m_endpointCombo.GetWindowText(endpoint);
    endpoint.Trim();
    CString parameter;
    m_baudCombo.GetWindowText(parameter);
    parameter.Trim();
    const int number = _tstoi(parameter);

    CString error;
    bool opened = false;
    if (m_transportCombo.GetCurSel() == 1) {
        // TCP: the address box holds the host, the parameter box holds the port (serial_bridge.py --tcp defaults to 18801)
        const UINT tcpPort = (number >= 1 && number <= 65535) ? static_cast<UINT>(number) : 18801;
        opened = m_transport.OpenTcp(endpoint.IsEmpty() ? _T("127.0.0.1") : LPCTSTR(endpoint),
                                     tcpPort, &error);
    } else {
        const DWORD baud = number > 0 ? static_cast<DWORD>(number) : CBR_115200;
        opened = m_transport.OpenSerial(endpoint.IsEmpty() ? _T("COM1") : LPCTSTR(endpoint), baud,
                                        &error);
    }

    if (!opened) {
        m_linkPhase = 2;
        m_linkDetail = error;
        RefreshLinkText();
        return;
    }
    m_connectButton.SetWindowText(Bi(_T("Disconnect"), _T("Disconnect")));
    m_linkPhase = 1;
    m_linkDetail = m_transport.Endpoint();
    RefreshLinkText();
    // As soon as the link is up, push the playback speed chosen in the UI; otherwise the bridge keeps its own startup pacing.
    SendSpeedCommand();
    if (!m_detectorReady) {
        SetStatusText(Bi(_T("connected, but no calibration loaded: curves only"),
                         _T("connected, but no calibration loaded: curves only")));
    }
}

void CSimUiDlg::OnTransportChange() {
    // Switching the data source syncs the defaults of the "address/parameter" boxes: COM port and baud for serial, address and port for TCP.
    if (m_transport.IsOpen()) {
        m_linkPhase = 5;
        m_linkDetail.Empty();
        RefreshLinkText();
        return;
    }
    if (m_transportCombo.GetCurSel() == 1) {
        m_endpointCombo.SetWindowText(_T("127.0.0.1"));
        m_baudCombo.SetWindowText(_T("18801"));
        SetStatusText(Bi(_T("TCP mode: run python scripts/serial_bridge.py --tcp, then Connect"),
                         _T("TCP mode: run python scripts/serial_bridge.py --tcp, then Connect")));
        return;
    }
    CString firstPort;
    if (m_endpointCombo.GetCount() > 0) m_endpointCombo.GetLBText(0, firstPort);
    m_endpointCombo.SetWindowText(firstPort.IsEmpty() ? _T("COM4") : LPCTSTR(firstPort));
    m_baudCombo.SetWindowText(_T("115200"));
    SetStatusText(Bi(_T("Serial mode: run python scripts/serial_bridge.py --port COM3, then Connect"),
                     _T("Serial mode: run python scripts/serial_bridge.py --port COM3, then Connect")));
}

// ---------------------------------------------------------------------------
// Manual fault injection
// ---------------------------------------------------------------------------

double CSimUiDlg::InjectResidualSigma() const {
    // The spike magnitude is based on the calibrated point-residual sigma; without calibration it degrades to 0.02 (same order).
    return m_calibration.residualSigma > 0.0 ? m_calibration.residualSigma : 0.02;
}

void CSimUiDlg::SetInjectInfoText(const CString& text, COLORREF color) {
    if (!m_injectInfo.GetSafeHwnd()) return;
    m_injectInfoColor = color;
    m_injectInfo.SetWindowText(text);
    m_injectInfo.Invalidate();
}

void CSimUiDlg::BeginInjection(bool spike, int frames, double sigmaMultiple) {
    m_injectSigmaMultiple = sigmaMultiple;
    m_injectElapsedFrames = 0;
    m_injectAwaitingAlarm = true;
    m_injectKind = spike ? 1 : 2;
    m_injectVerdict = 1;
    if (spike) {
        m_spikeFrames = frames;
        m_shadeFrames = 0;
    } else {
        m_shadeFrames = frames;
        m_spikeFrames = 0;
    }
    // A single-point spike is only 1 pixel wide in a 720 tick window, so injection narrows the curve to 120 tick,
    // letting that one step stand out (the user can still switch back to a longer window afterwards).
    if (m_displayPoints > kNarrowSpanTicks) {
        for (int i = 0; i < kTickspanChoiceCount; ++i) {
            if (kTickspanChoices[i] == kNarrowSpanTicks) {
                m_tickspanCombo.SetCurSel(i);
                break;
            }
        }
        m_displayPoints = kNarrowSpanTicks;
        TrimSeries();
        RefreshCurves();
    }
    RefreshInjectText();
    if (!m_transport.IsOpen()) {
        SetStatusText(Bi(_T("injection queued, but no data is flowing: run scripts\\run_sim.bat or press Connect"),
                         _T("injection queued, but no data is flowing: run scripts\\run_sim.bat or press Connect")));
    } else if (!m_detectorReady) {
        SetStatusText(Bi(_T("injection armed, but no calibration: no alarm will fire, press Load calibration"),
                         _T("injection armed, but no calibration: no alarm will fire, press Load calibration")));
    }
}

void CSimUiDlg::OnInjectSpikeClicked() {
    double multiple = kDefaultSpikeSigma;
    CString text;
    if (m_injectSpikeSigmaEdit.GetSafeHwnd()) {
        m_injectSpikeSigmaEdit.GetWindowText(text);
        const double parsed = _ttof(text);
        if (parsed > 0.0) multiple = parsed;
    }
    const bool narrowed = m_displayPoints > kNarrowSpanTicks;
    BeginInjection(true, 1, multiple);
    CString status;
    if (m_lang == Lang::Zh) {
        status.Format(_T("spike ready: next frame P += %.4f (%.1fx sigma)%s"),
                      multiple * InjectResidualSigma(), multiple,
                      narrowed ? _T("; curve window narrowed to 120 tick") : _T(""));
    } else {
        status.Format(_T("spike ready: next frame P += %.4f (%.1fx sigma)%s"),
                      multiple * InjectResidualSigma(), multiple,
                      narrowed ? _T("; curve window narrowed to 120 tick") : _T(""));
    }
    SetStatusText(status);
}

void CSimUiDlg::OnInjectShadeClicked() {
    int ticks = kDefaultShadeTicks;
    CString text;
    if (m_injectShadeTicksEdit.GetSafeHwnd()) {
        m_injectShadeTicksEdit.GetWindowText(text);
        const int parsed = _tstoi(text);
        if (parsed > 0) ticks = parsed;
    }
    const bool narrowed = m_displayPoints > kNarrowSpanTicks;
    BeginInjection(false, ticks, 0.0);
    CString status;
    if (m_lang == Lang::Zh) {
        status.Format(_T("shading ready: next %d frames P×%.2f%s"), ticks, kShadeFactor,
                      narrowed ? _T("; curve window narrowed to 120 tick") : _T(""));
    } else {
        status.Format(_T("shading ready: next %d frames P×%.2f%s"), ticks, kShadeFactor,
                      narrowed ? _T("; curve window narrowed to 120 tick") : _T(""));
    }
    SetStatusText(status);
}

// Playback speed: the combo only updates m_speedMs and sends a SPEED command; the actual pacing changes on the bridge side.
void CSimUiDlg::OnSpeedChange() {
    const int selection = m_speedCombo.GetCurSel();
    if (selection < 0 || selection >= kSpeedChoiceCount) return;
    m_speedMs = kSpeedMsChoices[selection];
    SendSpeedCommand();
}

void CSimUiDlg::SendSpeedCommand() {
    if (!m_transport.IsOpen()) return;   // Remembered while offline; OnConnectClicked re-sends it after connecting
    CString command;
    command.Format(_T("SPEED %.3f"), static_cast<double>(m_speedMs) / 1000.0);
    CString error;
    if (m_transport.SendLine(command, &error)) {
        SetStatusText(Bi(_T("playback speed sent: "), _T("playback speed sent: ")) +
                      FromLongLong(m_speedMs) + Bi(_T(" ms/tick"), _T(" ms/tick")));
    } else {
        SetStatusText(Bi(_T("speed command failed: "), _T("speed command failed: ")) + error);
    }
}

void CSimUiDlg::OnLanguageClicked() {
    m_lang = (m_lang == Lang::Zh) ? Lang::En : Lang::Zh;
    ApplyLanguage();
    Relayout();   // Chinese and English differ in length, so re-layout to restore the control widths
}

// ---------------------------------------------------------------------------
// Data processing
// ---------------------------------------------------------------------------

void CSimUiDlg::OnTimer(UINT_PTR timerId) {
    if (timerId == kPollTimer) {
        PollTransport();
        return;
    }
    CDialogEx::OnTimer(timerId);
}

void CSimUiDlg::PollTransport() {
    // When the peer closes, the receiver thread exits on its own and running_ becomes false; sync the button and messages here.
    if (!m_transport.IsOpen()) {
        if (m_connectButton.GetSafeHwnd()) {
            CString caption;
            m_connectButton.GetWindowText(caption);
            if (caption == Bi(_T("Disconnect"), _T("Disconnect"))) {
                m_connectButton.SetWindowText(Bi(_T("Connect"), _T("Connect")));
                m_linkPhase = 4;
                m_linkDetail.Empty();
                RefreshLinkText();
            }
        }
        return;
    }

    std::vector<CString> lines;
    const int taken = m_transport.Drain(lines, 256);
    bool changed = false;
    for (int i = 0; i < taken; ++i) {
        const tepad::Frame frame = tepad::ParseFrameLine(ToUtf8(lines[i]));
        if (!frame.valid) {
            ++m_badLineCount;
            continue;
        }
        if (m_paused) continue;
        HandleFrame(frame);
        changed = true;
    }
    if (changed) {
        TrimSeries();
        RefreshCurves();
    }
}

void CSimUiDlg::HandleFrame(const tepad::Frame& frame) {
    ++m_frameCount;
    m_lastTick = frame.tick;

    // Manual injection: a spike changes P for one frame (like inject_spike), shading applies P x factor for N frames (like inject_shading).
    // Injection happens before the detector, so curves, score and alarm log all see the injected data.
    tepad::Frame adjusted = frame;
    int marker = 0;
    if (m_spikeFrames > 0) {
        adjusted.p += m_injectSigmaMultiple * InjectResidualSigma();
        --m_spikeFrames;
        marker = 1;
    } else if (m_shadeFrames > 0) {
        adjusted.p *= kShadeFactor;
        --m_shadeFrames;
        marker = 2;
    }

    m_lastG = adjusted.g;
    m_lastP = adjusted.p;
    m_seriesG.push_back(adjusted.g);
    m_seriesP.push_back(adjusted.p);
    // The marker series has the same length and trimming as the curves; CurveCtrl uses it to draw spike lines/shading bands.
    m_seriesMarker.push_back(marker);

    tepad::DetectionResult result;
    if (m_detector && m_detector->Usable()) {
        result = m_detector->Step(adjusted.g, adjusted.p);
        m_seriesScore.push_back(result.score);
        m_seriesPointZ.push_back(result.pointZ);
        m_seriesShort.push_back(result.shortMahalanobis);
        m_seriesLong.push_back(result.longMahalanobis);
        if (result.alarm) {
            ++m_alarmCount;
            AppendAlarmRow(adjusted, result);
        }
    } else {
        m_seriesScore.push_back(0.0);
        m_seriesPointZ.push_back(0.0);
        m_seriesShort.push_back(0.0);
        m_seriesLong.push_back(0.0);
    }
    m_lastPointZ = result.pointZ;
    m_lastShort = result.shortMahalanobis;
    m_lastLong = result.longMahalanobis;
    m_lastScore = result.score;
    m_lastAlarm = result.alarm;
    m_lastWindowsReady = result.shortReady && result.longReady;

    // Alarm latency: count frames from the injection frame to the first alarm (1 tick = 1 minute).
    // Only state is updated here; RefreshInjectText() builds the sentence in the current language.
    if (m_injectAwaitingAlarm) {
        ++m_injectElapsedFrames;
        if (result.alarm) {
            m_injectVerdict = 2;
            m_injectAwaitingAlarm = false;
        } else if (m_injectElapsedFrames > kInjectDeadlineFrames) {
            m_injectVerdict = 3;
            m_injectAwaitingAlarm = false;
        } else {
            m_injectVerdict = 1;
        }
        RefreshInjectText();
    }

    RefreshStatsText();
}

void CSimUiDlg::AppendAlarmRow(const tepad::Frame& frame, const tepad::DetectionResult& result) {
    std::string channels;
    if (result.pointZ > m_calibration.residualThreshold) channels += "z";
    if (result.shortMahalanobis > m_calibration.tauShort) {
        if (!channels.empty()) channels += "+";
        channels += "L10";
    }
    if (result.longMahalanobis > m_calibration.tauLong) {
        if (!channels.empty()) channels += "+";
        channels += "L60";
    }

    SYSTEMTIME now;
    ::GetLocalTime(&now);
    CString time;
    time.Format(_T("%02d:%02d:%02d.%03d"), now.wHour, now.wMinute, now.wSecond, now.wMilliseconds);
    const CString tick = FromLongLong(frame.tick);

    const int row = m_alarmList.InsertItem(m_alarmList.GetItemCount(), time);
    if (row < 0) return;
    m_alarmList.SetItemText(row, 1, tick);
    m_alarmList.SetItemText(row, 2, FromDouble(frame.g, 4));
    m_alarmList.SetItemText(row, 3, FromDouble(frame.p, 4));
    m_alarmList.SetItemText(row, 4, FromDouble(result.pointZ, 3));
    m_alarmList.SetItemText(row, 5, FromDouble(result.shortMahalanobis, 3));
    m_alarmList.SetItemText(row, 6, FromDouble(result.longMahalanobis, 3));
    m_alarmList.SetItemText(row, 7, FromDouble(result.score, 3));
    m_alarmList.SetItemText(row, 8, ToCString(channels));

    while (m_alarmList.GetItemCount() > kMaxAlarmRows) m_alarmList.DeleteItem(0);
    m_alarmList.EnsureVisible(row, FALSE);
}

void CSimUiDlg::TrimSeries() {
    const std::size_t limit = static_cast<std::size_t>(std::max(10, m_displayPoints));
    std::deque<double>* series[] = {&m_seriesG, &m_seriesP, &m_seriesScore, &m_seriesPointZ,
                                    &m_seriesShort, &m_seriesLong};
    for (std::size_t i = 0; i < sizeof(series) / sizeof(series[0]); ++i) {
        while (series[i]->size() > limit) series[i]->pop_front();
    }
    // Markers are a deque<int> and cannot join the double* array above, so trim them separately.
    while (m_seriesMarker.size() > limit) m_seriesMarker.pop_front();
}

void CSimUiDlg::RefreshCurves() {
    std::vector<const std::deque<double>*> measure;
    measure.push_back(&m_seriesG);
    measure.push_back(&m_seriesP);
    std::vector<CString> measureLabels;
    measureLabels.push_back(Bi(_T("G irradiance"), _T("G irradiance")));
    measureLabels.push_back(Bi(_T("P power"), _T("P power")));
    std::vector<COLORREF> measureColors;
    measureColors.push_back(kColorG);
    measureColors.push_back(kColorP);
    m_measureCurve.SetSeries(measure, measureLabels, measureColors);

    std::vector<const std::deque<double>*> score;
    score.push_back(&m_seriesScore);
    std::vector<CString> scoreLabels;
    scoreLabels.push_back(_T("score"));
    std::vector<COLORREF> scoreColors;
    scoreColors.push_back(kColorScore);
    m_scoreCurve.SetSeries(score, scoreLabels, scoreColors);

    const CString markerHint = Bi(_T("▲ spike  ▨ shading"), _T("▲ spike  ▨ shading"));
    m_measureCurve.SetMarkers(&m_seriesMarker);
    m_measureCurve.SetMarkerHint(markerHint);
    m_scoreCurve.SetMarkers(&m_seriesMarker);
    m_scoreCurve.SetMarkerHint(markerHint);

    m_measureCurve.Refresh();
    m_scoreCurve.Refresh();
}

// ---------------------------------------------------------------------------
// Other interactions
// ---------------------------------------------------------------------------

void CSimUiDlg::OnPauseToggled() {
    m_paused = (m_pauseButton.GetCheck() == BST_CHECKED);
    SetStatusText(m_paused ? Bi(_T("paused: data still arrives, curves and detector frozen"),
                                _T("paused: data still arrives, curves and detector frozen"))
                           : Bi(_T("resumed"), _T("resumed")));
}

void CSimUiDlg::OnSendClicked() {
    CString text;
    m_sendEdit.GetWindowText(text);
    text.Trim();
    if (text.IsEmpty()) return;
    if (!m_transport.IsOpen()) {
        SetStatusText(Bi(_T("not connected, cannot send: "), _T("not connected, cannot send: ")) + text);
        return;
    }
    CString error;
    if (m_transport.SendLine(text, &error)) {
        SetStatusText(Bi(_T("sent: "), _T("sent: ")) + text);
    } else {
        SetStatusText(Bi(_T("send failed: "), _T("send failed: ")) + error);
    }
}

void CSimUiDlg::OnClearClicked() {
    m_frameCount = 0;
    m_alarmCount = 0;
    m_badLineCount = 0;
    m_lastTick = -1;
    m_spikeFrames = 0;
    m_shadeFrames = 0;
    m_injectAwaitingAlarm = false;
    m_injectElapsedFrames = 0;
    m_injectKind = 0;
    m_injectVerdict = 0;
    m_lastPointZ = 0.0;
    m_lastShort = 0.0;
    m_lastLong = 0.0;
    m_lastScore = 0.0;
    m_lastAlarm = false;
    m_lastWindowsReady = false;
    RefreshInjectText();
    m_seriesG.clear();
    m_seriesP.clear();
    m_seriesScore.clear();
    m_seriesPointZ.clear();
    m_seriesShort.clear();
    m_seriesLong.clear();
    m_seriesMarker.clear();
    if (m_alarmList.GetSafeHwnd()) m_alarmList.DeleteAllItems();
    RestartDetector();
    RefreshCurves();
    RefreshStatsText();
    SetStatusText(Bi(_T("curves, alarm log and detector state cleared"),
                     _T("curves, alarm log and detector state cleared")));
}

void CSimUiDlg::OnExportLogClicked() {
    const CString filter = Bi(_T("alarm log CSV (*.csv)|*.csv|All files (*.*)|*.*||"),
                              _T("alarm log CSV (*.csv)|*.csv|All files (*.*)|*.*||"));
    CFileDialog dialog(TRUE, _T("csv"), _T("alarm_log.csv"),
                       OFN_OVERWRITEPROMPT | OFN_HIDEREADONLY, filter, this);
    if (dialog.DoModal() != IDOK) return;

    CStdioFile file;
    if (!file.Open(dialog.GetPathName(), CFile::modeCreate | CFile::modeWrite | CFile::typeText)) {
        SetStatusText(Bi(_T("export failed, cannot write: "), _T("export failed, cannot write: ")) +
                      dialog.GetPathName());
        return;
    }
    file.WriteString(_T("time,tick,G,P,point_z,short_mahalanobis,long_mahalanobis,score,channels\n"));
    const int columns = 9;
    CString line;
    for (int row = 0; row < m_alarmList.GetItemCount(); ++row) {
        line.Empty();
        for (int column = 0; column < columns; ++column) {
            if (column > 0) line += _T(",");
            line += m_alarmList.GetItemText(row, column);
        }
        line += _T("\n");
        file.WriteString(line);
    }
    file.Close();
    SetStatusText(Bi(_T("exported "), _T("exported ")) + FromLongLong(m_alarmList.GetItemCount()) +
                  Bi(_T(" alarm rows to: "), _T(" alarm rows to: ")) + dialog.GetPathName());
}

void CSimUiDlg::OnTickspanChange() {
    const int selection = m_tickspanCombo.GetCurSel();
    if (selection >= 0 && selection < kTickspanChoiceCount) {
        m_displayPoints = kTickspanChoices[selection];
        TrimSeries();
        RefreshCurves();
    }
}

void CSimUiDlg::OnSize(UINT nType, int cx, int cy) {
    CDialogEx::OnSize(nType, cx, cy);
    Relayout();
}

void CSimUiDlg::OnOK() {
    // Enter sends by default instead of closing the dialog.
    CWnd* focus = GetFocus();
    if (focus == &m_sendEdit) {
        OnSendClicked();
        return;
    }
    CDialogEx::OnOK();
}

void CSimUiDlg::OnCancel() {
    KillTimer(kPollTimer);
    m_transport.Close();
    CDialogEx::OnCancel();
}
