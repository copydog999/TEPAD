// SimUiDlg.h : main dialog of the simulation detector UI.
//
// Responsibilities: receive tick,G,P frames over serial/TCP -> dual-window detector -> live curves, score curve, alarm log,
// send box, calibration loading. All controls are created dynamically with a pixel layout and re-laid out on resize.
#pragma once

#include <afxdialogex.h>

#include <deque>
#include <memory>
#include <vector>

#include "core/Calibration.h"
#include "core/DualWindowDetector.h"
#include "CurveCtrl.h"
#include "Transport.h"

class CSimUiDlg : public CDialogEx {
public:
    /// UI language. Chinese/English caption sets; the lower-right button switches at any time (control widths are re-measured from the actual text).
    enum class Lang { Zh, En };

    explicit CSimUiDlg(CWnd* pParent = nullptr);
    enum { IDD = IDD_SIM_UI_DIALOG };

    /// Command-line switch: auto-connect to the local data bridge with the TCP parameters after startup (pairs with scripts\run_sim.bat for a one-click demo).
    void SetAutoConnectTcp(bool enable) { m_autoConnectTcp = enable; }
    /// Startup language (Chinese by default; --lang=en starts the English UI directly).
    void SetLanguage(Lang lang) { m_lang = lang; }

protected:
    virtual void DoDataExchange(CDataExchange* pDX);
    virtual BOOL OnInitDialog();
    virtual void OnOK();
    virtual void OnCancel();

    afx_msg void OnTimer(UINT_PTR timerId);
    afx_msg void OnSize(UINT nType, int cx, int cy);
    afx_msg void OnConnectClicked();
    afx_msg void OnLoadCalibrationClicked();
    afx_msg void OnBrowseCalibrationClicked();
    afx_msg void OnSendClicked();
    afx_msg void OnClearClicked();
    afx_msg void OnPauseToggled();
    afx_msg void OnExportLogClicked();
    afx_msg void OnTickspanChange();
    afx_msg void OnTransportChange();
    afx_msg void OnInjectSpikeClicked();
    afx_msg void OnInjectShadeClicked();
    afx_msg void OnSpeedChange();
    afx_msg void OnLanguageClicked();
    afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
    DECLARE_MESSAGE_MAP()

private:
    // ---- UI construction ----
    void CreateChildControls();
    void Relayout();
    void ApplyUiFont();
    void ApplyLanguage();

    // ---- Bilingual ----
    /// Selects the Chinese/English caption for the current language (all user-facing text goes through here).
    CString Bi(LPCWSTR zh, LPCWSTR en) const;
    CString AlarmHeader(int column) const;

    // ---- Dynamic text: only state is stored; captions are rebuilt in the current language (state survives a switch) ----
    void RefreshStatsText();
    void RefreshLinkText();
    void RefreshCalibrationText();
    void RefreshInjectText();
    void SetStatusText(const CString& text);
    void SetLinkStateText(const CString& text, COLORREF color);
    void SetCalibrationInfoText(const CString& text, COLORREF color);
    void SetInjectInfoText(const CString& text, COLORREF color);
    CString DefaultCalibrationPath() const;

    // ---- Manual fault injection (added directly to the P entering the detector, same semantics as Python inject_spike/shading) ----
    double InjectResidualSigma() const;
    void BeginInjection(bool spike, int frames, double sigmaMultiple);

    // ---- Playback speed ----
    void SendSpeedCommand();

    // ---- Data link ----
    void PollTransport();
    void HandleFrame(const tepad::Frame& frame);
    void AppendAlarmRow(const tepad::Frame& frame, const tepad::DetectionResult& result);
    void TrimSeries();
    void RefreshCurves();
    void RestartDetector();

    // ---- Controls ----
    CComboBox m_transportCombo;
    CComboBox m_endpointCombo;   ///< serial port name (editable, auto-populated with detected COM ports) / TCP address
    CComboBox m_baudCombo;
    CButton m_connectButton;
    CStatic m_linkState;
    CEdit m_calibrationPathEdit;
    CButton m_loadCalibrationButton;
    CButton m_browseCalibrationButton;
    CStatic m_calibrationInfo;
    CComboBox m_tickspanCombo;
    CStatic m_tickspanLabel;
    CListCtrl m_alarmList;
    CButton m_pauseButton;
    CButton m_clearButton;
    CButton m_exportButton;
    CEdit m_sendEdit;
    CButton m_sendButton;
    CStatic m_statsText;
    CStatic m_injectLabel;
    CEdit m_injectSpikeSigmaEdit;
    CButton m_injectSpikeButton;
    CEdit m_injectShadeTicksEdit;
    CButton m_injectShadeButton;
    CStatic m_injectInfo;
    CStatic m_speedLabel;
    CComboBox m_speedCombo;
    CButton m_languageButton;
    CCurveCtrl m_measureCurve;
    CCurveCtrl m_scoreCurve;
    CFont m_uiFont;
    CBrush m_staticBrush;      ///< static-text background brush (painted in OnCtlColor)
    COLORREF m_linkStateColor;
    COLORREF m_calibrationInfoColor;
    COLORREF m_injectInfoColor;

    // ---- Detection core ----
    tepad::Calibration m_calibration;
    std::unique_ptr<tepad::DualWindowDetector> m_detector;
    CSimTransport m_transport;

    // ---- Rolling data (the newest tick is at the right edge) ----
    std::deque<double> m_seriesG;
    std::deque<double> m_seriesP;
    std::deque<double> m_seriesScore;
    std::deque<double> m_seriesPointZ;
    std::deque<double> m_seriesShort;
    std::deque<double> m_seriesLong;
    /// Injection markers (same length and trimming as the series above): 0 normal / 1 spike / 2 shading.
    std::deque<int> m_seriesMarker;

    long long m_frameCount;
    long long m_alarmCount;
    long long m_badLineCount;
    long long m_lastTick;

    // ---- Pending injection frames and alarm-latency observation ----
    int m_spikeFrames;
    int m_shadeFrames;
    double m_injectSigmaMultiple;
    long long m_injectElapsedFrames;
    bool m_injectAwaitingAlarm;
    int m_injectKind;      ///< 0 none / 1 spike / 2 shading
    int m_injectVerdict;   ///< 0 none / 1 waiting / 2 alarm fired / 3 deadline exceeded
    double m_lastG;
    double m_lastP;
    double m_lastPointZ;
    double m_lastShort;
    double m_lastLong;
    double m_lastScore;
    bool m_lastAlarm;
    bool m_lastWindowsReady;
    int m_linkPhase;       ///< 0 not connected / 1 connected / 2 failed
    CString m_linkDetail;  ///< endpoint or error detail (language-independent raw text)
    bool m_calibrationOk;
    int m_calibrationState;  ///< 0 not loaded / 1 empty path / 2 load failed / 3 loaded
    CString m_calibrationDetail;
    int m_displayPoints;   ///< number of rolling points shown on the curves
    int m_speedMs;         ///< milliseconds per tick (sent to the data bridge)
    Lang m_lang;
    bool m_paused;
    bool m_detectorReady;
    bool m_uiReady;      ///< whether the child controls exist: WM_SIZE arrives during window creation,
                         /// before OnInitDialog, so Relayout must return early until the UI is ready,
                         /// otherwise it would message controls without an HWND (Debug MFC asserts).
    bool m_autoConnectTcp;

    static constexpr int kPollTimer = 1;
    static constexpr UINT_PTR kPollIntervalMs = 33;   ///< ~30fps polling of the receive queue
    static constexpr int kMaxAlarmRows = 2000;
    static constexpr double kShadeFactor = 0.30;      ///< shading attenuation factor, same as the simulation definition
    static constexpr int kDefaultSpikeSigma = 8;      ///< default spike magnitude (times residual_sigma)
    static constexpr int kDefaultShadeTicks = 60;     ///< default shading duration in frames
    static constexpr long long kInjectDeadlineFrames = 180;  ///< past this many frames without an alarm, report "no alarm"
    static constexpr int kDefaultSpeedMs = 100;       ///< default 100ms/tick (1 second = 10 ticks)
    static constexpr int kNarrowSpanTicks = 120;      ///< on injection narrow to this window to see a single point
};
