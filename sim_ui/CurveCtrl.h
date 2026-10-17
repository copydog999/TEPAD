// CurveCtrl.h : two-channel live curve control (shared by the G/P curves and the detection score curve).
//
// The data is owned by the dialog (std::deque<double>, rolling window); the control only keeps pointers and draws by index,
// avoiding a copy per frame; an in-memory bitmap double buffer keeps it flicker-free at high serial refresh rates.
#pragma once

#include <deque>
#include <vector>

class CCurveCtrl : public CWnd {
public:
    CCurveCtrl();
    virtual ~CCurveCtrl();

    /// Created as a WS_CHILD window; nID is the child window ID (for Tab order), layout is handled by the parent.
    BOOL Create(CWnd* parent, UINT nID, const CRect& rect);

    /// Clear and re-assign the curve series; the data pointers must stay valid while the control is in use.
    void SetSeries(const std::vector<const std::deque<double>*>& series,
                   const std::vector<CString>& labels,
                   const std::vector<COLORREF>& colors);

    /// Y axis range; when hi<=lo the range is scaled automatically from the data.
    void SetRange(double lo, double hi);
    /// Horizontal reference line (threshold line of the score curve = 1.0); value<=0 means not drawn.
    void SetThreshold(double value, const CString& label);
    /// Top-left title.
    void SetTitle(const CString& title);

    /// Injection markers (same length and right alignment as the curve series): 0=normal, 1=spike (single point), 2=shading (contiguous run).
    /// A single-point spike is only 1 pixel wide in a 720 tick window, so the marker is what makes it visible.
    void SetMarkers(const std::deque<int>* markers);
    /// Marker legend text drawn to the right of the legend row (language is chosen by the caller).
    void SetMarkerHint(const CString& hint);

    /// Request a repaint (the dialog calls this once after a batch of data is appended).
    void Refresh() { InvalidateRect(nullptr, FALSE); }

protected:
    afx_msg void OnPaint();
    afx_msg BOOL OnEraseBkgnd(CDC* pDC);
    afx_msg void OnSize(UINT nType, int cx, int cy);
    DECLARE_MESSAGE_MAP()

private:
    void DrawGrid(CDC* dc, const CRect& plot, double yMin, double yMax, int maxPoints);
    void DrawSeries(CDC* dc, const CRect& plot, const std::deque<double>& values,
                    COLORREF color, int maxPoints, double yMin, double yMax);
    void DrawMarkers(CDC* dc, const CRect& plot, int maxPoints);
    void DrawLegend(CDC* dc, const CRect& plot);

    std::vector<const std::deque<double>*> series_;
    std::vector<CString> labels_;
    std::vector<COLORREF> colors_;
    const std::deque<int>* markers_ = nullptr;
    CString markerHint_;
    double yLo_ = 0.0;
    double yHi_ = 0.0;      ///< <=0 means automatic
    double threshold_ = 0.0;
    CString thresholdLabel_;
    CString title_;
    CFont font_;
    CPen axisPen_;
};
