// CurveCtrl.cpp : curve control drawing implementation.
#include "framework.h"
#include "CurveCtrl.h"

#include <algorithm>
#include <vector>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// Drawing margins (pixels)
static const int kMarginLeft = 48;
static const int kMarginRight = 10;
static const int kMarginTop = 20;
static const int kMarginBottom = 18;
static const int kGridDivY = 5;

// Injection marker colors: red for spikes, purple for shading, clearly distinct from the G/P/score curve colors.
static const COLORREF kColorSpikeMark = RGB(200, 40, 40);
static const COLORREF kColorShadeMark = RGB(125, 60, 185);
// Above this many consecutive markers draw a band (instead of repeated vertical lines).
static const int kBandMinRun = 3;

BEGIN_MESSAGE_MAP(CCurveCtrl, CWnd)
    ON_WM_PAINT()
    ON_WM_ERASEBKGND()
    ON_WM_SIZE()
END_MESSAGE_MAP()

CCurveCtrl::CCurveCtrl() : yLo_(0.0), yHi_(0.0), threshold_(0.0) {}

CCurveCtrl::~CCurveCtrl() {}

BOOL CCurveCtrl::Create(CWnd* parent, UINT nID, const CRect& rect) {
    if (parent == nullptr) return FALSE;
    // AfxRegisterWndClass registers the class and unregisters it at exit; the background brush is left null,
    // so OnEraseBkgnd + double buffering fill it, avoiding flicker.
    LPCTSTR lpszClass = AfxRegisterWndClass(CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS,
                                           ::LoadCursor(nullptr, IDC_CROSS), nullptr, nullptr);
    if (lpszClass == nullptr) return FALSE;
    // For child windows the 7th parameter of MFC's CWnd::CreateEx is the control ID (UINT), so pass nID directly.
    return CreateEx(WS_EX_CLIENTEDGE, lpszClass, _T(""),
                    WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN | WS_CLIPSIBLINGS,
                    rect, parent, nID, nullptr);
}

void CCurveCtrl::SetSeries(const std::vector<const std::deque<double>*>& series,
                           const std::vector<CString>& labels,
                           const std::vector<COLORREF>& colors) {
    series_ = series;
    labels_ = labels;
    colors_ = colors;
}

void CCurveCtrl::SetRange(double lo, double hi) {
    yLo_ = lo;
    yHi_ = hi;
}

void CCurveCtrl::SetThreshold(double value, const CString& label) {
    threshold_ = value;
    thresholdLabel_ = label;
}

void CCurveCtrl::SetTitle(const CString& title) { title_ = title; }

void CCurveCtrl::SetMarkers(const std::deque<int>* markers) { markers_ = markers; }

void CCurveCtrl::SetMarkerHint(const CString& hint) { markerHint_ = hint; }

void CCurveCtrl::OnSize(UINT nType, int cx, int cy) {
    CWnd::OnSize(nType, cx, cy);
    if (cx > 0 && cy > 0) InvalidateRect(nullptr, FALSE);
}

BOOL CCurveCtrl::OnEraseBkgnd(CDC* /*pDC*/) {
    return TRUE;  // background is filled in the double buffer inside OnPaint
}

void CCurveCtrl::DrawGrid(CDC* dc, const CRect& plot, double yMin, double yMax, int maxPoints) {
    dc->SelectObject(&axisPen_);
    dc->MoveTo(plot.left, plot.top);
    dc->LineTo(plot.left, plot.bottom);
    dc->LineTo(plot.right, plot.bottom);

    CString text;
    for (int i = 0; i <= kGridDivY; ++i) {
        const int y = plot.bottom - plot.Height() * i / kGridDivY;
        CPen dotPen(PS_DOT, 1, RGB(210, 214, 222));
        CPen* old = dc->SelectObject(&dotPen);
        if (i > 0) {
            dc->MoveTo(plot.left, y);
            dc->LineTo(plot.right, y);
        }
        dc->SelectObject(old);
        const double value = yMin + (yMax - yMin) * i / kGridDivY;
        text.Format(_T("%.3g"), value);
        CSize size = dc->GetTextExtent(text);
        dc->TextOut(plot.left - kMarginLeft + 4, y - size.cy / 2, text);
    }

    const int divX = 6;
    for (int i = 1; i <= divX; ++i) {
        const int x = plot.left + plot.Width() * i / divX;
        CPen dotPen(PS_DOT, 1, RGB(228, 232, 238));
        CPen* old = dc->SelectObject(&dotPen);
        dc->MoveTo(x, plot.top);
        dc->LineTo(x, plot.bottom);
        dc->SelectObject(old);
    }
    if (maxPoints > 0) {
        text.Format(_T("%d tick"), maxPoints);
        dc->TextOut(plot.right - dc->GetTextExtent(text).cx - 2, plot.bottom + 2, text);
    }
}

void CCurveCtrl::DrawSeries(CDC* dc, const CRect& plot, const std::deque<double>& values,
                            COLORREF color, int maxPoints, double yMin, double yMax) {
    if (values.empty() || maxPoints <= 1 || yMax <= yMin) return;
    CPen pen(PS_SOLID, 1, color);
    CPen* oldPen = dc->SelectObject(&pen);
    const int count = static_cast<int>(values.size());
    // Right aligned: when the window holds fewer than maxPoints, drawing starts after the left padding.
    const int offset = maxPoints - count;
    std::vector<CPoint> points;
    points.reserve(count);
    for (int i = 0; i < count; ++i) {
        const double xRatio = static_cast<double>(offset + i) / (maxPoints - 1);
        const double yRatio = (values[i] - yMin) / (yMax - yMin);
        const int x = plot.left + static_cast<int>(xRatio * plot.Width() + 0.5);
        const int topEdge = static_cast<int>(plot.top);
        const int bottomEdge = static_cast<int>(plot.bottom);
        const int raw = plot.bottom - static_cast<int>(yRatio * plot.Height() + 0.5);
        points.push_back(CPoint(x, std::min(std::max(raw, topEdge), bottomEdge)));
    }
    if (points.size() >= 2) dc->Polyline(points.data(), static_cast<int>(points.size()));
    dc->SelectObject(oldPen);
}

void CCurveCtrl::DrawMarkers(CDC* dc, const CRect& plot, int maxPoints) {
    if (markers_ == nullptr || markers_->empty() || maxPoints <= 1) return;
    const int count = static_cast<int>(markers_->size());
    const int offset = maxPoints - count;
    auto xOf = [&](int index) {
        const double ratio = static_cast<double>(offset + index) / (maxPoints - 1);
        return plot.left + static_cast<int>(ratio * plot.Width() + 0.5);
    };

    // Consecutive markers of the same kind are merged into one run: long runs draw a hatched band (shading), short runs/single points draw a dashed line + top triangle (spike).
    int index = 0;
    while (index < count) {
        const int kind = (*markers_)[index];
        if (kind <= 0) { ++index; continue; }
        int end = index;
        while (end + 1 < count && (*markers_)[end + 1] == kind) ++end;
        const int x0 = xOf(index);
        const int x1 = xOf(end);
        const COLORREF color = (kind == 1) ? kColorSpikeMark : kColorShadeMark;
        if (end - index + 1 >= kBandMinRun) {
            CBrush hatch;
            hatch.CreateHatchBrush(HS_BDIAGONAL, color);
            CRect band(x0, plot.top, x1 + 1, plot.bottom);
            dc->FillRect(&band, &hatch);
            CPen border(PS_DOT, 1, color);
            CPen* old = dc->SelectObject(&border);
            dc->Rectangle(x0, plot.top, x1 + 1, plot.bottom);
            dc->SelectObject(old);
        } else {
            CPen dash(PS_DASH, 1, color);
            CPen* oldPen = dc->SelectObject(&dash);
            dc->MoveTo(x0, plot.top);
            dc->LineTo(x0, plot.bottom);
            dc->SelectObject(oldPen);
            if (kind == 1) {
                CBrush triangle;
                triangle.CreateSolidBrush(color);
                CBrush* oldBrush = dc->SelectObject(&triangle);
                CPoint tips[3] = {CPoint(x0 - 4, plot.top), CPoint(x0 + 4, plot.top),
                                  CPoint(x0, plot.top + 8)};
                dc->Polygon(tips, 3);
                dc->SelectObject(oldBrush);
            }
        }
        index = end + 1;
    }
}

void CCurveCtrl::DrawLegend(CDC* dc, const CRect& plot) {
    int x = plot.left + 6;
    const int y = plot.top - 15;
    CString text;
    for (std::size_t i = 0; i < series_.size(); ++i) {
        CPen pen(PS_SOLID, 2, i < colors_.size() ? colors_[i] : RGB(0, 0, 0));
        CPen* old = dc->SelectObject(&pen);
        dc->MoveTo(x, y + 6);
        dc->LineTo(x + 14, y + 6);
        dc->SelectObject(old);
        x += 17;
        text = i < labels_.size() ? labels_[i] : CString(_T("?"));
        dc->TextOut(x, y, text);
        x += dc->GetTextExtent(text).cx + 12;
    }
    // The marker legend goes on the right, clear of the curves and the threshold label.
    if (!markerHint_.IsEmpty()) {
        const int hintWidth = dc->GetTextExtent(markerHint_).cx;
        const int hintX = std::max(x, static_cast<int>(plot.right) - hintWidth - 2);
        dc->SetTextColor(kColorSpikeMark);
        dc->TextOut(hintX, y, markerHint_);
        dc->SetTextColor(RGB(40, 44, 52));
    }
}

void CCurveCtrl::OnPaint() {
    CPaintDC dc(this);
    CRect client;
    GetClientRect(&client);
    if (client.Width() < 60 || client.Height() < 60) return;

    // ---- Double buffering ----
    CDC memory;
    memory.CreateCompatibleDC(&dc);
    CBitmap bitmap;
    bitmap.CreateCompatibleBitmap(&dc, client.Width(), client.Height());
    CBitmap* oldBitmap = memory.SelectObject(&bitmap);
    memory.SetBkMode(TRANSPARENT);
    memory.FillSolidRect(client, RGB(255, 255, 255));

    if (font_.m_hObject == nullptr) {
        LOGFONT lf;
        memset(&lf, 0, sizeof(lf));
        lf.lfHeight = -12;
        lf.lfWeight = FW_REGULAR;
        lf.lfCharSet = DEFAULT_CHARSET;
        _tcscpy_s(lf.lfFaceName, LF_FACESIZE, _T("Segoe UI"));
        font_.CreateFontIndirect(&lf);
    }
    if (axisPen_.m_hObject == nullptr) axisPen_.CreatePen(PS_SOLID, 1, RGB(120, 128, 140));
    CFont* oldFont = memory.SelectObject(&font_);

    CRect plot(client.left + kMarginLeft, client.top + kMarginTop,
               client.right - kMarginRight, client.bottom - kMarginBottom);

    // ---- Y range ----
    double yMin = yLo_;
    double yMax = yHi_;
    int maxPoints = 0;
    for (std::size_t i = 0; i < series_.size(); ++i) {
        const std::deque<double>* values = series_[i];
        if (values == nullptr) continue;
        maxPoints = std::max(maxPoints, static_cast<int>(values->size()));
        if (yMax <= yMin) {  // automatic range
            for (std::deque<double>::const_iterator it = values->begin(); it != values->end(); ++it) {
                yMin = std::min(yMin, *it);
                yMax = std::max(yMax, *it);
            }
        }
    }
    if (threshold_ > 0.0) yMax = std::max(yMax, threshold_ * 1.05);
    if (yMax <= yMin) {
        yMin = 0.0;
        yMax = 1.0;
    }
    const double pad = (yMax - yMin) * 0.08;
    if (yHi_ <= yLo_) {
        yMin -= pad;
        yMax += pad;
        if (yMin < 0.0 && yMin + pad >= -1e-12) yMin = 0.0;
    }

    DrawGrid(&memory, plot, yMin, yMax, maxPoints);

    // ---- Threshold reference line (score curve: score = 1 means alarm) ----
    if (threshold_ > 0.0 && yMax > yMin) {
        const double ratio = (threshold_ - yMin) / (yMax - yMin);
        if (ratio >= 0.0 && ratio <= 1.0) {
            const int y = plot.bottom - static_cast<int>(ratio * plot.Height() + 0.5);
            CPen dash(PS_DOT, 1, RGB(220, 60, 60));
            CPen* old = memory.SelectObject(&dash);
            memory.MoveTo(plot.left, y);
            memory.LineTo(plot.right, y);
            memory.SelectObject(old);
            if (!thresholdLabel_.IsEmpty()) {
                memory.SetTextColor(RGB(200, 40, 40));
                memory.TextOut(plot.left + 4, y - 15, thresholdLabel_);
                memory.SetTextColor(RGB(40, 44, 52));
            }
        }
    }

    for (std::size_t i = 0; i < series_.size(); ++i) {
        if (series_[i] == nullptr) continue;
        DrawSeries(&memory, plot, *series_[i],
                   i < colors_.size() ? colors_[i] : RGB(0, 0, 0),
                   maxPoints, yMin, yMax);
    }
    // Markers are drawn above the curves, otherwise a single point is hidden by the curve itself.
    DrawMarkers(&memory, plot, maxPoints);
    DrawLegend(&memory, plot);

    if (!title_.IsEmpty()) {
        memory.SetTextColor(RGB(70, 76, 88));
        memory.TextOut(client.left + 4, client.top + 2, title_);
    }

    dc.BitBlt(0, 0, client.Width(), client.Height(), &memory, 0, 0, SRCCOPY);
    memory.SelectObject(oldFont);
    memory.SelectObject(oldBitmap);
}
