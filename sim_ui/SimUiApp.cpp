// SimUiApp.cpp : application implementation.
#include "framework.h"
#include "SimUiApp.h"
#include "SimUiDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CSimUiApp, CWinApp)
END_MESSAGE_MAP()

CSimUiApp theApp;

CSimUiApp::CSimUiApp() { m_pMainWnd = nullptr; }

BOOL CSimUiApp::InitInstance() {
    CWinApp::InitInstance();

    // The alarm log uses CListCtrl (report view), so the common control classes must be registered first.
    INITCOMMONCONTROLSEX controls;
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_LISTVIEW_CLASSES | ICC_STANDARD_CLASSES | ICC_BAR_CLASSES;
    if (!::InitCommonControlsEx(&controls)) {
        AfxMessageBox(_T("Failed to initialize common controls; cannot start the UI."));
        return FALSE;
    }

    CSimUiDlg dialog;
    // Command line: --auto / /auto auto-connects to the local TCP data bridge after startup.
    const CString cmdLine(m_lpCmdLine);
    dialog.SetAutoConnectTcp(cmdLine.Find(_T("--auto")) >= 0 || cmdLine.Find(_T("/auto")) >= 0);
    // --lang=en / /lang=en: start with the English UI (it can be switched back at any time).
    if (cmdLine.Find(_T("--lang=en")) >= 0 || cmdLine.Find(_T("/lang=en")) >= 0) {
        dialog.SetLanguage(CSimUiDlg::Lang::En);
    }
    m_pMainWnd = &dialog;
    dialog.DoModal();
    return FALSE;  // exit the program when the dialog closes
}

int CSimUiApp::ExitInstance() { return CWinApp::ExitInstance(); }
