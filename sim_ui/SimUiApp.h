// SimUiApp.h : application object.
#pragma once

class CSimUiApp : public CWinApp {
public:
    CSimUiApp();

    virtual BOOL InitInstance();
    virtual int ExitInstance();

    DECLARE_MESSAGE_MAP()
};

extern CSimUiApp theApp;
