// framework.h : standard MFC framework include file.
//
// Note: MFC support requires the component "C++ MFC" to be installed in the Visual Studio Installer
// (component ID: Microsoft.VisualStudio.Component.VC.ATLMFC),
// otherwise headers such as afxwin.h are missing and the project cannot compile.
#pragma once

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN  // exclude rarely used material from the Windows headers
#endif
// Block winsock.h so Transport.cpp can safely include winsock2.h;
// also avoid min/max macros from windef.h clashing with std::min / std::max.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "targetver.h"

#define _AFX_ALL_WARNINGS

#include <afxwin.h>       // MFC core and standard components (CWinApp / CWnd)
#include <afxext.h>       // MFC extension controls
#include <afxdlgs.h>      // CFileDialog (browse calibration file / export alarm log)
#include <afxcmn.h>       // common controls: CListCtrl / CHeaderCtrl
#include <afxmt.h>        // CCriticalSection / CSingleLock (receiver thread and UI thread sync)

#include "Resource.h"
