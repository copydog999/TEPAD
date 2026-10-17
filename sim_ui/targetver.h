// targetver.h : target Windows version.
//
// This project uses CancelIoEx (Vista+) and InetPton (Win8+), so Win10 is targeted throughout.
#pragma once

#ifndef WINVER
#define WINVER 0x0A00
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef _WIN32_IE
#define _WIN32_IE 0x0A00
#endif
