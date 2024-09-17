#pragma once

#if defined(_WIN32) || defined(_WIN64)
#define LAB_PLATFORM_WINDOWS
#elif defined(__APPLE__) || defined(__MACH__)
#define LAB_PLATFORM_OSX
#elif defined(__linux__)
#define LAB_PLATFORM_LINUX
#else
#define LAB_PLATFORM_UNKNOWN
#endif
#ifdef LAB_PLATFORM_UNKNOWN
#error "Unknown platform. Compilation halted."
#endif
