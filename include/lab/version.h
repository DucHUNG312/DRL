#pragma once

#ifndef LAB_STR
#define LAB_STR(x) #x
#define LAB_MAKE_STR(x) STR(x)
#endif // !LAB_STR

#define LAB_VERSION_MAJOR 0
#define LAB_VERSION_MINOR 0
#define LAB_VERSION_PATCH 1
#define LAB_VERSION LAB_MAKE_STR(LAB_VERSION_MAJOR) "." \
                    LAB_MAKE_STR(LAB_VERSION_MINOR) "." \
                    LAB_MAKE_STR(LAB_VERSION_PATCH)