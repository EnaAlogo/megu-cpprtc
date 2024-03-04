#pragma once

#if defined(_WIN32) || defined (_WIN64)
#include <string>

namespace megu {

    std::string GetLastErrorMsg();

    std::string GetWinSdkPath();

    std::string GetWinSdkVersion();

    std::string GetArch();

    struct WindowsDependenciesPaths {
        std::string Lib;
        std::string Include;
    };

    WindowsDependenciesPaths GetWinSdkDependenciesPaths();

}//end megu
#endif