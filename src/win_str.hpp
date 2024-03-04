#pragma once

#if defined(_WIN32) || defined(_WIN64)
#include <string>

namespace megu{

    std::wstring to_wide(const std::string_view& str);

    std::string from_wide(const std::wstring& str);

}

#endif