#if defined(_WIN32) || defined(_WIN64)
#include "win_str.hpp"
#include "win_inc.hpp"
#include <stdexcept>

namespace megu {

    std::wstring to_wide(const std::string_view& str)
    {
        std::wstring out;

        if (str.empty())
            return out;

        int size = MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.size()), NULL, 0);
        if(!(size > 0)) throw std::runtime_error("Error converting string to wstring");
        out.resize(size);
        MultiByteToWideChar(CP_UTF8, 0, str.data(), static_cast<int>(str.size()), &out[0], size);
        return out;
    }

    std::string from_wide(const std::wstring& str) {
        if (str.empty()) {
            return std::string();
        }
        int size_needed = WideCharToMultiByte(
            CP_UTF8,
            0,
            str.c_str(),
            static_cast<int>(str.size()),
            NULL,
            0,
            NULL,
            NULL);
        if(!(size_needed > 0)) throw std::runtime_error("Error converting the content to UTF8");
        std::string outstr(size_needed, 0);
        WideCharToMultiByte(
            CP_UTF8,
            0,
            str.c_str(),
            static_cast<int>(str.size()),
            outstr.data(),
            size_needed,
            NULL,
            NULL);
        return outstr;
    }
}//end megu::detail

#endif