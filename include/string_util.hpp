#pragma once
#include <string>
#include <functional>
#include <sstream>

namespace megu::detail
{

    constexpr inline std::string& replace_first_
    (
        std::string& string,
        const std::string_view to_replace,
        const std::string_view replace_with
    )
    {
        std::size_t curr = 0;
        std::size_t pos = string.find(to_replace, curr);
        if (pos == std::string::npos)return string;
        string.replace(string.begin() + pos, string.begin() + pos + to_replace.size(), replace_with);

        return string;
    }


    constexpr inline std::string replace_first
    (
        const std::string_view string,
        const std::string_view to_replace,
        const std::string_view replace_with
    )
    {
        std::size_t curr = 0;
        std::size_t pos = string.find(to_replace, curr);
        std::string ss;
        MEGU_ENSURE(string.size() - to_replace.size() + replace_with.size() > 0, "Output string evaluated to have a negative size somehow");
        ss.reserve(string.size() - to_replace.size() + replace_with.size());
        if (pos == std::string::npos)
        {
            ss += string;  
            return ss;
        }
        ss += std::string_view{ string.data(),pos };
        ss += replace_with;
        ss += std::string_view{ string.data() + pos + to_replace.size() , string.size() - pos - to_replace.size() };
        return ss;
    }

    constexpr inline std::string& remove_dup_chars_(std::string& str, const char Char = ' ')
    {
        const std::string::iterator new_end =
            std::unique(str.begin(), str.end(),
                [Char](char lhs, char rhs) { return (lhs == rhs) && (lhs == Char); }
        );
        str.erase(new_end, str.end()); 
        return str;
    }

    constexpr inline std::string remove_dup_chars(const std::string_view str, const char Char = ' ')
    {
        std::string out(str);
        remove_dup_chars_(out, Char);
        return out;
    }

    constexpr inline std::string& right_trim_(std::string& s)
    {
        s.erase(s.find_last_not_of(" \t\n\r\f\v") + 1);
        return s;
    }

    constexpr inline std::string& left_trim_(std::string& s)
    {
        s.erase(0, s.find_first_not_of(" \t\n\r\f\v"));
        return s;
    }

    constexpr inline std::string& trim_(std::string& s)
    {
        left_trim_(s);
        right_trim_(s);
        return s;
    }
    
    constexpr std::string_view left_trim(const std::string_view view)
    {
        const std::size_t pos = view.find_first_not_of(" \t\n\r\f\v");
        return std::string_view{view.data() + pos, view.size() - pos};
    }

    constexpr static inline std::string_view right_trim(const std::string_view view)
    {
        const std::size_t pos = view.find_last_not_of(" \t\n\r\f\v") + 1;
        return std::string_view{ view.data() ,  pos }; 
    }

    constexpr static inline std::string_view trim(const std::string_view view)
    {
        return right_trim(left_trim(view));
    }

}
