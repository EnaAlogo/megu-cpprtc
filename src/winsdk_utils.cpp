#if defined(_WIN32) || defined(_WIN64)
#include "winsdk_utils.hpp"
#include <vector>
#include <filesystem>
#include "win_inc.hpp"
#include "win_str.hpp"
#include "macros.hpp"
#pragma comment(lib, "Version.lib")


namespace megu {
    std::string GetLastErrorMsg() {
        DWORD errorMessageID = ::GetLastError();
        LPSTR messageBuffer = nullptr;
        //Ask Win32 to give us the string version of that message ID.
        //The parameters we pass in, tell Win32 to create the buffer that holds the message for us (because we don't yet know how long the message string will be).
        size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
        //Copy the error message into a std::string.
        auto out = std::string(messageBuffer, size);
        //Free the Win32's string's buffer.
        LocalFree(messageBuffer);
        return out;
    }

    std::string GetWinSdkPath() {
        
        HKEY hKey;
        LSTATUS result = RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots", 0, KEY_READ, &hKey);

        MEGU_ENSURE(result == ERROR_SUCCESS, GetLastErrorMsg());

        wchar_t sdkPath[MAX_PATH];
        DWORD bufferSize = sizeof(sdkPath);
        result = RegQueryValueExW(hKey, L"KitsRoot10", nullptr, nullptr, reinterpret_cast<LPBYTE>(sdkPath), &bufferSize);
        RegCloseKey(hKey);
        MEGU_ENSURE(result == ERROR_SUCCESS, GetLastErrorMsg()); 

        return from_wide (sdkPath);

    }

    std::string GetWinSdkVersion() {
        wchar_t systemPath[MAX_PATH];
        MEGU_ENSURE(GetSystemDirectoryW(systemPath, MAX_PATH) != 0, "Error getting system directory.");

        std::wstring filePath = std::wstring(systemPath) + L"\\kernel32.dll";

        // Get the size of the version information
        DWORD versionInfoSize = GetFileVersionInfoSizeW(filePath.c_str(), nullptr);
        MEGU_ENSURE(versionInfoSize != 0, "Error getting version info size.")

            // Allocate memory for the version information
            std::vector<BYTE> versionInfoBuffer(versionInfoSize);
        MEGU_ENSURE(GetFileVersionInfoW(filePath.c_str(), 0, versionInfoSize, versionInfoBuffer.data()),
            "Error getting version info."
        );

        // Query the version information for the ProductVersion
        LPVOID versionValue;
        UINT versionValueSize;
        if (VerQueryValueW(versionInfoBuffer.data(), L"\\", &versionValue, &versionValueSize)) {
            VS_FIXEDFILEINFO* fileInfo = reinterpret_cast<VS_FIXEDFILEINFO*>(versionValue);

            // Extract the major and minor version numbers
            int majorVersion = HIWORD(fileInfo->dwProductVersionMS);
            int minorVersion = LOWORD(fileInfo->dwProductVersionMS);
            int buildNumber = HIWORD(fileInfo->dwProductVersionLS);
            int revisionNumber = LOWORD(fileInfo->dwProductVersionLS);

            std::ostringstream ss;
            ss
                << majorVersion << "." << minorVersion << "." << buildNumber;
            return ss.str();
        }
        else {
            MEGU_ENSURE(false, "Error querying version value.");
        }
    }

    std::string GetArch() {
#if defined(_M_X64)
        return "x64";
#elif defined(_M_X86) 
        return "x86";
#elif defined(_M_ARM64)
        return "arm64";
#elif defined(_M_ARM)
        return "arm"
#else
        MEGU_ENSURE(false, "Cannot detect cpu ach");
#endif 
    }

    static std::filesystem::path getIncludePath(std::filesystem::path const& winsdk, std::string const& version) {
        for (auto const& file : std::filesystem::directory_iterator(winsdk / "Include")) {
            if (file
                .path()
                .filename()
                .string()
                .starts_with(version)) {
                return file.path();
            }
        }
        MEGU_ENSURE(false, "Could not find include directory in windows sdk path : ", winsdk);
    }

    static std::filesystem::path getLibPath(std::filesystem::path const& winsdk, std::string const& version) {
        for (auto const& file : std::filesystem::directory_iterator(winsdk / "Lib")) {
            if (file
                .path()
                .filename()
                .string()
                .starts_with(version)) {
                return file.path();
            }
        }
        MEGU_ENSURE(false, "Could not find lib directory in windows sdk path : ", winsdk);
    }

    static std::string getLibDeps(std::filesystem::path const& winsdk, std::string const& ver) {
        std::ostringstream ss;
        auto libpath = getLibPath(winsdk, ver);
        for (auto const& lib : { "um" , "ucrt" }) {
            ss << " /LIBPATH:\"" << (libpath / lib / GetArch()).string() << "\"";
        }
        return ss.str();
    }
    static std::string getIncludeDeps(std::filesystem::path const& winsdk, std::string const& ver) {
        std::ostringstream ss;
        auto incpath = getIncludePath(winsdk, ver);
        for (auto const& lib : { "um" , "ucrt" , "shared" , "winrt" , "cppwinrt" }) {
            ss << " /I\"" << (incpath / lib).string() << "\"";
        }
        return ss.str();
    }


    WindowsDependenciesPaths GetWinSdkDependenciesPaths() {
        auto sdk = GetWinSdkPath(); 
        auto ver = GetWinSdkVersion();
        return {
            getLibDeps(sdk,ver),
            getIncludeDeps(sdk,ver)
        };
    }

}//end megu

#endif