#define _CRT_SECURE_NO_WARNINGS
#include "../include/program.hpp"
#include <string>
#include  <vector>
#include <optional>
#include <memory>
#include <filesystem>
#include <iostream>
#include "macros.hpp"
#include <assert.h>
#include "../include/string_util.hpp"
#include <fstream>

#if defined(_WIN32) || defined(_WIN64)
#include "win_inc.hpp"
#include <WinError.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdio.h>
#include <sys/stat.h>
#include <random>
#include "win_str.hpp"
#include "winsdk_utils.hpp"

#else
#include <unistd.h>
#endif


#if (defined(_MSC_VER) && !defined(_M_ARM64))

extern "C" int __isa_available;

#endif

namespace megu {

    struct CMD {

    private:
#if defined (_MSC_VER)
        std::vector<std::wstring> env_list;
#endif

    public:
#if defined(_MSC_VER)
    intptr_t
#else
    int
#endif
        run_cmd
        (std::string_view cmd) {
#if defined(_MSC_VER)


        // Getting the path of `cmd.exe`
        wchar_t const* comspec = _wgetenv(L"COMSPEC");
        if (!comspec) {
            comspec = L"C:\\Windows\\System32\\cmd.exe";
        }
        // Constructing the command line
        auto wCmd = to_wide(cmd);
        const wchar_t* a[] = { L"/c", wCmd.c_str(), nullptr };
        // Constructing the env array
        // If `env_list` is not empty, then add char pointers ending with nullptr.
        // Otherwise, it will be nullptr, which implies the default env.
        std::vector<const wchar_t*> e;
        if (!env_list.empty()) {
            for (auto& s : env_list) {
                e.push_back(s.c_str());
            }
            e.push_back(nullptr);
        }

        // Running the command
        intptr_t r = _wspawnve(_P_WAIT, comspec, a, e.data());

        return r;
#else
        return system(cmd.data());
#endif
    }


#ifdef _MSC_VER
    std::optional<std::wstring> exec(const std::wstring& cmd) {
        constexpr static int bufferSize = 128;

        wchar_t buffer[bufferSize];
        std::wstring result;
        std::unique_ptr<FILE, decltype(&_pclose)> pipe(
            _wpopen(cmd.c_str(), L"r"), _pclose);
        if (!pipe) {
            return std::nullopt;
        }
        while (fgetws(buffer, bufferSize , pipe.get()) !=  nullptr) {
            result += buffer;
        }
        return result;
    }

    private:
        
        friend struct JitConfig;

    void activate() {
        wchar_t* root = nullptr;
        std::wstring cmd;
        std::optional<std::wstring> exec_out;
        std::wstring path;
        std::wstring vcruntime_plat;
        std::wstring envvars;

        // Checking whether the environment is already activated
        if (_wgetenv(L"VSCMD_ARG_TGT_ARCH")) {
            return;
        }

        // Getting `ProgramFiles` through environment variable queries
        root = _wgetenv(L"ProgramFiles(x86)");
        if (!root) {
            root = _wgetenv(L"ProgramFiles");
        }
        if (!root) {
            return;
        }

        // Getting VS installation path using `vswhere`
        cmd = L"\"" + std::wstring(root) +
            L"\\Microsoft Visual Studio\\Installer\\vswhere.exe\""
            L" -latest -prerelease -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath";
        exec_out = exec(cmd);
        if (!exec_out) {
            return;
        }
        path = *exec_out;

        path.erase(path.find_last_not_of(L" \t\n\r\f\v") + 1);

        // Checking whether the activation script `vcvarsall.bat` exists
        path += L"\\VC\\Auxiliary\\Build";
        struct _stati64 st;
        if (_wstati64(path.c_str(), &st) == -1 || !(st.st_mode & _S_IFDIR)) {
            return;
        }
        path += L"\\vcvarsall.bat";
        if (_waccess(path.c_str(), 0) == -1) {
            return;
        }

        // Determining current platform
        if (sizeof(void*) == 8) {
            vcruntime_plat = L"x64";
        }
        else {
            vcruntime_plat = L"x86";
        }

        // Getting environment variables after activating VS development shell
        cmd = L"\"" + path + L"\" " + vcruntime_plat + L">NUL && set";
        exec_out = exec(cmd);
        if (!exec_out) {
            return;
        }
        envvars = *exec_out;
        // Setting environment variables to the current environment
        std::wistringstream f(envvars);
        std::wstring envvar;
        while (getline(f, envvar, L'\n')) {
            env_list.push_back(envvar);
        }
    }

#endif


};

#if (defined(_MSC_VER) && !defined(_M_ARM64))


static std::string getArchFlags() {
    if (__isa_available >= 6) {
        return "/arch:AVX512";
    }
    else if (__isa_available >= 5) {
        return "/arch:AVX2";
    }
    else if (__isa_available >= 4) {
        return "/arch:AVX";
    }
    else {
        return "";
    }
}
#endif

static CMD& getCMD() {
    static CMD cmd;
    return cmd;
}

void Program::disasm(std::string_view path)const
{

#ifdef _MSC_VER
    static const std::string disas_string =
        "dumpbin /DISASM:NOBYTES \"${so_file}\" > ${filename}";
#else
    static const std::string disas_string = "objdump -M  intel -d \"${so_file}\"";
#endif
    auto cmd = megu::detail::replace_first(disas_string, "${so_file}", so_file_);
    detail::replace_first_(cmd, "${filename}", path);
    auto r = getCMD().run_cmd(cmd); 
    assert(r == 0);
}
struct JitConfig {
    JitConfig() {
        
#ifdef _MSC_VER
        getCMD().activate(); 
#endif
        //this is broken lul lets just assume we always find a compiler 
        //MEGU_ENSURE(programExists(cxx),"Compiler for c++ not found");

    }

#ifdef _MSC_VER
    std::string cxx = "cl";
#elif defined(__clang__)
    std::string cxx = "clang++";
#else
    std::string cxx = "g++";
#endif

};

static JitConfig& Config() {
    static JitConfig config;
    return config;
}

constexpr static std::string getInclude(std::vector<std::string> const& dirs) {
    if (dirs.empty()) {
        return "";
    }
    std::string out_;
    for (auto const& path : dirs) {
#if defined(_MSC_VER)
        out_ += std::format(" /I\"{}\"", path);
#else
        out_ += std::format(" -I\"{}\"", path);
#endif
    }
    return out_;
}

constexpr static std::string getLib(std::vector<std::string> const& dirs) {
    if (dirs.empty()) {
        return "";
    }
    std::string out_;
    for (auto const& path : dirs) {
#if defined(_MSC_VER)
        out_ += std::format(" /LIBPATH:\"{}\"", path);
#else
        out_ += std::format(" -L\"{}\"", path);
#endif
    }
    return out_;
}

constexpr static std::string getDeps(std::vector<std::string> const& ls) {
    if (ls.empty()) {
        return "";
    }
    std::string out_;
    for (auto const& l : ls) {
#if defined(_MSC_VER)
        out_ += std::format(" {}.lib", l);
#else
        out_ += std::format(" -l{}", l);
#endif
    }
    return out_;
}

constexpr static std::string getOMP(bool enabled) {
    if (!enabled) {
        return "";
    }
#if defined(_MSC_VER)
    return "/openmp";
#else
    return "-fopenmp"
#endif
}

static inline void jit_impl(
    std::string_view cpp_file,
    std::string_view so_file,
    const CompilerArgs& args) {

    auto libs = getLib(args.getLibraryDirectories());
    auto const deps = getDeps(args.getDependencies());
    auto incs = getInclude(args.getIncludeDirectories());

#if (defined(_MSC_VER) && !defined(_M_ARM64))
    //TODO: how do i disable messages about msvc building the dlls and stuff do i just redirect to NUL is this good?
    static const std::string arch_flags = args.getArch().value_or(getArchFlags()); 
    const auto this_path = std::filesystem::absolute(cpp_file).parent_path().string();
    static const std::string compile_string = "cd /D \"" + this_path +
        "\" && "
        "${cxx} /nologo /MD /O${OPTLEVEL} " +
        arch_flags +
        " /LD /EHsc "
        " /std:${STDV} "
        "${INCLUDE}"
        "${fopenmp} \"${cpp_file}\" /link ${LIBPATH} ${ADDLINK} /out:\"${so_file}\"";

    static const WindowsDependenciesPaths WinDeps = GetWinSdkDependenciesPaths();

    libs += WinDeps.Lib;
    incs += WinDeps.Include;
#else
    static const std::string compile_string =
        "\"${cxx}\" ${INCLUDE} ${ADDLINK} -O${OPTLEVEL} -g "
        "-std=${STDV} -fPIC ${fopenmp} -shared \"${cpp_file}\" -o \"${so_file}\" -lm ${ADDLINK}";
#endif


    auto& config = Config();
    MEGU_ENSURE(
        !config.cxx.empty(),
        "Failed to jit compile c++ : Compiler not found");
    
    auto result = detail::replace_first(compile_string, "${cxx}", config.cxx);
    detail::replace_first_(result, "${fopenmp}", getOMP(args.getEnableOpenMP()));
    detail::replace_first_(result, "${cpp_file}", cpp_file);
    detail::replace_first_(result, "${so_file}", so_file);
    detail::replace_first_(result, "${LIBPATH}", libs);
    detail::replace_first_(result, "${INCLUDE}", incs); 
    detail::replace_first_(result, "${ADDLINK}", deps); 
    detail::replace_first_(result, "${OPTLEVEL}", args.getOptLevel());
    detail::replace_first_(result, "${STDV}", args.getLanguageStandard());

    auto r = getCMD().run_cmd(result);
#ifdef _WIN32
    MEGU_ENSURE(r == 0, "Failed to jit compile c++ program : ", GetLastErrorMsg());
#else
    MEGU_ENSURE(r == 0, "Failed to jit compile c++ program");
#endif
}

static inline void cleanup(std::unique_ptr<DynamicLib>& lib_, std::string& cpp_file_, std::string& so_file_) {
    lib_.reset(); 
    detail::replace_first_(cpp_file_, ".cpp", "");
    auto const dir = std::filesystem::current_path();


    auto cpp = dir / std::format("{}.cpp", cpp_file_);
    auto wxp = dir / std::format("{}.exp", cpp_file_);
    auto wlib = dir / std::format("{}.lib", cpp_file_);
    auto wobj = dir / std::format("{}.obj", cpp_file_);
    auto wdll = dir / so_file_;


    for (auto const& str : { cpp,wxp , wlib , wobj , wdll }) {
        std::filesystem::remove(str);
    }
}

Program::~Program() {
    if (delete_after_use_) {
        cleanup(lib_, cpp_file_, so_file_);
    }
}


Program::Program(CompilerArgs const& args) {
    std::string const& code = args.getCodeString();  
    std::string const& name = args.getFileName(); 
    try {
#ifdef    _MSC_VER
        auto dll_name = std::format("{}.dll", name);  
#else
        auto dll_name = std::format("{}.so", name);  
#endif
        auto cpp_name = std::format("{}.cpp", name);  
        std::ofstream so_file(dll_name); 
        std::ofstream cpp_file(cpp_name); 

        cpp_file.write(code.data(), code.size()); 
        cpp_file.flush(); 
#ifdef _MSC_VER
        so_file.close(); 
        cpp_file.close(); 
#endif
        cpp_file_ = cpp_name; 
        so_file_ = dll_name; 

        jit_impl(cpp_file_, so_file_,args);

        lib_ = std::make_unique<DynamicLib>(so_file_);
    }
    catch ([[maybe_unused]]  std::exception&  except) {
        cleanup(lib_, cpp_file_, so_file_);
        //omegalul
        throw;
    }

}



}//end megu::jit