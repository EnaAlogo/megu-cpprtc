#pragma once
#include <string>
#include <vector>
#include <optional>

namespace megu {
    //TDO add flags option like -Wextra -Wpedantic
    struct CompilerArgs {
        constexpr CompilerArgs(std::string full_path_and_name, std::string code)
            :file_name_(std::move(full_path_and_name)), code_(std::move(code)) {};

        constexpr std::string getOptLevel()const { return opt_level_; }

        constexpr CompilerArgs& setOptLevel(std::string level) {
            opt_level_ = std::move(level);
            return *this;
        }

        constexpr CompilerArgs& setIncludeDirectories(std::vector<std::string> dirs) {
            include_dirs_ = std::move(dirs);
            return *this;
        }
        constexpr std::vector<std::string> const& getIncludeDirectories()const {
            return include_dirs_;
        }

        constexpr CompilerArgs& setDependencies(std::vector<std::string> dirs) {
            libs_ = std::move(libs_);
            return *this;
        }
        constexpr std::vector<std::string> const& getDependencies()const {
            return libs_;
        }

        constexpr CompilerArgs& setEnableOpenMP(bool enable) {
            omp_ = enable;
            return *this;
        }
        constexpr bool getEnableOpenMP() const {
            return omp_;
        }

        constexpr CompilerArgs& setLanguageStandard(std::string std) {
            cpp_ver_ = std::move(std);
            return *this;
        }
        constexpr std::string const& getLanguageStandard()const {
            return cpp_ver_;
        }

        constexpr CompilerArgs& setLibraryDirectories(std::vector<std::string> dirs) {
            lib_dirs_ = std::move(dirs);
            return *this;
        }
        constexpr std::vector<std::string> const& getLibraryDirectories()const {
            return lib_dirs_;
        }

        constexpr std::string const& getCodeString()const {
            return code_;
        }
        constexpr std::string const& getFileName()const {
            return file_name_;
        }

        //setting the arch flag on other compilers is a bit more complicated it requries the full cpu name
        //(i have no idea maybe its not send a patch)
#if defined(_MSC_VER)
        constexpr CompilerArgs& setArch(std::string arch) {
            if (arch != "" && arch.find("/arch:") == std::string::npos) {
                arch = std::string("/arch:") + arch;
            }
            this->arch = std::move(arch);
            return *this;
        }
        constexpr std::optional<std::string> const& getArch() const {
            return arch;
        }
#endif

    private:
        //TODO make this a lil bit nicer instead of using 999 vectors and 999 strings 
        std::vector<std::string> lib_dirs_;
        std::vector<std::string> include_dirs_;
        std::vector<std::string> libs_;
        std::string code_;
        std::string file_name_;
        std::string cpp_ver_ = "c++20";
#if _MSC_VER
        std::string opt_level_ = "2";
        std::optional<std::string> arch = std::nullopt;
#else
        std::string opt_level_ = "3";
#endif
        bool omp_ = false;
    };


}//end megu