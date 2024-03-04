#pragma once
#include <optional>
#include <vector>
#include "dynamic_lib.hpp"
#include "compiler_args.hpp"

namespace megu{

    struct Program {
        Program(const Program&) = delete; 
        Program& operator=(const Program&) = delete;;

        Program(CompilerArgs const& args);

        ~Program();

        constexpr void deletedAfterUse(bool enable) {
            delete_after_use_ = enable;
        }

        void* getSymbol(std::string_view s)const {
            return lib_->symbol(s);
        }

        template<typename FnType>
        FnType* getFunction(std::string_view name) const
        {
            return static_cast<FnType*>(getSymbol(name)); 
        }

        void disasm(std::string_view path)const;

        bool isValid()const { return lib_ != nullptr; }

    private:
        std::unique_ptr<DynamicLib> lib_;
        std::string cpp_file_;
        std::string so_file_;
        bool delete_after_use_ = true;
    };


}//end megu::jit