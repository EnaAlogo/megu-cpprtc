#pragma once
#include "threadpool.hpp"
#include "program.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include "string_util.hpp"

struct LibraryContext {
    LibraryContext(IThreadPool* user_threadpool)
        :lib_(megu::CompilerArgs("./tinyreducer", get_template(user_threadpool))
#ifdef _MSC_VER
            .setArch("AVX2")
            //.setArch("SSE")
#endif
            .setLanguageStandard("c++20")
            //dot product for some [uwudacted] up reason breaks under O2 on MSVC 
            .setOptLevel("1")
            .setIncludeDirectories({ (std::filesystem::current_path()).string()})
        ),
        thread_pool_{ user_threadpool}
    {}

    float dot(float const* x, int64_t incx, float const* y, int64_t incy, int64_t size) {
        if (!functions.dot_) {
            functions.dot_ = lib_.getFunction<FnDot>("Dot");
        }
        return functions.dot_(thread_pool_, x, incx, y, incy, size);
    }

    float max(float const* x, int64_t incx, int64_t size) {
        if (!functions.max_) {
            functions.max_ = lib_.getFunction<FnType>("Max");
        }
        return functions.max_(thread_pool_, x, incx, size);
    }
    float min(float const* x, int64_t incx, int64_t size) {
        if (!functions.min_) {
            functions.min_ = lib_.getFunction<FnType>("Min");
        }
        return functions.min_(thread_pool_, x, incx, size);
    }
    float sum(float const* x, int64_t incx, int64_t size) {
        if (!functions.sum_) {
            functions.sum_ = lib_.getFunction<FnType>("Sum");
        }
        return functions.sum_(thread_pool_, x, incx, size);
    }
    float squared_sum(float const* x, int64_t incx, int64_t size) {
        if (!functions.ssum_) {
            functions.ssum_ = lib_.getFunction<FnType>("SquaredSum");
        }
        return functions.ssum_(thread_pool_, x, incx, size);
    }
    float mean(float const* x, int64_t incx, int64_t size) {
        return sum(x, incx, size) / size;
    }

    float variance(float const* x, int64_t incx, int64_t size) {
        auto const m = mean(x, incx, size);
        if (!functions.var_)
        {
            functions.var_ = lib_.getFunction<VType>("Variance");
        }
        return functions.var_(thread_pool_, x, incx, size, m);
    }

    float standard_deviation(float const* x, int64_t incx, int64_t size) {
        return std::sqrt(variance(x, incx, size));
    }

    float norm(float const* x, int64_t incx, int64_t size) {
        return std::sqrt(squared_sum(x, incx, size));
    }
    void axpy(float val, float const* x, int64_t incx, float* dst, int64_t incy, int64_t size) {
        if (!functions.axpy_) {
            functions.axpy_ = lib_.getFunction<AxpyT>("Axpy");
        }
        functions.axpy_(thread_pool_, val, x, incx, dst, incy, size);
    }

private:
    using VType = float(void*, float const*, int64_t, int64_t, float);
    using FnType = float(void*, float const*, int64_t, int64_t);
    using FnDot = float(void*, float const*, int64_t, float const*, int64_t, int64_t);
    using AxpyT = void (void* tp, float val, float const* x, int64_t incx, float* dst, int64_t incy, int64_t size);

    megu::Program lib_;
    IThreadPool* thread_pool_;

    struct {
        FnType* max_ = nullptr;
        FnType* min_ = nullptr;
        FnType* sum_ = nullptr;
        FnType* ssum_ = nullptr;
        VType* var_ = nullptr;
        FnDot* dot_ = nullptr;
        AxpyT* axpy_ = nullptr;
    }functions;

    const static int threads = 8;


    std::string get_template(IThreadPool* thread_pool_) {
        std::ifstream file("impl.hpp");
        if (file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            megu::detail::replace_first_(content, "${n_threads}", std::to_string(thread_pool_->size()));
            megu::detail::replace_first_(content, "${arch}", "AVX2");
            file.close();
            return content;
        }
        else {
            std::cerr << "Unable to open file." << std::endl;
            throw std::runtime_error("unable to open file");
        }
    }
};