#include <program.hpp>
#include <filesystem>
#include <vector>
#include <execution>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string_util.hpp>
#include <string>



struct LibraryContext {
    LibraryContext()
		:lib_(megu::CompilerArgs("./tinyreducer",get_template())
#ifdef _MSC_VER
		.setArch("AVX2")
#endif
		.setLanguageStandard("c++20")
        //dot product for some [uwudacted] up reason breaks under O2 on MSVC 
        .setOptLevel("1")
		),
		thread_pool_{nullptr,lib_.getFunction<void(void*)>("DestroyThreadPool")}
    {}

    float dot(float const* x, int64_t incx ,float const* y, int64_t incy, int64_t size) {
        lazyinit_threadpool();
        if (!functions.dot_) {
            functions.dot_ = lib_.getFunction<FnDot>("Dot");
        }
        return functions.dot_(thread_pool_.get(), x, incx, y,  incy, size); 
    }

    float max(float const* x, int64_t incx, int64_t size) {
        lazyinit_threadpool();
        if (!functions.max_) {
            functions.max_ = lib_.getFunction<FnType>("Max");
        }
        return functions.max_(thread_pool_.get(),x, incx, size);
    }
    float min(float const* x, int64_t incx, int64_t size) {
        lazyinit_threadpool();
        if (!functions.min_) {
            functions.min_ = lib_.getFunction<FnType>("Min");
        }
        return functions.min_(thread_pool_.get(), x, incx, size);
    }
    float sum(float const* x, int64_t incx, int64_t size) {
        lazyinit_threadpool();
        if (!functions.sum_) {
            functions.sum_ = lib_.getFunction<FnType>("Sum");
        }
        return functions.sum_(thread_pool_.get(), x, incx, size);
    }
    float squared_sum(float const* x, int64_t incx, int64_t size) {
        lazyinit_threadpool();
        if (!functions.ssum_) {
            functions.ssum_ = lib_.getFunction<FnType>("SquaredSum");
        }
        return functions.ssum_(thread_pool_.get(), x, incx, size);
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
        return functions.var_(thread_pool_.get(), x, incx, size,m);
    }

    float standard_deviation(float const* x, int64_t incx, int64_t size) {
        return std::sqrt(variance(x, incx, size));
    }

    float norm(float const* x, int64_t incx, int64_t size) {
        return std::sqrt(squared_sum(x, incx, size));
    }

private:
    using VType = float(void*, float const*, int64_t, int64_t,float);
    using FnType = float(void*, float const*, int64_t, int64_t);
    using FnDot = float(void*, float const*, int64_t, float const*, int64_t, int64_t);
    megu::Program lib_;
	std::unique_ptr<void, void(*)(void*)> thread_pool_;

    struct{
        FnType* max_ = nullptr; 
        FnType* min_ = nullptr; 
        FnType* sum_ = nullptr; 
        FnType* ssum_ = nullptr;
        VType* var_ = nullptr;
        FnDot* dot_ = nullptr; 
    }functions; 

    void lazyinit_threadpool() {
        if (thread_pool_ == nullptr) {
            thread_pool_.reset(lib_.getFunction<void* (int)>("CreateThreadPool")(std::thread::hardware_concurrency()));
        }
    }

    std::string get_template() {
        std::ifstream file("impl.hpp"); 
        if (file.is_open()) { 
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
			megu::detail::replace_first_(content, "${n_threads}", std::to_string(std::thread::hardware_concurrency()));
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

void benchmark(std::string_view hint , const auto& fun) {
    auto const now = std::chrono::high_resolution_clock::now();
    auto const val = fun();
    auto const then = std::chrono::high_resolution_clock::now();
    auto const time = std::chrono::duration_cast<std::chrono::microseconds>(then - now);
    std::cout << std::format("{} value : {} , elapsed {}\n", hint, val, time);
}


int main() {

    std::cout << std::fixed;

	const int sizes[] = { 1 << 1, 1 << 5 , 1 << 6 , 1 << 10, 1 << 14, 1 << 16 , 1 << 25  };

	LibraryContext lib;

    for (auto const size : sizes) {
        std::vector<float> f_(size, 1.f);
        std::shared_ptr<float> f((float*)_aligned_malloc(sizeof(float) * size,32), [](void* x) {_aligned_free(x); });
        std::memcpy(f.get(), f_.data(), sizeof(float) * size);

        std::cout << std::format("-----------Problem Size : {}---------------\n",size);
        benchmark("sum", [&] {return lib.sum(f.get(), 1, size); });
        benchmark("variance", [&] {return lib.variance(f.get(), 1,size); });
        benchmark("max", [&] {return lib.max(f.get(), 1, size); });
        benchmark("min", [&] {return lib.min(f.get(), 1, size); });
        benchmark("self dot", [&] {return lib.dot(f.get(), 1,f.get(),1, size); });

        benchmark("std sum", [&] {
            return std::reduce(std::execution::par_unseq,
            f.get(), f.get()+size, 0, std::plus<float>{});
            });

        benchmark("std max", [&] {
            return std::reduce(std::execution::par_unseq,
            f.get(), f.get()+size, std::numeric_limits<float>::lowest(), [](float x, float y) {return std::max(x, y); });
            });

        benchmark("std dot", [&] {
            return std::transform_reduce(std::execution::par_unseq,
            f.get(), f.get()+size, f.get(),0.f, std::plus<float>{}, std::multiplies<float>{});
            });
    }

	return 0;
}