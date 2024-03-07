#include <program.hpp>
#include "library_context.hpp"
#include "user_threadpool.hpp"
#include <vector>
#include <algorithm>
#include <execution>
#include <ranges>


void benchmark(std::string_view hint , const auto& fun) {
    auto const now = std::chrono::high_resolution_clock::now();
    auto const val = fun();
    auto const then = std::chrono::high_resolution_clock::now();
    auto const time = std::chrono::duration_cast<std::chrono::microseconds>(then - now);
    std::cout << std::format("{} value : {} , elapsed {}\n", hint, val, time);
}

void std_axpy(float val, float const* x, float* dst, int64_t size) {
    auto range = std::ranges::iota_view{ 0ll ,size };
    std::for_each(std::execution::par_unseq, range.begin(), range.end(), 
        [&](int64_t idx) {
            dst[idx] += x[idx] * val;
        });
}

int main() {

    std::cout << std::fixed;

	const int sizes[] = { 1 << 1, 1 << 5 , 1 << 6 , 1 << 10, 1 << 14, 1 << 16 , 1 << 25  };

    ThreadPool tp(std::thread::hardware_concurrency());

	LibraryContext lib(&tp);

    for (auto const size : sizes) {
        std::vector<float> f_(size, 1.f);
        std::shared_ptr<float> f( (float*)::operator new(size * sizeof(float), std::align_val_t(32)), [size](void* x) {
            ::operator delete(x, std::align_val_t(32));
            });
        std::memcpy(f.get(), f_.data(), sizeof(float) * size);

        std::cout << std::format("-----------Problem Size : {}---------------\n",size);
        benchmark("sum", [&] {return lib.sum(f.get(), 1, size); });
        benchmark("variance", [&] {return lib.variance(f.get(), 1,size); });
        benchmark("max", [&] {return lib.max(f.get(), 1, size); });
        benchmark("min", [&] {return lib.min(f.get(), 1, size); });
        benchmark("self dot", [&] {return lib.dot(f.get(), 1,f.get(),1, size); });

        benchmark("axpy", [&] {lib.axpy(0.f, f.get(), 1, f.get(), 1, size); return 0; });

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

        benchmark("std axpy", [&] {
            std_axpy(0.f, f.get(), f.get(), size);
            return 0;
            });
    }

	return 0;
}