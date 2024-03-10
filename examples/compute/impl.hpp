#pragma once
//Most of the threading stuff is from pytorch slightly modified to 
//use jthreads and stop tokens isntead of a stop flag and also no NUMA

//LICENSE:

/*From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.*/
#include <vector>
#include <thread>
#include <queue>
#include <atomic>
#include <mutex>
#include <functional>
#include <iostream>
#include <assert.h>
#include "vectypes.hpp" 
#include <array>

#include <threadpool.hpp>

constexpr static int NUM_THREADS = ${n_threads};
constexpr static Arch MAX_ARCH =   Arch::${arch};

constexpr static int64_t divup(int64_t x, int64_t y) {
	return (x + y - 1) / y;
}

using ThreadPool = IThreadPool;


static constexpr std::tuple<size_t, size_t> calc_num_tasks_and_chunk_size(
	ThreadPool const& tp,
	int64_t begin, int64_t end, int64_t grain_size) {
	if ((end - begin) < grain_size) {
		return std::make_tuple(1, std::max((int64_t)0, end - begin));
	}
	// Choose number of tasks based on grain size and number of threads.
	size_t chunk_size = divup((end - begin), tp.size() + 1);
	// Make sure each task is at least grain_size size.
	chunk_size = std::max((size_t)grain_size, chunk_size);
	size_t num_tasks = divup((end - begin), chunk_size);
	return std::make_tuple(num_tasks, chunk_size);
}

struct ParallelContext {
	static int thread_id() {
		return threadid();
	}
	static bool in_region() {
		return inregion();
	}

	struct ParallelModeGuard {
		ParallelModeGuard(bool enable) :flag_(inregion()) {
			inregion() = enable;
		}
		~ParallelModeGuard() {
			inregion() = flag_;
		}
		
	private:
		bool flag_;
	};

	struct ParallelRegionGuard {
		ParallelRegionGuard(int tid) {
			threadid() = tid;
			inregion() = true;
		}
		~ParallelRegionGuard() {
			threadid() = 0;
			inregion() = false;
		};
	};
	struct ThreadIDGuard {
		ThreadIDGuard(int tid) :ptidx_(threadid()) {
			threadid() = tid; 
		}
		~ThreadIDGuard() {
			threadid() = ptidx_; 
		};

	private:
		int ptidx_;
	};

private:
	static bool& inregion() {
		static thread_local bool in_par_ = 0;
		return in_par_;
	}
	static int& threadid() {
		static thread_local int tid = 0;
		return tid;
	}

};

template<typename Fn>
static void run_pool(ThreadPool& tp, const Fn& fn, size_t range) {
for (auto i = 1; i < range;  ++i) {
		tp.schedule([fn, i]() { fn((int)i, i); }); 
	}
	// Run the first task on the current thread directly.
	fn(0, 0);
}



static void par_impl(
	ThreadPool& tp,
	const int64_t begin,
	const int64_t end,
	const int64_t grain_size,
	const std::function<void(int64_t, int64_t)>& f) {
	
	size_t num_tasks = 0, chunk_size = 0;
	std::tie(num_tasks, chunk_size) = calc_num_tasks_and_chunk_size(tp,begin, end, grain_size);

	struct {
		std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
		std::exception_ptr eptr;
		std::mutex mutex;
		volatile size_t remaining{ 0 };
		std::condition_variable cv;
	} state;

	auto sd = [] {return -1; };


	auto task = [f, &state, begin, end, chunk_size]
	(int /**/, size_t task_id) ->void 
	{
		int64_t local_start = begin + task_id * chunk_size;
		if (local_start < end) {
			int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
			try {
				ParallelContext::ParallelRegionGuard guard(task_id); 
				f(local_start, local_end);
			}
			catch (...) {
				if (!state.err_flag.test_and_set()) {
					state.eptr = std::current_exception();
				}
			}
		}
		{
			std::unique_lock<std::mutex> lk(state.mutex);
			if (--state.remaining == 0) {
				state.cv.notify_one();
			}
		}
	};
	
	state.remaining = num_tasks;
	run_pool(tp,std::move(task), num_tasks);

	// Wait for all tasks to finish.
	{
		std::unique_lock<std::mutex> lk(state.mutex);
		if (state.remaining != 0) {
			state.cv.wait(lk);
		}
	}
	if (state.eptr) {
		std::rethrow_exception(state.eptr);
	}
}



static bool in_parallel_region(ThreadPool const& tp) {
	return ParallelContext::in_region() || tp.this_thread_in_pool();
}

#define USE_PARALLEL_ATLEAST1(tp,begin,end,grain_size)\
(((end) - (begin)) > grain_size && ((end) - (begin)>1) && !in_parallel_region(tp) &&\
(tp.size()+1) > 1)

#define USE_PARALLEL(tp,begin,end,grain_size)\
((end - begin) > grain_size && !in_parallel_region(tp) &&\
(tp.size()+1) > 1)


template <class F>
static inline void parallel_for(
	ThreadPool& tp,
	const int64_t begin,
	const int64_t end,
	const int64_t grain_size,
	const F& f) {
	assert(grain_size >= 0); 
	if (begin >= end) {
		return;
	}
	const bool use_parallel = USE_PARALLEL_ATLEAST1(tp, begin, end, grain_size); 
	if (!use_parallel) {
		ParallelContext::ThreadIDGuard tid_guard(0); 
		ParallelContext::ParallelModeGuard guard(true);  
		f(begin, end);
		return;
	}

	par_impl( 
		tp, 
		begin, end, grain_size, [&](int64_t begin, int64_t end) {
			ParallelContext::ParallelModeGuard guard(true); 
			f(begin, end);
		});
}


template<typename scalar_t>
struct AccumulationBuffer {
	
	constexpr static int size() { return BUFFER_SIZE; }

	constexpr AccumulationBuffer(scalar_t const ident) {
		for (size_t i = 0; i < BUFFER_SIZE; ++i) {
			buf_[i] = ident;
		}
	}

	constexpr void push(scalar_t const value) {
		buf_[ParallelContext::thread_id()] = value;
 	}

	template<typename SF>
	constexpr scalar_t reduce(SF const& sf,const scalar_t ident) const {
		scalar_t acc = ident;
		for (scalar_t const partial : buf_) {
			acc = sf(partial, acc);
		}
		return acc;
	}

private:
	constexpr static int BUFFER_SIZE = NUM_THREADS + 1;

	std::array<scalar_t, BUFFER_SIZE> buf_{};
};

template <class scalar_t, class F, class SF>
static inline scalar_t parallel_reduce(
	ThreadPool& tp,
	const int64_t begin,
	const int64_t end,
	const int64_t grain_size,
	const scalar_t ident,
	const F& f,
	const SF& sf) 
{
	assert(grain_size >= 0); 
	if (begin >= end) {
		return ident;
	}
	const bool use_parallel = USE_PARALLEL(tp, begin, end, grain_size);
	if (!use_parallel) {
		ParallelContext::ThreadIDGuard tid_guard(0);
		ParallelContext::ParallelModeGuard guard(true);  
		return f(begin, end); 
	}
	AccumulationBuffer buffer(ident);
	par_impl(tp,
		begin,
		end,
		grain_size,
		[&](const int64_t b, const int64_t e) {
			ParallelContext::ParallelModeGuard guard(true);
			buffer.push(f(b, e));
		}
	);
	return buffer.reduce(sf, ident);
}

#undef USE_PARALLEL
#undef USE_PARALLEL_ATLEAST1

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

//pytorch internal::GRAIN_SIZE their tests showed that theres no benefit
//in parallelizing less than this and works well cuz its a power of 2 and aligns with all simd types
constexpr static int MIN_SIZE = 32768;



#if defined(__GNUC__) // For GCC and compatible compilers
#define NO_INLINE __attribute__((noinline))
#elif defined(_MSC_VER) // For Microsoft Visual C++
#define NO_INLINE __declspec(noinline)
#else
#define NO_INLINE
#endif

template<Arch arch>
struct AlignmentRequirements {};

template<>
struct AlignmentRequirements<Arch::NONE> {
	constexpr static bool get(void const*) { return true; }
};

template<>
struct AlignmentRequirements<Arch::SSE> {
	constexpr static bool get(void const*ptr) { 
		uintptr_t constexpr mask = 15; 
		return ((uintptr_t)ptr & mask) == 0;
	}
};

template<>
struct AlignmentRequirements<Arch::AVX2> {
	constexpr static bool get(void const* ptr) {
		uintptr_t constexpr mask = 31; 
		return ((uintptr_t)ptr & mask) == 0;
	}
};

template<>
struct AlignmentRequirements<Arch::AVX512> {
	constexpr static bool get(void const* ptr) {
		uintptr_t constexpr mask = 64;
		return ((uintptr_t)ptr & mask) == 0;
	}
};

template<Arch arc = MAX_ARCH>
constexpr static Arch get_arch(void const* x) { 
	if constexpr (arc == Arch::NONE) {
		return Arch::NONE;
	}
	else {
		return AlignmentRequirements<arc>::get(x) ? arc : get_arch<static_cast<Arch>(static_cast<int>(arc) - 1)>(x);
	}
}

template<Arch arc>
struct Dot_impl {
	template<typename T, typename T2>
	auto dot_seq(T const* x, int64_t incx, T2 const* y, int64_t incy, int64_t size)
		-> std::common_type_t<T, T2>
	{
		using acc_t = std::common_type_t<T, T2>;

		acc_t ac1(0.), ac2(0.);
		size_t i;
		for (i = 0; i < size; i += 2) {
			ac1 += x[i * incx] * y[i * incy];
			ac2 += x[(i + 1) * incx] * y[(i + 1) * incy];
		}
		if (i == size - 1) {
			ac1 += x[i * incx] * y[i * incy];
		}
		acc_t outer = ac1 + ac2;
		return outer;
	}
	template<std::floating_point T>
	NO_INLINE
		T operator()(T const* x, T const* y, int64_t size) {
		if constexpr (arc == Arch::NONE) {
			return dot_seq(x, 1, y, 1, size);
		}
		else{
			using Vec = Vec<T, arc>;
			constexpr int simd_s4 = Vec::size() * 4;
			size_t i; 
			int64_t const s8 = size - size % simd_s4;
			T s;
			Vec vs1(0.f), vs2(0.f), vs3(0.f), vs4(0.f);
			for (i = 0; i < s8; i += simd_s4) {
				Vec const x0 = Vec::load(x+i);
				Vec const y0 = Vec::load(y+i);
				vs1 = x0.fmadd(y0, vs1);
				Vec const x1 = Vec::load(x + i + Vec::size());
				Vec const y1 = Vec::load(y + i + Vec::size());
				vs2 = x1.fmadd(y1, vs2);
				Vec const x2 = Vec::load(x + i + Vec::size() * 2);
				Vec const y2 = Vec::load(y + i + Vec::size() * 2);
				vs3 = x2.fmadd(y2, vs3);
				Vec const x3 = Vec::load(x + i + Vec::size() * 3);
				Vec const y3 = Vec::load(y + i + Vec::size() * 3);
				vs4 = x3.fmadd(y3, vs4);
			}
			for (s = 0; i < size; ++i) s += x[i] * y[i];
			vs1 += vs2;
			vs3 += vs4;
			vs1 += vs3;
			s += vs1.reduce_add();
			return s;
		}
	}
};



template<Arch arc>
struct Reducer {
	
	template<std::floating_point T,
		typename P,
		typename R>
	auto reduce_seq(P const& combine ,R const& reduce,T const* x, int64_t incx,int64_t size, T const ident)
		-> decltype(reduce(x[0],x[0],0))
	{
		T ac1(ident), ac2(ident);
		size_t i;
		for (i = 0; i < size; i += 2) {
			ac1 = reduce(ac1,x[i * incx],i);
			ac2 = reduce(ac2,x[(i + 1) * incx],i+1);
		}
		T outer = combine(ac1 , ac2);
		if (i == size - 1) {
			outer = combine(outer, reduce(outer,x[i * incx],i));
		}
		return outer;
	}
	
	template<std::floating_point T,
		typename P,
		typename VP,
		typename R, 
		typename V, 
		typename S>
	NO_INLINE
		auto operator()(
			P const& combine,
			VP const& vcombine,
			R const& reduce, 
			V const& vreduce,
			S const& shfl_down,
			T const* x, 
			int64_t size, 
			T ident) -> decltype(reduce(x[0],x[0],0))
	{
		if constexpr (arc == Arch::NONE) {
			return reduce_seq(combine,reduce,x, 1,size,ident);   
		}
		else {
			using Vec = Vec<T, arc>;
			constexpr int simd_s4 = Vec::size() * 4;
			size_t i;
			int64_t const s8 = size - size % simd_s4;
			float s;
			Vec vs1(ident), vs2(ident), vs3(ident), vs4(ident);
			for (i = 0; i < s8; i += simd_s4) {
				Vec const x0 = Vec::load(x+i);
				vs1 = vreduce(vs1,x0,i);
				Vec const x1 = Vec::load(x +i + Vec::size());
				vs2 = vreduce(vs2, x1,i + Vec::size());
				Vec const x2 = Vec::load(x + i + Vec::size() * 2);
				vs3 = vreduce(vs3, x2,i + Vec::size() * 2);
				Vec const x3 = Vec::load(x + i + Vec::size() * 3);
				vs4 = vreduce(vs4, x3 , i + Vec::size() * 3);
			}
			for (s = ident; i < size; ++i) s = reduce(s, x[i], i );
			vs1 = vcombine(vs1,vs2); 
			vs3 = vcombine(vs3,vs4); 
			vs1 = vcombine(vs1,vs3); 
			s = combine(s, shfl_down(vs1));  
			return s;
		}
	}
};

struct Dispatcher {
	template<typename Rt,template<Arch>typename Fn, typename...Args>
	static Rt dispatch(Arch dynamic_arch,Args&&...args) {
		return Fn<MAX_ARCH>{}(std::forward<Args>(args)...);
		//return dispatch_impl<MAX_ARCH,Rt,Fn>(dynamic_arch,std::forward<Args>(args)...); 
	}
private:
	template<Arch arc,typename Rt, template<Arch>typename Fn, typename...Args>
	static Rt dispatch_impl(Arch dynamic,Args&&...args) {
		if constexpr (arc == Arch::NONE) {
			return Fn<arc>{}(std::forward<Args>(args)...);
		}
		else {
			return arc == dynamic ? Fn<arc>{}(std::forward<Args>(args)...)
				: dispatch_impl<static_cast<Arch>(static_cast<int>(arc) - 1), Rt, Fn>(dynamic, std::forward<Args>(args)...);
		}
	}
};

template<typename T ,typename T2>
static auto dot_template(void* _tp ,T const* a,int64_t incx, T2 const* b,int64_t incy, int64_t size) ->std::common_type_t<T,T2> {
	ThreadPool* tp = reinterpret_cast<ThreadPool*>(_tp);
	const T ident = std::common_type_t < T, T2>(0.);
	return parallel_reduce(*tp, 0, size, MIN_SIZE, ident, 
		[&](int64_t start, int64_t end) -> std::common_type_t<T, T2> {
			T const* x = a + start*incx; 
			T const* y = b + start*incy;
			if (incx == 1 && incy == 1) {
				auto const arch = std::min(get_arch(x), get_arch(y));
				return Dispatcher::dispatch< std::common_type_t<T, T2>, Dot_impl>(arch, x, y, end - start);
			}
			else {
				return Dot_impl<Arch::NONE>{}.dot_seq(x, incx, y, incy, end - start);
			}

		}, std::plus<std::common_type_t<T,T2>>{}); 
}

template<typename T,
	typename Combine,
	typename VCombine,
	typename Reduce, 
	typename VReduce, 
	typename ShflDown>
static auto reduce_template(void* _tp, 
	Combine const& c,
	VCombine const& vc,
	Reduce const& r,
	VReduce const& vr ,
	ShflDown const& sf,
	T const* a, 
	int64_t incx,
	int64_t size, 
	T ident) -> decltype(r(a[0],a[0],0)) {
	ThreadPool* tp = reinterpret_cast<ThreadPool*>(_tp);
	using out_t = decltype(r(std::declval<T>(), std::declval<T>(),0));
	return parallel_reduce(*tp, 0, size, MIN_SIZE, ident, 
		[&](int64_t start, int64_t end) -> out_t {
			T const* x = a + start*incx; 
			if (incx == 1) {
				auto const arch = get_arch(x);
				return Dispatcher::dispatch<out_t, Reducer>(arch,c,vc,r,vr,sf, x,end - start,ident); 
			}
			else {
				return Reducer<Arch::NONE>{}.reduce_seq(c,r,x, incx, end - start, ident);
			}

		}, [&c](out_t x , out_t y) -> out_t{
			return c(x, y);
		}
		);
}

extern "C"
EXPORT
float SquaredSum(void* tp, float const* x, int64_t incx, int64_t size) { 
	return reduce_template(tp,
		[](float x, float y) {return x + y; },
		[](auto x, auto y) {return x+y; },
		[](float x, float y, int64_t) {return x + y*y; },
		[](auto x, auto y, int64_t) {return y.fmadd(y,x); },
		[](auto const x)->float {
			return x.reduce_add();
		},
		x,
			incx,
			size,
			0.f
			);
}


extern "C"
EXPORT
float Variance(void* tp, float const* x, int64_t incx, int64_t size,float m) {
	return reduce_template(tp,
		[](float x, float y) {return x + y; },
		[](auto x, auto y) {return x + y; },
		[m](float x, float y, int64_t) {
			auto const m_ = y - m;
			return x + (m_ * m_);
		},
		[m](auto x, auto y, int64_t) {
			auto const m_ = y - decltype(x)(m);
			return m_.fmadd(m_, x);
		},
		[](auto const x)->float {
			return x.reduce_add();
		},
		x,
		incx,
		size,
		0.f
		);
}

extern "C"
EXPORT
float Sum(void* tp, float const* x, int64_t incx, int64_t size) {
	return reduce_template(tp,
		[](float x, float y) {return x + y; },
		[](auto x, auto y) {return x+y; },
		[](float x, float y, int64_t) {return x + y; },
		[](auto x, auto y, int64_t) {return x + y; },
		[](auto const x)->float {
			return x.reduce_add();
		},
		x,
		incx,
		size,
		0.f
		);
}

extern "C"
EXPORT
float Max(void* tp ,float const* x,int64_t incx, int64_t size) {
	return reduce_template(tp, 
		[](float x, float y) {return std::max(x, y); },
		[](auto x, auto y) {return x.max(y); },
		[](float x, float y,int64_t) {return std::max(x, y); },
		[](auto x, auto y,int64_t) {return x.max(y); },
		[](auto const x)->float {
			float tmp_buffer[x.size()];
			x.store(tmp_buffer);
			float ret = std::numeric_limits<float>::lowest();
			for (size_t i = 0; i < decltype(x)::size(); ++i) {
				ret = std::max(ret, tmp_buffer[i]);
			}
			return ret;
		},
		x,
		incx,
		size,
		std::numeric_limits<float>::lowest()
		);
}

extern "C"
EXPORT
float Min(void* tp, float const* x, int64_t incx, int64_t size) {
	return reduce_template(tp,
		[](float x, float y) {return std::min(x, y); },
		[](auto x, auto y) {return x.min(y); },
		[](float x, float y, int64_t) {return std::min(x, y); },
		[](auto x, auto y, int64_t) {return x.min(y); },
		[](auto const x)->float {
			float tmp_buffer[x.size()];
			x.store(tmp_buffer);
			float ret = tmp_buffer[0];
			for (size_t i = 1; i < x.size(); ++i) {
				ret = std::min(ret, tmp_buffer[i]);
			}
			return ret;
		},
		x,
			incx,
			size,
			std::numeric_limits<float>::max()
			);
}

extern "C"
EXPORT
int64_t Dot(void* tp, float const* x, int64_t incx, float const* y , int64_t incy, int64_t size) { 
	return dot_template(tp, x, incx, y, incy, size);
}

static void axpy_seq(float val, float const* x,int64_t incx,  float* dst,int64_t incy, int64_t size) {
	for (size_t i = 0; i < size; ++i) {
		dst[i * incy] += x[incx * i] * val;
	}
}

NO_INLINE
static void axpy(float val, float const* x, float* dst, int64_t size) {
	using Vec = Vec<float, MAX_ARCH>;
	auto const mul = Vec(val);
	int64_t i = 0;
	for (; i <= size - Vec::size(); i += Vec::size()) {
		auto const iy = Vec::loadu(x + i);
		auto d1 = Vec::loadu(dst + i);
		d1 = iy.fmadd(mul, d1);
		
		d1.storeu(dst + i);

	}
	if (i == size)return;
	const int64_t remaining = size - i;
	for (int64_t j = 0; j < remaining; ++j, ++i) {
		dst[i] += val * x[i];
	}
}

extern "C"
EXPORT
void Axpy(void* tp, float val, float const* x, int64_t incx, float* dst, int64_t incy, int64_t size) {
	parallel_for(*reinterpret_cast<ThreadPool*>(tp), 0, size, MIN_SIZE, [&](int64_t b , int64_t e) {
		auto const local_size = e - b;
		auto const* local_x = x + b * incx;
		auto* local_dst = dst + b * incy;
		if (incx == 1 && incy == 1 && MAX_ARCH != Arch::NONE) {
			axpy(val, local_x, local_dst, local_size);
		}
		else {
			axpy_seq(val, local_x, incx, local_dst, incy, local_size);
		}
	});
}
