#pragma once
#include <functional>

class IThreadPool {
public:
	virtual ~IThreadPool() = default;

	virtual void schedule(std::function<void()> task) = 0;

	virtual size_t size() const = 0;

	virtual size_t threads_available() const = 0;

	virtual bool this_thread_in_pool() const = 0;
};