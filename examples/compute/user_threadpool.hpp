#pragma once
#include "threadpool.hpp"
#include <mutex>
#include <thread>
#include <queue>
#include <iostream>

//Most of the threading stuff is from pytorch slightly modified to 
//use jthreads and stop tokens isntead of a stop flag and also no NUMA
//see impl.hpp for license

struct ThreadPool : public IThreadPool {
	ThreadPool(int pool_size)
		: threads_(pool_size < 0 ? std::max<int>(1, std::thread::hardware_concurrency()) : pool_size),
		complete_(true),
		available_(threads_.size()),
		total_(threads_.size()) {
		for (std::size_t i = 0; i < threads_.size(); ++i) {
			threads_[i] = std::jthread([this, i](std::stop_token t) {
				this->main_loop(t, i);
				});
		}
	}
	~ThreadPool() {
		{
			std::unique_lock<std::mutex> lock(mutex_);
			for (auto& t : threads_) {
				t.request_stop();
			}
		}
		condition_.notify_all();
	}

	constexpr size_t size() const override {
		return threads_.size();
	}

	size_t threads_available() const override{
		std::unique_lock<std::mutex> lock(mutex_);
		return available_;
	}

	void schedule(std::function<void()> func) override {
		if (threads_.empty()) {
			throw std::runtime_error("No threads to run a task");
		}
		{
			std::unique_lock<std::mutex> lock(mutex_);
			tasks_.emplace(std::move(func));
			complete_ = false;
		}
		condition_.notify_one();
	}

	void wait() {
		std::unique_lock<std::mutex> lock(mutex_);
		completed_.wait(lock, [&]() { return complete_; });
	}

	bool this_thread_in_pool() const override {
		for (auto& thread : threads_) {
			if (thread.get_id() == std::this_thread::get_id()) {
				return true;
			}
		}
		return false;
	}

private:
	void main_loop(std::stop_token token, std::size_t index) {
		std::unique_lock<std::mutex> lock(mutex_);
		while (!token.stop_requested()) {
			// Wait on condition variable while the task is empty and
			// the pool is still running.
			condition_.wait(lock, [&]() { return !tasks_.empty() || token.stop_requested(); });
			// If pool is no longer running, break out of loop.
			if (token.stop_requested()) {
				break;
			}

			// Copy task locally and remove from the queue.  This is
			// done within its own scope so that the task object is
			// destructed immediately after running the task.  This is
			// useful in the event that the function contains
			// shared_ptr arguments bound via bind.
			{
				TaskElement tasks = std::move(tasks_.front());
				tasks_.pop();
				// Decrement count, indicating thread is no longer available.
				--available_;

				lock.unlock();

				// Run the task.
				try {
					if (tasks.run_with_id) {
						tasks.with_id(index);
					}
					else {
						tasks.no_id();
					}
				}
				catch (const std::exception& e) {
					std::cerr << "Exception in thread pool task: " << e.what();
				}
				catch (...) {
					std::cerr << "Exception in thread pool task: unknown";
				}

				// Destruct tasks before taking the lock.  As tasks
				// are user provided std::function, they can run
				// arbitrary code during destruction, including code
				// that can reentrantly call into ThreadPool (which would
				// cause a deadlock if we were holding the lock).
			}

			// Update status of empty, maybe
			// Need to recover the lock first
			lock.lock();

			// Increment count, indicating thread is available.
			++available_;
			if (tasks_.empty() && available_ == total_) {
				complete_ = true;
				completed_.notify_one();
			}

			// Deliberately hold the lock on the backedge, so this thread has an
			// opportunity to acquire a new task before another thread acquires
			// the lock.
		} // while running_
	}

	struct TaskElement {
		const std::function<void()> no_id;
		const std::function<void(std::size_t)> with_id;
		bool run_with_id;

		explicit TaskElement(std::function<void()> f)
			: run_with_id(false), no_id(std::move(f)), with_id(nullptr) {}
		explicit TaskElement(std::function<void(std::size_t)> f)
			: run_with_id(true), no_id(nullptr), with_id(std::move(f)) {}
	};
	std::queue<TaskElement> tasks_;
	std::vector<std::jthread> threads_;
	mutable std::mutex mutex_;
	std::condition_variable condition_;
	std::condition_variable completed_;
	bool complete_;
	std::size_t available_;
	std::size_t total_;

};