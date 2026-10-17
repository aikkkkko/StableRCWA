// stablercwa/parallel.hpp —— 极简线程池与 parallel_for（只依赖 C++17 标准库）
//
//   stablercwa::set_num_threads(8);            // 默认 1，即完全串行
//   stablercwa::parallel_for(0, n, [&](int i) { ... });
//
// 线程池在第一次并行调用时创建，之后复用。任务按连续区间均分给各线程；
// 在并行区内部（无论是工作线程还是发起并行的那个线程）再次调用 parallel_for 时直接串行执行，不会嵌套开线程。
// 几个互不相干的线程同时发起并行区时，它们排队轮流使用线程池。
//
// 写法上的一个约定（整个模块通用）：std::min / std::max 一律写成 (std::min)(a, b)。Windows 的 <windows.h> 把
// min、max 定义成了带参数的宏，用户若先包含了它又没有定义 NOMINMAX，不加这层括号的调用会被宏展开而编译不过。
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace stablercwa {
namespace detail {

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    int threads() const { return target_.load(std::memory_order_relaxed); }

    // 改线程数。要等正在进行的并行区结束才生效，所以可以在任何线程里调用——只有一处例外：
    // 不能在并行区内部（parallel_for 的循环体里）调用，那是在等自己结束，会死锁。
    void resize(int n) {
        n = (std::max)(1, n);
        std::lock_guard<std::mutex> guard(run_mutex_);
        if (n == target_.load()) return;
        stop_workers();
        target_.store(n);
        start_workers();
    }

    // 把 [begin, end) 分成至多 threads() 段，各段调用 body(lo, hi)；调用线程自己也干一段。
    template <class Body>
    void run(int begin, int end, int min_chunk, Body&& body) {
        const int n = end - begin;
        if (n <= 0) return;
        const int wanted = (std::max)(1, n / (std::max)(1, min_chunk));
        if (wanted <= 1 || inside_worker() || threads() <= 1) {
            body(begin, end);
            return;
        }
        std::unique_lock<std::mutex> guard(run_mutex_);  // 一次只跑一个并行区；线程数在持锁期间不会变
        const int parts = (std::min)(target_.load(), wanted);
        if (parts <= 1) {                                // 等锁的时候线程数被改成了 1
            guard.unlock();
            body(begin, end);
            return;
        }
        regions_started_.fetch_add(1, std::memory_order_relaxed);
        std::function<void(int, int)> fn = [&](int lo, int hi) { body(lo, hi); };
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = &fn;
            job_begin_ = begin;
            job_end_ = end;
            job_parts_ = parts;
            next_part_ = 1;                 // 第 0 段留给调用线程
            pending_ = parts - 1;
            error_ = nullptr;
            ++generation_;
        }
        wake_.notify_all();
        try {
            // 调用线程自己干第 0 段。这期间它也算“在并行区内”：段里再调用 parallel_for 时直接串行执行，
            // 否则会第二次去锁 run_mutex_ 而死锁。
            InsideGuard inside(inside_worker());
            fn(part_lo(0), part_hi(0));
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return pending_ == 0; });
        job_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

    ~ThreadPool() { stop_workers(); }

    // 调用线程当前是否在并行区内：线程池的工作线程，或正在执行第 0 段的发起线程。
    static bool in_region() { return inside_worker(); }

    // 到目前为止真正分给了多个线程的并行区个数（只在线程池里开了段的才算，串行执行的不算）。
    long long regions_started() const { return regions_started_.load(std::memory_order_relaxed); }

private:
    ThreadPool() = default;

    static bool& inside_worker() {
        thread_local bool flag = false;
        return flag;
    }
    struct InsideGuard {
        bool& flag;
        bool saved;
        explicit InsideGuard(bool& f) : flag(f), saved(f) { flag = true; }
        ~InsideGuard() { flag = saved; }
    };
    int part_lo(int p) const {
        return job_begin_ + static_cast<int>(static_cast<long long>(job_end_ - job_begin_) * p / job_parts_);
    }
    int part_hi(int p) const { return part_lo(p + 1); }

    void start_workers() {
        quit_ = false;
        for (int i = 1; i < target_.load(); ++i) workers_.emplace_back([this] { loop(); });
    }
    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            quit_ = true;
            ++generation_;
        }
        wake_.notify_all();
        for (auto& t : workers_) t.join();
        workers_.clear();
    }
    void loop() {
        inside_worker() = true;
        unsigned long long seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            wake_.wait(lock, [&] { return quit_ || generation_ != seen; });
            if (quit_) return;
            seen = generation_;
            while (job_ && next_part_ < job_parts_) {
                const int p = next_part_++;
                const int lo = part_lo(p), hi = part_hi(p);
                auto* fn = job_;
                lock.unlock();
                std::exception_ptr err;
                try {
                    (*fn)(lo, hi);
                } catch (...) {
                    err = std::current_exception();
                }
                lock.lock();
                if (err && !error_) error_ = err;
                if (--pending_ == 0) done_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_, run_mutex_;
    std::condition_variable wake_, done_;
    std::function<void(int, int)>* job_ = nullptr;
    std::exception_ptr error_;
    std::atomic<int> target_{1};
    std::atomic<long long> regions_started_{0};
    int job_begin_ = 0, job_end_ = 0, job_parts_ = 1, next_part_ = 0, pending_ = 0;
    unsigned long long generation_ = 0;
    bool quit_ = false;
};

}  // namespace detail

// 设定之后所有并行区使用的线程数（含调用线程）。n <= 1 表示串行。
inline void set_num_threads(int n) { detail::ThreadPool::instance().resize(n); }
inline int num_threads() { return detail::ThreadPool::instance().threads(); }
inline int hardware_threads() { return static_cast<int>((std::max)(1u, std::thread::hardware_concurrency())); }

// 调用线程是否已在并行区内（此时再调用 parallel_for 会直接串行执行）。会按线程数切块的算法用它判断：
// 嵌套时按一个线程做，不必再去切块、攒队列。
inline bool in_parallel_region() { return detail::ThreadPool::in_region(); }

// 到目前为止真正分给了多个线程的并行区个数（串行执行的不算）。测试用它确认并行路径确实被走到了：
// 取调用前后的差，差为零就是整个调用都在一个线程里做的。每个区只多一次宽松序的原子加，可以忽略。
#define STABLERCWA_PARALLEL_REGION_COUNT 1
inline long long parallel_regions_started() { return detail::ThreadPool::instance().regions_started(); }

// 分段版本：body(lo, hi) 处理下标 [lo, hi)。min_chunk 是每段至少包含的下标数，太小的循环不值得并行。
template <class Body>
inline void parallel_ranges(int begin, int end, Body&& body, int min_chunk = 1) {
    detail::ThreadPool::instance().run(begin, end, min_chunk, std::forward<Body>(body));
}

// 逐下标版本：body(i)。
template <class Body>
inline void parallel_for(int begin, int end, Body&& body, int min_chunk = 1) {
    parallel_ranges(begin, end, [&](int lo, int hi) { for (int i = lo; i < hi; ++i) body(i); }, min_chunk);
}

}  // namespace stablercwa
