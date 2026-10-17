// tests/test_parallel.cpp —— 线程池与 parallel_for 的测试
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

#include "stablercwa/matrix.hpp"

using namespace stablercwa;

// 文字一律经 say() 输出（用法同 std::printf），不把带中文的格式串直接交给 printf 系列函数：
// MSVC 检查 printf 的格式串时按系统代码页（中文 Windows 上是 GBK）而不是按 /utf-8 去读它，UTF-8 的汉字
// 会被读错——报 C4819；汉字后面紧跟的 % 还可能被当成半个汉字吞掉，于是误报参数个数、类型不符
// （C4474、C4477）。经 say() 转一道手，MSVC 不再解析格式串；GCC、Clang 仍照 printf 的规则核对参数
// （MinGW 除外：那里 printf 这个原型按微软 C 库的旧规则检查，不认 %zu）。
#if defined(__GNUC__) && !defined(__MINGW32__)
[[gnu::format(printf, 1, 2)]]
#endif
static void say(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
}

static int g_total = 0, g_failed = 0;

// 断言的计数与报错在这个普通函数里；宏只负责取行号和表达式的文字，宏里不出现中文。
static void check(bool ok, int line, const char* text) {
    ++g_total;
    if (!ok) {
        ++g_failed;
        say("失败：%s（第 %d 行）\n", text, line);
    }
}
#define CHECK(cond) check(static_cast<bool>(cond), __LINE__, #cond)

static Matrix test_matrix(int n, unsigned seed) {
    Matrix m(n, n);
    unsigned state = seed;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            const double re = (state >> 8) / 16777216.0 - 0.5;
            state = state * 1664525u + 1013904223u;
            const double im = (state >> 8) / 16777216.0 - 0.5;
            m(i, j) = cplx(re, im);
        }
    return m;
}

static bool same_bits(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (int j = 0; j < a.cols(); ++j)
        for (int i = 0; i < a.rows(); ++i)
            if (a(i, j) != b(i, j)) return false;
    return true;
}

int main() {
    // 1. 每个下标恰好执行一次，线程数从 1 到 8
    for (int threads : {1, 2, 3, 4, 8}) {
        set_num_threads(threads);
        CHECK(num_threads() == threads);
        std::vector<std::atomic<int>> hits(1000);
        for (auto& h : hits) h = 0;
        parallel_for(0, 1000, [&](int i) { hits[i]++; });
        bool once = true;
        for (auto& h : hits) once = once && h == 1;
        CHECK(once);
        parallel_for(5, 5, [&](int) { CHECK(false); });      // 空区间不调用
    }

    // 2. 矩阵乘法：多线程结果与单线程逐位相同
    const Matrix a = test_matrix(120, 1), b = test_matrix(120, 2);
    set_num_threads(1);
    const Matrix ref = a * b;
    for (int threads : {2, 4, 8}) {
        set_num_threads(threads);
        CHECK(same_bits(a * b, ref));
    }

    // 3. 并行区里再并行（嵌套）：外层并行、内层的矩阵乘法自动退回串行，不死锁、结果不变
    set_num_threads(4);
    {
        std::vector<Matrix> out(12);
        parallel_for(0, 12, [&](int i) { out[i] = a * b; });
        bool ok = true;
        for (const auto& m : out) ok = ok && same_bits(m, ref);
        CHECK(ok);
        std::atomic<int> count{0};
        parallel_for(0, 8, [&](int) { parallel_for(0, 50, [&](int) { count++; }); });
        CHECK(count == 400);
    }

    // 4. 几个互不相干的线程同时发起并行区
    {
        std::vector<Matrix> out(4);
        std::vector<std::thread> users;
        for (int t = 0; t < 4; ++t) users.emplace_back([&, t] { out[t] = a * b; });
        for (auto& u : users) u.join();
        bool ok = true;
        for (const auto& m : out) ok = ok && same_bits(m, ref);
        CHECK(ok);
    }

    // 5. 并行区里抛出的异常会传回调用者，之后线程池仍可用
    {
        bool caught = false;
        try {
            parallel_for(0, 100, [&](int i) { if (i == 57) throw std::runtime_error("x"); });
        } catch (const std::runtime_error&) {
            caught = true;
        }
        CHECK(caught);
        std::atomic<int> count{0};
        parallel_for(0, 100, [&](int) { count++; });
        CHECK(count == 100);
    }

    // 6. 分段版本：各段互不重叠、合起来覆盖整个区间
    {
        set_num_threads(3);
        std::vector<std::atomic<int>> hits(97);
        for (auto& h : hits) h = 0;
        parallel_ranges(0, 97, [&](int lo, int hi) { for (int i = lo; i < hi; ++i) hits[i]++; });
        bool once = true;
        for (auto& h : hits) once = once && h == 1;
        CHECK(once);
    }

    // 7. 一个线程不停地改线程数，另外两个线程同时在跑并行区：不死锁、不漏算、不重算
    {
        std::atomic<bool> stop{false};
        std::thread resizer([&] {
            int k = 0;
            while (!stop) set_num_threads(1 + (k++ % 4));
        });
        std::atomic<long long> sum{0};
        std::vector<std::thread> users;
        for (int t = 0; t < 2; ++t)
            users.emplace_back([&] {
                for (int rep = 0; rep < 300; ++rep) {
                    long long local = 0;
                    std::atomic<long long> acc{0};
                    parallel_for(0, 200, [&](int i) { acc += i; });
                    local = acc;
                    sum += local;
                }
            });
        for (auto& u : users) u.join();
        stop = true;
        resizer.join();
        CHECK(sum == 2LL * 300 * (199 * 200 / 2));
    }

    set_num_threads(1);
    say("通过 %d / 共 %d\n", g_total - g_failed, g_total);
    return g_failed ? 1 : 0;
}
