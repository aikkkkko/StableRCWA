// examples/say.hpp —— 示例程序输出中文提示用的 say() / say_err()
//
// 用法同 std::printf：say("共 %d 项\n", n) 写到标准输出，say_err(...) 写到标准错误。
//
// 为什么不直接用 std::printf：MSVC 检查 printf 系列函数的格式串时，按系统代码页（中文 Windows 上是 GBK）
// 而不是按 /utf-8 去读它，UTF-8 的汉字会被读错——报 C4819；汉字后面紧跟的 % 还可能被当成半个汉字吞掉，
// 于是误报参数个数、类型不符（C4474、C4477），真正写错的地方反而可能漏掉。经这里转一道手，MSVC 不再解析
// 格式串；GCC、Clang 仍照 printf 的规则核对参数（MinGW 除外：那里 printf 这个原型按微软 C 库的旧规则检查，
// 不认 %zu）。格式串全是 ASCII 的地方照常用 std::printf 即可。
#pragma once

#include <cstdarg>
#include <cstdio>

namespace examples {

#if defined(__GNUC__) && !defined(__MINGW32__)
[[gnu::format(printf, 1, 2)]]
#endif
inline void say(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::vfprintf(stdout, format, args);
    va_end(args);
}

#if defined(__GNUC__) && !defined(__MINGW32__)
[[gnu::format(printf, 1, 2)]]
#endif
inline void say_err(const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
}

}  // namespace examples
