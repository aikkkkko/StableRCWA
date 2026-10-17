// examples/rcwa_json.cpp —— JSON 进、JSON 出的命令行求解程序（stablercwa/rcwa_json.hpp 外面的一层壳）
//
//   c++ -std=c++17 -O2 -I include examples/rcwa_json.cpp -o rcwa_json          （Linux 上另加 -pthread）
//   cl /std:c++17 /O2 /EHsc /utf-8 /I include examples\rcwa_json.cpp
//
//   ./rcwa_json request.json                      # 应答写到标准输出
//   ./rcwa_json request.json --out reply.json     # 应答写进文件
//   ./rcwa_json < request.json > reply.json       # 不给文件名（或者写 -）时从标准输入读请求
//   ./rcwa_json request.json --pretty             # 应答分行、缩进，便于阅读（默认写成一行）
//
// 请求、应答的格式见 stablercwa/rcwa_json.hpp 开头的说明。
// 返回码：0 —— 应答里 ok 为 true；
//         1 —— 应答里 ok 为 false：请求有错，原因在应答的 error、where 里，标准错误上也说一句；
//         2 —— 读不到输入（文件打不开、输入是空的）、写不出应答，或者命令行写错了。这时没有应答。
//
// 文件名里可以有空格和非 ASCII 字符。Windows 上用 MSVC 编译时程序取宽字符的命令行，任何文件名都认；
// 用 MinGW 编译时默认取本机代码页的命令行，文件名里的字符要能用本机代码页表示（中文系统上的中文文件名没有问题），
// 编译时加 -municode 则与 MSVC 相同。由别的程序调用时，走标准输入、标准输出最省事，与文件名无关。
// 应答和标准错误上的提示一律是 UTF-8（提示里的文件名也换成 UTF-8）。Windows 上标准输出按文本方式写，
// 应答里的换行是 \r\n；--out 写的文件里是 \n。
#include <clocale>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "stablercwa/rcwa_json.hpp"
#include "say.hpp"

// 取宽字符命令行的条件：Windows 上的 MSVC（链接器见到 wmain 自己会选对入口），或者 MinGW 加了 -municode（它定义 UNICODE）。
#if !defined(RCWA_JSON_WIDE_ARGS)
#if defined(_WIN32) && (defined(_MSC_VER) || defined(UNICODE) || defined(_UNICODE))
#define RCWA_JSON_WIDE_ARGS 1
#else
#define RCWA_JSON_WIDE_ARGS 0
#endif
#endif
#if RCWA_JSON_WIDE_ARGS
#include <filesystem>
#endif

using examples::say;          // 用法同 std::printf，格式串里可以有中文（缘由见 say.hpp）
using examples::say_err;      // 同上，写到标准错误
namespace rcwa = stablercwa::rcwa;
namespace json = stablercwa::json;

namespace {

#if RCWA_JSON_WIDE_ARGS || defined(_WIN32)
// UTF-16（Windows）或 UTF-32 的宽字符串换成 UTF-8；落单的代理、越界的码点换成 U+FFFD。
std::string wide_to_utf8(const wchar_t* s) {
    std::string out;
    for (std::size_t i = 0; s[i] != 0; ++i) {
        unsigned long cp = static_cast<unsigned long>(s[i]);
        if (cp >= 0xD800 && cp <= 0xDBFF) {
            const unsigned long low = static_cast<unsigned long>(s[i + 1]);
            if (low >= 0xDC00 && low <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                ++i;
            } else {
                cp = 0xFFFD;
            }
        } else if ((cp >= 0xDC00 && cp <= 0xDFFF) || cp > 0x10FFFF) {
            cp = 0xFFFD;
        }
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}
#endif

// 命令行参数的字符类型，以及两个转换：换成 UTF-8（认选项、显示文件名用），换成打开文件用的路径。
#if RCWA_JSON_WIDE_ARGS
using ArgChar = wchar_t;

std::string to_utf8(const wchar_t* s) { return wide_to_utf8(s); }
std::filesystem::path file_path(const wchar_t* s) { return std::filesystem::path(s); }
#else
using ArgChar = char;

#if defined(_WIN32)
// Windows 上窄字符的命令行是本机代码页的（中文系统上是 GBK），不是 UTF-8：借 C 库按本机代码页（locale 名 ".ACP"）
// 换成宽字符，再换成 UTF-8，提示里的文件名才与其余文字是同一种编码。只在换算期间改 LC_CTYPE，用完改回；
// 换不了时原样返回。
std::string to_utf8(const char* s) {
    std::string out = s;
    const char* current = std::setlocale(LC_CTYPE, nullptr);
    const std::string saved = current ? current : "C";
    if (std::setlocale(LC_CTYPE, ".ACP") != nullptr) {
        std::vector<wchar_t> wide(std::strlen(s) + 1);            // 宽字符的个数不会比字节数多
        const std::size_t count = std::mbstowcs(wide.data(), s, wide.size());
        if (count < wide.size()) {                                // 失败时 count 是 (size_t)-1
            wide[count] = 0;
            out = wide_to_utf8(wide.data());
        }
        std::setlocale(LC_CTYPE, saved.c_str());
    }
    return out;
}
#else
std::string to_utf8(const char* s) { return s; }
#endif
std::string file_path(const char* s) { return s; }
#endif

void print_usage(bool to_stderr) {
    const char* const usage =
        "用法：rcwa_json [请求.json] [--out 应答.json] [--pretty]\n"
        "  请求.json    请求文件；不给（或者写 -）时从标准输入读\n"
        "  --out 文件   应答写进这个文件；不给时写到标准输出\n"
        "  --pretty     应答分行、缩进（默认写成一行）\n"
        "返回码：0 应答里 ok 为 true；1 应答里 ok 为 false；2 读不到输入、写不出应答或命令行有错\n";
    if (to_stderr) say_err("%s", usage);
    else say("%s", usage);
}

// 整个读进来。读的过程中出错返回 false。
bool read_stdin(std::string& out) {
    std::vector<char> buf(1 << 16);
    for (;;) {
        const std::size_t got = std::fread(buf.data(), 1, buf.size(), stdin);
        out.append(buf.data(), got);
        if (got < buf.size()) break;
    }
    return std::ferror(stdin) == 0;
}
bool read_file(const ArgChar* name, std::string& out) {
    std::ifstream in(file_path(name), std::ios::in | std::ios::binary);
    if (!in) return false;
    std::vector<char> buf(1 << 16);
    while (in.read(buf.data(), static_cast<std::streamsize>(buf.size())) || in.gcount() > 0)
        out.append(buf.data(), static_cast<std::size_t>(in.gcount()));
    return !in.bad();
}

bool write_stdout(const std::string& text) {
    return std::fwrite(text.data(), 1, text.size(), stdout) == text.size() && std::fflush(stdout) == 0;
}
bool write_file(const ArgChar* name, const std::string& text) {
    std::ofstream out(file_path(name), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.flush();
    return out.good();
}

int run(int argc, ArgChar** argv) {
    int input = 0, output = 0;                // 请求文件、应答文件在 argv 里的下标；0 表示没给
    bool pretty = false, only_files = false;  // only_files：遇到过 "--"，后面的参数一律当文件名
    for (int i = 1; i < argc; ++i) {
        const std::string arg = to_utf8(argv[i]);
        if (!only_files && arg == "--") {
            only_files = true;
        } else if (!only_files && (arg == "--help" || arg == "-h")) {
            print_usage(false);
            return 0;
        } else if (!only_files && arg == "--pretty") {
            pretty = true;
        } else if (!only_files && arg == "--out") {
            if (i + 1 >= argc || output != 0) {
                say_err("rcwa_json: --out 后面要跟一个文件名，而且只能写一次\n");
                print_usage(true);
                return 2;
            }
            output = ++i;
        } else if (!only_files && arg == "-") {
            // 标准输入：与不给文件名相同
        } else if (!only_files && arg.size() > 1 && arg[0] == '-') {
            say_err("rcwa_json: 不认识的选项 %s\n", arg.c_str());
            print_usage(true);
            return 2;
        } else {
            if (input != 0) {
                say_err("rcwa_json: 请求文件只能给一个\n");
                print_usage(true);
                return 2;
            }
            input = i;
        }
    }

    std::string request;
    if (input != 0) {
        if (!read_file(argv[input], request)) {
            say_err("rcwa_json: 读不了请求文件 %s\n", to_utf8(argv[input]).c_str());
            return 2;
        }
    } else if (!read_stdin(request)) {
        say_err("rcwa_json: 读标准输入时出错\n");
        return 2;
    }
    if (request.empty()) {
        say_err("rcwa_json: 输入是空的（%s）\n", input != 0 ? to_utf8(argv[input]).c_str() : "标准输入");
        return 2;
    }

    rcwa::JsonReply reply = rcwa::solve_json_reply(request, pretty ? 2 : -1);
    reply.text.push_back('\n');
    if (output != 0) {
        if (!write_file(argv[output], reply.text)) {
            say_err("rcwa_json: 写不了应答文件 %s\n", to_utf8(argv[output]).c_str());
            return 2;
        }
    } else if (!write_stdout(reply.text)) {
        say_err("rcwa_json: 写标准输出时出错\n");
        return 2;
    }
    if (reply.ok) return 0;

    // 请求有错：在标准错误上也说一句（应答可能进了文件，或者被别的程序接走了）
    const json::Value answer = json::parse(reply.text);
    const std::string where = answer.at("where").as_string();
    say_err("rcwa_json: 请求有错%s%s%s：%s\n", where.empty() ? "" : "（", where.c_str(), where.empty() ? "" : "）",
            answer.at("error").as_string().c_str());
    return 1;
}

int guarded_run(int argc, ArgChar** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        say_err("rcwa_json: %s\n", e.what());
    } catch (...) {
        say_err("rcwa_json: 出现了无法识别的错误\n");
    }
    return 2;
}

}  // namespace

#if RCWA_JSON_WIDE_ARGS
int wmain(int argc, wchar_t** argv) { return guarded_run(argc, argv); }
#else
int main(int argc, char** argv) { return guarded_run(argc, argv); }
#endif
