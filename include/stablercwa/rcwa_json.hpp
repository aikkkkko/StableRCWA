// stablercwa/rcwa_json.hpp —— 一维 RCWA 的 JSON 入口：请求是一段 JSON 文字，应答也是一段 JSON 文字
//
//   std::string reply = stablercwa::rcwa::solve_json(R"({
//       "mode": "planar",
//       "stack": {"nI": 1.0, "nII": 1.46, "period": 0.574,
//                 "layers": [{"d": 0.38, "edges": [0.0, 0.1722, 0.574], "n_values": [1.46, 1.0]},
//                            {"d": 0.115, "n": [1.96, 0.0]}]},
//       "M": 20, "polarization": "TE", "wavelength": 1.053, "theta_deg": 66.36})");
//
// 它只是 rcwa::solve / rcwa::sweep（rcwa1d.hpp）和 rcwa::solve_conical / rcwa::sweep_conical（rcwa1d_conical.hpp）
// 外面的一层：把请求换成这些函数的参数，把结果原样写出来。应答里的数读回来与直接调用这些函数得到的逐位相同。
// 命令行程序见 examples/rcwa_json.cpp。物理约定（级次的方向、振幅的含义、s 与 p 的定义）见那两个头文件的开头。
// 二维（交叉光栅）用 rcwa::solve2d（rcwa2d.hpp），写法见本说明最后的“二维：mode = periodic2d”一节；下面各节讲一维。
//
// 请求（各项的次序随意；不认识的键一律报错，免得拼错了悄悄被忽略）：
//   mode          "planar"：平面内入射的 TE 或 TM；"conical"：圆锥入射（方位角 φ 任意），给出各级的 Jones 矩阵；
//                 "periodic2d"：二维交叉光栅（见最后一节）。
//   stack         结构，自上而下：
//       nI          入射介质的折射率（必须是正实数）
//       nII         出射介质的折射率（conical 模式下必须是实数）
//       period      横向周期 Λ
//       layers      各层的数组，没有层时写 []。每一层有厚度 d，另外
//                     均匀层写      "n": 折射率
//                     光栅层写      "edges": [x0, x1, …, xK]（一个周期内的分段边界，递增，首尾相差一个周期）
//                                   "n_values": [n1, …, nK]（各段的折射率）
//                   一维层可选 "fourier_method": "analytic"（默认）或 "fft"；"fft_samples": N（默认 0 自动选择）。
//                   FFT 使用 pocketfft 做均匀中点采样，N >= 4*M+1，最大 4194304；均匀层仍走解析路径。
//                   两种写法不能混在同一层里。
//     折射率写成一个实数，或者 [实部, 虚部]；吸收介质的虚部是正的。
//   M             单侧截断阶数：保留 −M … M 共 2M+1 级。planar 模式不超过 8191，conical 模式不超过 4095（见 kJsonMaxMatrixOrder）。
//   polarization  "TE" 或 "TM"。planar 模式必填；conical 模式不用它。
//   入射点        两种写法取一种：
//                   "points": [{"wavelength": λ, "theta_deg": θ}, …]      至少一个点
//                   或者把 "wavelength"、"theta_deg" 直接写在最外层，表示单独一个点。
//                 conical 模式每个点另有 "phi_deg"（方位角，不写就是 0）。
//                 只有一个点时直接调用 solve / solve_conical；多个点时用 sweep / sweep_conical 把各点分给线程池。
//   input         只用于 conical 模式，可以不写：入射偏振。
//                   {"psi_deg": ψ, "delta_deg": δ}       [Es, Ep] = [sin ψ·exp(iδ), cos ψ]；delta_deg 不写就是 0
//                   或者 {"jones": [Es, Ep]}             任意的入射 Jones 矢量（s 在前），每个分量是实数或 [实部, 虚部]
//                 写了它，每个结果里多一段 polarized；不写时只给 Jones 矩阵和四个通道的效率。
//   options       可以不写，里面各项也都可以不写；不写的取 rcwa::Options 的默认值：
//       backend                 "builtin"（默认）：全用本库的运算；"mkl"：本征分解、稠密乘法、稠密线性求解都交给 oneMKL
//                               （mkl/mkl_backend.hpp，做的是 mkl::use_mkl），只有编译时定义了 STABLERCWA_WITH_MKL 才能用
//                               （见本说明末尾）。各层的折射率都是实数时打开 mkl::Settings::real_path（对称结构的层矩阵走
//                               dgeev），有一层有吸收就一律走 zgeev。MKL 用 threads 个线程；多个入射点分给线程池、或
//                               parallel_layers 把几个层分给线程池时，每次调用只用 1 个。"mkl" 时 eig 只能不写或写 "qr"，
//                               linear 只能不写或写 "lu"，写了别的是请求错误
//       eig                     "qr"（默认）或 "lr"：层矩阵的本征分解用哪种迭代
//       linear                  "lu"（默认，按列选主元）、"lu_nopivot"（不选主元）、"qr"（Householder QR）
//       homogeneous_fast_path   true（默认）/ false
//       parallel_layers         false（默认）/ true
//       diagnostics             false（默认）/ true：填写 max_eig_residual、max_solve_backward_error
//       threads                 线程数，1 ~ 256。写了就在这次调用期间 set_num_threads(threads)，结束时改回原值；
//                               不写则沿用当前的设定（进程里没有设过就是 1）。结果与线程数无关。
//
// 应答（各项的次序固定，就是下面的次序）：
//   成功：
//     {"ok": true, "version": 1, "mode": "planar", "M": 20, "orders": [-20, …, 20], "threads": 1,
//      "results": [ 每个入射点一项，次序同请求 ]}
//   planar 的每个结果：
//     wavelength、theta_deg                      照抄请求
//     R、T                                       各级复振幅，每级是 [实部, 虚部]
//     DEr、DEt                                   各级反射、透射效率
//     R_total、T_total                           效率之和
//     eig_info、n_eig、n_dense_solve、max_eig_residual、max_solve_backward_error    同 rcwa::Result 的同名成员
//     timing: {total, eig, linear, product}      用时，秒
//   conical 的每个结果：
//     wavelength、theta_deg、phi_deg             照抄请求
//     kx、ky、phi_m_deg、kzI、kzII、flux_r、flux_t    同 rcwa::ConicalResult 的同名成员（kzI、kzII 每级是 [实部, 虚部]）
//     Jr、Jt                                     各级的 Jones 矩阵，每级是 [ss, sp, ps, pp]，每个元是 [实部, 虚部]
//     DEr_channels、DEt_channels                 各级四个通道的效率，每级是 [ss, sp, ps, pp]
//     flux_unitarity                             四个复数，依次是 [0][0]、[0][1]、[1][0]、[1][1]
//     polarized（请求里有 input 时才有）：
//       {jones_in: [Es, Ep]（归一化之后的入射矢量）, Rs, Rp, Ts, Tp（各级复振幅）,
//        DEr, DEt, DEr_s, DEr_p, DEt_s, DEt_p（各级效率）, R_total, T_total}
//     eig_info … timing                          同 planar
//     n_schur                                    切换到完整 2N 阶 Schur 传播的层数
//     min_abs_lambda                             同 rcwa::ConicalResult 的同名成员：做了本征分解的各层全部本征值的模的最小值；
//                                                没有这样的层（例如纯膜系）时那个成员是 +∞，这里写成 null
//     degenerate_mode_warning                    true / false，同 rcwa::ConicalResult 的同名成员。true：这个波长、角度离圆锥入射
//                                                约化模式退化点太近，相关层改用完整 Schur 传播。警告本身不表示求解失败；
//                                                结合 n_schur、eig_info 与结果是否有限判断。详见 rcwa1d_conical.hpp 开头的说明
//   数值失败（本征迭代未收敛、方程组奇异等）不算请求出错：ok 仍为 true，那个点里受影响的数写成 null，eig_info 照实给出
//   （conical 的 min_abs_lambda、degenerate_mode_warning 在求解失败时也照常填写）。
//   请求有错：
//     {"ok": false, "error": "中文说明", "where": "出错的字段路径"}
//   路径形如 stack.layers[1].n_values[0]、points[2].theta_deg、options.eig。请求不是合法的 JSON（这时 error 里有
//   出错的行、列），或者最外层不是对象时，where 是空字符串。
//
// solve_json 不抛异常，任何错误都写进应答。它可以被多个线程同时调用；但线程数是整个进程共用的设定，
// 几个线程同时调用时不要各自在 options 里写不同的 threads。
//
// 圆锥入射的头文件 rcwa1d_conical.hpp 与本文件放在一起时自动包含；找不到它时 conical 模式的请求得到一条说明原因的错误。
// （包含本文件之前定义 STABLERCWA_RCWA_JSON_CONICAL 为 0，可以在有那个头文件的情况下也不编入圆锥入射。）
// 二维的头文件 rcwa2d.hpp 同样自动包含，找不到时 periodic2d 模式的请求得到一条说明原因的错误
// （包含本文件之前定义 STABLERCWA_RCWA_JSON_2D 为 0，可以不编入二维）。
// oneMKL 后端不会自动编入（它要链接 mkl_rt）：包含本文件之前定义 STABLERCWA_WITH_MKL 为 1（写 1，不要写 ON；CMake 的
// stablercwa_mkl 目标会定义它），才包含 mkl/mkl_backend.hpp、接受 "backend": "mkl"；没有定义时这样的请求得到一条说明原因的错误。
// 同一个程序里所有包含本文件的源文件，这三个宏的取值要一致（例如不要让一个静态库按 stablercwa 编、程序按 stablercwa_mkl 编），
// 否则同名的内联函数有两种定义，用到哪一种取决于链接。
// "backend": "mkl" 的结果与默认的只在舍入误差以内一致（实测效率差 1e-12 量级），不再是任何线程数下逐位相同；
// 详见 mkl_backend.hpp。
//
// ====================================================================================================
// 二维：mode = periodic2d（rcwa2d.hpp 的 solve2d；协议的完整规格见 gui/validation/field_reference_2d/PROTOCOL_2D.md）
//
//   {"mode": "periodic2d",
//    "stack": {"nI": 1.0, "nII": 1.46, "Lx": 0.8, "Ly": 0.8,
//              "layers": [{"d": 0.3, "y_edges": [0.0, 0.25, 0.55, 0.8],
//                          "x_profiles": [{"edges": [0.0, 0.8],              "n_values": [1.0]},
//                                         {"edges": [0.0, 0.25, 0.55, 0.8], "n_values": [1.0, 1.8, 1.0]},
//                                         {"edges": [0.0, 0.8],              "n_values": [1.0]}]},
//                         {"d": 0.115, "n": [1.96, 0.0]}]},
//    "M": 4, "My": 4,
//    "points": [{"wavelength": 1.053, "theta_deg": 42.0, "phi_deg": -25.0}],
//    "input": {"psi_deg": 90.0, "delta_deg": 0.0},
//    "options": {"eig": "qr", "factorization": "li", "homogeneous_fast_path": true,
//                "parallel_layers": false, "diagnostics": false, "threads": 1}}
//
//   请求（不认识的键一律报错；中文错误信息，where 是出错的字段路径）：
//     stack         nI（正实数）、nII（实数或 [实部, 虚部]，吸收出射介质合法）、Lx、Ly（x、y 方向的周期，正数）、layers。
//                   层：均匀层 {"d", "n"}；结构层 {"d", "y_edges": [K+1 个递增边界，首尾差一个 Ly],
//                   "x_profiles": [K 个 {"edges": [递增，首尾差一个 Lx], "n_values": [...]}]}（第 k 个剖面对应第 k 个 y 分段）。
//                   两种写法不能混用；零厚度层合法（保留）。几何非法（边界不递增、不恰好覆盖一个周期、个数不符、折射率非有限…）
//                   是请求错误，where 带层号；不写 period（二维用 Lx、Ly）。
//     M、My         x、y 方向的截断：保留 m = −M…M、n = −My…My。两个都必须写（二维没有默认的 M），My 可以是 0；
//                   总谐波数 N = (2M+1)(2My+1) 不得超过 kJsonMax2dHarmonics = 2048，在分配任何东西之前检查。
//     入射点        同一维 conical（points 数组，或者写在最外层的单点；phi_deg 不写取 0）。
//     input         二维必须写（不写不取默认值，免得悄悄取 s 偏振）：{"psi_deg", "delta_deg"} 或 {"jones": [Es, Ep]}，
//                   写法与取值同一维 conical；Jones 矢量不归一（psi/delta 写法给出的模为 1）。
//     options       backend（同一维；"mkl" 时 eig 只能不写或写 "qr"）、
//                   eig（"qr"（默认）或 "lr"；二维上 LR 常因本征残差检查而返回数值失败，见 rcwa2d_types.hpp）、
//                   factorization（"li"（默认）或 "laurent"）、homogeneous_fast_path、parallel_layers、diagnostics、threads，
//                   后四项同一维。没有 linear：二维求解没有这个选项，写了就报错。
//     不写 polarization、period（一维的写法，同时出现是请求错误）。
//   应答（各项的次序固定）：
//     {"ok": true, "version": 1, "mode": "periodic2d", "M": M, "My": My, "orders": [[m, n], …], "threads": t, "results": […]}
//     orders 长 N，次序就是 harmonic_index_2d（m 在外层、n 在内层）。每个入射点一个结果：
//       wavelength、theta_deg、phi_deg（照抄请求）、status（"ok"、"rayleigh_anomaly"、"numerical_failure"）、message（Result2D::message）、
//       Kx、Ky（实数）、kzI、kzII（[实部, 虚部]，都以 k0 为单位）、Rs、Rp、Ts、Tp、Ex_r、Ey_r、Ez_r、Ex_t、Ey_t、Ez_t（[实部, 虚部]）、
//       DEr、DEt、prop_r、prop_t（0 或 1）、eig_info、n_eig、n_dense_solve、max_eig_residual、max_solve_backward_error、
//       timing: {total, operators, eig, linear, product}。数组的长度都是 N，次序同 orders；数读回来与直接调用 solve2d 逐位相同。
//       Ez_r、Ez_t 是远场约定的 Ez（−sign·(Kx·Ex + Ky·Ey)/kz，只在传播级），与近场的 Ez 符号相反（见 rcwa2d_fields.hpp 开头）。
//       diagnostics = false 时 max_eig_residual 是 0；二维目前没有求解后向误差，max_solve_backward_error 恒为 0。
//     求解失败（status 不是 "ok"：均匀介质里 kz 恰为 0 的 Rayleigh 异常，或本征迭代未收敛等）不算请求出错：ok 仍为 true，
//     message 是 solve2d 给出的原文（Rayleigh 例含区域与 (m, n)），Kx…DEt 里所有的数写成 null（prop_r、prop_t 照常），
//     eig_info 等照实给出。请求有错（几何、M、N 超限、角度……）才是 {"ok": false, …}。
//   近场（切面、相位）见 rcwa_fields_json.hpp。
#pragma once

#include <cmath>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"
#include "parallel.hpp"
#include "rcwa1d.hpp"

#if !defined(STABLERCWA_RCWA_JSON_CONICAL)
#if __has_include("rcwa1d_conical.hpp")
#define STABLERCWA_RCWA_JSON_CONICAL 1
#else
#define STABLERCWA_RCWA_JSON_CONICAL 0
#endif
#endif
#if STABLERCWA_RCWA_JSON_CONICAL
#include <array>

#include "rcwa1d_conical.hpp"
#endif

#if !defined(STABLERCWA_RCWA_JSON_2D)
#if __has_include("rcwa2d.hpp")
#define STABLERCWA_RCWA_JSON_2D 1
#else
#define STABLERCWA_RCWA_JSON_2D 0
#endif
#endif
#if STABLERCWA_RCWA_JSON_2D
#include "rcwa2d.hpp"
#endif

#if !defined(STABLERCWA_WITH_MKL)
#define STABLERCWA_WITH_MKL 0
#endif
#if STABLERCWA_WITH_MKL
#include "mkl/mkl_backend.hpp"
#endif

namespace stablercwa {
namespace rcwa {

inline constexpr int kJsonVersion = 1;        // 应答格式的版本号
inline constexpr int kJsonMaxThreads = 256;   // options.threads 的上限
// 二维：总谐波数 N = (2M+1)(2My+1) 的上限（在分配任何东西之前检查）。GUI 自己的上限是 625；这里的 2048 对应 2N = 4096 阶稠密复矩阵。
inline constexpr int kJsonMax2dHarmonics = 2048;
// M 的上限：planar 的 2M+1、conical 的 2(2M+1)（稠密复矩阵的阶数）不得超过它，即 planar M ≤ 8191，conical M ≤ 4095；
// 这时单个矩阵是 4 GiB。缘由：macOS、Linux 上申请内存不会失败（不抛 std::bad_alloc），M 写错成几万、几亿时进程会一直涨到
// 被系统终止，连应答都没有。直接调用 rcwa::solve / solve_conical 不受这个限制。
inline constexpr int kJsonMaxMatrixOrder = 16384;

struct JsonReply {
    bool ok = false;     // 就是应答里的 ok
    std::string text;    // 应答的 JSON 文字（末尾没有换行）
};

namespace json_detail {

// 请求里的错误：在读请求的各个函数里抛出，solve_json_reply() 接住后写成应答。
struct RequestError {
    std::string message;   // 中文说明
    std::string where;     // 出错的字段路径
};
[[noreturn]] inline void fail(const std::string& where, const std::string& message) {
    throw RequestError{message, where};
}

// ------------------------------------------------------------------ 字段路径与取值

// 太长的键只显示开头一段（不把一个多字节字符切成两半）
inline std::string shown_key(const std::string& key) {
    if (key.size() <= 48) return key;
    std::size_t cut = 48;
    while (cut > 0 && (static_cast<unsigned char>(key[cut]) & 0xC0) == 0x80) --cut;
    return key.substr(0, cut) + "...";
}
inline std::string child(const std::string& path, const std::string& key) {
    return path.empty() ? key : path + "." + key;
}
inline std::string child(const std::string& path, std::size_t index) {
    return path + "[" + std::to_string(index) + "]";
}

// 对象 object（路径 path）里不许出现 allowed 之外的键
inline void check_keys(const json::Value& object, const std::string& path, std::initializer_list<const char*> allowed) {
    for (std::size_t i = 0; i < object.size(); ++i) {
        const std::string& key = object.key(i);
        bool known = false;
        for (const char* name : allowed) known = known || key == name;
        if (known) continue;
        std::string list;
        for (const char* name : allowed) {
            if (!list.empty()) list += "、";
            list += name;
        }
        fail(child(path, shown_key(key)), "不认识的键（这里可以写：" + list + "）");
    }
}

inline const json::Value& member(const json::Value& object, const char* key, const std::string& path) {
    const json::Value* v = object.find(key);
    if (!v) fail(child(path, key), "缺少这一项");
    return *v;
}
inline void need_object(const json::Value& v, const std::string& where) {
    if (!v.is_object()) fail(where, "应当是一个对象 {…}");
}
inline void need_array(const json::Value& v, const std::string& where) {
    if (!v.is_array()) fail(where, "应当是一个数组 […]");
}
inline double number(const json::Value& v, const std::string& where) {
    if (!v.is_number()) fail(where, "应当是一个数");
    return v.as_double();
}
inline int integer(const json::Value& v, const std::string& where) {
    int out = 0;
    if (!v.to_int(out)) fail(where, "应当是一个整数");
    return out;
}
inline bool truth(const json::Value& v, const std::string& where) {
    if (!v.is_bool()) fail(where, "应当是 true 或 false");
    return v.as_bool();
}
inline std::string text(const json::Value& v, const std::string& where) {
    if (!v.is_string()) fail(where, "应当是一个字符串");
    return v.as_string();
}
// 复数：一个实数，或者 [实部, 虚部]
inline cplx complex_number(const json::Value& v, const std::string& where) {
    if (v.is_number()) return cplx{v.as_double(), 0.0};
    if (!v.is_array() || v.size() != 2 || !v.at(0).is_number() || !v.at(1).is_number())
        fail(where, "应当是一个实数，或者 [实部, 虚部]");
    return cplx{v.at(0).as_double(), v.at(1).as_double()};
}

// ------------------------------------------------------------------ 请求

struct PointSpec {
    double wavelength = 1.0, theta_deg = 0.0, phi_deg = 0.0;
};

enum class InputKind { None, Angles, Jones };

struct Request {
    bool conical = false;
    Stack stack;
    int M = 0;
    Polarization polarization = Polarization::TE;
    std::vector<PointSpec> points;
    Options options;
    int threads = 0;                     // 0：请求里没写，沿用当前的线程数
    InputKind input = InputKind::None;   // 以下几项只用于 conical 与 periodic2d 模式
    double psi_deg = 0.0, delta_deg = 0.0;
    cplx Es, Ep;
    bool periodic2d = false;             // 二维：用下面几项，stack、options 不用；M 是 x 方向的截断，points、input、threads 同上
    int My = 0;
#if STABLERCWA_RCWA_JSON_2D
    Stack2D stack2d;
    Options2D options2d;
    Jones2D jones2d{};                   // 二维的入射 Jones 矢量：psi/delta 写法给出 jones_from_angles_2d，jones 写法原样（不归一）
#endif
};

// 以下各项检查与 rcwa1d.hpp 里 check_index / check_layer / check_setup 的条件相同，只是报错时带上字段路径。

inline void check_index(cplx n, const std::string& where) {
    if (!detail::is_finite(n) || std::abs(n) == 0.0 || n.real() < 0.0 || n.imag() < 0.0)
        fail(where, "折射率不能是零，实部、虚部都不能是负的（吸收介质写成 [n, k]，k > 0）");
}

inline Layer read_layer(const json::Value& v, const std::string& path, double period) {
    need_object(v, path);
    check_keys(v, path, {"d", "n", "edges", "n_values", "fourier_method", "fft_samples"});
    const double d = number(member(v, "d", path), child(path, "d"));
    if (!std::isfinite(d) || d < 0.0) fail(child(path, "d"), "层厚不能是负的");
    const json::Value* n = v.find("n");
    const json::Value* edges = v.find("edges");
    const json::Value* values = v.find("n_values");
    if (n && (edges || values)) fail(path, "均匀层只写 n，光栅层只写 edges 和 n_values，两种写法不能混在一层里");
    if (!n && !edges && !values) fail(path, "缺少折射率：均匀层写 n，光栅层写 edges 和 n_values");

    Layer layer;
    if (n) {
        const cplx uniform_index = complex_number(*n, child(path, "n"));
        check_index(uniform_index, child(path, "n"));
        layer = Layer::uniform(d, uniform_index);
    } else {
        const std::string edges_path = child(path, "edges"), values_path = child(path, "n_values");
        if (!edges) fail(edges_path, "光栅层还要写 edges（一个周期内的分段边界）");
        if (!values) fail(values_path, "光栅层还要写 n_values（各段的折射率）");
        need_array(*edges, edges_path);
        need_array(*values, values_path);
        std::vector<double> e(edges->size());
        for (std::size_t j = 0; j < e.size(); ++j) e[j] = number(edges->at(j), child(edges_path, j));
        std::vector<cplx> indices(values->size());
        for (std::size_t j = 0; j < indices.size(); ++j) {
            indices[j] = complex_number(values->at(j), child(values_path, j));
            check_index(indices[j], child(values_path, j));
        }
        if (indices.empty() || e.size() != indices.size() + 1)
            fail(values_path, "段数对不上：K 段折射率要配 K+1 个边界（现在是 " + std::to_string(indices.size()) +
                                  " 段折射率、" + std::to_string(e.size()) + " 个边界）");
        for (std::size_t j = 0; j < e.size(); ++j)
            if (!std::isfinite(e[j]) || (j > 0 && e[j] <= e[j - 1]))
                fail(child(edges_path, j), "分段边界必须严格递增");
        if (std::abs((e.back() - e.front()) - period) > 1e-10 * period)
            fail(edges_path, "分段边界的首尾必须正好相差一个周期（stack.period）");
        layer = Layer::grating(d, std::move(e), std::move(indices));
    }
    if (const json::Value* method = v.find("fourier_method")) {
        const std::string where = child(path, "fourier_method");
        const std::string name = text(*method, where);
        if (name == "analytic") layer.fourier_method = FourierMethod::Analytic;
        else if (name == "fft") layer.fourier_method = FourierMethod::FFT;
        else fail(where, "只能是 \"analytic\" 或 \"fft\"");
    }
    if (const json::Value* samples = v.find("fft_samples")) {
        const std::string where = child(path, "fft_samples");
        layer.fft_samples = integer(*samples, where);
        if (layer.fft_samples < 0 || layer.fft_samples > kMaxFFTSamples)
            fail(where, "FFT 采样数必须为 0..4194304（0 表示自动选择）");
    }
    try {                                    // 上面的检查若与求解器自己的有出入，在这里补上
        detail::check_layer(layer, period);
    } catch (const std::invalid_argument& e) {
        fail(path, std::string("这一层的参数不合法：") + e.what());
    }
    return layer;
}

inline Stack read_stack(const json::Value& v, bool conical) {
    const std::string path = "stack";
    need_object(v, path);
    check_keys(v, path, {"nI", "nII", "period", "layers"});
    Stack stack;
    stack.nI = complex_number(member(v, "nI", path), "stack.nI");
    check_index(stack.nI, "stack.nI");
    if (stack.nI.imag() != 0.0 || stack.nI.real() <= 0.0) fail("stack.nI", "入射介质必须无吸收，折射率是正实数");
    stack.nII = complex_number(member(v, "nII", path), "stack.nII");
    check_index(stack.nII, "stack.nII");
    if (conical && stack.nII.imag() != 0.0) fail("stack.nII", "圆锥入射要求出射介质的折射率是实数");
    stack.period = number(member(v, "period", path), "stack.period");
    if (!std::isfinite(stack.period) || stack.period <= 0.0) fail("stack.period", "周期必须是正数");
    const json::Value& layers = member(v, "layers", path);
    need_array(layers, "stack.layers");
    stack.layers.reserve(layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i)
        stack.layers.push_back(read_layer(layers.at(i), child("stack.layers", i), stack.period));
    return stack;
}

// 一个入射点。v 是 points 里的一项，或者（单点写在最外层时）整个请求；path 相应是 "points[k]" 或空串。
inline PointSpec read_point(const json::Value& v, const std::string& path, bool conical) {
    PointSpec p;
    p.wavelength = number(member(v, "wavelength", path), child(path, "wavelength"));
    if (!std::isfinite(p.wavelength) || p.wavelength <= 0.0) fail(child(path, "wavelength"), "波长必须是正数");
    p.theta_deg = number(member(v, "theta_deg", path), child(path, "theta_deg"));
    if (!std::isfinite(p.theta_deg) || std::abs(p.theta_deg) >= 90.0)
        fail(child(path, "theta_deg"), "入射角必须严格在 -90 度与 90 度之间");
    if (conical) {
        if (const json::Value* phi = v.find("phi_deg")) {
            p.phi_deg = number(*phi, child(path, "phi_deg"));
            if (!std::isfinite(p.phi_deg)) fail(child(path, "phi_deg"), "方位角必须是有限的数");
        }
    }
    return p;
}

inline const char* planar_phi_message() { return "planar 模式没有方位角；方位角不为零的入射请用 conical 模式"; }

inline void read_points(const json::Value& root, Request& rq) {
    const json::Value* points = root.find("points");
    const bool single = root.contains("wavelength") || root.contains("theta_deg") || root.contains("phi_deg");
    if (points && single)
        fail("points", "入射点只能用一种写法：points 数组，或者写在最外层的 wavelength、theta_deg");
    if (!points && !single)
        fail("points", "缺少入射点：写 points 数组，或者把 wavelength、theta_deg 写在最外层");
    const bool azimuth = rq.conical || rq.periodic2d;      // 有方位角的模式：conical、periodic2d
    if (!points) {                           // 最外层的键在 read_request 里已经查过
        rq.points.push_back(read_point(root, "", azimuth));
        return;
    }
    need_array(*points, "points");
    if (points->size() == 0) fail("points", "至少要有一个入射点");
    rq.points.reserve(points->size());
    for (std::size_t k = 0; k < points->size(); ++k) {
        const std::string path = child("points", k);
        const json::Value& item = points->at(k);
        need_object(item, path);
        if (azimuth) {
            check_keys(item, path, {"wavelength", "theta_deg", "phi_deg"});
        } else {
            if (item.contains("phi_deg")) fail(child(path, "phi_deg"), planar_phi_message());
            check_keys(item, path, {"wavelength", "theta_deg"});
        }
        rq.points.push_back(read_point(item, path, azimuth));
    }
}

// 各层的折射率是否都是实数（没有吸收）。出射介质不进层矩阵，不看。
inline bool layers_lossless(const Stack& stack) {
    for (const Layer& layer : stack.layers) {
        if (layer.homogeneous && layer.n.imag() != 0.0) return false;
        for (const cplx& n : layer.n_values)
            if (n.imag() != 0.0) return false;
    }
    return true;
}

// options.backend："builtin"（默认）或 "mkl"；返回是否用 MKL。编译时没有编入 MKL 后端时 "mkl" 是请求错误。
inline bool read_backend(const json::Value& v) {
    const json::Value* backend = v.find("backend");
    if (!backend) return false;
    const std::string name = text(*backend, "options.backend");
    if (name == "builtin") return false;
    if (name != "mkl") fail("options.backend", "只能是 \"builtin\"（默认）或 \"mkl\"");
#if !STABLERCWA_WITH_MKL
    fail("options.backend", "这个程序编译时没有编入 oneMKL 后端（STABLERCWA_WITH_MKL），不能用 \"mkl\"；请去掉这一项或写 \"builtin\"");
#endif
    return true;
}

// "backend": "mkl"：本征分解、稠密乘法、稠密线性求解都交给 oneMKL（mkl::use_mkl）。各层都无吸收（lossless）时打开
// real_path，对称结构的层矩阵走 dgeev：弱到与舍入误差分不开的吸收光看层矩阵认不出来，所以由这里按请求里的折射率决定
// （见 mkl_backend.hpp）。O 是 Options 或 Options2D。
template <class O>
inline void use_mkl_backend(O& options, bool lossless) {
#if STABLERCWA_WITH_MKL
    mkl::Settings settings;
    settings.real_path = lossless;
    mkl::use_mkl(options, settings);
#else
    (void)options;
    (void)lossless;
#endif
}

inline void read_options(const json::Value& v, Request& rq) {
    const std::string path = "options";
    need_object(v, path);
    check_keys(v, path, {"backend", "eig", "linear", "homogeneous_fast_path", "parallel_layers", "diagnostics", "threads"});
    const bool mkl_backend = read_backend(v);
    if (const json::Value* method = v.find("eig")) {
        const std::string name = text(*method, "options.eig");
        if (name == "qr") rq.options.eig = EigMethod::QR;
        else if (name == "lr") rq.options.eig = EigMethod::LR;
        else fail("options.eig", "只能是 \"qr\" 或 \"lr\"");
        if (mkl_backend && rq.options.eig != EigMethod::QR)
            fail("options.eig", "backend 是 \"mkl\" 时本征分解由 MKL 做，eig 只能不写或写 \"qr\"");
    }
    if (const json::Value* solver = v.find("linear")) {
        const std::string name = text(*solver, "options.linear");
        if (name == "lu") rq.options.linear = LinearSolver::LUPartial;
        else if (name == "lu_nopivot") rq.options.linear = LinearSolver::LUNoPivot;
        else if (name == "qr") rq.options.linear = LinearSolver::QRHouseholder;
        else fail("options.linear", "只能是 \"lu\"、\"lu_nopivot\" 或 \"qr\"");
        if (mkl_backend && rq.options.linear != LinearSolver::LUPartial)
            fail("options.linear", "backend 是 \"mkl\" 时线性求解由 MKL 的按列选主元 LU 做，linear 只能不写或写 \"lu\"");
    }
    if (mkl_backend) use_mkl_backend(rq.options, layers_lossless(rq.stack));   // stack 已经读过
    if (const json::Value* fast = v.find("homogeneous_fast_path"))
        rq.options.homogeneous_fast_path = truth(*fast, "options.homogeneous_fast_path");
    if (const json::Value* layers = v.find("parallel_layers"))
        rq.options.parallel_layers = truth(*layers, "options.parallel_layers");
    if (const json::Value* diagnostics = v.find("diagnostics"))
        rq.options.diagnostics = truth(*diagnostics, "options.diagnostics");
    if (const json::Value* threads = v.find("threads")) {
        rq.threads = integer(*threads, "options.threads");
        if (rq.threads < 1 || rq.threads > kJsonMaxThreads)
            fail("options.threads", "线程数应当在 1 到 " + std::to_string(kJsonMaxThreads) + " 之间");
    }
}

// conical 模式的入射偏振
inline void read_input(const json::Value& v, Request& rq) {
    const std::string path = "input";
    need_object(v, path);
    check_keys(v, path, {"psi_deg", "delta_deg", "jones"});
    if (const json::Value* jones = v.find("jones")) {
        if (v.contains("psi_deg") || v.contains("delta_deg"))
            fail(path, "入射偏振只能用一种写法：psi_deg（和 delta_deg），或者 jones");
        if (!jones->is_array() || jones->size() != 2) fail("input.jones", "应当是两个分量 [Es, Ep]（s 在前）");
        rq.Es = complex_number(jones->at(0), "input.jones[0]");
        rq.Ep = complex_number(jones->at(1), "input.jones[1]");
        const double norm = std::hypot(std::abs(rq.Es), std::abs(rq.Ep));     // 与 ConicalResult::for_input 的判断相同
        if (!(norm > 0.0) || !std::isfinite(norm)) fail("input.jones", "入射 Jones 矢量不能是零矢量");
        rq.input = InputKind::Jones;
        return;
    }
    if (!v.contains("psi_deg")) fail("input.psi_deg", "缺少这一项（入射偏振写 psi_deg，或者改写 jones）");
    rq.psi_deg = number(v.at("psi_deg"), "input.psi_deg");
    if (const json::Value* delta = v.find("delta_deg")) rq.delta_deg = number(*delta, "input.delta_deg");
    if (!std::isfinite(rq.psi_deg)) fail("input.psi_deg", "必须是有限的数");
    if (!std::isfinite(rq.delta_deg)) fail("input.delta_deg", "必须是有限的数");
    rq.input = InputKind::Angles;
}

#if STABLERCWA_RCWA_JSON_2D

// ------------------------------------------------------------------ 二维（mode = periodic2d）的请求

// 一串边界的数：必须是数组，每个元素是数。个数、递增、周期由 check_edges_2d 与调用者另查。
inline std::vector<double> read_edge_numbers(const json::Value& v, const std::string& path) {
    need_array(v, path);
    std::vector<double> e(v.size());
    for (std::size_t j = 0; j < e.size(); ++j) e[j] = number(v.at(j), child(path, j));
    return e;
}

// 边界（非空）：有限、严格递增，首尾正好相差一个周期（容差与 rcwa2d.hpp 的 check_edge_list 相同：1e-9 倍周期）。
inline void check_edges_2d(const std::vector<double>& e, const std::string& path, double period, const char* period_name) {
    for (std::size_t j = 0; j < e.size(); ++j)
        if (!std::isfinite(e[j]) || (j > 0 && e[j] <= e[j - 1])) fail(child(path, j), "分段边界必须严格递增");
    if (std::abs((e.back() - e.front()) - period) > 1e-9 * period)
        fail(path, std::string("分段边界的首尾必须正好相差一个周期（stack.") + period_name + "）");
}

inline Layer2D read_layer_2d(const json::Value& v, const std::string& path, double Lx, double Ly) {
    need_object(v, path);
    check_keys(v, path, {"d", "n", "y_edges", "x_profiles"});
    const double d = number(member(v, "d", path), child(path, "d"));
    if (!std::isfinite(d) || d < 0.0) fail(child(path, "d"), "层厚不能是负的");
    const json::Value* n = v.find("n");
    const json::Value* y_edges = v.find("y_edges");
    const json::Value* profiles = v.find("x_profiles");
    if (n && (y_edges || profiles))
        fail(path, "均匀层只写 n，结构层只写 y_edges 和 x_profiles，两种写法不能混在一层里");
    if (!n && !y_edges && !profiles) fail(path, "缺少折射率：均匀层写 n，结构层写 y_edges 和 x_profiles");

    Layer2D layer;
    if (n) {
        const cplx uniform_index = complex_number(*n, child(path, "n"));
        check_index(uniform_index, child(path, "n"));
        layer = Layer2D::uniform(d, uniform_index);
    } else {
        const std::string y_path = child(path, "y_edges"), x_path = child(path, "x_profiles");
        if (!y_edges) fail(y_path, "结构层还要写 y_edges（y 方向的分段边界）");
        if (!profiles) fail(x_path, "结构层还要写 x_profiles（每个 y 分段里沿 x 的折射率剖面）");
        std::vector<double> ye = read_edge_numbers(*y_edges, y_path);
        need_array(*profiles, x_path);
        if (profiles->size() == 0 || ye.size() != profiles->size() + 1)
            fail(x_path, "段数对不上：K 个 x 剖面要配 K+1 个 y 边界（现在是 " + std::to_string(profiles->size()) +
                             " 个 x 剖面、" + std::to_string(ye.size()) + " 个 y 边界）");
        check_edges_2d(ye, y_path, Ly, "Ly");
        std::vector<XProfile2D> rows(profiles->size());
        for (std::size_t k = 0; k < rows.size(); ++k) {
            const std::string row_path = child(x_path, k);
            const json::Value& item = profiles->at(k);
            need_object(item, row_path);
            check_keys(item, row_path, {"edges", "n_values"});
            const std::string e_path = child(row_path, "edges"), n_path = child(row_path, "n_values");
            rows[k].edges = read_edge_numbers(member(item, "edges", row_path), e_path);
            const json::Value& values = member(item, "n_values", row_path);
            need_array(values, n_path);
            rows[k].n_values.resize(values.size());
            for (std::size_t j = 0; j < values.size(); ++j) {
                rows[k].n_values[j] = complex_number(values.at(j), child(n_path, j));
                check_index(rows[k].n_values[j], child(n_path, j));
            }
            if (rows[k].n_values.empty() || rows[k].edges.size() != rows[k].n_values.size() + 1)
                fail(n_path, "段数对不上：K 段折射率要配 K+1 个边界（现在是 " + std::to_string(rows[k].n_values.size()) +
                                 " 段折射率、" + std::to_string(rows[k].edges.size()) + " 个边界）");
            check_edges_2d(rows[k].edges, e_path, Lx, "Lx");
        }
        layer = Layer2D::patterned(d, std::move(ye), std::move(rows));
    }
    try {                                    // 上面的检查若与求解器自己的有出入，在这里补上（带层的路径，路径里的下标从 0 数起）
        detail2d::check_layer_core(layer, Lx, Ly);
    } catch (const std::invalid_argument& e) {
        fail(path, path + " 的参数不合法：" + e.what());
    }
    return layer;
}

inline Stack2D read_stack_2d(const json::Value& v) {
    const std::string path = "stack";
    need_object(v, path);
    if (v.contains("period"))
        fail("stack.period", "periodic2d 模式没有 period：x、y 方向的周期写成 Lx、Ly（period 是一维的写法）");
    check_keys(v, path, {"nI", "nII", "Lx", "Ly", "layers"});
    Stack2D stack;
    const cplx nI = complex_number(member(v, "nI", path), "stack.nI");
    check_index(nI, "stack.nI");
    if (nI.imag() != 0.0 || nI.real() <= 0.0) fail("stack.nI", "入射介质必须无吸收，折射率是正实数");
    stack.nI = nI.real();
    stack.nII = complex_number(member(v, "nII", path), "stack.nII");
    check_index(stack.nII, "stack.nII");
    stack.Lx = number(member(v, "Lx", path), "stack.Lx");
    if (!std::isfinite(stack.Lx) || stack.Lx <= 0.0) fail("stack.Lx", "周期必须是正数");
    stack.Ly = number(member(v, "Ly", path), "stack.Ly");
    if (!std::isfinite(stack.Ly) || stack.Ly <= 0.0) fail("stack.Ly", "周期必须是正数");
    const json::Value& layers = member(v, "layers", path);
    need_array(layers, "stack.layers");
    stack.layers.reserve(layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i)
        stack.layers.push_back(read_layer_2d(layers.at(i), child("stack.layers", i), stack.Lx, stack.Ly));
    return stack;
}

// 二维：各结构层、均匀层的折射率是否都是实数。
inline bool layers_lossless(const Stack2D& stack) {
    for (const Layer2D& layer : stack.layers) {
        if (layer.homogeneous && layer.n.imag() != 0.0) return false;
        for (const XProfile2D& profile : layer.x_profiles)
            for (const cplx& n : profile.n_values)
                if (n.imag() != 0.0) return false;
    }
    return true;
}

inline void read_options_2d(const json::Value& v, Request& rq) {
    const std::string path = "options";
    need_object(v, path);
    if (v.contains("linear"))
        fail("options.linear", "二维不适用：二维求解没有 linear 选项（它是一维的线性求解器选项）");
    check_keys(v, path, {"backend", "eig", "factorization", "homogeneous_fast_path", "parallel_layers", "diagnostics", "threads"});
    const bool mkl_backend = read_backend(v);
    if (const json::Value* method = v.find("eig")) {
        const std::string name = text(*method, "options.eig");
        if (name == "qr") rq.options2d.eig = EigMethod::QR;
        else if (name == "lr") rq.options2d.eig = EigMethod::LR;
        else fail("options.eig", "只能是 \"qr\" 或 \"lr\"");
        if (mkl_backend && rq.options2d.eig != EigMethod::QR)
            fail("options.eig", "backend 是 \"mkl\" 时本征分解由 MKL 做，eig 只能不写或写 \"qr\"");
    }
    if (mkl_backend) use_mkl_backend(rq.options2d, layers_lossless(rq.stack2d));   // stack2d 已经读过
    if (const json::Value* rule = v.find("factorization")) {
        const std::string name = text(*rule, "options.factorization");
        if (name == "li") rq.options2d.factorization = Factorization2D::Li;
        else if (name == "laurent") rq.options2d.factorization = Factorization2D::Laurent;
        else fail("options.factorization", "只能是 \"li\" 或 \"laurent\"");
    }
    if (const json::Value* fast = v.find("homogeneous_fast_path"))
        rq.options2d.homogeneous_fast_path = truth(*fast, "options.homogeneous_fast_path");
    if (const json::Value* layers = v.find("parallel_layers"))
        rq.options2d.parallel_layers = truth(*layers, "options.parallel_layers");
    if (const json::Value* diagnostics = v.find("diagnostics"))
        rq.options2d.diagnostics = truth(*diagnostics, "options.diagnostics");
    if (const json::Value* threads = v.find("threads")) {
        rq.threads = integer(*threads, "options.threads");
        if (rq.threads < 1 || rq.threads > kJsonMaxThreads)
            fail("options.threads", "线程数应当在 1 到 " + std::to_string(kJsonMaxThreads) + " 之间");
    }
}

// 截断阶数 M 或 My（name 是字段名）：不能是负的；单独一个方向的 2M+1 就不能超过总谐波数的上限
inline void check_truncation_2d(int order, const char* name) {
    const long long max_order = (kJsonMax2dHarmonics - 1) / 2;
    if (order < 0 || order > max_order)
        fail(name, std::string("截断阶数不能是负的，也不能超过 ") + std::to_string(max_order) +
                       "（总谐波数 (2M+1)(2My+1) 不得超过 " + std::to_string(kJsonMax2dHarmonics) + "）");
}

inline Request read_request_2d(const json::Value& root) {
    Request rq;
    rq.periodic2d = true;
    if (root.contains("polarization"))
        fail("polarization", "periodic2d 模式不用 polarization：入射偏振写在 input 里（psi_deg、delta_deg 或 jones）");
    if (root.contains("period"))
        fail("period", "periodic2d 模式没有 period：x、y 方向的周期写成 stack.Lx、stack.Ly");
    check_keys(root, "", {"mode", "stack", "M", "My", "points", "wavelength", "theta_deg", "phi_deg", "input", "options"});

    rq.stack2d = read_stack_2d(member(root, "stack", ""));

    rq.M = integer(member(root, "M", ""), "M");
    check_truncation_2d(rq.M, "M");
    rq.My = integer(member(root, "My", ""), "My");
    check_truncation_2d(rq.My, "My");
    const long long harmonics = (2LL * rq.M + 1) * (2LL * rq.My + 1);
    if (harmonics > kJsonMax2dHarmonics)
        fail("My", "总谐波数 N = (2M+1)(2My+1) = " + std::to_string(harmonics) + " 超过了上限 " +
                       std::to_string(kJsonMax2dHarmonics));

    read_points(root, rq);
    const json::Value* input = root.find("input");
    if (!input) fail("input", "缺少这一项：二维必须指定入射偏振，写 psi_deg（和 delta_deg），或者 jones");
    read_input(*input, rq);
    rq.jones2d = rq.input == InputKind::Angles ? jones_from_angles_2d(rq.psi_deg, rq.delta_deg) : Jones2D{rq.Es, rq.Ep};
    if (const json::Value* options = root.find("options")) read_options_2d(*options, rq);
    return rq;
}

#endif   // STABLERCWA_RCWA_JSON_2D

inline Request read_request(const json::Value& root) {
    if (!root.is_object()) fail("", "请求的最外层应当是一个对象 {…}");
    Request rq;
    const std::string mode = text(member(root, "mode", ""), "mode");
    if (mode == "conical") rq.conical = true;
#if STABLERCWA_RCWA_JSON_2D
    else if (mode == "periodic2d") return read_request_2d(root);
    else if (mode != "planar")
        fail("mode", "只能是 \"planar\"（平面内入射，TE 或 TM）、\"conical\"（圆锥入射）或 \"periodic2d\"（二维交叉光栅）");
#else
    else if (mode == "periodic2d")
        fail("mode", "这份程序编译时没有包含二维的头文件 rcwa2d.hpp，算不了 periodic2d 模式");
    else if (mode != "planar") fail("mode", "只能是 \"planar\"（平面内入射，TE 或 TM）或 \"conical\"（圆锥入射）");
#endif
#if !STABLERCWA_RCWA_JSON_CONICAL
    if (rq.conical)
        fail("mode", "这份程序编译时没有包含圆锥入射的头文件 rcwa1d_conical.hpp，算不了 conical 模式");
#endif

    if (rq.conical) {
        if (root.contains("polarization"))
            fail("polarization", "conical 模式不用 polarization：入射偏振写在 input 里；不写 input 时给出各级的 Jones 矩阵");
        check_keys(root, "", {"mode", "stack", "M", "points", "wavelength", "theta_deg", "phi_deg", "input", "options"});
    } else {
        if (root.contains("phi_deg")) fail("phi_deg", planar_phi_message());
        if (root.contains("input"))
            fail("input", "planar 模式的入射偏振由 polarization 指定；input 只用于 conical 模式");
        check_keys(root, "", {"mode", "stack", "M", "polarization", "points", "wavelength", "theta_deg", "options"});
    }

    rq.stack = read_stack(member(root, "stack", ""), rq.conical);

    rq.M = integer(member(root, "M", ""), "M");
    const int max_order = (kJsonMaxMatrixOrder / (rq.conical ? 2 : 1) - 1) / 2;   // 比 rcwa1d.hpp 的 check_setup 严：见 kJsonMaxMatrixOrder
    if (rq.M < 0 || rq.M > max_order)
        fail("M", "截断阶数不能是负的，也不能超过 " + std::to_string(max_order) + "（再大，单个稠密矩阵就超过 4 GiB）");

    for (std::size_t j = 0; j < rq.stack.layers.size(); ++j) {
        const Layer& layer = rq.stack.layers[j];
        if (!layer.homogeneous && layer.fourier_method == FourierMethod::FFT) {
            try { (void)resolve_fft_samples(rq.M, layer.fft_samples); }
            catch (const std::invalid_argument& e) { fail(child(child("stack.layers", j), "fft_samples"), e.what()); }
        }
    }

    if (!rq.conical) {
        const std::string name = text(member(root, "polarization", ""), "polarization");
        if (name == "TE") rq.polarization = Polarization::TE;
        else if (name == "TM") rq.polarization = Polarization::TM;
        else fail("polarization", "只能是 \"TE\" 或 \"TM\"");
    }

    read_points(root, rq);
    if (rq.conical) {
        if (const json::Value* input = root.find("input")) read_input(*input, rq);
    }
    if (const json::Value* options = root.find("options")) read_options(*options, rq);

    // 上面的检查若与求解器自己的有出入，在这里补上：逐点用求解器的 check_setup 再过一遍
    for (std::size_t k = 0; k < rq.points.size(); ++k) {
        try {
            detail::check_setup(rq.stack.period, rq.points[k].wavelength, rq.stack.nI, rq.points[k].theta_deg, rq.M,
                                rq.polarization);
        } catch (const std::invalid_argument& e) {
            fail(root.contains("points") ? child("points", k) : std::string(),
                 std::string("求解器不接受这组参数：") + e.what());
        }
    }
    return rq;
}

// ------------------------------------------------------------------ 线程数

// 请求里写了 threads 时临时改线程数，离开时改回原值。
class ThreadSetting {
public:
    explicit ThreadSetting(int wanted) : saved_(num_threads()) {
        if (wanted <= 0 || wanted == saved_) return;
        try {
            set_num_threads(wanted);
            changed_ = true;
        } catch (...) {                      // 线程开不出来：先把线程池恢复原状，再报错
            restore();
            fail("options.threads", "开不出这么多线程");
        }
    }
    ~ThreadSetting() {
        if (changed_) restore();
    }
    ThreadSetting(const ThreadSetting&) = delete;
    ThreadSetting& operator=(const ThreadSetting&) = delete;

private:
    void restore() noexcept {
        try {
            set_num_threads(saved_);
        } catch (...) {
            try {
                set_num_threads(1);          // 串行不需要创建线程，不会失败
            } catch (...) {
            }
        }
    }
    int saved_;
    bool changed_ = false;
};

// 调用求解器。走到这里的请求已经查过参数，求解器不应当再抛异常；真抛了也写成应答，不往外传。
template <class Call>
inline auto guarded(Call&& call) -> decltype(call()) {
    try {
        return call();
    } catch (const std::invalid_argument& e) {
        fail("", std::string("求解器不接受这组参数：") + e.what());
    } catch (const std::bad_alloc&) {
        fail("M", "内存不够：截断阶数 M 太大");
    } catch (const std::length_error&) {
        fail("M", "内存不够：截断阶数 M 太大");
    }
}

// ------------------------------------------------------------------ 应答

inline void write_complex(json::Writer& w, cplx z) {
    w.begin_array(true);
    w.value(z.real());
    w.value(z.imag());
    w.end_array();
}
inline void write_complex_array(json::Writer& w, const Vector& v) {
    w.begin_array(true);
    for (const cplx& z : v) write_complex(w, z);
    w.end_array();
}
inline void write_real_array(json::Writer& w, const RealVector& v) {
    w.begin_array(true);
    for (double x : v) w.value(x);
    w.end_array();
}

// 应答开头到 "results": [ 为止
inline void write_head(json::Writer& w, const Request& rq) {
    w.begin_object();
    w.key("ok");
    w.value(true);
    w.key("version");
    w.value(kJsonVersion);
    w.key("mode");
    w.value(rq.periodic2d ? "periodic2d" : rq.conical ? "conical" : "planar");
    w.key("M");
    w.value(rq.M);
    if (rq.periodic2d) {                     // 二维：再写 My，orders 是 [m, n] 对，次序同 harmonic_index_2d（m 在外层、n 在内层）
        w.key("My");
        w.value(rq.My);
        w.key("orders");
        w.begin_array(true);
        for (int m = -rq.M; m <= rq.M; ++m)
            for (int n = -rq.My; n <= rq.My; ++n) {
                w.begin_array(true);
                w.value(m);
                w.value(n);
                w.end_array();
            }
        w.end_array();
    } else {
        w.key("orders");
        w.begin_array(true);
        for (int m = -rq.M; m <= rq.M; ++m) w.value(m);
        w.end_array();
    }
    w.key("threads");
    w.value(num_threads());
    w.key("results");
    w.begin_array();
}
inline JsonReply finish(json::Writer& w) {
    w.end_array();
    w.end_object();
    JsonReply reply;
    reply.ok = true;
    reply.text = w.take();
    return reply;
}

// 每个结果末尾的几项：本征求解的 info、计数、诊断量、用时。Result 与 ConicalResult 的这几个成员同名。
template <class AnyResult>
inline void write_bookkeeping(json::Writer& w, const AnyResult& r) {
    w.key("eig_info");
    w.value(r.eig_info);
    w.key("n_eig");
    w.value(r.n_eig);
    w.key("n_dense_solve");
    w.value(r.n_dense_solve);
    w.key("max_eig_residual");
    w.value(r.max_eig_residual);
    w.key("max_solve_backward_error");
    w.value(r.max_solve_backward_error);
    w.key("timing");
    w.begin_object(true);
    w.key("total");
    w.value(r.timing.total);
    w.key("eig");
    w.value(r.timing.eig);
    w.key("linear");
    w.value(r.timing.linear);
    w.key("product");
    w.value(r.timing.product);
    w.end_object();
}

inline void write_result(json::Writer& w, const PointSpec& p, const Result& r) {
    w.begin_object();
    w.key("wavelength");
    w.value(p.wavelength);
    w.key("theta_deg");
    w.value(p.theta_deg);
    w.key("R");
    write_complex_array(w, r.R);
    w.key("T");
    write_complex_array(w, r.T);
    w.key("DEr");
    write_real_array(w, r.DEr);
    w.key("DEt");
    write_real_array(w, r.DEt);
    w.key("R_total");
    w.value(r.reflectance());
    w.key("T_total");
    w.value(r.transmittance());
    write_bookkeeping(w, r);
    w.end_object();
}

inline JsonReply run_planar(const Request& rq, int indent) {
    const std::vector<Result> results = guarded([&] {
        std::vector<Result> out;
        if (rq.points.size() == 1) {
            out.push_back(solve(rq.stack, rq.points[0].wavelength, rq.points[0].theta_deg, rq.M, rq.polarization,
                                rq.options));
        } else {
            std::vector<Point> points(rq.points.size());
            for (std::size_t k = 0; k < points.size(); ++k)
                points[k] = Point{rq.points[k].wavelength, rq.points[k].theta_deg};
            out = sweep(rq.stack, points, rq.M, rq.polarization, rq.options);
        }
        return out;
    });
    json::Writer w(indent);
    write_head(w, rq);
    for (std::size_t k = 0; k < results.size(); ++k) write_result(w, rq.points[k], results[k]);
    return finish(w);
}

#if STABLERCWA_RCWA_JSON_CONICAL

inline void write_jones_array(json::Writer& w, const std::vector<Jones>& v) {
    w.begin_array(true);
    for (const Jones& j : v) {
        w.begin_array(true);
        write_complex(w, j.ss);
        write_complex(w, j.sp);
        write_complex(w, j.ps);
        write_complex(w, j.pp);
        w.end_array();
    }
    w.end_array();
}
inline void write_channel_array(json::Writer& w, const std::vector<JonesEfficiency>& v) {
    w.begin_array(true);
    for (const JonesEfficiency& e : v) {
        w.begin_array(true);
        w.value(e.ss);
        w.value(e.sp);
        w.value(e.ps);
        w.value(e.pp);
        w.end_array();
    }
    w.end_array();
}

inline void write_result(json::Writer& w, const Request& rq, const PointSpec& p, const ConicalResult& r) {
    w.begin_object();
    w.key("wavelength");
    w.value(p.wavelength);
    w.key("theta_deg");
    w.value(p.theta_deg);
    w.key("phi_deg");
    w.value(p.phi_deg);
    w.key("kx");
    write_real_array(w, r.kx);
    w.key("ky");
    w.value(r.ky);
    w.key("phi_m_deg");
    write_real_array(w, r.phi_m_deg);
    w.key("kzI");
    write_complex_array(w, r.kzI);
    w.key("kzII");
    write_complex_array(w, r.kzII);
    w.key("flux_r");
    write_real_array(w, r.flux_r);
    w.key("flux_t");
    write_real_array(w, r.flux_t);
    w.key("Jr");
    write_jones_array(w, r.Jr);
    w.key("Jt");
    write_jones_array(w, r.Jt);
    w.key("DEr_channels");
    write_channel_array(w, r.DEr);
    w.key("DEt_channels");
    write_channel_array(w, r.DEt);
    w.key("flux_unitarity");
    const std::array<cplx, 4> unitarity = r.flux_unitarity();
    w.begin_array(true);
    for (const cplx& u : unitarity) write_complex(w, u);
    w.end_array();
    if (rq.input != InputKind::None) {
        const ConicalPolarized c = rq.input == InputKind::Angles ? r.for_angles(rq.psi_deg, rq.delta_deg)
                                                                 : r.for_input(rq.Es, rq.Ep);
        w.key("polarized");
        w.begin_object();
        w.key("jones_in");
        w.begin_array(true);
        write_complex(w, c.Es);
        write_complex(w, c.Ep);
        w.end_array();
        w.key("Rs");
        write_complex_array(w, c.Rs);
        w.key("Rp");
        write_complex_array(w, c.Rp);
        w.key("Ts");
        write_complex_array(w, c.Ts);
        w.key("Tp");
        write_complex_array(w, c.Tp);
        w.key("DEr");
        write_real_array(w, c.DEr);
        w.key("DEt");
        write_real_array(w, c.DEt);
        w.key("DEr_s");
        write_real_array(w, c.DEr_s);
        w.key("DEr_p");
        write_real_array(w, c.DEr_p);
        w.key("DEt_s");
        write_real_array(w, c.DEt_s);
        w.key("DEt_p");
        write_real_array(w, c.DEt_p);
        w.key("R_total");
        w.value(c.reflectance());
        w.key("T_total");
        w.value(c.transmittance());
        w.end_object();
    }
    write_bookkeeping(w, r);
    w.key("n_schur");
    w.value(r.n_schur);
    w.key("min_abs_lambda");                 // 没有做本征分解的层时是 +∞，写出来是 null
    w.value(r.min_abs_lambda);
    w.key("degenerate_mode_warning");
    w.value(r.degenerate_mode_warning);
    w.end_object();
}

inline JsonReply run_conical(const Request& rq, int indent) {
    json::Writer w(indent);
    guarded([&] {                            // for_input 也可能抛 std::invalid_argument，所以连同写结果一起包住
        std::vector<ConicalResult> results;
        if (rq.points.size() == 1) {
            results.push_back(solve_conical(rq.stack, rq.points[0].wavelength, rq.points[0].theta_deg,
                                            rq.points[0].phi_deg, rq.M, rq.options));
        } else {
            std::vector<ConicalPoint> points(rq.points.size());
            for (std::size_t k = 0; k < points.size(); ++k)
                points[k] = ConicalPoint{rq.points[k].wavelength, rq.points[k].theta_deg, rq.points[k].phi_deg};
            results = sweep_conical(rq.stack, points, rq.M, rq.options);
        }
        write_head(w, rq);
        for (std::size_t k = 0; k < results.size(); ++k) write_result(w, rq, rq.points[k], results[k]);
    });
    return finish(w);
}

#endif   // STABLERCWA_RCWA_JSON_CONICAL

#if STABLERCWA_RCWA_JSON_2D

inline const char* status_name_2d(Status2D status) {
    switch (status) {
    case Status2D::Ok: return "ok";
    case Status2D::RayleighAnomaly: return "rayleigh_anomaly";
    case Status2D::NumericalFailure: return "numerical_failure";
    }
    return "numerical_failure";              // 枚举之外的值（不应出现）：按失败处理
}

// 求解失败时的数组：count 个 null（复数数组里每一项是 [null, null]，与一维里写 NaN 的复数一样）
inline void write_null_array(json::Writer& w, std::size_t count, bool complex_values) {
    w.begin_array(true);
    for (std::size_t j = 0; j < count; ++j) {
        if (!complex_values) {
            w.null();
            continue;
        }
        w.begin_array(true);
        w.null();
        w.null();
        w.end_array();
    }
    w.end_array();
}

inline void write_flags(json::Writer& w, const std::vector<char>& flags) {
    w.begin_array(true);
    for (char flag : flags) w.value(flag != 0 ? 1 : 0);
    w.end_array();
}

// 二维的一个结果；count 是谐波总数 N。status 不是 Ok 时所有数值数组写成 null（prop_r、prop_t 照常），其余各项照实给出。
inline void write_result_2d(json::Writer& w, const PointSpec& p, const Result2D& r, std::size_t count) {
    const bool solved = r.status == Status2D::Ok;
    const auto reals = [&](const char* key, const RealVector& v) {
        w.key(key);
        if (solved) write_real_array(w, v);
        else write_null_array(w, count, false);
    };
    const auto complexes = [&](const char* key, const Vector& v) {
        w.key(key);
        if (solved) write_complex_array(w, v);
        else write_null_array(w, count, true);
    };
    w.begin_object();
    w.key("wavelength");
    w.value(p.wavelength);
    w.key("theta_deg");
    w.value(p.theta_deg);
    w.key("phi_deg");
    w.value(p.phi_deg);
    w.key("status");
    w.value(status_name_2d(r.status));
    w.key("message");
    w.value(r.message);
    reals("Kx", r.Kx);
    reals("Ky", r.Ky);
    complexes("kzI", r.kzI);
    complexes("kzII", r.kzII);
    complexes("Rs", r.Rs);
    complexes("Rp", r.Rp);
    complexes("Ts", r.Ts);
    complexes("Tp", r.Tp);
    complexes("Ex_r", r.Ex_r);
    complexes("Ey_r", r.Ey_r);
    complexes("Ez_r", r.Ez_r);
    complexes("Ex_t", r.Ex_t);
    complexes("Ey_t", r.Ey_t);
    complexes("Ez_t", r.Ez_t);
    reals("DEr", r.DEr);
    reals("DEt", r.DEt);
    w.key("prop_r");
    write_flags(w, r.prop_r);
    w.key("prop_t");
    write_flags(w, r.prop_t);
    w.key("eig_info");
    w.value(r.eig_info);
    w.key("n_eig");
    w.value(r.n_eig);
    w.key("n_dense_solve");
    w.value(r.n_dense_solve);
    w.key("max_eig_residual");
    w.value(r.max_eig_residual);
    w.key("max_solve_backward_error");       // 二维目前没有求解后向误差：恒为 0
    w.value(0.0);
    w.key("timing");
    w.begin_object(true);
    w.key("total");
    w.value(r.timing.total);
    w.key("operators");
    w.value(r.timing.operators);
    w.key("eig");
    w.value(r.timing.eig);
    w.key("linear");
    w.value(r.timing.linear);
    w.key("product");
    w.value(r.timing.product);
    w.end_object();
    w.end_object();
}

// 谐波总数 N（请求已检查过 ≤ kJsonMax2dHarmonics）
inline std::size_t harmonic_count_2d(const Request& rq) {
    return static_cast<std::size_t>(2 * rq.M + 1) * static_cast<std::size_t>(2 * rq.My + 1);
}

inline JsonReply run_periodic2d(const Request& rq, int indent) {
    json::Writer w(indent);
    guarded([&] {                            // 各点依次求解、写出（每个点内部的矩阵运算用线程池）
        write_head(w, rq);
        for (const PointSpec& p : rq.points) {
            const Result2D r = solve2d(rq.stack2d, p.wavelength, p.theta_deg, p.phi_deg, rq.M, rq.My, rq.jones2d, rq.options2d);
            write_result_2d(w, p, r, harmonic_count_2d(rq));
        }
    });
    return finish(w);
}

#endif   // STABLERCWA_RCWA_JSON_2D

inline JsonReply error_reply(const std::string& message, const std::string& where, int indent) {
    json::Writer w(indent);
    w.begin_object();
    w.key("ok");
    w.value(false);
    w.key("error");
    w.value(message);
    w.key("where");
    w.value(where);
    w.end_object();
    JsonReply reply;
    reply.ok = false;
    reply.text = w.take();
    return reply;
}

inline JsonReply answer(const std::string& request, int indent) {
    Request rq;
    {
        json::Value root;
        try {
            root = json::parse(request);
        } catch (const json::ParseError& e) {
            fail("", "请求不是合法的 JSON（第 " + std::to_string(e.line()) + " 行第 " + std::to_string(e.column()) +
                         " 列，字节偏移 " + std::to_string(e.offset()) + "）：" + e.reason());
        }
        rq = read_request(root);
    }                                        // 请求的那棵树到这里就释放了
    const ThreadSetting threads(rq.threads);
#if STABLERCWA_RCWA_JSON_2D
    if (rq.periodic2d) return run_periodic2d(rq, indent);
#endif
#if STABLERCWA_RCWA_JSON_CONICAL
    if (rq.conical) return run_conical(rq, indent);
#endif
    return run_planar(rq, indent);
}

}  // namespace json_detail

// 解请求，给出应答和它的 ok。indent < 0：应答写成一行；indent >= 0：分行、逐层缩进 indent 个空格（见 json.hpp）。
// 不抛异常。
inline JsonReply solve_json_reply(const std::string& request, int indent = -1) {
    namespace jd = json_detail;
    try {
        try {
            return jd::answer(request, indent);
        } catch (const jd::RequestError& e) {
            return jd::error_reply(e.message, e.where, indent);
        } catch (const std::bad_alloc&) {
            return jd::error_reply("内存不够", "", indent);
        } catch (const std::exception& e) {
            return jd::error_reply(std::string("求解时出错：") + e.what(), "", indent);
        } catch (...) {
            return jd::error_reply("求解时出现了无法识别的错误", "", indent);
        }
    } catch (...) {                          // 连出错的应答都写不出来（内存耗尽）：退回一条固定的文字
    }
    JsonReply reply;
    try {
        reply.text = "{\"ok\": false, \"error\": \"out of memory\", \"where\": \"\"}";
    } catch (...) {
    }
    return reply;
}

// 请求进、应答出（都是 JSON 文字，应答写成一行）。不抛异常：请求有错时应答是 {"ok": false, "error": …, "where": …}。
inline std::string solve_json(const std::string& request) { return solve_json_reply(request).text; }

}  // namespace rcwa
}  // namespace stablercwa
