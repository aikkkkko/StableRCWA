// tests/test_rcwa1d.cpp —— rcwa1d.hpp（一维 RCWA）的测试
//
//   c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I include tests/test_rcwa1d.cpp -o build/test_rcwa1d
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I include tests/test_rcwa1d.cpp -o build/test_rcwa1d_san
//
// 只依赖 C++17 标准库，参考数值全部写在本文件里：
//   · 均匀平板：Airy 公式（解析解）；
//   · 光栅、多层膜光栅、四脊大周期光栅加银膜、银光栅、随机膜系：同一截断阶数下各级效率的参考表（逐级比较，容差 1e-9）；
//   · 平移、镜像、分层这几种等价的写法互相比较，不需要参考值；
//   · 级次与方向的约定：薄光栅的一阶微扰（解析式）。
// 全部 23 张参考表都能用两条与本库无关的路线重现（脚本在 tests/reference_gen/，各表的结构与入射条件写在 rcwa1d_cases.py 里）：
//   · make_rcwa1d_tables.py 用 Python 求解器 gui/rcwa1d.py（S 矩阵级联）：不带参数时重算全部表并与本文件逐级比较，
//     最大偏差 4.3e-13；--emit 写出 C++ 表（四脊大周期光栅加银膜的两张就是这样写出的）；
//   · crosscheck_rcwa1d_tables.py：另写的整体矩阵法（所有界面的边界条件装成一个方程组一次解出，不调用 gui/ 的代码），
//     最大偏差 1.2e-13。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "stablercwa/eig_qr.hpp"
#include "stablercwa/rcwa1d.hpp"

using namespace stablercwa;
namespace rcwa = stablercwa::rcwa;
using rcwa::Polarization;

// ---------------------------------------------------------------------------------------------
// 输出与计数
//
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

static int g_pass = 0, g_total = 0;

// 断言的计数与报错都在下面几个普通函数里；宏只负责取文件名、行号和表达式的文字，宏里不出现中文
// （带 # 运算符的宏里若有中文的字符串，MSVC 会在每一处展开上报 C4819）。
static void check(bool ok, const char* file, int line, const char* text) {
    ++g_total;
    if (ok) ++g_pass;
    else say("  失败 %s:%d  %s\n", file, line, text);
}
// value <= bound；失败时打印实际数值。NaN 也算失败。
static void check_le(double value, double bound, const char* file, int line, const char* text) {
    ++g_total;
    if (value <= bound) ++g_pass;
    else say("  失败 %s:%d  %s = %.3e > %.3e\n", file, line, text, value, bound);
}
static void check_throws(bool caught, const char* file, int line, const char* text, const char* type) {
    ++g_total;
    if (caught) ++g_pass;
    else say("  失败 %s:%d  %s 没有抛出 %s\n", file, line, text, type);
}

#define CHECK(cond) check(static_cast<bool>(cond), __FILE__, __LINE__, #cond)

#define CHECK_LE(value, bound)                                                    \
    do {                                                                          \
        const double v_ = (value), b_ = (bound);                                  \
        check_le(v_, b_, __FILE__, __LINE__, #value);                             \
    } while (0)

#define CHECK_THROWS(expr, exception_type)                                        \
    do {                                                                          \
        bool caught_ = false;                                                     \
        try { (void)(expr); } catch (const exception_type&) { caught_ = true; }   \
        check_throws(caught_, __FILE__, __LINE__, #expr, #exception_type);        \
    } while (0)

// ---------------------------------------------------------------------------------------------
// 参考效率表：{级次, 反射效率, 透射效率}。只列效率不为零的级次，没列出的级次应当为零。
struct OrderRef {
    int order;
    double der, det;
};

// 单层无吸收光栅，λ = 1.053，θ = 66.36°，M = 12，TE
static const OrderRef kGratingTE[] = {
    {0, 0.031920623454517905, 0.6463974501559708},
    {1, 0.10407895934452913, 0.21760296704498022},
};

// 单层无吸收光栅，λ = 1.053，θ = 66.36°，M = 12，TM
static const OrderRef kGratingTM[] = {
    {0, 0.014520963754696536, 0.8764508042353972},
    {1, 0.07879482590979917, 0.030233406100106323},
};

// 单层吸收光栅（齿 n = 1.8 + 0.13i），λ = 1.053，θ = 42°，M = 14，TE
static const OrderRef kLossyTE[] = {
    {0, 0.027551073050931442, 0.28997415863052456},
    {1, 0.0, 0.3168915066378712},
};

// 单层吸收光栅（齿 n = 1.8 + 0.13i），λ = 1.053，θ = 42°，M = 14，TM
static const OrderRef kLossyTM[] = {
    {0, 0.03306765093865813, 0.7507493599487344},
    {1, 0.0, 0.08956699287036905},
};

// 光栅 + 19 层介质膜（共 20 层），λ = 1.053，θ = 66.36°，M = 20，TE
static const OrderRef kMdgTE[] = {
    {0, 0.0032976240198281325, 0.36694798058630257},
    {1, 0.41861300166719917, 0.2111413937267152},
};

// 光栅 + 19 层介质膜（共 20 层），λ = 1.053，θ = 66.36°，M = 20，TM
static const OrderRef kMdgTM[] = {
    {0, 0.040177187478985125, 0.8337107487972006},
    {1, 0.08894979111495854, 0.037162272608813246},
};

// 两个光栅层夹一层吸收膜，入射介质 n = 1.2，出射介质 n = 1.5 + 0.03i，λ = 0.83，θ = 27°，M = 12，TE
static const OrderRef kMultilayerTE[] = {
    {-12, 0.0, 9.266054028494248e-12},
    {-11, 0.0, 1.3789850694950022e-10},
    {-10, 0.0, 1.6041783319089122e-10},
    {-9, 0.0, 4.671395118259019e-11},
    {-8, 0.0, 1.0916105793645675e-09},
    {-7, 0.0, 2.1521379681502767e-09},
    {-6, 0.0, 9.44785905690279e-10},
    {-5, 0.0, 2.3216448193633126e-08},
    {-4, 0.0, 1.0837268079419198e-07},
    {-3, 0.0, 3.5960641704005127e-07},
    {-2, 0.0, 3.1047162640177546e-05},
    {-1, 0.0, 0.07128675542441114},
    {0, 0.03131011529305081, 0.3780206174109231},
    {1, 0.11745356444274715, 0.06775892736958519},
    {2, 0.0, 0.02980253035751372},
    {3, 0.0, 2.4754320059643814e-05},
    {4, 0.0, 4.776555096980643e-07},
    {5, 0.0, 1.3272607175993034e-07},
    {6, 0.0, 5.331673559998519e-09},
    {7, 0.0, 3.4165085608997058e-09},
    {8, 0.0, 2.991539509210957e-09},
    {9, 0.0, 2.414452771462732e-10},
    {10, 0.0, 1.7334983751732677e-10},
    {11, 0.0, 2.745666908520834e-10},
    {12, 0.0, 3.7341629108590824e-11},
};

// 两个光栅层夹一层吸收膜，入射介质 n = 1.2，出射介质 n = 1.5 + 0.03i，λ = 0.83，θ = 27°，M = 12，TM
static const OrderRef kMultilayerTM[] = {
    {-12, 0.0, 1.0346986379520517e-06},
    {-11, 0.0, 1.3276651551085525e-06},
    {-10, 0.0, 1.0839982950792177e-06},
    {-9, 0.0, 2.3887218989412502e-06},
    {-8, 0.0, 3.1964307794349226e-06},
    {-7, 0.0, 2.7219494350141666e-06},
    {-6, 0.0, 7.769256825112355e-06},
    {-5, 0.0, 1.3080892428302352e-05},
    {-4, 0.0, 1.2296044926521088e-05},
    {-3, 0.0, 7.264360850104495e-05},
    {-2, 0.0, 0.0004368439018416324},
    {-1, 0.0, 0.0615955868298624},
    {0, 0.023592241358946536, 0.4548502219861373},
    {1, 0.02254382052479617, 0.1887391581674398},
    {2, 0.0, 0.016897664693795098},
    {3, 0.0, 0.00023879367086146743},
    {4, 0.0, 1.586335688412386e-05},
    {5, 0.0, 1.6119341803501309e-06},
    {6, 0.0, 7.502825742782968e-06},
    {7, 0.0, 2.3690286857576895e-06},
    {8, 0.0, 2.5441771877077796e-07},
    {9, 0.0, 2.018433592824168e-06},
    {10, 0.0, 1.008116898852317e-06},
    {11, 0.0, 1.646907977439646e-07},
    {12, 0.0, 1.072702639720259e-06},
};

// 四脊大周期光栅（周期 6.3），下接吸收膜、介质间隔层和 40 nm 银膜，λ = 0.8，θ = 23°，M = 50，TE
static const OrderRef kRidgesSilver50TE[] = {
    {-8, 0.0, 1.881791408613599e-05},
    {-7, 0.0, 2.3715286298706092e-05},
    {-6, 0.0, 0.00014707392176275006},
    {-5, 0.0, 0.000262678506793101},
    {-4, 0.0795629239456752, 0.0008631008113681442},
    {-3, 0.010246212793944345, 0.0002887419768441246},
    {-2, 0.10434793786663206, 0.00187282993620335},
    {-1, 0.0168871585338964, 0.003874071174771557},
    {0, 0.031753696377394953, 0.009349142801841917},
    {1, 0.11496559464190406, 0.0027246422838180253},
    {2, 0.03337136705635213, 0.003803794650656493},
    {3, 0.059420036223644765, 0.003029382778166549},
    {4, 0.01318588418994586, 0.0009014250439108209},
    {5, 0.1073646854349063, 0.0026537638519077405},
    {6, 0.0001495008237413991, 0.0013765689187013603},
    {7, 0.009301291016465887, 1.0019599715834334e-05},
    {8, 0.03641537607709477, 3.27789834056511e-05},
    {9, 0.012545364841779594, 0.0003403165765551208},
    {10, 0.05090220175705197, 0.0001376873167700127},
    {11, 0.0, 3.707637394445553e-05},
    {12, 0.0, 2.375050499914856e-05},
    {13, 0.0, 2.5460700539079284e-05},
    {14, 0.0, 5.903718402610031e-05},
};

// 四脊大周期光栅（周期 6.3），下接吸收膜、介质间隔层和 40 nm 银膜，λ = 0.8，θ = 23°，M = 50，TM
static const OrderRef kRidgesSilver50TM[] = {
    {-8, 0.0, 0.00017042004286912281},
    {-7, 0.0, 0.00018590073140845818},
    {-6, 0.0, 0.0031330890900644953},
    {-5, 0.0, 0.0004399426134477887},
    {-4, 0.02767062769747271, 0.0022941459524908813},
    {-3, 0.028844396704549462, 0.00045717131391310767},
    {-2, 0.13365554770454177, 0.002222570041052171},
    {-1, 0.016709323594780973, 0.001155871449588495},
    {0, 0.06427896199032623, 0.014988387738076514},
    {1, 0.10194455740368198, 0.0032440430237871977},
    {2, 0.05256800297651084, 0.0032846812084216066},
    {3, 0.05527938994674007, 0.0027150377669918116},
    {4, 0.002208652486193338, 0.0011947734356977924},
    {5, 0.10532860526852393, 0.0025176607947402204},
    {6, 0.004734625833079391, 0.00048821955220222266},
    {7, 0.0018786196842453845, 0.0002694154211178121},
    {8, 0.01617366262609785, 0.0013752097900933547},
    {9, 0.01923461386460697, 0.0014467679910377187},
    {10, 0.045440224430545295, 0.0015509579972904839},
    {11, 0.0, 0.00045248442600357694},
    {12, 0.0, 0.0004899479622195968},
    {13, 0.0, 0.0010135227493637408},
    {14, 0.0, 0.0016141599456155821},
};

// 银光栅（脊 n = 0.079095 + 5.42875i，槽为空气，厚 0.08）在玻璃（n = 1.5）上，周期 0.6，λ = 0.8，θ = 30°，M = 15，TE
static const OrderRef kSilverGlassTE[] = {
    {0, 0.516347061653725, 0.23309931562041372},
    {1, 0.09137114586451126, 0.15005671999581485},
};

// 银光栅（脊 n = 0.079095 + 5.42875i，槽为空气，厚 0.08）在玻璃（n = 1.5）上，周期 0.6，λ = 0.8，θ = 30°，M = 15，TM
static const OrderRef kSilverGlassTM[] = {
    {0, 0.37914528456199764, 0.20788745768762176},
    {1, 0.22693651076553212, 0.17518982637129096},
};

// 银光栅（厚 0.10，边界 {−0.1, 0.15, 0.5}）在银衬底上，周期 0.6，λ = 0.8，θ = 30°，M = 15，TE
static const OrderRef kSilverBulkTE[] = {
    {-15, 0.0, 3.256305696569167e-11},
    {-14, 0.0, 6.584696348304062e-11},
    {-13, 0.0, 2.2255338469107963e-10},
    {-12, 0.0, 1.2711560673304247e-11},
    {-11, 0.0, 8.329564718869339e-10},
    {-10, 0.0, 3.0259057696306106e-10},
    {-9, 0.0, 1.893971856905416e-09},
    {-8, 0.0, 5.526164158074456e-09},
    {-7, 0.0, 1.133255806618735e-09},
    {-6, 0.0, 4.964047106115399e-08},
    {-5, 0.0, 2.6191234855000693e-08},
    {-4, 0.0, 3.2633323810165404e-07},
    {-3, 0.0, 2.2254807014934872e-06},
    {-2, 0.0, 7.611816601821701e-07},
    {-1, 0.0, 0.0002590380958408845},
    {0, 0.8879990710256762, 0.0014693165498930894},
    {1, 0.10125836095566869, 0.0007922439435913065},
    {2, 0.0, 2.0548083856252485e-05},
    {3, 0.0, 7.272172098373404e-06},
    {4, 0.0, 1.798943839811169e-08},
    {5, 0.0, 3.2361254013347527e-07},
    {6, 0.0, 3.692014493542904e-08},
    {7, 0.0, 1.208496784858264e-08},
    {8, 0.0, 1.2109573633037458e-08},
    {9, 0.0, 4.738775436034868e-11},
    {10, 0.0, 2.4426322723966506e-09},
    {11, 0.0, 3.8724134522608614e-10},
    {12, 0.0, 3.033296860340134e-10},
    {13, 0.0, 3.170740729876827e-10},
    {14, 0.0, 7.288708201246009e-12},
    {15, 0.0, 1.3374066775168833e-10},
};

// 银光栅（厚 0.10，边界 {−0.1, 0.15, 0.5}）在银衬底上，周期 0.6，λ = 0.8，θ = 30°，M = 15，TM
static const OrderRef kSilverBulkTM[] = {
    {-15, 0.0, 0.0001333443720978434},
    {-14, 0.0, 3.22610084765835e-05},
    {-13, 0.0, 3.285058679040118e-05},
    {-12, 0.0, 2.9053016755611e-05},
    {-11, 0.0, 2.6069338443474727e-05},
    {-10, 0.0, 2.5835536719373185e-05},
    {-9, 0.0, 3.232435327054659e-05},
    {-8, 0.0, 2.970052095776869e-05},
    {-7, 0.0, 3.822172527368491e-05},
    {-6, 0.0, 5.707199178215756e-05},
    {-5, 0.0, 4.028184114058334e-05},
    {-4, 0.0, 0.00012496391255520175},
    {-3, 0.0, 0.00010545442836931614},
    {-2, 0.0, 0.0002194675452818899},
    {-1, 0.0, 0.0014895812459311162},
    {0, 0.3678269684405447, 0.004147110625437078},
    {1, 0.605834499060213, 0.0011294809870072095},
    {2, 0.0, 0.0009300115058138694},
    {3, 0.0, 4.95731925916206e-05},
    {4, 0.0, 0.00016361575075176415},
    {5, 0.0, 4.0405836621094e-05},
    {6, 0.0, 3.174636541405395e-05},
    {7, 0.0, 4.313760489594093e-05},
    {8, 0.0, 1.5345985788598333e-06},
    {9, 0.0, 2.6708216062987393e-05},
    {10, 0.0, 4.422783234183036e-06},
    {11, 0.0, 9.118846760474954e-06},
    {12, 0.0, 1.121777478926723e-05},
    {13, 0.0, 2.9549379114961134e-06},
    {14, 0.0, 1.1732412389573382e-05},
    {15, 0.0, 4.611657333000556e-05},
};

// ---------------------------------------------------------------------------------------------
// 测试用的结构（长度单位：微米）

// 单层矩形光栅：脊中心在 x = 0，边界从负坐标开始。
static rcwa::Stack grating_stack(bool lossy = false) {
    const double p = 0.5747, f = 0.30;
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.45;
    s.period = p;
    s.layers.push_back(rcwa::Layer::grating(0.38, {-f * p / 2, f * p / 2, p - f * p / 2},
                                            {lossy ? cplx(1.8, 0.13) : cplx(1.45), 1.0}));
    return s;
}
// 多层介质膜光栅：光栅下面 9 组 H/2L 膜再加一层 H，共 20 层。
static rcwa::Stack mdg_stack() {
    rcwa::Stack s = grating_stack();
    for (int j = 0; j < 9; ++j) {
        s.layers.push_back(rcwa::Layer::uniform(1.053 / (4 * 2.05), 2.05));
        s.layers.push_back(rcwa::Layer::uniform(2 * 1.053 / (4 * 1.45), 1.45));
    }
    s.layers.push_back(rcwa::Layer::uniform(1.053 / (4 * 2.05), 2.05));
    return s;
}
// 两个不同的光栅层夹一层吸收膜；入射介质 n = 1.2，出射介质有吸收。
static rcwa::Stack multilayer_stack() {
    rcwa::Stack s;
    s.nI = 1.2;
    s.nII = cplx(1.5, 0.03);
    s.period = 0.9;
    s.layers = {rcwa::Layer::grating(0.23, {0.0, 0.21, 0.47, 0.9}, {1.7, 1.0, cplx(2.1, 0.05)}),
                rcwa::Layer::uniform(0.19, cplx(1.4, 0.02)),
                rcwa::Layer::grating(0.31, {-0.12, 0.17, 0.78}, {1.0, 1.9})};
    return s;
}
// 光栅层下面的 20 层介质膜：3L · H · (2L · H)^9
static void add_film_stack(std::vector<rcwa::Layer>& layers) {
    const double n_low = 1.46, n_high = 1.96, d_high = 0.115306, d_2low = 0.30958, d_3low = 0.46438;
    layers.push_back(rcwa::Layer::uniform(d_3low, n_low));
    layers.push_back(rcwa::Layer::uniform(d_high, n_high));
    for (int k = 0; k < 9; ++k) {
        layers.push_back(rcwa::Layer::uniform(d_2low, n_low));
        layers.push_back(rcwa::Layer::uniform(d_high, n_high));
    }
}
// 侧壁倾斜的齿切成 slices 片，每片是一个光栅层（多个光栅层，各要一次本征分解）。
static rcwa::Stack trapezoid_stack(int slices) {
    const double base = 0.574, depth = 0.38;
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.46;
    s.period = base;
    for (int k = 0; k < slices; ++k) {
        const double w = base * (0.2 + 0.25 * (k + 0.5) / slices), x0 = 0.5 * (base - w);
        s.layers.push_back(rcwa::Layer::grating(depth / slices, {0.0, x0, x0 + w, base}, {1.0, 1.46, 1.0}));
    }
    add_film_stack(s.layers);
    return s;
}
// 银光栅（光栅层本身强吸收）
static const cplx kSilver{0.079095, 5.42875};
static rcwa::Stack silver_on_glass() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.5;
    s.period = 0.6;
    s.layers = {rcwa::Layer::grating(0.08, {0.0, 0.3, 0.6}, {kSilver, 1.0})};
    return s;
}
static rcwa::Stack silver_on_silver() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = kSilver;
    s.period = 0.6;
    s.layers = {rcwa::Layer::grating(0.10, {-0.1, 0.15, 0.5}, {kSilver, 1.0})};
    return s;
}
// 四脊大周期光栅：周期 6.3，一个周期里四条宽度、折射率各不相同的脊（其余是空气），刻深 0.32。
// 周期是波长的七八倍：级次很密，传播级多，截断阶数可以取得很大。
static const double kRidgesPeriod = 6.3;
static rcwa::Layer ridges_layer() {
    return rcwa::Layer::grating(0.32, {0.0, 0.20, 1.05, 1.60, 2.05, 2.90, 4.25, 5.10, 5.45, 6.30},
                                {1.0, 1.52, 1.0, 2.10, 1.0, 1.75, 1.0, 1.38, 1.0});
}
// 四脊大周期光栅 + 20 层介质高反膜 (H·L)^10（H：n = 2.25，厚 0.105；L：n = 1.48，厚 0.160），玻璃衬底。
// 无吸收；λ = 0.94，θ = 22°。
static const double kRidgesWavelength = 0.94, kRidgesTheta = 22.0;
static rcwa::Stack ridges_on_mirror() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.50;
    s.period = kRidgesPeriod;
    s.layers.push_back(ridges_layer());
    for (int k = 0; k < 10; ++k) {
        s.layers.push_back(rcwa::Layer::uniform(0.105, 2.25));
        s.layers.push_back(rcwa::Layer::uniform(0.160, 1.48));
    }
    return s;
}
// 四脊大周期光栅，下面是吸收膜（n = 2.0 + 0.08i，厚 0.12）、介质间隔层（n = 1.45，厚 0.20）和 40 nm 银膜，玻璃衬底。
// 用在 λ = 0.8，θ = 23°；银膜薄，透射级不是零。
static rcwa::Stack ridges_on_silver() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.50;
    s.period = kRidgesPeriod;
    s.layers = {ridges_layer(), rcwa::Layer::uniform(0.12, cplx(2.0, 0.08)), rcwa::Layer::uniform(0.20, 1.45),
                rcwa::Layer::uniform(0.040, kSilver)};
    return s;
}

// ---------------------------------------------------------------------------------------------
// 比较工具

template <std::size_t N>
static double table_error(const rcwa::Result& r, int M, const OrderRef (&table)[N]) {
    std::vector<double> der(static_cast<std::size_t>(2 * M + 1), 0.0), det(static_cast<std::size_t>(2 * M + 1), 0.0);
    for (const OrderRef& row : table) {
        der[row.order + M] = row.der;
        det[row.order + M] = row.det;
    }
    if (static_cast<int>(r.DEr.size()) != 2 * M + 1 || static_cast<int>(r.DEt.size()) != 2 * M + 1)
        return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (int j = 0; j <= 2 * M; ++j) {
        const double e = std::max(std::abs(r.DEr[j] - der[j]), std::abs(r.DEt[j] - det[j]));
        if (!(e <= worst)) worst = e;            // NaN 也算不合格
    }
    return worst;
}

// 两次求解各级效率之差的最大值
static double efficiency_gap(const rcwa::Result& a, const rcwa::Result& b) {
    if (a.DEr.size() != b.DEr.size() || a.DEt.size() != b.DEt.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t j = 0; j < a.DEr.size(); ++j) {
        const double e = std::max(std::abs(a.DEr[j] - b.DEr[j]), std::abs(a.DEt[j] - b.DEt[j]));
        if (!(e <= worst)) worst = e;
    }
    return worst;
}

// 复振幅和效率是否逐位相同
static bool same_bits(const rcwa::Result& a, const rcwa::Result& b) {
    const auto same = [](const void* x, const void* y, std::size_t bytes) { return bytes == 0 || std::memcmp(x, y, bytes) == 0; };
    return a.orders == b.orders && a.R.size() == b.R.size() && a.T.size() == b.T.size() &&
           a.DEr.size() == b.DEr.size() && a.DEt.size() == b.DEt.size() &&
           same(a.R.data(), b.R.data(), a.R.size() * sizeof(cplx)) &&
           same(a.T.data(), b.T.data(), a.T.size() * sizeof(cplx)) &&
           same(a.DEr.data(), b.DEr.data(), a.DEr.size() * sizeof(double)) &&
           same(a.DEt.data(), b.DEt.data(), a.DEt.size() * sizeof(double));
}

static bool all_nan(const rcwa::Result& r) {
    if (r.R.empty() || r.R.size() != r.T.size() || r.DEr.size() != r.R.size() || r.DEt.size() != r.R.size()) return false;
    for (std::size_t j = 0; j < r.R.size(); ++j)
        if (!std::isnan(r.R[j].real()) || !std::isnan(r.T[j].real()) || !std::isnan(r.DEr[j]) || !std::isnan(r.DEt[j]))
            return false;
    return std::isnan(r.reflectance()) && std::isnan(r.transmittance());
}

static double energy_defect(const rcwa::Result& r) { return std::abs(r.reflectance() + r.transmittance() - 1.0); }

static const char* name(Polarization pol) { return pol == Polarization::TE ? "TE" : "TM"; }

// ---------------------------------------------------------------------------------------------
// 均匀平板的 Airy 公式。n0 | n1（厚 d）| n2，主场（TE：Ey；TM：Hy）的反射、透射振幅：
//     r = (r01 + r12·φ²)/(1 + r01·r12·φ²)，   t = t01·t12·φ/(1 + r01·r12·φ²)，   φ = exp(i·k0·kz1·d)，
//     r_ab = (p_a − p_b)/(p_a + p_b)，   t_ab = 2 p_a/(p_a + p_b)，   p = kz（TE）或 kz/n²（TM），kz = sqrt(n² − kx²)。
struct Airy {
    cplx r, t;
    double der, det;
};
static Airy airy(cplx n0, cplx n1, cplx n2, double d, double wavelength, double theta_deg, Polarization pol) {
    const double k0 = 2.0 * kPi / wavelength;
    const cplx kx = n0 * std::sin(theta_deg * kPi / 180.0);
    const auto kz = [&](cplx n) {
        cplx v = std::sqrt(n * n - kx * kx);
        return v.imag() < 0.0 ? -v : v;
    };
    const auto p = [&](cplx n) { return pol == Polarization::TE ? kz(n) : kz(n) / (n * n); };
    const cplx r01 = (p(n0) - p(n1)) / (p(n0) + p(n1)), r12 = (p(n1) - p(n2)) / (p(n1) + p(n2));
    const cplx t01 = 2.0 * p(n0) / (p(n0) + p(n1)), t12 = 2.0 * p(n1) / (p(n1) + p(n2));
    const cplx phase = std::exp(kI * k0 * kz(n1) * d);
    const cplx den = 1.0 + r01 * r12 * phase * phase;
    Airy a;
    a.r = (r01 + r12 * phase * phase) / den;
    a.t = t01 * t12 * phase / den;
    a.der = std::norm(a.r);
    a.det = std::norm(a.t) * p(n2).real() / p(n0).real();
    return a;
}

static void test_slab() {
    say("[均匀平板：Airy 公式]\n");
    struct Case {
        cplx n1, n2;
        double d, theta;
        bool lossless;     // 入射介质和平板都无吸收：R + T = 1
    };
    const Case cases[] = {
        {1.72, 1.45, 0.62, 41.0, true},                        // 无吸收
        {1.72, 1.45, 0.62, 0.0, true},                         // 正入射：±m 级的 kx² 相同（简并），稠密路径另取角度，见下
        {cplx(1.72, 0.05), 1.45, 0.62, 41.0, false},           // 平板有吸收
        {cplx(0.079095, 5.42875), 1.45, 0.03, 30.0, false},    // 30 nm 的金属膜
        {1.72, cplx(1.45, 0.02), 0.62, 41.0, true},            // 只有出射介质吸收：T 是进入它的能流，R + T 仍是 1
        {2.3, 1.0, 0.25, 60.0, true},                          // 两侧都是空气的高折射率膜
    };
    const double wavelength = 1.053, period = 0.8;
    const int M = 6;
    double worst_fast = 0.0, worst_dense = 0.0;
    for (const Case& c : cases) {
        rcwa::Stack s;
        s.nI = 1.0;
        s.nII = c.n2;
        s.period = period;
        s.layers = {rcwa::Layer::uniform(c.d, c.n1)};
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            for (bool fast : {true, false}) {
                // 稠密路径在恰好正入射时精度没有保证（均匀层的层矩阵里 ±m 两级的本征值相等，见 rcwa1d.hpp 里 solve() 的说明），
                // 出不出问题取决于舍入的细节，换一个编译器就可能不同。所以稠密路径把正入射换成偏开 1e-7° 的角度，
                // Airy 公式也按同一个角度算（与正入射的结果只差 1e-18 量级）。
                const double theta = (!fast && c.theta == 0.0) ? 1e-7 : c.theta;
                const Airy ref = airy(1.0, c.n1, c.n2, c.d, wavelength, theta, pol);
                rcwa::Options opt;
                opt.homogeneous_fast_path = fast;
                const rcwa::Result r = rcwa::solve(s, wavelength, theta, M, pol, opt);
                double err = std::max({std::abs(r.R[M] - ref.r), std::abs(r.T[M] - ref.t), std::abs(r.DEr[M] - ref.der),
                                       std::abs(r.DEt[M] - ref.det)});
                for (int j = 0; j <= 2 * M; ++j)                // 平板不产生衍射：其余各级都是零
                    if (j != M) err = std::max({err, std::abs(r.R[j]), std::abs(r.T[j]), std::abs(r.DEr[j]), std::abs(r.DEt[j])});
                CHECK_LE(err, fast ? 1e-13 : 1e-11);
                (fast ? worst_fast : worst_dense) = std::max(fast ? worst_fast : worst_dense, err);
                if (c.lossless) CHECK_LE(energy_defect(r), fast ? 1e-13 : 1e-11);
                else CHECK(r.reflectance() + r.transmittance() < 1.0 - 1e-6 && r.reflectance() > 0.0 && r.transmittance() >= 0.0);
                CHECK(r.n_eig == (fast ? 0 : 1));
                if (fast) CHECK(r.n_dense_solve == 0);          // 全是对角运算
            }
        }
    }
    say("  与 Airy 公式的最大偏差：解析路径 %.1e，稠密路径 %.1e\n", worst_fast, worst_dense);

    // 没有任何层：就是一个 Fresnel 界面
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        rcwa::Stack bare;
        bare.nI = 1.0;
        bare.nII = 1.45;
        bare.period = 0.8;
        const double theta = 41.0 * kPi / 180.0;
        const cplx kz1 = std::cos(theta), kz2 = std::sqrt(cplx(1.45 * 1.45 - std::sin(theta) * std::sin(theta)));
        const cplx p1 = kz1, p2 = pol == Polarization::TE ? kz2 : kz2 / (1.45 * 1.45);
        for (bool fast : {true, false}) {
            rcwa::Options opt;
            opt.homogeneous_fast_path = fast;
            const rcwa::Result r = rcwa::solve(bare, 1.053, 41.0, 4, pol, opt);
            CHECK_LE(std::abs(r.R[4] - (p1 - p2) / (p1 + p2)), 1e-13);
            CHECK_LE(std::abs(r.T[4] - 2.0 * p1 / (p1 + p2)), 1e-13);
            CHECK_LE(energy_defect(r), 1e-13);
        }
    }

    // 厚度为零的层不改变结果；M = 0 也能算
    {
        rcwa::Stack s;
        s.nI = 1.0;
        s.nII = 1.45;
        s.period = 0.8;
        s.layers = {rcwa::Layer::uniform(0.62, 1.72)};
        const rcwa::Result base = rcwa::solve(s, 1.053, 41.0, 3);
        s.layers.push_back(rcwa::Layer::uniform(0.0, 2.2));
        const rcwa::Result with_zero = rcwa::solve(s, 1.053, 41.0, 3);
        CHECK_LE(std::abs(with_zero.R[3] - base.R[3]) + std::abs(with_zero.T[3] - base.T[3]), 1e-14);
        const rcwa::Result m0 = rcwa::solve(s, 1.053, 41.0, 0);
        CHECK(m0.orders.size() == 1 && m0.orders[0] == 0);
        CHECK_LE(std::abs(m0.R[0] - base.R[3]), 1e-14);
    }
}

// ---------------------------------------------------------------------------------------------
static void test_reference_tables() {
    say("[参考效率：逐级比较]\n");
    double worst = 0.0;
    const auto run = [&](const char* label, const rcwa::Stack& s, double wavelength, double theta, int M, Polarization pol,
                         double err_of(const rcwa::Result&)) {
        const rcwa::Result r = rcwa::solve(s, wavelength, theta, M, pol);
        const double err = err_of(r);
        CHECK_LE(err, 1e-9);
        CHECK(r.eig_info == 0);
        if (!(err <= worst)) worst = err;
        say("  %-18s %s  M = %2d  最大偏差 %.1e   R = %.9f  T = %.9f\n", label, name(pol), M, err, r.reflectance(),
                    r.transmittance());
    };
    run("无吸收光栅", grating_stack(), 1.053, 66.36, 12, Polarization::TE, [](const rcwa::Result& r) { return table_error(r, 12, kGratingTE); });
    run("无吸收光栅", grating_stack(), 1.053, 66.36, 12, Polarization::TM, [](const rcwa::Result& r) { return table_error(r, 12, kGratingTM); });
    run("吸收光栅", grating_stack(true), 1.053, 42.0, 14, Polarization::TE, [](const rcwa::Result& r) { return table_error(r, 14, kLossyTE); });
    run("吸收光栅", grating_stack(true), 1.053, 42.0, 14, Polarization::TM, [](const rcwa::Result& r) { return table_error(r, 14, kLossyTM); });
    run("20 层膜光栅", mdg_stack(), 1.053, 66.36, 20, Polarization::TE, [](const rcwa::Result& r) { return table_error(r, 20, kMdgTE); });
    run("20 层膜光栅", mdg_stack(), 1.053, 66.36, 20, Polarization::TM, [](const rcwa::Result& r) { return table_error(r, 20, kMdgTM); });
    run("双光栅吸收多层", multilayer_stack(), 0.83, 27.0, 12, Polarization::TE, [](const rcwa::Result& r) { return table_error(r, 12, kMultilayerTE); });
    run("双光栅吸收多层", multilayer_stack(), 0.83, 27.0, 12, Polarization::TM, [](const rcwa::Result& r) { return table_error(r, 12, kMultilayerTM); });
    run("四脊光栅 + 银膜", ridges_on_silver(), 0.8, 23.0, 50, Polarization::TE, [](const rcwa::Result& r) { return table_error(r, 50, kRidgesSilver50TE); });
    run("四脊光栅 + 银膜", ridges_on_silver(), 0.8, 23.0, 50, Polarization::TM, [](const rcwa::Result& r) { return table_error(r, 50, kRidgesSilver50TM); });
    // 光栅层本身强吸收（一般复矩阵的本征分解）；参考值来自独立的 Python 求解器
    run("银光栅 / 玻璃", silver_on_glass(), 0.8, 30.0, 15, Polarization::TE, [](const rcwa::Result& r) { return table_error(r, 15, kSilverGlassTE); });
    run("银光栅 / 玻璃", silver_on_glass(), 0.8, 30.0, 15, Polarization::TM, [](const rcwa::Result& r) { return table_error(r, 15, kSilverGlassTM); });
    run("银光栅 / 银", silver_on_silver(), 0.8, 30.0, 15, Polarization::TE, [](const rcwa::Result& r) { return table_error(r, 15, kSilverBulkTE); });
    run("银光栅 / 银", silver_on_silver(), 0.8, 30.0, 15, Polarization::TM, [](const rcwa::Result& r) { return table_error(r, 15, kSilverBulkTM); });
    say("  全部算例的最大偏差 %.1e\n", worst);
}

// ---------------------------------------------------------------------------------------------
static void test_energy() {
    say("[能量守恒]\n");
    struct Lossless {
        const char* label;
        rcwa::Stack stack;
        double wavelength, theta;
        int M;
    };
    const Lossless lossless[] = {
        {"无吸收光栅", grating_stack(), 1.053, 66.36, 12},
        {"无吸收光栅（正入射）", grating_stack(), 1.053, 0.0, 12},
        {"20 层膜光栅", mdg_stack(), 1.053, 66.36, 20},
        {"梯形齿 4 片 + 20 层膜", trapezoid_stack(4), 1.053, 66.36, 12},
        {"四脊大周期光栅 + 20 层膜", ridges_on_mirror(), kRidgesWavelength, kRidgesTheta, 40},
    };
    for (const Lossless& c : lossless)
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const rcwa::Result r = rcwa::solve(c.stack, c.wavelength, c.theta, c.M, pol);
            CHECK_LE(energy_defect(r), 1e-10);
            bool sane = true;                           // 每一级的效率都在 [0, 1] 之内
            for (std::size_t j = 0; j < r.DEr.size(); ++j)
                sane = sane && r.DEr[j] >= 0.0 && r.DEr[j] <= 1.0 + 1e-12 && r.DEt[j] >= 0.0 && r.DEt[j] <= 1.0 + 1e-12;
            CHECK(sane);
            say("  %-24s %s  M = %2d  |R + T − 1| = %.1e\n", c.label, name(pol), c.M, energy_defect(r));
        }

    // 有吸收：R + T < 1
    struct Lossy {
        rcwa::Stack stack;
        double wavelength, theta;
        int M;
    };
    const Lossy lossy[] = {
        {grating_stack(true), 1.053, 42.0, 14},
        {multilayer_stack(), 0.83, 27.0, 12},
        {ridges_on_silver(), 0.8, 23.0, 30},
        {silver_on_glass(), 0.8, 30.0, 15},
        {silver_on_silver(), 0.8, 30.0, 15},
    };
    for (const Lossy& c : lossy)
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const rcwa::Result r = rcwa::solve(c.stack, c.wavelength, c.theta, c.M, pol);
            const double total = r.reflectance() + r.transmittance();
            CHECK(total < 1.0 - 1e-4 && total > 0.0 && r.reflectance() >= 0.0 && r.transmittance() >= 0.0);
        }
}

// ---------------------------------------------------------------------------------------------
static void test_option_equivalence() {
    say("[选项互换：结果应当一致]\n");
    struct Case {
        const char* label;
        rcwa::Stack stack;
        double wavelength, theta;
        int M;          // 快速路径开/关、线性求解方法的比较用这个阶数
        int M_small;    // LR 与 QR 的比较用这个阶数（<= 10）
    };
    const Case cases[] = {
        {"无吸收光栅", grating_stack(), 1.053, 66.36, 12, 10},
        {"吸收光栅", grating_stack(true), 1.053, 42.0, 14, 10},
        {"20 层膜光栅", mdg_stack(), 1.053, 66.36, 12, 10},
        {"20 层膜光栅（正入射）", mdg_stack(), 1.053, 0.0, 12, 10},     // 稠密路径另取偏开 1e-7° 的角度，见下
        {"双光栅吸收多层", multilayer_stack(), 0.83, 27.0, 12, 8},
        {"四脊光栅 + 银膜", ridges_on_silver(), 0.8, 23.0, 20, 10},
        {"银光栅 / 银", silver_on_silver(), 0.8, 30.0, 15, 10},
    };
    double gap_dense = 0.0, gap_qr = 0.0, gap_qr_dense = 0.0, gap_lr = 0.0;
    for (const Case& c : cases)
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const rcwa::Result base = rcwa::solve(c.stack, c.wavelength, c.theta, c.M, pol);

            // 均匀层走解析路径 ↔ 当作一般层走稠密路径（本征分解都用 QR 迭代）。
            // 稠密路径在恰好正入射时精度没有保证（出不出问题取决于舍入的细节，见 rcwa1d.hpp 里 solve() 的说明），
            // 所以正入射的算例在稠密路径下换成偏开 1e-7° 的角度，并与同一个角度的快速路径比较。
            const double theta_dense = c.theta == 0.0 ? 1e-7 : c.theta;
            const rcwa::Result base_dense = theta_dense == c.theta ? base : rcwa::solve(c.stack, c.wavelength, theta_dense, c.M, pol);
            rcwa::Options dense;
            dense.homogeneous_fast_path = false;
            const rcwa::Result rd = rcwa::solve(c.stack, c.wavelength, theta_dense, c.M, pol, dense);
            CHECK_LE(efficiency_gap(base_dense, rd), 1e-9);
            CHECK(rd.n_eig == static_cast<int>(c.stack.layers.size()));
            gap_dense = std::max(gap_dense, efficiency_gap(base_dense, rd));

            // 选主元的 LU ↔ Householder QR
            rcwa::Options qr;
            qr.linear = rcwa::LinearSolver::QRHouseholder;
            const rcwa::Result rq = rcwa::solve(c.stack, c.wavelength, c.theta, c.M, pol, qr);
            CHECK_LE(efficiency_gap(base, rq), 1e-9);
            CHECK(rq.n_dense_solve == base.n_dense_solve);
            gap_qr = std::max(gap_qr, efficiency_gap(base, rq));
            qr.homogeneous_fast_path = false;
            const rcwa::Result rqd = rcwa::solve(c.stack, c.wavelength, theta_dense, c.M, pol, qr);
            CHECK_LE(efficiency_gap(base_dense, rqd), 1e-9);
            gap_qr_dense = std::max(gap_qr_dense, efficiency_gap(base_dense, rqd));

            // LR 迭代 ↔ QR 迭代，小阶数、快速路径
            const rcwa::Result low_order = rcwa::solve(c.stack, c.wavelength, c.theta, c.M_small, pol);
            rcwa::Options lr;
            lr.eig = EigMethod::LR;
            const rcwa::Result rl = rcwa::solve(c.stack, c.wavelength, c.theta, c.M_small, pol, lr);
            CHECK_LE(efficiency_gap(low_order, rl), 1e-8);
            CHECK(rl.eig_info == 0);
            gap_lr = std::max(gap_lr, efficiency_gap(low_order, rl));
        }
    say("  各级效率的最大差别：快速路径开/关 %.1e；LU/QR 求解 %.1e（稠密路径 %.1e）；LR/QR 迭代（M <= 10）%.1e\n", gap_dense,
                gap_qr, gap_qr_dense, gap_lr);
}

// ---------------------------------------------------------------------------------------------
// 定性结论：把均匀层也交给本征迭代（层矩阵 = 对角矩阵 + Fourier 系数里的舍入误差）。
// QR 迭代不受影响；LR 迭代在这类矩阵上累积出极病态的变换，阶数一高整个解就被毁掉。
// 只把光栅层交给 LR（均匀层用解析解）时，这个结构在 M = 120 还没有这个问题（单周期光栅的光栅层本身在 M = 150 时
// 已经算坏，见 examples/rcwa_demo.cpp）。LR 的具体数值对舍入极其敏感，这里只断言量级。
static void test_lr_on_all_layers() {
    say("[均匀层也交给本征迭代：QR 无恙，LR 失效]\n");
    const rcwa::Stack stack = ridges_on_mirror();
    const int M = 120;
    set_num_threads(2);                                  // 只为省时间；结果与线程数无关（另有测试逐位核对）
    const auto total = [&](EigMethod method, bool fast, rcwa::Result* out = nullptr) {
        rcwa::Options opt;
        opt.eig = method;
        opt.homogeneous_fast_path = fast;
        opt.parallel_layers = true;
        const rcwa::Result r = rcwa::solve(stack, kRidgesWavelength, kRidgesTheta, M, Polarization::TE, opt);
        if (out) *out = r;
        return r.reflectance() + r.transmittance();
    };
    rcwa::Result qr_all_result, qr_fast_result;
    const double qr_all = total(EigMethod::QR, false, &qr_all_result);
    const double qr_fast = total(EigMethod::QR, true, &qr_fast_result);
    const double lr_all = total(EigMethod::LR, false);
    const double lr_grating = total(EigMethod::LR, true);
    set_num_threads(1);
    CHECK_LE(std::abs(qr_all - 1.0), 1e-9);
    CHECK_LE(std::abs(qr_fast - 1.0), 1e-9);
    CHECK(qr_all_result.n_eig == 21 && qr_fast_result.n_eig == 1);
    CHECK_LE(efficiency_gap(qr_all_result, qr_fast_result), 1e-9);
    CHECK(!(std::abs(lr_all - 1.0) <= 1e-6));            // 写成“不满足”：结果是 NaN 时同样算失效
    CHECK_LE(std::abs(lr_grating - 1.0), 1e-4);
    say("  M = %d（%d 阶）：T + R = %.8f%%（QR，全部 21 层）  %.8f%%（LR，只有光栅层）  %.6g%%（LR，全部 21 层）\n", M,
                2 * M + 1, 100.0 * qr_all, 100.0 * lr_grating, 100.0 * lr_all);
}

// ---------------------------------------------------------------------------------------------
static void test_sweep_and_threads() {
    say("[扫描与线程数]\n");
    const rcwa::Stack stack = mdg_stack();
    std::vector<rcwa::Point> points;
    for (int k = 0; k < 7; ++k) points.push_back({1.03 + 0.008 * k, 60.0 + 1.5 * k});
    const int M = 10;

    set_num_threads(1);
    std::vector<rcwa::Result> serial;
    for (const rcwa::Point& p : points) serial.push_back(rcwa::solve(stack, p.wavelength, p.theta_deg, M, Polarization::TE));
    CHECK(!same_bits(serial[0], serial[1]));                  // 各点确实不同
    for (int threads : {1, 2, 4}) {
        set_num_threads(threads);
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const std::vector<rcwa::Result> swept = rcwa::sweep(stack, points, M, pol);
            bool same = swept.size() == points.size();
            for (std::size_t i = 0; same && i < points.size(); ++i) {
                set_num_threads(1);
                const rcwa::Result one = rcwa::solve(stack, points[i].wavelength, points[i].theta_deg, M, pol);
                set_num_threads(threads);
                same = same_bits(swept[i], one) && swept[i].n_eig == 1 && swept[i].eig_info == 0;
                if (pol == Polarization::TE) same = same && same_bits(swept[i], serial[i]);
            }
            CHECK(same);
        }
    }
    set_num_threads(4);
    CHECK(rcwa::sweep(stack, {}, M).empty());
    // 选项原样传给每个点
    {
        rcwa::Options opt;
        opt.homogeneous_fast_path = false;
        opt.linear = rcwa::LinearSolver::QRHouseholder;
        const std::vector<rcwa::Result> swept = rcwa::sweep(stack, points, 6, Polarization::TM, opt);
        set_num_threads(1);
        bool same = true;
        for (std::size_t i = 0; i < points.size(); ++i)
            same = same && swept[i].n_eig == 20 &&
                   same_bits(swept[i], rcwa::solve(stack, points[i].wavelength, points[i].theta_deg, 6, Polarization::TM, opt));
        CHECK(same);
    }

    // 单次求解内部的并行（矩阵乘法、LU 的列并行）也不改变任何一位：阶数取大一些让并行分支真正用上
    {
        const rcwa::Stack wide = ridges_on_mirror();
        for (bool fast : {true, false}) {
            rcwa::Options opt;
            opt.homogeneous_fast_path = fast;
            const int m = fast ? 45 : 32;
            set_num_threads(1);
            const rcwa::Result one = rcwa::solve(wide, kRidgesWavelength, kRidgesTheta, m, Polarization::TE, opt);
            for (int threads : {2, 4}) {
                set_num_threads(threads);
                CHECK(same_bits(one, rcwa::solve(wide, kRidgesWavelength, kRidgesTheta, m, Polarization::TE, opt)));
            }
        }
    }
    set_num_threads(1);
}

static void test_parallel_layers_and_custom_eig() {
    say("[各层本征分解并行、自定义本征求解器]\n");
    const rcwa::Stack stack = trapezoid_stack(5);             // 5 个光栅层 + 20 层膜
    const int M = 10;
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        set_num_threads(1);
        const rcwa::Result base = rcwa::solve(stack, 1.053, 66.36, M, pol);
        CHECK(base.n_eig == 5);

        rcwa::Options par;
        par.parallel_layers = true;
        CHECK(same_bits(base, rcwa::solve(stack, 1.053, 66.36, M, pol, par)));      // 线程数为 1：等同串行
        for (int threads : {2, 4}) {
            set_num_threads(threads);
            const rcwa::Result r = rcwa::solve(stack, 1.053, 66.36, M, pol, par);
            CHECK(same_bits(base, r) && r.n_eig == 5);
        }
        // 均匀层也做本征分解时共 25 层
        par.homogeneous_fast_path = false;
        set_num_threads(1);
        const rcwa::Result dense_serial = rcwa::solve(stack, 1.053, 66.36, M, pol, par);
        set_num_threads(4);
        const rcwa::Result dense_parallel = rcwa::solve(stack, 1.053, 66.36, M, pol, par);
        CHECK(same_bits(dense_serial, dense_parallel) && dense_parallel.n_eig == 25);
        CHECK_LE(efficiency_gap(base, dense_parallel), 1e-9);
        set_num_threads(1);

        // custom_eig 里调用 eig_qr：与默认结果逐位相同；它取代 options.eig
        int calls = 0;
        rcwa::Options custom;
        custom.custom_eig = [&calls](const Matrix& a) {
            ++calls;
            return eig_qr(a);
        };
        const rcwa::Result rc = rcwa::solve(stack, 1.053, 66.36, M, pol, custom);
        CHECK(same_bits(base, rc) && calls == 5 && rc.n_eig == 5);
        custom.eig = EigMethod::LR;
        calls = 0;
        CHECK(same_bits(base, rcwa::solve(stack, 1.053, 66.36, M, pol, custom)) && calls == 5);
        rcwa::Options lr;
        lr.eig = EigMethod::LR;
        CHECK(!same_bits(base, rcwa::solve(stack, 1.053, 66.36, M, pol, lr)));      // 确认 LR 的结果在末位上确实不同

        // 本征对的排列顺序、本征向量的归一和相位不影响物理结果
        rcwa::Options shuffled;
        shuffled.custom_eig = [](const Matrix& a) {
            EigResult r = eig_qr(a);
            const int n = a.rows();
            EigResult out = r;
            for (int j = 0; j < n; ++j) {
                const int src = n - 1 - j;
                const cplx factor = (0.3 + 0.1 * j) * std::exp(cplx(0.0, 0.13 + j));
                out.values[j] = r.values[src];
                for (int i = 0; i < n; ++i) out.vectors(i, j) = r.vectors(i, src) * factor;
            }
            return out;
        };
        CHECK_LE(efficiency_gap(base, rcwa::solve(stack, 1.053, 66.36, M, pol, shuffled)), 1e-10);

        // 可被多个线程同时调用的 custom_eig 与 parallel_layers、sweep 同用
        rcwa::Options both;
        both.parallel_layers = true;
        both.custom_eig = [](const Matrix& a) { return eig_qr(a); };
        set_num_threads(4);
        CHECK(same_bits(base, rcwa::solve(stack, 1.053, 66.36, M, pol, both)));
        const std::vector<rcwa::Result> swept = rcwa::sweep(stack, {{1.053, 66.36}, {1.06, 64.0}, {1.053, 66.36}}, M, pol, both);
        CHECK(swept.size() == 3 && same_bits(base, swept[0]) && same_bits(base, swept[2]) && !same_bits(base, swept[1]));
        set_num_threads(1);
    }
}

// 把 double 舍入成只有 24 位有效数字（单精度的精度）：最近舍入，恰在中点时取偶。
// 直接在位模式上用整数运算做，不经过 float 类型的转换，所以与编译器怎样处理浮点类型转换（扩展精度的寄存器、
// 放宽浮点语义的优化选项）无关，精度一定降下来。数值在 float 的正规数范围内时，结果与 (double)(float)x 逐位相同。
static double round_to_single(double x) {
    if (!std::isfinite(x)) return x;
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    const std::uint64_t unit = std::uint64_t{1} << 29;         // 尾数 53 位里要舍掉的是低 29 位
    const std::uint64_t low = bits & (unit - 1), half = unit >> 1;
    bits &= ~(unit - 1);
    if (low > half || (low == half && (bits & unit) != 0)) bits += unit;   // 进位进到指数位也是对的（IEEE 754 的编码）
    std::memcpy(&x, &bits, sizeof x);
    return x;
}
// x 是否只有 24 位有效数字（低 29 位全为零）
static bool has_single_precision(double x) {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return (bits & ((std::uint64_t{1} << 29) - 1)) == 0;
}

// 本征分解只有单精度时（例如单精度的 GPU 后端通过 custom_eig 接入）：求解照常进行，不会因为残差偏大而拒绝，
// 结果的精度随之降到单精度水平，诊断量如实反映。这里用“层矩阵压成单精度 → eig_qr → 结果再压成单精度”来模拟。
// 这个模拟偏乐观（迭代本身仍是双精度，残差约 1e-8）；换成全程 float 运算的本征分解（gpu/cuda_backend.hpp 的
// Precision::Single），在 ridges_on_mirror 上（61 ~ 601 阶）实测残差 7e-7 ~ 2e-6、各级效率偏差 2e-7 ~ 5e-5。
static void test_single_precision_eig() {
    say("[单精度的本征求解器]\n");
    // 先核对“压成单精度”这一步本身：精度确实丢了，而且与硬件的 double → float 转换逐位一致
    {
        CHECK(round_to_single(0.1) != 0.1 && round_to_single(0.1) == 0.100000001490116119384765625);   // 最接近 0.1 的 float
        CHECK(round_to_single(1.0 + 1e-9) == 1.0 && round_to_single(-1.0 - 1e-9) == -1.0);
        CHECK(round_to_single(0.0) == 0.0 && round_to_single(1.5) == 1.5 && round_to_single(-3.0e7) == -3.0e7);   // float 能精确表示的数不变
        CHECK(!has_single_precision(kPi) && !has_single_precision(0.1) && has_single_precision(round_to_single(kPi)) &&
              has_single_precision(round_to_single(0.1)) && round_to_single(round_to_single(kPi)) == round_to_single(kPi));
        // 最后三个数恰好在（或紧挨着）两个相邻 float 的中点上
        const double samples[] = {0.1, -1.0 / 3.0, kPi, 1e10 + 0.5, 1.23456789e-20, -9.87654321e20,
                                  1.0 + 0x1p-24, 1.0 + 0x1p-23 + 0x1p-24, 1.0 + 0x1p-24 + 0x1p-50};
        bool same = true;
        double worst = 0.0;
        for (double x : samples) {
            volatile float as_float = static_cast<float>(x);   // volatile：这次转换和存取不能被优化掉
            const double back = static_cast<double>(as_float);
            same = same && back == round_to_single(x);
            worst = std::max(worst, std::abs(round_to_single(x) - x) / std::abs(x));
        }
        CHECK(same);
        CHECK(worst > 1e-8 && worst <= 0x1p-24);               // 相对舍入误差不超过 2^-24 ≈ 6e-8
    }

    // 回调里的统计：层矩阵有多少元素被改动、改动的相对大小，返回的量是不是逐个都只有单精度
    struct Stats {
        int calls = 0;
        long long entries = 0, changed = 0, not_single = 0;
        double matrix_change = 0.0;    // ||A − A_单精度||_F / ||A||_F
    };
    Stats stats;
    const auto narrow = [](cplx z) { return cplx(round_to_single(z.real()), round_to_single(z.imag())); };
    const auto single_precision = [](cplx z) { return has_single_precision(z.real()) && has_single_precision(z.imag()); };
    rcwa::Options single;
    single.diagnostics = true;
    single.custom_eig = [&stats, narrow, single_precision](const Matrix& a) {
        Matrix b = a;
        for (int j = 0; j < b.cols(); ++j)
            for (int i = 0; i < b.rows(); ++i) {
                b(i, j) = narrow(a(i, j));
                ++stats.entries;
                if (b(i, j) != a(i, j)) ++stats.changed;
            }
        stats.matrix_change = std::max(stats.matrix_change, norm_fro(a - b) / norm_fro(a));
        EigResult r = eig_qr(b);
        for (cplx& v : r.values) v = narrow(v);
        for (int j = 0; j < r.vectors.cols(); ++j)
            for (int i = 0; i < r.vectors.rows(); ++i) r.vectors(i, j) = narrow(r.vectors(i, j));
        for (const cplx& v : r.values)
            if (!single_precision(v)) ++stats.not_single;
        for (int j = 0; j < r.vectors.cols(); ++j)
            for (int i = 0; i < r.vectors.rows(); ++i)
                if (!single_precision(r.vectors(i, j))) ++stats.not_single;
        ++stats.calls;
        return r;
    };
    struct Case {
        rcwa::Stack stack;
        double wavelength, theta;
        int M;
    };
    const Case cases[] = {{mdg_stack(), 1.053, 66.36, 12}, {ridges_on_mirror(), kRidgesWavelength, kRidgesTheta, 40},
                          {ridges_on_silver(), 0.8, 23.0, 30}};
    double worst_gap = 0.0, worst_res = 0.0, least_res = 1.0, least_change = 1.0;
    set_num_threads(1);                                         // 回调里的统计没有加锁
    for (const Case& c : cases)
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const rcwa::Result d = rcwa::solve(c.stack, c.wavelength, c.theta, c.M, pol);
            stats = Stats();
            const rcwa::Result f = rcwa::solve(c.stack, c.wavelength, c.theta, c.M, pol, single);
            CHECK(f.eig_info == 0 && f.n_eig == 1 && stats.calls == 1);
            // 交给本征迭代的矩阵确实只有单精度：大多数元素被改动，整体的相对改动是 float 舍入的量级；返回的量逐个都是单精度数
            CHECK(stats.changed > stats.entries / 2 && stats.not_single == 0);
            CHECK(stats.matrix_change > 1e-9 && stats.matrix_change <= 0x1p-24);
            CHECK_LE(efficiency_gap(d, f), 1e-6);
            CHECK(!same_bits(d, f));                                               // 结果确实变了
            CHECK(f.max_eig_residual > 1e-11 && f.max_eig_residual < 1e-6);      // 远大于双精度的 1e-15，如实报告
            CHECK_LE(f.max_solve_backward_error, 1e-13);                          // 线性求解仍是双精度
            worst_gap = std::max(worst_gap, efficiency_gap(d, f));
            worst_res = std::max(worst_res, f.max_eig_residual);
            least_res = std::min(least_res, f.max_eig_residual);
            least_change = std::min(least_change, stats.matrix_change);
        }
    say("  与双精度相比各级效率最多差 %.1e；本征对残差 %.1e ~ %.1e；层矩阵压成单精度的相对改动最小 %.1e\n", worst_gap, least_res,
                worst_res, least_change);
}

// ---------------------------------------------------------------------------------------------
static void test_layer_matrix() {
    say("[layer_matrix]\n");
    const rcwa::Stack g = grating_stack();
    const int M = 8, n = 2 * M + 1;
    const double wavelength = 1.053, theta = 66.36;

    // 无吸收光栅：TE 的层矩阵是厄米的；对角元 = kx² − ε 的周期平均
    {
        const Matrix a = rcwa::layer_matrix(g.layers[0], g.period, wavelength, g.nI, theta, M, Polarization::TE);
        CHECK(a.rows() == n && a.cols() == n);
        CHECK(is_hermitian(a, 1e-14));
        const double eps_mean = 0.30 * 1.45 * 1.45 + 0.70;
        double worst = 0.0;
        for (int j = 0; j < n; ++j) {
            const double kx = std::sin(theta * kPi / 180.0) - (j - M) * wavelength / g.period;
            worst = std::max(worst, std::abs(a(j, j) - (kx * kx - eps_mean)));
        }
        CHECK_LE(worst, 1e-13);
        // 第 (m, p) 元只依赖 m − p（卷积矩阵）
        CHECK_LE(std::abs(a(3, 1) - a(9, 7)), 1e-15);
    }
    // 吸收光栅：一般复矩阵（非厄米）
    {
        const rcwa::Stack lossy = grating_stack(true);
        const Matrix a = rcwa::layer_matrix(lossy.layers[0], lossy.period, wavelength, lossy.nI, 42.0, M, Polarization::TE);
        CHECK(!is_hermitian(a, 1e-6));
    }

    // 均匀层：按“只有一段的光栅层”构造，得到对角矩阵 + 舍入误差；非对角元很小但不全是严格的零
    for (cplx index : {cplx(1.96, 0.0), cplx(0.079095, 5.42875)})
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const int m_big = 30, size = 2 * m_big + 1;
            const double period = kRidgesPeriod;
            const Matrix a = rcwa::layer_matrix(rcwa::Layer::uniform(0.1, index), period, wavelength, 1.0, theta, m_big, pol);
            double diag_err = 0.0, off = 0.0;
            int nonzero_off = 0;
            for (int j = 0; j < size; ++j)
                for (int i = 0; i < size; ++i) {
                    if (i == j) {
                        const double kx = std::sin(theta * kPi / 180.0) - (j - m_big) * wavelength / period;
                        diag_err = std::max(diag_err, std::abs(a(j, j) - (kx * kx - index * index)) / std::abs(a(j, j)));
                    } else {
                        off = std::max(off, std::abs(a(i, j)));
                        if (a(i, j) != cplx{}) ++nonzero_off;
                    }
                }
            CHECK_LE(diag_err, 1e-12);
            CHECK_LE(off, 1e-12);
            CHECK(nonzero_off > 0);
            if (pol == Polarization::TE && index.imag() == 0.0)
                say("  均匀层（n = %.2f，%d 阶）的层矩阵：非对角元最大 %.1e，其中 %d 个不为零\n", index.real(), size, off, nonzero_off);
        }

    // layer_matrix 给出的就是 solve() 交给本征求解器的那个矩阵（第二个结构有两个光栅层，入射介质的折射率不是 1）
    for (Polarization pol : {Polarization::TE, Polarization::TM})
        for (const rcwa::Stack& s : {mdg_stack(), multilayer_stack()}) {
            const std::size_t bytes = sizeof(cplx) * static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
            std::vector<Matrix> seen;
            rcwa::Options opt;
            opt.custom_eig = [&seen](const Matrix& a) {
                seen.push_back(a);
                return eig_qr(a);
            };
            rcwa::solve(s, wavelength, theta, M, pol, opt);
            std::vector<Matrix> expect;                       // 各光栅层的层矩阵，自上而下
            for (const rcwa::Layer& layer : s.layers)
                if (!layer.homogeneous) expect.push_back(rcwa::layer_matrix(layer, s.period, wavelength, s.nI, theta, M, pol));
            bool same = !expect.empty() && seen.size() == expect.size();
            for (std::size_t j = 0; same && j < expect.size(); ++j)
                same = seen[j].rows() == n && seen[j].cols() == n && std::memcmp(seen[j].data(), expect[j].data(), bytes) == 0;
            CHECK(same);

            seen.clear();
            opt.homogeneous_fast_path = false;                    // 均匀层也交给本征求解器
            rcwa::solve(s, wavelength, theta, M, pol, opt);
            same = seen.size() == s.layers.size();
            for (std::size_t j = 0; same && j < s.layers.size(); ++j) {
                const Matrix e = rcwa::layer_matrix(s.layers[j], s.period, wavelength, s.nI, theta, M, pol);
                same = seen[j].rows() == n && seen[j].cols() == n && std::memcmp(seen[j].data(), e.data(), bytes) == 0;
            }
            CHECK(same);
        }

    CHECK_THROWS(rcwa::layer_matrix(g.layers[0], g.period, wavelength, g.nI, theta, -1, Polarization::TE), std::invalid_argument);
    CHECK_THROWS(rcwa::layer_matrix(g.layers[0], 0.9, wavelength, g.nI, theta, M, Polarization::TE), std::invalid_argument);
    CHECK_THROWS(rcwa::layer_matrix(g.layers[0], g.period, wavelength, cplx(1.0, 0.1), theta, M, Polarization::TE), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------------
static void test_bookkeeping() {
    say("[计数、计时、诊断量]\n");
    const rcwa::Stack s = mdg_stack();
    const int M = 12;
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        const rcwa::Result r = rcwa::solve(s, 1.053, 66.36, M, pol);
        CHECK(static_cast<int>(r.orders.size()) == 2 * M + 1 && r.orders.front() == -M && r.orders.back() == M && r.orders[M] == 0);
        CHECK(r.R.size() == r.orders.size() && r.T.size() == r.orders.size() && r.DEr.size() == r.orders.size() &&
              r.DEt.size() == r.orders.size());
        CHECK(r.n_eig == 1 && r.eig_info == 0);
        // 稠密求解：TM 组装层矩阵 2 次；上界面 Mm、Nn、T+ 各 1 次，入射列 1 次；下界面 T+ 1 次、叠到膜系上 1 次
        CHECK(r.n_dense_solve == (pol == Polarization::TE ? 6 : 8));
        CHECK(r.max_eig_residual == 0.0 && r.max_solve_backward_error == 0.0);       // 没开诊断
        const rcwa::Timing& t = r.timing;                      // 三项互不重叠，都包含在 total 里
        CHECK(t.total > 0.0 && t.eig >= 0.0 && t.linear >= 0.0 && t.product >= 0.0);
        CHECK(t.eig + t.linear + t.product <= t.total * (1.0 + 1e-9));

        rcwa::Options diag;
        diag.diagnostics = true;
        const rcwa::Result d = rcwa::solve(s, 1.053, 66.36, M, pol, diag);
        CHECK(same_bits(r, d));                               // 诊断不改变结果
        CHECK(d.max_eig_residual > 0.0 && d.max_eig_residual < 1e-13);
        CHECK(d.max_solve_backward_error > 0.0 && d.max_solve_backward_error < 1e-13);

        diag.eig = EigMethod::LR;                             // LR 迭代的残差明显大于 QR 迭代
        diag.homogeneous_fast_path = false;
        const rcwa::Result dl = rcwa::solve(s, 1.053, 66.36, M, pol, diag);
        CHECK(dl.n_eig == 20 && dl.max_eig_residual > d.max_eig_residual && dl.max_eig_residual < 1e-8);
        if (pol == Polarization::TE)
            say("  本征对残差：QR %.1e，LR（全部 20 层）%.1e；稠密求解的后向误差 %.1e\n", d.max_eig_residual,
                        dl.max_eig_residual, d.max_solve_backward_error);
    }
    // 不选主元的 LU：后向误差如实变大
    {
        rcwa::Options np;
        np.diagnostics = true;
        np.linear = rcwa::LinearSolver::LUNoPivot;
        const rcwa::Result r = rcwa::solve(ridges_on_silver(), 0.8, 23.0, 30, Polarization::TE, np);
        rcwa::Options pp;
        pp.diagnostics = true;
        const rcwa::Result q = rcwa::solve(ridges_on_silver(), 0.8, 23.0, 30, Polarization::TE, pp);
        CHECK(r.max_solve_backward_error > 100.0 * q.max_solve_backward_error);
        CHECK(q.max_solve_backward_error < 1e-13);
        say("  四脊光栅 + 银膜 M = 30 稠密求解的后向误差：选主元 %.1e，不选主元 %.1e\n", q.max_solve_backward_error,
                    r.max_solve_backward_error);
    }
}

// ---------------------------------------------------------------------------------------------
static void test_failures() {
    say("[数值失败的报告方式]\n");
    const rcwa::Stack s = grating_stack();
    const int M = 6;

    // 本征迭代未收敛：不抛异常，结果全部是 NaN，eig_info 给出 info
    {
        rcwa::Options opt;
        opt.custom_eig = [](const Matrix& a) {
            EigOptions one;
            one.max_iter = 1;
            return eig_qr(a, one);
        };
        const rcwa::Result r = rcwa::solve(s, 1.053, 66.36, M, Polarization::TE, opt);
        CHECK(r.eig_info > 0 && all_nan(r) && r.n_eig == 1);
        CHECK(static_cast<int>(r.orders.size()) == 2 * M + 1 && r.timing.total > 0.0);
        opt.diagnostics = true;
        CHECK(std::isnan(rcwa::solve(s, 1.053, 66.36, M, Polarization::TE, opt).max_eig_residual));
    }
    // 本征向量矩阵奇异（全零）：三种线性求解方法都报告失败
    for (rcwa::LinearSolver lin : {rcwa::LinearSolver::LUPartial, rcwa::LinearSolver::LUNoPivot, rcwa::LinearSolver::QRHouseholder}) {
        rcwa::Options opt;
        opt.linear = lin;
        opt.custom_eig = [](const Matrix& a) {
            EigResult r = eig_qr(a);
            r.vectors = Matrix(a.rows(), a.cols());
            return r;
        };
        const rcwa::Result r = rcwa::solve(s, 1.053, 66.36, M, Polarization::TE, opt);
        CHECK(all_nan(r) && r.eig_info == 0);
    }
    // 本征求解器返回的尺寸不对：这是调用方的错误
    {
        rcwa::Options opt;
        opt.custom_eig = [](const Matrix& a) { return eig_qr(Matrix::identity(a.rows() - 1)); };
        CHECK_THROWS(rcwa::solve(s, 1.053, 66.36, M, Polarization::TE, opt), std::invalid_argument);
        opt.custom_eig = [](const Matrix& a) {               // 本征向量齐全，本征值多一个
            EigResult r = eig_qr(a);
            r.values.push_back(1.0);
            return r;
        };
        CHECK_THROWS(rcwa::solve(s, 1.053, 66.36, M, Polarization::TE, opt), std::invalid_argument);
        opt.custom_eig = [](const Matrix& a) {               // 本征值少一个
            EigResult r = eig_qr(a);
            r.values.pop_back();
            return r;
        };
        CHECK_THROWS(rcwa::solve(s, 1.053, 66.36, M, Polarization::TE, opt), std::invalid_argument);
        opt.custom_eig = [](const Matrix& a) {               // 本征向量矩阵多一行
            EigResult r = eig_qr(a);
            Matrix tall(a.rows() + 1, a.cols());
            for (int j = 0; j < a.cols(); ++j)
                for (int i = 0; i < a.rows(); ++i) tall(i, j) = r.vectors(i, j);
            r.vectors = tall;
            return r;
        };
        CHECK_THROWS(rcwa::solve(s, 1.053, 66.36, M, Polarization::TE, opt), std::invalid_argument);
    }

    // 恰好落在 Rayleigh 异常点（λ = Λ、正入射，±1 级在入射介质里恰好掠射，kz 精确为零）：
    // 入射介质一侧的异常点不妨碍求解，能量仍然守恒。
    {
        rcwa::Stack g = grating_stack();
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const rcwa::Result r = rcwa::solve(g, g.period, 0.0, 8, pol);
            CHECK_LE(energy_defect(r), 1e-10);
            CHECK(r.DEr[8 + 1] == 0.0 && r.DEr[8 - 1] == 0.0);           // 掠射级不带走能量
        }
        // 出射介质一侧恰好掠射、又紧挨着光栅层：界面矩阵奇异，按失败处理（NaN），不抛异常
        g.nII = 2.0;                                                    // ±2 级：n² − kx² = 4 − 4 = 0
        const rcwa::Result bad = rcwa::solve(g, g.period, 0.0, 8, Polarization::TE);
        CHECK(all_nan(bad));
        // 两侧都是均匀层的界面则没有问题
        rcwa::Stack films;
        films.nI = 1.0;
        films.nII = 2.0;
        films.period = 1.0;
        films.layers = {rcwa::Layer::uniform(0.2, 1.5)};
        CHECK_LE(energy_defect(rcwa::solve(films, 1.0, 0.0, 4, Polarization::TE)), 1e-13);
    }
}

static void test_arguments() {
    say("[参数检查]\n");
    const rcwa::Stack s = grating_stack();
    const auto run = [](const rcwa::Stack& st, double wavelength = 1.053, double theta = 30.0, int M = 4) {
        return rcwa::solve(st, wavelength, theta, M);
    };
    CHECK(run(s).eig_info == 0);
    CHECK_THROWS(run(s, 1.053, 30.0, -1), std::invalid_argument);
    CHECK_THROWS(run(s, 0.0), std::invalid_argument);
    CHECK_THROWS(run(s, -1.0), std::invalid_argument);
    CHECK_THROWS(run(s, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, 90.0), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, -90.0), std::invalid_argument);
    CHECK_THROWS(run(s, 1.053, std::numeric_limits<double>::infinity()), std::invalid_argument);
    CHECK(run(s, 1.053, -89.0).eig_info == 0);                // 负的入射角是合法的
    {
        rcwa::Stack t = s;
        t.period = 0.0;
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.nI = cplx(1.0, 0.01);                              // 入射介质必须无吸收
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.nI = 0.0;
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.nII = cplx(1.5, -0.1);                             // 增益介质不支持
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.nII = cplx(1.5, 0.1);                              // 吸收的出射介质是合法的
        CHECK(run(t).eig_info == 0);
        t = s;
        t.layers[0].d = -0.1;
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers[0].n_values.pop_back();                     // 段数与边界数对不上
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers[0].edges.back() += 0.01;                    // 边界没有正好跨一个周期
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        std::swap(t.layers[0].edges[0], t.layers[0].edges[1]);   // 边界不递增
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers[0].n_values[0] = cplx(std::numeric_limits<double>::quiet_NaN(), 0.0);
        CHECK_THROWS(run(t), std::invalid_argument);
        t = s;
        t.layers.push_back(rcwa::Layer::uniform(0.1, 0.0));  // 折射率为零
        CHECK_THROWS(run(t), std::invalid_argument);
    }
    {
        rcwa::Options opt;
        opt.linear = static_cast<rcwa::LinearSolver>(7);
        CHECK_THROWS(rcwa::solve(s, 1.053, 30.0, 4, Polarization::TE, opt), std::invalid_argument);
        opt = rcwa::Options();
        opt.eig = static_cast<EigMethod>(7);
        CHECK_THROWS(rcwa::solve(s, 1.053, 30.0, 4, Polarization::TE, opt), std::invalid_argument);
        CHECK_THROWS(rcwa::solve(s, 1.053, 30.0, 4, static_cast<Polarization>(5)), std::invalid_argument);
    }
    // 工厂函数
    {
        const rcwa::Layer u = rcwa::Layer::uniform(0.3, cplx(1.5, 0.2));
        CHECK(u.homogeneous && u.d == 0.3 && u.n == cplx(1.5, 0.2) && u.edges.empty());
        const rcwa::Layer gr = rcwa::Layer::grating(0.4, {0.0, 0.5, 1.0}, {2.0, 1.0});
        CHECK(!gr.homogeneous && gr.d == 0.4 && gr.edges.size() == 3 && gr.n_values.size() == 2);
        const rcwa::Stack def;
        CHECK(def.nI == cplx(1.0) && def.nII == cplx(1.0) && def.period == 1.0 && def.layers.empty());
        const rcwa::Options opt;
        CHECK(opt.eig == EigMethod::QR && opt.linear == rcwa::LinearSolver::LUPartial && opt.homogeneous_fast_path &&
              !opt.diagnostics && !opt.parallel_layers && !opt.custom_eig);
    }
}

// ---------------------------------------------------------------------------------------------
// 随机膜系：用固定种子随机生成一次，结构照原样写在下面；参考效率来自独立的 Python 求解器（重算与核对见文件开头）。
// 它们补上前面的结构没有覆盖到的组合：
// 均匀膜盖在光栅层上面、入射介质折射率大于 1 的 TM、负的入射角、金属衬底、零厚度的光栅层、3 ~ 5 段的光栅。

// 随机膜系 #3：TM，M = 6，λ = 1.53394，θ = 37.5193°，nI = 1.5，层序（自上而下）UGU
static rcwa::Stack random_stack_3() {
    rcwa::Stack s;
    s.nI = 1.5;
    s.nII = cplx(1.0367, 3.9027);
    s.period = 3.00592;
    s.layers.push_back(rcwa::Layer::uniform(0.06429, cplx(3.0646, 0.070776)));
    s.layers.push_back(rcwa::Layer::grating(0.48278, {1.2216037660790897, 1.5834740458158494, 3.0604920858391766, 3.367905016258635, 3.7419360997352666, 4.22752376607909},
                                            {2.5123, 1.0, cplx(2.0595, 0.045929), 2.261, 1.0}));
    s.layers.push_back(rcwa::Layer::uniform(0.57245, cplx(0.0672, 4.5789)));
    return s;
}
static const double kRandom3Der[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0754058076701966, 0.10026329186740567, 0.09654419767948305,
    0.20689828823308495, 0.28248539300436526, 0.04145254801948253, 0.0,
    0.0,
};
static const double kRandom3Det[] = {  // 第 −M … +M 级
    9.543115818037185e-15, 3.283662387337089e-14, 8.005869089226478e-14, 2.5510832295779902e-12,
    2.5688888326541862e-12, 1.3215065556670574e-11, 1.2789835708723911e-11, 1.2742805503467246e-11,
    4.8771927281465326e-11, 2.4267610374802648e-11, 1.1482951935852367e-12, 1.4847974176240752e-11,
    2.502429889190215e-12,
};

// 随机膜系 #12：TE，M = 4，λ = 0.69177，θ = 54.4633°，nI = 2.0，层序（自上而下）GUGGG
static rcwa::Stack random_stack_12() {
    rcwa::Stack s;
    s.nI = 2.0;
    s.nII = 3.2766;
    s.period = 0.52506;
    s.layers.push_back(rcwa::Layer::grating(0.28899, {0.0, 0.21470852461949538, 0.291635488282107, 0.52506},
                                            {1.5952, 2.6941, 1.0}));
    s.layers.push_back(rcwa::Layer::uniform(0.15519, 1.7212));
    s.layers.push_back(rcwa::Layer::grating(0.59828, {0.0, 0.1348671539039357, 0.38103017701846575, 0.52506},
                                            {2.3259, 2.9847, 1.0}));
    s.layers.push_back(rcwa::Layer::grating(0.09588, {0.23793965557385166, 0.30118145894615533, 0.599632279980103, 0.7629996555738516},
                                            {1.0, 2.8048, 1.8829}));
    s.layers.push_back(rcwa::Layer::grating(0.46157, {0.0, 0.04675228866800866, 0.2649523734345849, 0.37714726161756695, 0.52506},
                                            {1.0, 1.0, 3.0051, 2.0576}));
    return s;
}
static const double kRandom12Der[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.0,
    0.3832157826809998, 0.2883721645071102, 0.21500402726448029, 0.0,
    0.0,
};
static const double kRandom12Det[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.005707913361484256,
    0.06391531736719605, 0.03474921510645482, 0.00604705494112533, 0.0029885247711499674,
    0.0,
};

// 随机膜系 #19：TM，M = 3，λ = 0.57215，θ = 37.1167°，nI = 1.0，层序（自上而下）GGGUG
static rcwa::Stack random_stack_19() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.3489;
    s.period = 1.34061;
    s.layers.push_back(rcwa::Layer::grating(0.0, {0.5156650515534138, 0.8318325263653672, 1.8562750515534139},
                                            {1.0, cplx(3.3889, 0.000229)}));
    s.layers.push_back(rcwa::Layer::grating(0.5349, {-0.5658312694886117, -0.4696050566510656, 0.05782139376129273, 0.7747787305113883},
                                            {1.0, cplx(1.2847, 4.8405), 2.7505}));
    s.layers.push_back(rcwa::Layer::grating(0.79773, {-0.10816328923120395, -0.009302224283890199, 0.6818917680133255, 0.9252570590935646, 1.0791902226686423, 1.2324467107687962},
                                            {1.0, cplx(2.5548, 0.000201), cplx(2.5451, 0.080789), cplx(0.7868, 2.4888), 1.0}));
    s.layers.push_back(rcwa::Layer::uniform(0.1625, cplx(1.21, 0.00255)));
    s.layers.push_back(rcwa::Layer::grating(0.39106, {0.6239481182065285, 0.685927668979927, 1.2835994804498256, 1.5469979778127105, 1.6730792318828505, 1.9645581182065286},
                                            {1.0, cplx(2.3476, 0.000169), 1.0, 1.8527, 1.0}));
    return s;
}
static const double kRandom19Der[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.20855077411345602,
    0.07150325301282726, 0.009207480085788231, 0.019342583334965816,
};
static const double kRandom19Det[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.005398810056788206, 0.02823702618140616,
    0.04767995398120614, 0.06473721886758191, 0.02647271053562467,
};

// 随机膜系 #41：TE，M = 8，λ = 0.99384，θ = 65.3981°，nI = 1.0，层序（自上而下）UGGGGG
static rcwa::Stack random_stack_41() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.1801;
    s.period = 1.13519;
    s.layers.push_back(rcwa::Layer::uniform(0.33267, cplx(1.6936, 0.006139)));
    s.layers.push_back(rcwa::Layer::grating(0.43378, {0.0, 0.16492250206616804, 0.39486054905994067, 0.9978827011848254, 1.0570393072884265, 1.13519},
                                            {cplx(2.2508, 0.010382), 2.5607, 1.4733, 1.0, 1.0}));
    s.layers.push_back(rcwa::Layer::grating(0.29255, {0.0, 0.5632513103596241, 1.13519},
                                            {cplx(1.4522, 0.00052), cplx(3.2144, 0.001193)}));
    s.layers.push_back(rcwa::Layer::grating(0.64326, {0.0, 0.10309602954957174, 0.41613457044054325, 0.6386076340808057, 0.7604868811390587, 1.13519},
                                            {1.2506, cplx(3.0035, 0.006697), 3.3129, 2.2244, cplx(2.3656, 0.000767)}));
    s.layers.push_back(rcwa::Layer::grating(0.0, {0.28439787828427904, 0.5892782848773083, 1.0542527740472492, 1.419587878284279},
                                            {cplx(1.604, 5.5012), cplx(2.8767, 0.066147), 1.0}));
    s.layers.push_back(rcwa::Layer::grating(0.41493, {0.27259822061918854, 0.7395007185144321, 0.8643499284689117, 1.1409946360379553, 1.4077882206191885},
                                            {cplx(0.0642, 2.8712), 3.4365, cplx(1.4507, 3.2998), 1.5234}));
    return s;
}
static const double kRandom41Der[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.4221868626768658, 0.20895444932963717, 0.030046706239548002, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0,
};
static const double kRandom41Det[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.011296730770633572, 0.030436371037198373, 0.00550782678902443, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0,
};

// 随机膜系 #45：TE，M = 3，λ = 0.99554，θ = 11.0152°，nI = 2.0，层序（自上而下）UGU
static rcwa::Stack random_stack_45() {
    rcwa::Stack s;
    s.nI = 2.0;
    s.nII = cplx(0.1468, 5.3027);
    s.period = 1.21537;
    s.layers.push_back(rcwa::Layer::uniform(0.49364, 1.6185));
    s.layers.push_back(rcwa::Layer::grating(0.03897, {0.06487027800333718, 0.11559842057396437, 0.1923385450658246, 0.2413254782316344, 1.2802402780033373},
                                            {cplx(3.1367, 0.141471), 1.3106, 1.8873, cplx(1.8873, 5.7138)}));
    s.layers.push_back(rcwa::Layer::uniform(0.36862, cplx(1.5923, 0.001829)));
    return s;
}
static const double kRandom45Der[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.015102765645356984, 0.6173332982910414,
    0.01070698390378844, 0.011918498647142561, 0.0,
};
static const double kRandom45Det[] = {  // 第 −M … +M 级
    1.5406001150460165e-09, 2.70847643493388e-07, 0.0003438631300036473, 0.0011835371542866565,
    0.00015664524361772388, 0.00011465373967583268, 1.2392137630542709e-07,
};

// 随机膜系 #55：TM，M = 6，λ = 1.46171，θ = -74.5678°，nI = 1.0，层序（自上而下）UUGUGU
static rcwa::Stack random_stack_55() {
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = cplx(1.9323, 0.027469);
    s.period = 3.56473;
    s.layers.push_back(rcwa::Layer::uniform(0.09677, 1.9021));
    s.layers.push_back(rcwa::Layer::uniform(0.5005, 1.459));
    s.layers.push_back(rcwa::Layer::grating(0.7394, {1.1965621850955124, 1.674869829777083, 1.9360266248213347, 3.2059797244276256, 3.6518692009143945, 4.761292185095512},
                                            {cplx(3.1595, 0.084433), 1.0, cplx(3.43, 0.021694), 1.0, cplx(1.8723, 1.8283)}));
    s.layers.push_back(rcwa::Layer::uniform(0.31727, cplx(0.4842, 2.9218)));
    s.layers.push_back(rcwa::Layer::grating(0.65018, {-0.06740477119207551, 0.26711249145034227, 0.9542194343337393, 2.3133782387418105, 3.1524893366562927, 3.4973252288079246},
                                            {cplx(2.8621, 0.03192), cplx(1.2083, 5.1586), 1.0, cplx(0.0871, 5.3898), cplx(2.963, 0.159421)}));
    s.layers.push_back(rcwa::Layer::uniform(0.14492, 1.6278));
    return s;
}
static const double kRandom55Der[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0026299453302002763, 0.028462920220367276,
    0.008982911765036683, 0.06264471514385646, 0.08893929396727249, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0,
};
static const double kRandom55Det[] = {  // 第 −M … +M 级
    8.877845133773017e-08, 1.5356845365621905e-05, 8.444016291324286e-06, 2.541195946265575e-05,
    1.2672954546759521e-06, 1.914427068074564e-06, 3.6875083419265464e-06, 4.63385833509127e-07,
    8.703604682977829e-07, 1.1354556082151584e-08, 4.7502950763221304e-09, 1.8968622676295928e-09,
    1.57371580323456e-09,
};

// 随机膜系 #62：TM，M = 6，λ = 0.93068，θ = 74.7309°，nI = 1.33，层序（自上而下）UGGGUU
static rcwa::Stack random_stack_62() {
    rcwa::Stack s;
    s.nI = 1.33;
    s.nII = 3.433;
    s.period = 1.67758;
    s.layers.push_back(rcwa::Layer::uniform(0.76402, 1.569));
    s.layers.push_back(rcwa::Layer::grating(0.74761, {0.0, 0.5517663329263505, 1.0378497820342552, 1.1806586012144589, 1.5058924320839342, 1.67758},
                                            {1.0, 2.4293, 1.1122, 1.0, 1.0}));
    s.layers.push_back(rcwa::Layer::grating(0.67934, {0.0, 0.8817153077149015, 1.6053185351937984, 1.67758},
                                            {2.4334, 1.069, 2.4808}));
    s.layers.push_back(rcwa::Layer::grating(0.70328, {0.7952418057740204, 1.169295627613309, 1.5435843940554042, 2.254297633825864, 2.4728218057740206},
                                            {1.449, 1.0, 3.2397, 3.1695}));
    s.layers.push_back(rcwa::Layer::uniform(0.0, 2.6041));
    s.layers.push_back(rcwa::Layer::uniform(0.02951, 2.7073));
    return s;
}
static const double kRandom62Der[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.14309972383363825, 0.02823190815189106,
    0.06251067064214529, 0.0011199472789374268, 0.04782662168263417, 0.0,
    0.0,
};
static const double kRandom62Det[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.0017742721221865763,
    0.05872234355972177, 0.11932508212989679, 0.026556963093545403, 0.09815787172521566,
    0.1284545456398007, 0.04514715844328932, 0.11196423894557352, 0.0571651415934116,
    0.06994351115810994,
};

// 随机膜系 #71：TM，M = 3，λ = 0.77593，θ = -64.3562°，nI = 2.0，层序（自上而下）G
static rcwa::Stack random_stack_71() {
    rcwa::Stack s;
    s.nI = 2.0;
    s.nII = 1.4127;
    s.period = 1.26482;
    s.layers.push_back(rcwa::Layer::grating(0.23688, {-0.48966611139860494, -0.1948413813954841, 0.03509971762771468, 0.6509504632831165, 0.7751538886013951},
                                            {1.726, cplx(1.0404, 5.2965), cplx(2.5719, 0.200919), 1.0}));
    return s;
}
static const double kRandom71Der[] = {  // 第 −M … +M 级
    0.015903741276719105, 0.003562503414693614, 0.0237101589735809, 0.006992615767829193,
    0.0, 0.0, 0.0,
};
static const double kRandom71Det[] = {  // 第 −M … +M 级
    0.0477469685045834, 0.057978345499016266, 0.12174897727175378, 0.0,
    0.0, 0.0, 0.0,
};

// 随机膜系 #76：TE，M = 5，λ = 0.47496，θ = 24.135°，nI = 1.5，层序（自上而下）UGUUUG
static rcwa::Stack random_stack_76() {
    rcwa::Stack s;
    s.nI = 1.5;
    s.nII = 2.6042;
    s.period = 0.50524;
    s.layers.push_back(rcwa::Layer::uniform(0.71632, 1.47));
    s.layers.push_back(rcwa::Layer::grating(0.26979, {0.09359053359145031, 0.2826544079523479, 0.5988305335914503},
                                            {1.0, 1.7448}));
    s.layers.push_back(rcwa::Layer::uniform(0.38688, 1.7637));
    s.layers.push_back(rcwa::Layer::uniform(0.1942, 3.2247));
    s.layers.push_back(rcwa::Layer::uniform(0.10354, 1.9487));
    s.layers.push_back(rcwa::Layer::grating(0.23767, {0.0, 0.2181867785222636, 0.2924767618694831, 0.50524},
                                            {2.2155, 1.0, 3.3026}));
    return s;
}
static const double kRandom76Der[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.06639252674484011, 0.06164559495668007, 0.10516225446884214,
    0.0, 0.0, 0.0,
};
static const double kRandom76Det[] = {  // 第 −M … +M 级
    0.0, 0.0, 0.0, 0.09858765795261774,
    0.11258236794483147, 0.06894867629013128, 0.17142421834977067, 0.050916995604148765,
    0.26433970768813764, 0.0, 0.0,
};

struct RandomCase {
    int id;
    Polarization pol;
    double wavelength, theta;
    int M;
    rcwa::Stack (*stack)();
    const double* der;
    const double* det;
};
static const RandomCase kRandomCases[] = {
    {3, Polarization::TM, 1.53394, 37.5193, 6, random_stack_3, kRandom3Der, kRandom3Det},
    {12, Polarization::TE, 0.69177, 54.4633, 4, random_stack_12, kRandom12Der, kRandom12Det},
    {19, Polarization::TM, 0.57215, 37.1167, 3, random_stack_19, kRandom19Der, kRandom19Det},
    {41, Polarization::TE, 0.99384, 65.3981, 8, random_stack_41, kRandom41Der, kRandom41Det},
    {45, Polarization::TE, 0.99554, 11.0152, 3, random_stack_45, kRandom45Der, kRandom45Det},
    {55, Polarization::TM, 1.46171, -74.5678, 6, random_stack_55, kRandom55Der, kRandom55Det},
    {62, Polarization::TM, 0.93068, 74.7309, 6, random_stack_62, kRandom62Der, kRandom62Det},
    {71, Polarization::TM, 0.77593, -64.3562, 3, random_stack_71, kRandom71Der, kRandom71Det},
    {76, Polarization::TE, 0.47496, 24.135, 5, random_stack_76, kRandom76Der, kRandom76Det},
};

static double random_error(const rcwa::Result& r, const RandomCase& c) {
    const std::size_t n = static_cast<std::size_t>(2 * c.M + 1);
    if (r.DEr.size() != n || r.DEt.size() != n) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        const double e = std::max(std::abs(r.DEr[j] - c.der[j]), std::abs(r.DEt[j] - c.det[j]));
        if (!(e <= worst)) worst = e;            // NaN 也算不合格
    }
    return worst;
}

static void test_random_stacks() {
    say("[随机膜系：各级效率与独立的 Python 求解器逐级比较]\n");
    double worst_qr = 0.0, worst_lr = 0.0;
    for (const RandomCase& c : kRandomCases) {
        const rcwa::Stack s = c.stack();
        set_num_threads(1);
        const rcwa::Result base = rcwa::solve(s, c.wavelength, c.theta, c.M, c.pol);
        CHECK(base.eig_info == 0);
        // 快速路径开/关 × 两种线性求解 × 两种本征迭代（阶数都不超过 17，LR 迭代在这里还是准确的）
        for (bool fast : {true, false})
            for (rcwa::LinearSolver lin : {rcwa::LinearSolver::LUPartial, rcwa::LinearSolver::QRHouseholder})
                for (EigMethod method : {EigMethod::QR, EigMethod::LR}) {
                    rcwa::Options opt;
                    opt.homogeneous_fast_path = fast;
                    opt.linear = lin;
                    opt.eig = method;
                    const double err = random_error(rcwa::solve(s, c.wavelength, c.theta, c.M, c.pol, opt), c);
                    CHECK_LE(err, 1e-9);
                    double& worst = method == EigMethod::QR ? worst_qr : worst_lr;
                    if (!(err <= worst)) worst = err;
                }
        // 各层并行、扫描：与串行逐位相同
        rcwa::Options par;
        par.parallel_layers = true;
        set_num_threads(3);
        CHECK(same_bits(base, rcwa::solve(s, c.wavelength, c.theta, c.M, c.pol, par)));
        const std::vector<rcwa::Result> swept = rcwa::sweep(s, {{c.wavelength, c.theta}, {c.wavelength, c.theta}}, c.M, c.pol);
        CHECK(swept.size() == 2 && same_bits(base, swept[0]) && same_bits(base, swept[1]));
        set_num_threads(1);
    }
    say("  %d 个膜系 × 8 种选项组合：最大偏差 %.1e（QR 迭代）、%.1e（LR 迭代）\n",
                static_cast<int>(sizeof kRandomCases / sizeof kRandomCases[0]), worst_qr, worst_lr);
}

// ---------------------------------------------------------------------------------------------
// 不依赖参考值的检查：把结构换一种等价的写法，结果不应当变。

// 所有光栅层的边界整体平移 dx
static rcwa::Stack shifted(rcwa::Stack s, double dx) {
    for (rcwa::Layer& layer : s.layers)
        for (double& e : layer.edges) e += dx;
    return s;
}
// 左右翻转：x → −x
static rcwa::Stack mirrored(rcwa::Stack s) {
    for (rcwa::Layer& layer : s.layers) {
        if (layer.homogeneous) continue;
        std::vector<double> e(layer.edges.rbegin(), layer.edges.rend());
        for (double& x : e) x = -x;
        layer.edges = e;
        std::reverse(layer.n_values.begin(), layer.n_values.end());
    }
    return s;
}
// 每一层切成 0.3 d 和 0.7 d 两片，中间夹一个厚度为零的别的材料的层（均匀层、光栅层轮流）
static rcwa::Stack split_layers(const rcwa::Stack& s) {
    rcwa::Stack out = s;
    out.layers.clear();
    int k = 0;
    for (const rcwa::Layer& layer : s.layers) {
        rcwa::Layer a = layer, b = layer;
        a.d = 0.3 * layer.d;
        b.d = 0.7 * layer.d;
        out.layers.push_back(a);
        if (k++ % 2 == 0) out.layers.push_back(rcwa::Layer::uniform(0.0, cplx(2.2, 0.1)));
        else out.layers.push_back(rcwa::Layer::grating(0.0, {0.0, 0.4 * s.period, s.period}, {cplx(1.3, 0.2), 2.5}));
        out.layers.push_back(b);
    }
    return out;
}

static void test_invariances() {
    say("[等价的写法：平移、镜像、分层]\n");
    struct Case {
        rcwa::Stack stack;
        double wavelength, theta;
        int M;
    };
    const Case cases[] = {
        {multilayer_stack(), 0.83, 27.0, 10},          // 两个不对称的光栅层夹吸收膜，入射介质 n = 1.2
        {random_stack_55(), 1.46171, -74.5678, 6},     // 均匀膜盖在光栅层上面，负的入射角
        {mdg_stack(), 1.053, 66.36, 8},
    };
    double worst_shift = 0.0, worst_mirror = 0.0, worst_split = 0.0;
    for (const Case& c : cases)
        for (Polarization pol : {Polarization::TE, Polarization::TM})
            for (bool fast : {true, false}) {
                rcwa::Options opt;
                opt.homogeneous_fast_path = fast;
                const int M = c.M;
                const rcwa::Result base = rcwa::solve(c.stack, c.wavelength, c.theta, M, pol, opt);

                // 1) 整体平移（不到一个周期、负方向超过一个周期、正好两个周期）：各级效率不变
                for (double fraction : {0.37, -1.3, 2.0}) {
                    const rcwa::Result r = rcwa::solve(shifted(c.stack, fraction * c.stack.period), c.wavelength, c.theta, M, pol, opt);
                    const double gap = efficiency_gap(base, r);
                    CHECK_LE(gap, 1e-10);
                    if (!(gap <= worst_shift)) worst_shift = gap;
                }
                // 2) 结构左右翻转、入射角取反：第 m 级与第 −m 级互换
                {
                    const rcwa::Result r = rcwa::solve(mirrored(c.stack), c.wavelength, -c.theta, M, pol, opt);
                    double gap = r.DEr.size() == base.DEr.size() ? 0.0 : std::numeric_limits<double>::infinity();
                    for (int j = 0; gap == gap && std::isfinite(gap) && j <= 2 * M; ++j) {
                        const double e = std::max(std::abs(base.DEr[j] - r.DEr[2 * M - j]), std::abs(base.DEt[j] - r.DEt[2 * M - j]));
                        if (!(e <= gap)) gap = e;
                    }
                    CHECK_LE(gap, 1e-10);
                    if (!(gap <= worst_mirror)) worst_mirror = gap;
                }
                // 3) 每层切成两片、中间插入零厚度的层：复振幅和效率都不变
                {
                    const rcwa::Result r = rcwa::solve(split_layers(c.stack), c.wavelength, c.theta, M, pol, opt);
                    double gap = efficiency_gap(base, r);
                    for (int j = 0; gap == gap && std::isfinite(gap) && j <= 2 * M; ++j) {
                        const double e = std::max(std::abs(base.R[j] - r.R[j]), std::abs(base.T[j] - r.T[j]));
                        if (!(e <= gap)) gap = e;
                    }
                    CHECK_LE(gap, 1e-10);
                    CHECK(r.n_eig > base.n_eig);
                    if (!(gap <= worst_split)) worst_split = gap;
                }
            }
    say("  最大差别：平移 %.1e，镜像 %.1e，分层 %.1e\n", worst_shift, worst_mirror, worst_split);

    // 折射率的虚部写成 −0.0（例如对实数取共轭的结果）与 +0.0 等价：均匀介质里的倏逝级仍然取沿传播方向衰减的一支
    {
        rcwa::Stack plus = grating_stack();
        plus.layers.push_back(rcwa::Layer::uniform(0.05, 1.2));     // 低折射率薄膜：高级次在里面是倏逝的
        rcwa::Stack minus = plus;
        minus.nI = cplx(1.0, -0.0);
        minus.nII = cplx(1.45, -0.0);
        minus.layers[1].n = cplx(1.2, -0.0);
        for (Polarization pol : {Polarization::TE, Polarization::TM})
            for (bool fast : {true, false}) {
                rcwa::Options opt;
                opt.homogeneous_fast_path = fast;
                const rcwa::Result a = rcwa::solve(plus, 1.053, 30.0, 8, pol, opt), b = rcwa::solve(minus, 1.053, 30.0, 8, pol, opt);
                CHECK_LE(efficiency_gap(a, b), 1e-13);
                CHECK_LE(energy_defect(b), 1e-12);
            }
    }

    // 只有一段的光栅层、各段折射率相同的光栅层，与同样折射率的均匀层等价（斜入射；它们要做本征分解，均匀层不用）
    {
        rcwa::Stack films;
        films.nI = 1.0;
        films.nII = 1.45;
        films.period = 0.8;
        films.layers = {rcwa::Layer::uniform(0.3, 1.9), rcwa::Layer::grating(0.2, {0.0, 0.3, 0.8}, {1.6, 1.0}),
                        rcwa::Layer::uniform(0.25, cplx(1.5, 0.05))};
        rcwa::Stack as_gratings = films;
        as_gratings.layers[0] = rcwa::Layer::grating(0.3, {0.1, 0.9}, {1.9});
        as_gratings.layers[2] = rcwa::Layer::grating(0.25, {0.0, 0.5, 0.8}, {cplx(1.5, 0.05), cplx(1.5, 0.05)});
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const rcwa::Result a = rcwa::solve(films, 1.0, 33.0, 8, pol), b = rcwa::solve(as_gratings, 1.0, 33.0, 8, pol);
            CHECK_LE(efficiency_gap(a, b), 1e-11);
            CHECK(a.n_eig == 1 && b.n_eig == 3);
        }
    }
    // M = 0（只保留零级）：TE 时光栅层等价于介电常数取周期平均值的均匀层
    {
        rcwa::Stack g = grating_stack();                              // 占空比 0.30，齿 1.45，槽 1.0
        rcwa::Stack film = g;
        film.layers[0] = rcwa::Layer::uniform(g.layers[0].d, std::sqrt(0.30 * 1.45 * 1.45 + 0.70));
        const rcwa::Result a = rcwa::solve(g, 1.053, 35.0, 0, Polarization::TE), b = rcwa::solve(film, 1.053, 35.0, 0, Polarization::TE);
        CHECK(a.n_eig == 1 && b.n_eig == 0 && a.R.size() == 1);
        CHECK_LE(std::abs(a.R[0] - b.R[0]) + std::abs(a.T[0] - b.T[0]), 1e-13);
        CHECK_LE(energy_defect(rcwa::solve(g, 1.053, 35.0, 0, Polarization::TM)), 1e-13);
    }

    // 稠密路径在入射角极小（但不为零）时与快速路径一致，QR、LR 迭代都是。
    // （恰好正入射时均匀层的 ±m 级本征值严格相等，稠密路径的精度没有保证，见 rcwa1d.hpp 里 solve() 的说明。）
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        const rcwa::Stack s = mdg_stack();
        const rcwa::Result fast = rcwa::solve(s, 1.053, 1e-6, 12, pol);
        for (EigMethod method : {EigMethod::QR, EigMethod::LR}) {
            rcwa::Options opt;
            opt.homogeneous_fast_path = false;
            opt.eig = method;
            CHECK_LE(efficiency_gap(fast, rcwa::solve(s, 1.053, 1e-6, 12, pol, opt)), 1e-9);
        }
        CHECK_LE(efficiency_gap(fast, rcwa::solve(s, 1.053, 0.0, 12, pol)), 1e-6);      // 快速路径对入射角是连续的
    }
}

// ---------------------------------------------------------------------------------------------
// 级次与方向的约定：第 m 级随 x 的变化是 exp(−i·k0·kx_m·x)，kx_m = nI·sin θ − m·λ/Λ，而 ε(x) = Σ c_h·exp(+i·2πh·x/Λ)，
// 所以第 m 级的横向波矢比入射波多 +m·2π/Λ，入射波到第 m 级的耦合由 c_m 决定。
// 用一层很薄、折射率起伏很小的光栅核对：一阶微扰（Born 近似）下第 m 级的振幅正比于 c_m / kz_m，于是两个传播级之间
//     DE(+1) / DE(−1) = |c_{+1} / c_{−1}|² · kz_{−1} / kz_{+1}        （反射、透射都是；TE）。
// 剖面取左右不对称、带吸收的三段，|c_{+1}|² 与 |c_{−1}|² 相差五倍；约定里任何一处符号弄反，比值就变成倒数。
static void test_order_convention() {
    say("[级次与方向的约定：薄光栅的一阶微扰]\n");
    const double period = 1.7, wavelength = 1.0, depth = 1e-3;
    const std::vector<double> edges{0.0, 0.2 * period, 0.55 * period, period};
    const std::vector<cplx> index{cplx(1.02, 0.03), cplx(1.05, 0.0), cplx(1.0, 0.0)};
    // ε 的 Fourier 系数按定义逐段积分，写成“段宽的 sinc · 段中心的相位”（与 rcwa1d.hpp 里端点相位之差的写法不同）
    const auto coefficient = [&](int h) {
        cplx sum{};
        for (std::size_t j = 0; j < index.size(); ++j) {
            const double width = (edges[j + 1] - edges[j]) / period, center = 0.5 * (edges[j + 1] + edges[j]) / period;
            sum += index[j] * index[j] * (std::sin(kPi * h * width) / (kPi * h)) * std::exp(cplx(0.0, -2.0 * kPi * h * center));
        }
        return sum;
    };
    const double coupling = std::norm(coefficient(+1)) / std::norm(coefficient(-1));     // ≈ 0.189
    CHECK(coupling < 0.25);
    rcwa::Stack s;
    s.nI = 1.0;
    s.nII = 1.0;
    s.period = period;
    s.layers = {rcwa::Layer::grating(depth, edges, index)};
    const int M = 6;
    for (double theta : {0.0, 20.0, -20.0}) {
        const double kx0 = std::sin(theta * kPi / 180.0), step = wavelength / period;
        const double kz_plus = std::sqrt(1.0 - (kx0 - step) * (kx0 - step)), kz_minus = std::sqrt(1.0 - (kx0 + step) * (kx0 + step));
        const double expect = coupling * kz_minus / kz_plus;
        const rcwa::Result r = rcwa::solve(s, wavelength, theta, M, Polarization::TE);
        CHECK(r.DEt[M + 1] > 0.0 && r.DEt[M - 1] > 0.0 && r.DEr[M + 1] > 0.0 && r.DEr[M - 1] > 0.0);
        CHECK_LE(std::abs(r.DEt[M + 1] / r.DEt[M - 1] / expect - 1.0), 5e-3);
        CHECK_LE(std::abs(r.DEr[M + 1] / r.DEr[M - 1] / expect - 1.0), 5e-3);
        say("  θ = %+5.1f°：DEt(+1)/DEt(−1) = %.6f，一阶微扰 %.6f\n", theta, r.DEt[M + 1] / r.DEt[M - 1], expect);
    }
    // TM 在正入射时两级的偏振因子相同，比值同样是 |c_{+1} / c_{−1}|²
    {
        const rcwa::Result r = rcwa::solve(s, wavelength, 0.0, M, Polarization::TM);
        CHECK_LE(std::abs(r.DEt[M + 1] / r.DEt[M - 1] / coupling - 1.0), 5e-3);
    }
    // 把结构左右翻转，c_{+1} 与 c_{−1} 互换，比值变成倒数
    {
        const rcwa::Result r = rcwa::solve(mirrored(s), wavelength, 0.0, M, Polarization::TE);
        CHECK_LE(std::abs(r.DEt[M + 1] / r.DEt[M - 1] * coupling - 1.0), 5e-3);
    }
}

// ---------------------------------------------------------------------------------------------
// 恰好落在 Rayleigh 异常点上（某一级在某个均匀介质里 kz 精确为零）时怎么处理，约定见 rcwa1d.hpp 开头。
// 这里都取 λ = Λ = 1、正入射：第 ±m 级的 kx = ∓m，在折射率恰好是整数 m 的介质里这两级恰好掠射。
static void test_rayleigh_anomalies() {
    say("[恰好落在 Rayleigh 异常点上]\n");
    const rcwa::Layer grating = rcwa::Layer::grating(0.3, {0.0, 0.3, 1.0}, {1.5, 1.0});
    const auto make = [](std::vector<rcwa::Layer> layers, cplx nII) {
        rcwa::Stack s;
        s.nI = 1.0;
        s.nII = nII;
        s.period = 1.0;
        s.layers = std::move(layers);
        return s;
    };
    const int M = 6;

    // (1) 有限均匀层的精确零根保留线性场，耦合光栅时仍能量守恒。
    const rcwa::Stack film_on_grating = make({rcwa::Layer::uniform(0.2, 2.0), grating}, 1.5);
    const rcwa::Stack film_below = make({grating, rcwa::Layer::uniform(0.2, 2.0)}, 1.5);
    const rcwa::Stack film_buried = make({grating, rcwa::Layer::uniform(0.1, 1.7), rcwa::Layer::uniform(0.2, 2.0), rcwa::Layer::uniform(0.1, 1.7)}, 1.5);
    const rcwa::Stack film_zero_thickness = make({rcwa::Layer::uniform(0.0, 2.0), grating}, 1.5);
    for (Polarization pol : {Polarization::TE, Polarization::TM})
        for (rcwa::LinearSolver lin : {rcwa::LinearSolver::LUPartial, rcwa::LinearSolver::QRHouseholder}) {
            rcwa::Options opt;
            opt.linear = lin;
            for (const rcwa::Stack* s : {&film_on_grating, &film_below, &film_buried, &film_zero_thickness}) {
                const rcwa::Result r = rcwa::solve(*s, 1.0, 0.0, M, pol, opt);
                CHECK(r.eig_info == 0);
                CHECK_LE(energy_defect(r), 2e-11);
            }
        }
    //     M 取小一些、不包含那一级时没有问题；波长挪开一点点（相对 1e-9）也照常求解，能量守恒
    for (Polarization pol : {Polarization::TE, Polarization::TM})
        for (const rcwa::Stack* s : {&film_on_grating, &film_below, &film_buried}) {
            CHECK_LE(energy_defect(rcwa::solve(*s, 1.0, 0.0, 1, pol)), 1e-12);
            CHECK_LE(energy_defect(rcwa::solve(*s, 1.0 + 1e-9, 0.0, M, pol)), 1e-9);
            CHECK_LE(energy_defect(rcwa::solve(*s, 1.0 - 1e-9, 0.0, M, pol)), 1e-9);
        }

    // 闭式截止膜：主场线性变化，TE 和 TM 的阻抗给出不同结果。
    {
        const double kx = 2.0 * std::sin(30.0 * kPi / 180.0);
        rcwa::Stack cutoff;
        cutoff.nI = cutoff.nII = 2.0;
        cutoff.period = 1.0;
        cutoff.layers = {rcwa::Layer::uniform(1.0 / (kPi * std::sqrt(4.0 - kx * kx)), kx)};
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const auto r = rcwa::solve(cutoff, 1.0, 30.0, 0, pol);
            const double expected = pol == Polarization::TE ? 0.5 : 1.0 / 17.0;
            CHECK_LE(std::abs(r.reflectance() - expected), 2e-15);
            CHECK_LE(std::abs(r.transmittance() - (1.0 - expected)), 2e-15);
            // 稠密层本征值由回调明确返回零，验证本征模式路径的相同分支。
            rcwa::Options dense;
            dense.homogeneous_fast_path = false;
            dense.custom_eig = [](const Matrix& a) {
                auto e = stablercwa::eig(a);
                e.values[0] = cplx{};
                return e;
            };
            const auto rd = rcwa::solve(cutoff, 1.0, 30.0, 0, pol, dense);
            CHECK_LE(std::abs(rd.reflectance() - expected), 2e-15);
            CHECK_LE(std::abs(rd.transmittance() - (1.0 - expected)), 2e-15);
        }
    }
    // 辅助传播仅显式启用；极大有限 h 的散射系数不因 2+h 溢出。
    {
        namespace dt = rcwa::detail;
        const Vector q{cplx{}};
        const auto ordinary = dt::propagation_block(q, 1.0, 2.0, true);
        CHECK(ordinary.Tdd.m(0, 0) == cplx(1.0) && ordinary.Rdu.m(0, 0) == cplx{});
        const auto huge = dt::propagation_block(q, 1e308, 1.0, true, true);
        CHECK(huge.Tdd.m(0, 0).real() > 0.0 && huge.Rdu.m(0, 0) == cplx(1.0));
        bool failed = false;
        try { (void)dt::propagation_block(q, 1e308, 10.0, true, true); }
        catch (const dt::NumericalFailure&) { failed = true; }
        CHECK(failed);
    }
    // 双重截止（±2 级）耦合光栅：检查邻近非零根的对称外推和零厚度消去。
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        rcwa::Options opt;
        opt.diagnostics = true;
        const auto exact = rcwa::solve(film_on_grating, 1.0, 0.0, M, pol, opt);
        auto shifted = [&](double delta) {
            auto s = film_on_grating;
            s.layers[0].n += delta;
            return rcwa::solve(s, 1.0, 0.0, M, pol);
        };
        const auto p = shifted(1e-4), m = shifted(-1e-4);
        const auto p2 = shifted(5e-5), m2 = shifted(-5e-5);
        for (int j = 0; j <= 2 * M; ++j) {
            const cplx rr = (4.0 * (p2.R[j] + m2.R[j]) - (p.R[j] + m.R[j])) / 6.0;
            const cplx tt = (4.0 * (p2.T[j] + m2.T[j]) - (p.T[j] + m.T[j])) / 6.0;
            CHECK_LE(std::abs(exact.R[j] - rr), 2e-9);
            CHECK_LE(std::abs(exact.T[j] - tt), 2e-9);
        }
        const auto zero = rcwa::solve(film_zero_thickness, 1.0, 0.0, M, pol);
        const auto absent = rcwa::solve(make({grating}, 1.5), 1.0, 0.0, M, pol);
        for (int j = 0; j <= 2 * M; ++j) {
            CHECK_LE(std::abs(zero.R[j] - absent.R[j]), 2e-14);
            CHECK_LE(std::abs(zero.T[j] - absent.T[j]), 2e-14);
        }
        set_num_threads(3);
        const auto swept = rcwa::sweep(film_on_grating, {{1.0, 0.0}, {1.01, 0.0}}, M, pol, opt);
        set_num_threads(1);
        CHECK(same_bits(swept[0], exact));
        CHECK(std::isfinite(exact.max_eig_residual));
    }

    // (2) 纯膜系不产生衍射：入射级之外的各级没有被激发，振幅是零，它们恰好掠射不影响结果——
    //     与只保留入射级（M = 0）的结果相同。几种情形：膜里掠射；膜与入射介质相同、界面两侧同时掠射；
    //     膜与出射介质相同；只有出射介质里掠射；两层膜里各有一级掠射。
    {
        struct Films {
            rcwa::Stack stack;
            double wavelength;
            int orders;
        };
        const Films films[] = {
            {make({rcwa::Layer::uniform(0.2, 2.0)}, 1.5), 1.0, 4},
            {make({rcwa::Layer::uniform(0.2, 1.0)}, 1.5), 1.0, 4},
            {make({rcwa::Layer::uniform(0.2, 2.0)}, 2.0), 1.0, 4},
            {make({rcwa::Layer::uniform(0.2, 1.7)}, 2.0), 1.0, 4},
            {make({rcwa::Layer::uniform(0.1, 2.0), rcwa::Layer::uniform(0.3, 1.5), rcwa::Layer::uniform(0.0, 1.0)}, cplx(1.3, 0.2)), 0.5, 5},
        };
        for (const Films& f : films)
            for (Polarization pol : {Polarization::TE, Polarization::TM}) {
                const int m = f.orders;
                const rcwa::Result r = rcwa::solve(f.stack, f.wavelength, 0.0, m, pol), r0 = rcwa::solve(f.stack, f.wavelength, 0.0, 0, pol);
                CHECK(r.eig_info == 0 && r.n_eig == 0 && r.R.size() == static_cast<std::size_t>(2 * m + 1));
                bool others_zero = r.R.size() == static_cast<std::size_t>(2 * m + 1);
                for (int j = 0; others_zero && j <= 2 * m; ++j)
                    if (j != m) others_zero = r.R[j] == cplx{} && r.T[j] == cplx{} && r.DEr[j] == 0.0 && r.DEt[j] == 0.0;
                CHECK(others_zero);
                CHECK_LE(others_zero ? std::abs(r.R[m] - r0.R[0]) + std::abs(r.T[m] - r0.T[0]) : 1.0, 1e-15);
                CHECK_LE(others_zero ? std::abs(r.DEr[m] - r0.DEr[0]) + std::abs(r.DEt[m] - r0.DEt[0]) : 1.0, 1e-15);
            }
    }

    // (3) 扫描里有一个点落在异常点上（出射介质恰好掠射，上面紧挨光栅层）：只有那个点是 NaN，其余各点不受影响
    {
        const rcwa::Stack s = make({grating}, 2.0);
        const std::vector<rcwa::Point> points{{1.01, 0.0}, {1.0, 0.0}, {0.93, 12.0}, {1.0, 1e-3}};
        set_num_threads(3);
        const std::vector<rcwa::Result> swept = rcwa::sweep(s, points, M, Polarization::TM);
        set_num_threads(1);
        bool ok = swept.size() == points.size() && all_nan(swept[1]) && swept[1].eig_info == 0 && swept[1].n_eig == 1;
        for (std::size_t i = 0; ok && i < points.size(); ++i)
            if (i != 1) ok = same_bits(swept[i], rcwa::solve(s, points[i].wavelength, points[i].theta_deg, M, Polarization::TM)) &&
                             energy_defect(swept[i]) < 1e-10;
        CHECK(ok);
    }

    // (4) 振幅有限、效率却溢出（折射率取了极端的数值）：也按失败处理，不留下“一部分是 NaN/Inf”的结果。
    //     这几个结构都没有吸收，所以结果要么全部是 NaN，要么全部有限而且能量守恒（包括纯膜系：入射级自己出了问题时
    //     不能被当成“没有被激发的级次”还原成零）。
    for (Polarization pol : {Polarization::TE, Polarization::TM})
        for (double extreme : {1e300, 1e160, 1e-160, 1e-300})
            for (const rcwa::Stack& s : {make({grating}, extreme), make({}, extreme), make({rcwa::Layer::uniform(0.2, 1.7)}, extreme),
                                         make({rcwa::Layer::uniform(0.2, extreme)}, 1.5)})
                for (int m : {0, 3}) {
                    const rcwa::Result r = rcwa::solve(s, 0.9, 20.0, m, pol);
                    bool finite = r.R.size() == static_cast<std::size_t>(2 * m + 1);
                    for (std::size_t j = 0; finite && j < r.R.size(); ++j)
                        finite = std::isfinite(r.R[j].real()) && std::isfinite(r.R[j].imag()) && std::isfinite(r.T[j].real()) &&
                                 std::isfinite(r.T[j].imag()) && std::isfinite(r.DEr[j]) && std::isfinite(r.DEt[j]);
                    CHECK(all_nan(r) || (finite && energy_defect(r) < 1e-9));
                }
    //     齿的折射率取极端值：层矩阵里出现 Inf。同样按失败处理，不抛异常
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        const rcwa::Stack s = make({rcwa::Layer::grating(0.3, {0.0, 0.3, 1.0}, {1e200, 1.0})}, 1.5);
        CHECK(all_nan(rcwa::solve(s, 0.9, 20.0, 3, pol)));
    }
}

// ---------------------------------------------------------------------------------------------
// solve() 可重入（没有全局可变状态）：用户自己的线程并发调用、在 custom_eig 回调里再调用 solve()，
// 结果都与单独调用逐位相同。
static void test_reentrancy() {
    say("[可重入：用户线程并发调用、回调里嵌套调用]\n");
    struct Job {
        rcwa::Stack stack;
        double wavelength, theta;
        int M;
        Polarization pol;
        rcwa::Options opt;
    };
    std::vector<Job> jobs;
    {
        rcwa::Options plain, dense, lr, diag, par;
        dense.homogeneous_fast_path = false;
        dense.linear = rcwa::LinearSolver::QRHouseholder;
        lr.eig = EigMethod::LR;
        diag.diagnostics = true;
        par.parallel_layers = true;
        jobs.push_back({mdg_stack(), 1.053, 66.36, 10, Polarization::TE, plain});
        jobs.push_back({multilayer_stack(), 0.83, 27.0, 9, Polarization::TM, dense});
        jobs.push_back({trapezoid_stack(4), 1.053, 66.36, 8, Polarization::TE, par});
        jobs.push_back({silver_on_silver(), 0.8, 30.0, 10, Polarization::TM, lr});
        jobs.push_back({random_stack_55(), 1.46171, -74.5678, 6, Polarization::TM, diag});
        jobs.push_back({ridges_on_mirror(), kRidgesWavelength, kRidgesTheta, 30, Polarization::TE, par});   // 61 阶：内部的乘法也去用线程池
        jobs.push_back({grating_stack(true), 1.053, 42.0, 12, Polarization::TE, dense});
    }
    set_num_threads(1);
    std::vector<rcwa::Result> expect;
    for (const Job& j : jobs) expect.push_back(rcwa::solve(j.stack, j.wavelength, j.theta, j.M, j.pol, j.opt));
    for (int pool : {1, 3}) {                                   // 线程池的线程数；用户线程另有 4 个
        set_num_threads(pool);
        std::vector<rcwa::Result> got(3 * jobs.size());
        std::vector<std::thread> users;
        for (std::size_t t = 0; t < 4; ++t)
            users.emplace_back([&, t] {
                for (std::size_t i = t; i < got.size(); i += 4) {
                    const Job& j = jobs[i % jobs.size()];
                    got[i] = rcwa::solve(j.stack, j.wavelength, j.theta, j.M, j.pol, j.opt);
                }
            });
        for (std::thread& t : users) t.join();
        bool same = true;
        for (std::size_t i = 0; i < got.size(); ++i) same = same && same_bits(got[i], expect[i % jobs.size()]);
        CHECK(same);
    }

    // 回调里再完整地解另一个问题（嵌套调用）
    {
        set_num_threads(1);
        const rcwa::Stack outer = trapezoid_stack(4), inner = multilayer_stack();
        const rcwa::Result outer_expect = rcwa::solve(outer, 1.053, 66.36, 7, Polarization::TM);
        const rcwa::Result inner_expect = rcwa::solve(inner, 0.83, 27.0, 5, Polarization::TE);
        std::mutex guard;
        int inner_calls = 0, inner_same = 0;
        rcwa::Options nested;
        nested.parallel_layers = true;
        nested.custom_eig = [&](const Matrix& a) {
            const rcwa::Result r = rcwa::solve(inner, 0.83, 27.0, 5, Polarization::TE);
            {
                std::lock_guard<std::mutex> lock(guard);
                ++inner_calls;
                if (same_bits(r, inner_expect)) ++inner_same;
            }
            return eig_qr(a);
        };
        for (int pool : {1, 4}) {
            set_num_threads(pool);
            CHECK(same_bits(outer_expect, rcwa::solve(outer, 1.053, 66.36, 7, Polarization::TM, nested)));
        }
        set_num_threads(1);
        CHECK(inner_calls == 8 && inner_same == 8);
    }
}

// ---------------------------------------------------------------------------------------------
// 各个选项确实起作用（结果相同的选项，光比较结果看不出它有没有被忽略）
static void test_option_wiring() {
    say("[选项确实生效]\n");
    // 三种线性求解方法是三种不同的算法：结果都对（见前面的比较），但末位上互不相同
    {
        const rcwa::Stack s = multilayer_stack();
        const rcwa::Result lu = rcwa::solve(s, 0.83, 27.0, 12, Polarization::TE);
        rcwa::Options qr;
        qr.linear = rcwa::LinearSolver::QRHouseholder;
        const rcwa::Result rq = rcwa::solve(s, 0.83, 27.0, 12, Polarization::TE, qr);
        CHECK(!same_bits(lu, rq));
        CHECK_LE(efficiency_gap(lu, rq), 1e-11);
        rcwa::Options np;
        np.linear = rcwa::LinearSolver::LUNoPivot;
        const rcwa::Result rn = rcwa::solve(s, 0.83, 27.0, 12, Polarization::TE, np);
        CHECK(!same_bits(lu, rn) && !same_bits(rq, rn));
    }
    // homogeneous_fast_path = false：每一层都做本征分解，每个界面、每个传播块都按稠密矩阵参加递推，没有任何捷径。
    // L 层时稠密求解的次数：L+1 个界面各 3 次（W、V 各解一次，T+ 求逆一次），最上面的块之外的 2L 个块叠加时各 1 次，
    // 入射列 1 次，共 5L + 4 次；TM 每层组装层矩阵再多 2 次。
    for (Polarization pol : {Polarization::TE, Polarization::TM}) {
        struct Case {
            rcwa::Stack stack;
            double wavelength, theta;
        };
        const Case cases[] = {{mdg_stack(), 1.053, 66.36}, {multilayer_stack(), 0.83, 27.0}, {random_stack_55(), 1.46171, -74.5678}};
        for (const Case& c : cases) {
            rcwa::Options dense;
            dense.homogeneous_fast_path = false;
            const rcwa::Result r = rcwa::solve(c.stack, c.wavelength, c.theta, 5, pol, dense);
            const int layers = static_cast<int>(c.stack.layers.size());
            CHECK(r.n_eig == layers);
            CHECK(r.n_dense_solve == (pol == Polarization::TE ? 5 : 7) * layers + 4);
            CHECK(rcwa::solve(c.stack, c.wavelength, c.theta, 5, pol).n_dense_solve < r.n_dense_solve);
        }
    }
    // 计时的三项各自记到了该记的地方：有光栅层时三项都大于零；纯膜系全是逐元素运算，三项都严格为零
    {
        const rcwa::Timing g = rcwa::solve(mdg_stack(), 1.053, 66.36, 24, Polarization::TM).timing;
        CHECK(g.eig > 0.0 && g.linear > 0.0 && g.product > 0.0 && g.eig + g.linear + g.product <= g.total * (1.0 + 1e-9));
        rcwa::Stack films = mdg_stack();
        films.layers.erase(films.layers.begin());
        const rcwa::Result f = rcwa::solve(films, 1.053, 66.36, 24, Polarization::TM);
        CHECK(f.timing.eig == 0.0 && f.timing.linear == 0.0 && f.timing.product == 0.0 && f.timing.total > 0.0);
        CHECK(f.n_eig == 0 && f.n_dense_solve == 0);
        CHECK_LE(energy_defect(f), 1e-13);
    }
    // 几层的本征分解都没收敛时，eig_info 取（自上而下）第一个失败的层的 info，与各层完成的先后无关
    for (bool parallel : {false, true}) {
        const rcwa::Stack s = trapezoid_stack(4);            // 4 个光栅层，齿宽各不相同
        std::vector<Matrix> layer_matrices;
        for (int j = 0; j < 4; ++j)
            layer_matrices.push_back(rcwa::layer_matrix(s.layers[j], s.period, 1.053, s.nI, 66.36, 5, Polarization::TE));
        rcwa::Options opt;
        opt.parallel_layers = parallel;
        opt.custom_eig = [&layer_matrices](const Matrix& a) {
            EigResult r = eig_qr(a);
            for (int j = 1; j < 4; ++j)                       // 第 1、2、3 层（0 起）分别报告 info = 5、2、9
                if (std::memcmp(a.data(), layer_matrices[j].data(), sizeof(cplx) * a.rows() * a.cols()) == 0) {
                    r.info = j == 1 ? 5 : j == 2 ? 2 : 9;
                    r.vectors = Matrix();
                }
            return r;
        };
        set_num_threads(parallel ? 4 : 1);
        const rcwa::Result r = rcwa::solve(s, 1.053, 66.36, 5, Polarization::TE, opt);
        set_num_threads(1);
        CHECK(r.eig_info == 5 && all_nan(r) && r.n_eig == 4);
    }
    // parallel_layers、sweep 确实把工作分给了线程池里别的线程：用 custom_eig 记下是哪些线程在调用它
    {
        std::mutex guard;
        std::set<std::thread::id> callers;
        rcwa::Options opt;
        opt.custom_eig = [&](const Matrix& a) {
            {
                std::lock_guard<std::mutex> lock(guard);
                callers.insert(std::this_thread::get_id());
            }
            return eig_qr(a);
        };
        const rcwa::Stack s = trapezoid_stack(6);            // 6 个光栅层
        const std::vector<rcwa::Point> points(8, rcwa::Point{1.053, 66.36});
        set_num_threads(4);
        rcwa::solve(s, 1.053, 66.36, 6, Polarization::TE, opt);
        CHECK(callers.size() == 1 && *callers.begin() == std::this_thread::get_id());    // 没开 parallel_layers：都在调用线程上
        callers.clear();
        opt.parallel_layers = true;
        const rcwa::Result r = rcwa::solve(s, 1.053, 66.36, 6, Polarization::TE, opt);
        CHECK(callers.size() >= 2 && callers.size() <= 4);
        // 并行时计时仍然自洽：三项互不重叠，都包含在 total 里
        CHECK(r.timing.eig > 0.0 && r.timing.eig + r.timing.linear + r.timing.product <= r.timing.total * (1.0 + 1e-9));
        callers.clear();
        opt.parallel_layers = false;
        const std::vector<rcwa::Result> swept = rcwa::sweep(s, points, 6, Polarization::TE, opt);
        CHECK(callers.size() >= 2 && callers.size() <= 4);
        bool timing_ok = swept.size() == points.size();
        for (const rcwa::Result& p : swept)
            timing_ok = timing_ok && p.timing.total > 0.0 && p.timing.eig + p.timing.linear + p.timing.product <= p.timing.total * (1.0 + 1e-9);
        CHECK(timing_ok);
        set_num_threads(1);                                   // 线程数为 1：全部在调用线程上
        callers.clear();
        opt.parallel_layers = true;
        rcwa::sweep(s, points, 6, Polarization::TE, opt);
        CHECK(callers.size() == 1 && *callers.begin() == std::this_thread::get_id());
    }
    // 并行区里抛出的异常原样传给调用者，之后线程池照常可用
    {
        const rcwa::Stack s = trapezoid_stack(4);
        set_num_threads(4);
        std::vector<rcwa::Point> points(6, rcwa::Point{1.053, 66.36});
        points[4].wavelength = -1.0;                          // 扫描里有一个点的参数非法
        CHECK_THROWS(rcwa::sweep(s, points, 5), std::invalid_argument);
        rcwa::Options bad;                                    // custom_eig 自己抛的异常不会被当成数值失败吞掉
        bad.parallel_layers = true;
        bad.custom_eig = [](const Matrix& a) -> EigResult {
            if (a.rows() > 0) throw std::runtime_error("backend lost");
            return EigResult();
        };
        CHECK_THROWS(rcwa::solve(s, 1.053, 66.36, 5, Polarization::TE, bad), std::runtime_error);
        bad.parallel_layers = false;
        CHECK_THROWS(rcwa::solve(s, 1.053, 66.36, 5, Polarization::TE, bad), std::runtime_error);
        points[4].wavelength = 1.06;
        const std::vector<rcwa::Result> swept = rcwa::sweep(s, points, 5);
        set_num_threads(1);
        CHECK(swept.size() == 6 && same_bits(swept[0], rcwa::solve(s, 1.053, 66.36, 5)) &&
              same_bits(swept[4], rcwa::solve(s, 1.06, 66.36, 5)));
    }
}

// 接近掠入射：裸界面对 Fresnel 公式。入射能流要用入射级自己的 kz 来归一：若改用 nI·cos θ，两者相差约 ε/cos²θ，
// θ = 89.9999° 时 R + T 会偏离 1 约 1e-5，再靠近 90° 就完全不对了。
static void test_grazing_incidence() {
    say("[接近掠入射的裸界面]\n");
    rcwa::Stack bare;
    bare.nI = 1.0;
    bare.nII = 1.5;
    bare.period = 1.0;
    double worst_energy = 0.0, worst_r = 0.0;
    for (double theta_deg : {89.9, 89.99, 89.9999, 89.999999}) {
        const double c1 = std::sin((90.0 - theta_deg) * kPi / 180.0);          // cos θ，这样算在 90° 附近不丢有效数字
        const double k2 = std::sqrt((1.5 - 1.0) * (1.5 + 1.0) + c1 * c1);      // nII·cos θ_II
        const double rs = (c1 - k2) / (c1 + k2);
        const double rp = (1.5 * 1.5 * c1 - k2) / (1.5 * 1.5 * c1 + k2);
        for (Polarization pol : {Polarization::TE, Polarization::TM}) {
            const rcwa::Result r = rcwa::solve(bare, 0.9, theta_deg, 2, pol);
            const double exact = pol == Polarization::TE ? rs * rs : rp * rp;
            const double err = std::abs(r.DEr[2] - exact);
            worst_energy = (std::max)(worst_energy, energy_defect(r));
            worst_r = (std::max)(worst_r, err);
            CHECK_LE(energy_defect(r), 1e-12);
            // 场里用的 kz0 = sqrt(nI² − kx0²) 本身在掠入射时有 ε/cos²θ 的相对误差，反射率因此差 4·kz0·(ε/cos²θ) 的量级
            CHECK_LE(err, theta_deg > 89.99999 ? 1e-7 : 1e-9);
        }
    }
    say("  θ = 89.9° … 89.999999°：|R + T − 1| 最大 %.1e，反射率与 Fresnel 公式最多相差 %.1e\n", worst_energy, worst_r);
}

int main() {
    test_slab();
    test_reference_tables();
    test_energy();
    test_option_equivalence();
    test_lr_on_all_layers();
    test_sweep_and_threads();
    test_parallel_layers_and_custom_eig();
    test_single_precision_eig();
    test_layer_matrix();
    test_bookkeeping();
    test_failures();
    test_arguments();
    test_random_stacks();
    test_invariances();
    test_order_convention();
    test_rayleigh_anomalies();
    test_reentrancy();
    test_option_wiring();
    test_grazing_incidence();
    say("通过 %d / 共 %d\n", g_pass, g_total);
    return g_pass == g_total ? 0 : 1;
}
