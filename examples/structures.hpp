// examples/structures.hpp —— 示例与数据生成程序共用的几个结构
//
// 长度单位一律是微米。膜系自上而下排列：光栅层在最上面，下面依次是各层介质膜，最后是衬底。
#pragma once

#include <vector>

#include "stablercwa/rcwa1d.hpp"

namespace examples {

using stablercwa::cplx;
namespace rcwa = stablercwa::rcwa;

// 单周期多层介质膜光栅的参数：每个周期一个矩形齿，下面是 20 层介质膜
struct Mdg {
    double wavelength = 1.053;      // 波长
    double theta_deg = 66.36;       // 入射角
    double period = 0.574;          // 周期
    double duty = 0.3;              // 齿宽 / 周期
    double depth = 0.38;            // 刻深
    cplx n_tooth{1.46, 0.0};        // 齿的折射率；虚部大于 0 表示有吸收
    double n_air = 1.0, n_sub = 1.46;
    double n_low = 1.46, n_high = 1.96;
    double d_high = 0.115306, d_2low = 0.30958, d_3low = 0.46438;
};

// 光栅层下面的 20 层介质膜：3L · H · (2L · H)^9
inline void add_film_stack(std::vector<rcwa::Layer>& layers, const Mdg& c) {
    layers.push_back(rcwa::Layer::uniform(c.d_3low, c.n_low));
    layers.push_back(rcwa::Layer::uniform(c.d_high, c.n_high));
    for (int k = 0; k < 9; ++k) {
        layers.push_back(rcwa::Layer::uniform(c.d_2low, c.n_low));
        layers.push_back(rcwa::Layer::uniform(c.d_high, c.n_high));
    }
}

// 单周期光栅：一个周期里一个矩形齿，下面是 20 层介质膜。
inline rcwa::Stack single_period(const Mdg& c = {}) {
    rcwa::Stack stack;
    stack.nI = c.n_air;
    stack.nII = c.n_sub;
    stack.period = c.period;
    stack.layers.push_back(rcwa::Layer::grating(c.depth, {0.0, c.period * c.duty, c.period},
                                                {c.n_tooth, cplx(c.n_air)}));
    add_film_stack(stack.layers, c);
    return stack;
}

// 侧壁倾斜的梯形齿：把刻深切成 slices 片，每片是一个齿宽不同的光栅层（用于演示多个光栅层并行求本征值）。
inline rcwa::Stack trapezoid(int slices, const Mdg& c = {}, double top_duty = 0.2, double bottom_duty = 0.45) {
    rcwa::Stack stack;
    stack.nI = c.n_air;
    stack.nII = c.n_sub;
    stack.period = c.period;
    for (int s = 0; s < slices; ++s) {
        const double f = top_duty + (bottom_duty - top_duty) * (s + 0.5) / slices;
        const double w = c.period * f;
        const double x0 = 0.5 * (c.period - w);
        stack.layers.push_back(rcwa::Layer::grating(c.depth / slices, {0.0, x0, x0 + w, c.period},
                                                    {cplx(c.n_air), c.n_tooth, cplx(c.n_air)}));
    }
    add_film_stack(stack.layers, c);
    return stack;
}

// 四脊大周期光栅的参数：周期是波长的七倍左右，一个周期里四条宽度、折射率各不相同的脊，其余是空气；
// 下面是 H、L 交替的介质高反膜。级次很密，用来把截断阶数取大（基准测试、高阶稳定性的数据）。
// 与 tests/test_rcwa1d.cpp 里的 ridges_on_mirror() 是同一个结构。
struct Ridges {
    double wavelength = 0.94;       // 波长
    double theta_deg = 22.0;        // 入射角
    double period = 6.3;            // 周期
    double depth = 0.32;            // 刻深
    std::vector<double> edges{0.0, 0.20, 1.05, 1.60, 2.05, 2.90, 4.25, 5.10, 5.45, 6.30};   // 各段的分界，首末相差一个周期
    std::vector<cplx> values{1.0, 1.52, 1.0, 2.10, 1.0, 1.75, 1.0, 1.38, 1.0};             // 各段的折射率，1.0 是空气
    double n_air = 1.0, n_sub = 1.50;
    double n_high = 2.25, n_low = 1.48;
    double d_high = 0.105, d_low = 0.160;
    int order = -2;                 // 示例程序报告这一级的反射效率（最强的衍射级：TE 约 15 %，TM 约 12 %）
};

// 四脊光栅层下面 films 层介质膜，自上而下 H · L · H · L …；films = 20 就是 ridges_on_mirror。
inline rcwa::Stack ridges_on_films(const Ridges& r, int films) {
    rcwa::Stack stack;
    stack.nI = r.n_air;
    stack.nII = r.n_sub;
    stack.period = r.period;
    stack.layers.push_back(rcwa::Layer::grating(r.depth, r.edges, r.values));
    for (int k = 0; k < films; ++k)
        stack.layers.push_back(k % 2 == 0 ? rcwa::Layer::uniform(r.d_high, r.n_high) : rcwa::Layer::uniform(r.d_low, r.n_low));
    return stack;
}

// 四脊大周期光栅 + 20 层介质高反膜 (H · L)^10
inline rcwa::Stack ridges_on_mirror(const Ridges& r = {}) { return ridges_on_films(r, 20); }

}  // namespace examples
