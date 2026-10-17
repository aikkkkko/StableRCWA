// examples/eig_minimal.cpp —— 最小的用法：包含头文件，填一个复矩阵，调用本征求解
//
//   c++ -std=c++17 -O2 -I include examples/eig_minimal.cpp -o eig_minimal && ./eig_minimal
#include "stablercwa/eig.hpp"
#include "stablercwa/svd.hpp"
#include "say.hpp"

using namespace stablercwa;
using examples::say;                                 // 用法同 std::printf，格式串里可以有中文（缘由见 say.hpp）

int main() {
    // 一个 4×4 的一般复矩阵（非厄米）。Matrix 按行书写初始化，A(i, j) 读写元素。
    Matrix A{{cplx(2.0, 0.5), 1.0, 0.0, cplx(0.0, 1.0)},
             {1.0, cplx(-1.0, 0.2), 0.5, 0.0},
             {0.0, 0.5, cplx(3.0, -0.3), 1.0},
             {cplx(0.0, -1.0), 0.0, 1.0, cplx(0.5, 0.1)}};

    EigOptions opt;
    opt.keep_transform = true;                       // 顺便取回累积相似变换 Z，用来看它的条件数

    for (EigMethod method : {EigMethod::QR, EigMethod::LR}) {
        const EigResult r = eig(A, method, opt);
        say("%s 迭代：%d 次，info = %d，本征对残差 = %.2e，κ(Z) = %.4g\n",
            method == EigMethod::QR ? "QR" : "LR", r.sweeps, r.info, eig_residual(A, r), cond2(r.Z));
        for (std::size_t j = 0; j < r.values.size(); ++j)
            say("    λ[%zu] = %+.12f %+.12fi\n", j, r.values[j].real(), r.values[j].imag());
    }
    return 0;
}
