// stablercwa/stablercwa.hpp —— 总头文件：一次包含整个模块
//
//   #include "stablercwa/stablercwa.hpp"
//
//   using namespace stablercwa;
//   Matrix x = solve(a, b);                         // LU 分解解方程（lu.hpp）
//   Matrix y = solve_qr(a, b);                      // Householder QR 解方程（qr.hpp）
//   EigResult e = eig(a, EigMethod::QR);            // 本征分解（eig.hpp；LR 迭代传 EigMethod::LR）
//   double kappa = cond2(e.vectors);                // 2-范数条件数（svd.hpp）
//   rcwa::Result r = rcwa::solve(stack, 1.053, 66.36, 100);   // 一维 RCWA（rcwa1d.hpp）
//   rcwa::Result2D r2 = rcwa::solve2d(stack2d, 1.053, 42.0, -25.0, 4, 4, rcwa::jones_from_angles_2d(90.0));   // 二维（交叉光栅）RCWA（rcwa2d.hpp）
//
// 全部是头文件、全部 inline，只依赖 C++17 标准库；需要多线程时先调用 set_num_threads(n)，默认串行。
// 可选的 GPU 后端（gpu/ 目录）要另外的开发库，不在这里包含，用到时单独 #include。
// 近场（rcwa1d_fields.hpp、rcwa1d_conical_fields.hpp、rcwa2d_fields.hpp）与场协议（rcwa_fields_json.hpp）同样不在这里包含，用到时单独 #include。
#pragma once

#include "parallel.hpp"    // 线程池、parallel_for
#include "matrix.hpp"      // Matrix、范数、乘法
#include "eig_types.hpp"   // EigMethod / EigOptions / EigResult / EigProgress
#include "lu.hpp"          // LU 分解、解方程、求逆
#include "qr.hpp"          // Householder QR 分解、解方程
#include "svd.hpp"         // 奇异值、2-范数条件数
#include "eig_qr.hpp"      // eig_qr()：CORTH + COMQR2
#include "eig_lr.hpp"      // eig_lr()：COMHES + COMLR2
#include "eig.hpp"         // 统一入口 eig()、残差与正交性度量
#include "rcwa1d.hpp"      // 一维 RCWA
#include "rcwa1d_conical.hpp"   // 一维 RCWA 的圆锥入射：各级的 2×2 Jones 矩阵
#include "rcwa2d_types.hpp"      // 二维 RCWA 的公共类型：Stack2D、Layer2D、Options2D、Result2D
#include "rcwa2d_operators.hpp"  // 二维 RCWA 的 Fourier 算子（detail2d::layer_operators）
#include "rcwa2d.hpp"            // 二维 RCWA 的求解核心 solve2d（交叉光栅）
#include "rcwa_json.hpp"        // RCWA 的 JSON 入口 solve_json()（连带 json.hpp；含一维 planar、conical 与二维 periodic2d）
