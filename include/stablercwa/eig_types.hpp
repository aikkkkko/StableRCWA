// stablercwa/eig_types.hpp —— 本征求解的公共类型（eig_qr.hpp 与 eig_lr.hpp 共用）
#pragma once

#include <functional>

#include "matrix.hpp"

namespace stablercwa {

enum class EigMethod {
    QR,   // 酉变换：Householder 反射化 Hessenberg 形 + 带位移的 QR 迭代（EISPACK CORTH + COMQR2）
    LR,   // 消元变换：带行交换的消元化 Hessenberg 形 + 带位移的 LR 迭代（EISPACK COMHES + COMLR2）
};

// 迭代过程中回调给观察者的快照。引用只在回调期间有效，需要保留时请自行拷贝。
struct EigProgress {
    const char* stage;    // "hessenberg"：刚化成 Hessenberg 形；"iteration"：完成一次迭代；"done"：全部收敛
    int sweep;            // 已完成的迭代次数
    int active;           // 当前正在求的本征值下标 en（其下方的行已收敛）；全部收敛后为 -1
    cplx shift;           // 累计位移 t：迭代中存放的是 A' - t·I，对角元加上 t 才是相似于 A 的矩阵
    const Matrix& H;      // 当前矩阵（对角元未加回位移）
    const Matrix& Z;      // 到目前为止累积的相似变换，Z^{-1} A Z = H + t·I
};

struct EigOptions {
    bool vectors = true;           // 是否计算本征向量
    bool keep_transform = false;   // 是否在结果里保留累积变换 Z 与三角矩阵 T
    double eps = kEps;             // 次对角元“可忽略”的相对阈值
    int max_iter = 30;             // 每个本征值允许的迭代次数；第 10、20 次用例外位移
    std::function<void(const EigProgress&)> observer;   // 可选；只在设置了的时候调用
};

struct EigResult {
    Vector values;     // 本征值 λ_j
    Matrix vectors;    // 第 j 列是 λ_j 对应的右本征向量，已归一到 2-范数为 1；options.vectors=false 时为空
    Matrix Z;          // 累积相似变换（含化 Hessenberg 的变换），Z^{-1} A Z = T；仅 keep_transform=true 时填写
    Matrix T;          // 迭代结束时的上三角矩阵；仅 keep_transform=true 时填写
    int sweeps = 0;    // 迭代总次数
    int info = 0;      // 0 表示全部收敛；k>0 表示第 k 个本征值（1 起）在 max_iter 次内未收敛，下标 >= k 的本征值有效
};

}  // namespace stablercwa
