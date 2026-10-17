# -*- coding: utf-8 -*-
"""tests/test_rcwa1d.cpp 里各张参考效率表对应的结构与入射条件（两个生成/核对脚本共用）

本文件只描述结构，不做任何计算：
  · make_rcwa1d_tables.py      用 gui/rcwa1d.py（Python 写的 S 矩阵求解器）算各表，写出 C++ 表或与 C++ 文件里的表比较；
  · crosscheck_rcwa1d_tables.py 用另写的整体矩阵法（不调用 gui/ 的任何代码）重算，与 C++ 文件里的表比较。

这里的每个结构与 test_rcwa1d.cpp 里的同名函数逐项相同（数值、各层次序、边界坐标的算式都照抄），
抄错了 C++ 测试本身会失败（表与 C++ 求解结果逐级比较，容差 1e-9）。
长度单位微米；膜系自上而下排列；级次约定与 C++ 相同：kx_m = k0·(nI·sin θ − m·λ/Λ)。
"""
import re
from collections import namedtuple

# 均匀层：n 是折射率，edges、values 为 None；光栅层：n 为 None，edges 是 K+1 个分界坐标（首末相差一个周期），
# values 是 K 段各自的折射率。
Layer = namedtuple("Layer", "d n edges values")
Stack = namedtuple("Stack", "nI nII period layers")


def uniform(d, n):
    return Layer(float(d), complex(n), None, None)


def grating(d, edges, values):
    return Layer(float(d), None, [float(x) for x in edges], [complex(v) for v in values])


# ---------------------------------------------------------------------------------------------
# 结构（与 test_rcwa1d.cpp 同名函数一一对应）

def grating_stack(lossy=False):
    """单层矩形光栅：脊中心在 x = 0，边界从负坐标开始。"""
    p, f = 0.5747, 0.30
    return Stack(1.0, 1.45, p, [grating(0.38, [-f * p / 2, f * p / 2, p - f * p / 2],
                                        [complex(1.8, 0.13) if lossy else 1.45, 1.0])])


def mdg_stack():
    """多层介质膜光栅：光栅下面 9 组 H/2L 膜再加一层 H，共 20 层。"""
    s = grating_stack()
    layers = list(s.layers)
    for _ in range(9):
        layers.append(uniform(1.053 / (4 * 2.05), 2.05))
        layers.append(uniform(2 * 1.053 / (4 * 1.45), 1.45))
    layers.append(uniform(1.053 / (4 * 2.05), 2.05))
    return s._replace(layers=layers)


def multilayer_stack():
    """两个不同的光栅层夹一层吸收膜；入射介质 n = 1.2，出射介质有吸收。"""
    return Stack(1.2, complex(1.5, 0.03), 0.9, [
        grating(0.23, [0.0, 0.21, 0.47, 0.9], [1.7, 1.0, complex(2.1, 0.05)]),
        uniform(0.19, complex(1.4, 0.02)),
        grating(0.31, [-0.12, 0.17, 0.78], [1.0, 1.9]),
    ])


SILVER = complex(0.079095, 5.42875)     # 银在 λ = 0.8 附近的折射率


# 四脊大周期光栅：周期 6.3，一个周期里四条宽度、折射率各不相同的脊，其余是空气，刻深 0.32。
RIDGE_PERIOD = 6.3
RIDGE_EDGES = [0.0, 0.20, 1.05, 1.60, 2.05, 2.90, 4.25, 5.10, 5.45, 6.30]
RIDGE_VALUES = [1.0, 1.52, 1.0, 2.10, 1.0, 1.75, 1.0, 1.38, 1.0]


def ridges_layer():
    return grating(0.32, RIDGE_EDGES, RIDGE_VALUES)


def ridges_on_silver():
    """四脊大周期光栅，下面是吸收膜、介质间隔层和 40 nm 银膜，玻璃衬底。"""
    return Stack(1.0, 1.50, RIDGE_PERIOD, [
        ridges_layer(),
        uniform(0.12, complex(2.0, 0.08)),
        uniform(0.20, 1.45),
        uniform(0.040, SILVER),
    ])


def silver_on_glass():
    return Stack(1.0, 1.5, 0.6, [grating(0.08, [0.0, 0.3, 0.6], [SILVER, 1.0])])


def silver_on_silver():
    return Stack(1.0, SILVER, 0.6, [grating(0.10, [-0.1, 0.15, 0.5], [SILVER, 1.0])])


# ---------------------------------------------------------------------------------------------
# 随机膜系（固定种子生成一次后照原样写进 C++，这里逐项照抄）

def random_stack_3():
    return Stack(1.5, complex(1.0367, 3.9027), 3.00592, [
        uniform(0.06429, complex(3.0646, 0.070776)),
        grating(0.48278, [1.2216037660790897, 1.5834740458158494, 3.0604920858391766, 3.367905016258635, 3.7419360997352666, 4.22752376607909],
                [2.5123, 1.0, complex(2.0595, 0.045929), 2.261, 1.0]),
        uniform(0.57245, complex(0.0672, 4.5789)),
    ])


def random_stack_12():
    return Stack(2.0, 3.2766, 0.52506, [
        grating(0.28899, [0.0, 0.21470852461949538, 0.291635488282107, 0.52506],
                [1.5952, 2.6941, 1.0]),
        uniform(0.15519, 1.7212),
        grating(0.59828, [0.0, 0.1348671539039357, 0.38103017701846575, 0.52506],
                [2.3259, 2.9847, 1.0]),
        grating(0.09588, [0.23793965557385166, 0.30118145894615533, 0.599632279980103, 0.7629996555738516],
                [1.0, 2.8048, 1.8829]),
        grating(0.46157, [0.0, 0.04675228866800866, 0.2649523734345849, 0.37714726161756695, 0.52506],
                [1.0, 1.0, 3.0051, 2.0576]),
    ])


def random_stack_19():
    return Stack(1.0, 1.3489, 1.34061, [
        grating(0.0, [0.5156650515534138, 0.8318325263653672, 1.8562750515534139],
                [1.0, complex(3.3889, 0.000229)]),
        grating(0.5349, [-0.5658312694886117, -0.4696050566510656, 0.05782139376129273, 0.7747787305113883],
                [1.0, complex(1.2847, 4.8405), 2.7505]),
        grating(0.79773, [-0.10816328923120395, -0.009302224283890199, 0.6818917680133255, 0.9252570590935646, 1.0791902226686423, 1.2324467107687962],
                [1.0, complex(2.5548, 0.000201), complex(2.5451, 0.080789), complex(0.7868, 2.4888), 1.0]),
        uniform(0.1625, complex(1.21, 0.00255)),
        grating(0.39106, [0.6239481182065285, 0.685927668979927, 1.2835994804498256, 1.5469979778127105, 1.6730792318828505, 1.9645581182065286],
                [1.0, complex(2.3476, 0.000169), 1.0, 1.8527, 1.0]),
    ])


def random_stack_41():
    return Stack(1.0, 1.1801, 1.13519, [
        uniform(0.33267, complex(1.6936, 0.006139)),
        grating(0.43378, [0.0, 0.16492250206616804, 0.39486054905994067, 0.9978827011848254, 1.0570393072884265, 1.13519],
                [complex(2.2508, 0.010382), 2.5607, 1.4733, 1.0, 1.0]),
        grating(0.29255, [0.0, 0.5632513103596241, 1.13519],
                [complex(1.4522, 0.00052), complex(3.2144, 0.001193)]),
        grating(0.64326, [0.0, 0.10309602954957174, 0.41613457044054325, 0.6386076340808057, 0.7604868811390587, 1.13519],
                [1.2506, complex(3.0035, 0.006697), 3.3129, 2.2244, complex(2.3656, 0.000767)]),
        grating(0.0, [0.28439787828427904, 0.5892782848773083, 1.0542527740472492, 1.419587878284279],
                [complex(1.604, 5.5012), complex(2.8767, 0.066147), 1.0]),
        grating(0.41493, [0.27259822061918854, 0.7395007185144321, 0.8643499284689117, 1.1409946360379553, 1.4077882206191885],
                [complex(0.0642, 2.8712), 3.4365, complex(1.4507, 3.2998), 1.5234]),
    ])


def random_stack_45():
    return Stack(2.0, complex(0.1468, 5.3027), 1.21537, [
        uniform(0.49364, 1.6185),
        grating(0.03897, [0.06487027800333718, 0.11559842057396437, 0.1923385450658246, 0.2413254782316344, 1.2802402780033373],
                [complex(3.1367, 0.141471), 1.3106, 1.8873, complex(1.8873, 5.7138)]),
        uniform(0.36862, complex(1.5923, 0.001829)),
    ])


def random_stack_55():
    return Stack(1.0, complex(1.9323, 0.027469), 3.56473, [
        uniform(0.09677, 1.9021),
        uniform(0.5005, 1.459),
        grating(0.7394, [1.1965621850955124, 1.674869829777083, 1.9360266248213347, 3.2059797244276256, 3.6518692009143945, 4.761292185095512],
                [complex(3.1595, 0.084433), 1.0, complex(3.43, 0.021694), 1.0, complex(1.8723, 1.8283)]),
        uniform(0.31727, complex(0.4842, 2.9218)),
        grating(0.65018, [-0.06740477119207551, 0.26711249145034227, 0.9542194343337393, 2.3133782387418105, 3.1524893366562927, 3.4973252288079246],
                [complex(2.8621, 0.03192), complex(1.2083, 5.1586), 1.0, complex(0.0871, 5.3898), complex(2.963, 0.159421)]),
        uniform(0.14492, 1.6278),
    ])


def random_stack_62():
    return Stack(1.33, 3.433, 1.67758, [
        uniform(0.76402, 1.569),
        grating(0.74761, [0.0, 0.5517663329263505, 1.0378497820342552, 1.1806586012144589, 1.5058924320839342, 1.67758],
                [1.0, 2.4293, 1.1122, 1.0, 1.0]),
        grating(0.67934, [0.0, 0.8817153077149015, 1.6053185351937984, 1.67758],
                [2.4334, 1.069, 2.4808]),
        grating(0.70328, [0.7952418057740204, 1.169295627613309, 1.5435843940554042, 2.254297633825864, 2.4728218057740206],
                [1.449, 1.0, 3.2397, 3.1695]),
        uniform(0.0, 2.6041),
        uniform(0.02951, 2.7073),
    ])


def random_stack_71():
    return Stack(2.0, 1.4127, 1.26482, [
        grating(0.23688, [-0.48966611139860494, -0.1948413813954841, 0.03509971762771468, 0.6509504632831165, 0.7751538886013951],
                [1.726, complex(1.0404, 5.2965), complex(2.5719, 0.200919), 1.0]),
    ])


def random_stack_76():
    return Stack(1.5, 2.6042, 0.50524, [
        uniform(0.71632, 1.47),
        grating(0.26979, [0.09359053359145031, 0.2826544079523479, 0.5988305335914503],
                [1.0, 1.7448]),
        uniform(0.38688, 1.7637),
        uniform(0.1942, 3.2247),
        uniform(0.10354, 1.9487),
        grating(0.23767, [0.0, 0.2181867785222636, 0.2924767618694831, 0.50524],
                [2.2155, 1.0, 3.3026]),
    ])


# ---------------------------------------------------------------------------------------------
# 参考表。kind = "orders"：C++ 里的 OrderRef 表，只列效率不为零的级次 {级次, 反射效率, 透射效率}；
#         kind = "full"：C++ 里的一对 double 数组 <name>Der、<name>Det，依次是第 −M … +M 级。

Case = namedtuple("Case", "name kind stack wavelength theta M pol title")

CASES = [
    Case("kGratingTE", "orders", grating_stack, 1.053, 66.36, 12, "TE",
         "单层无吸收光栅，λ = 1.053，θ = 66.36°，M = 12，TE"),
    Case("kGratingTM", "orders", grating_stack, 1.053, 66.36, 12, "TM",
         "单层无吸收光栅，λ = 1.053，θ = 66.36°，M = 12，TM"),
    Case("kLossyTE", "orders", lambda: grating_stack(True), 1.053, 42.0, 14, "TE",
         "单层吸收光栅（齿 n = 1.8 + 0.13i），λ = 1.053，θ = 42°，M = 14，TE"),
    Case("kLossyTM", "orders", lambda: grating_stack(True), 1.053, 42.0, 14, "TM",
         "单层吸收光栅（齿 n = 1.8 + 0.13i），λ = 1.053，θ = 42°，M = 14，TM"),
    Case("kMdgTE", "orders", mdg_stack, 1.053, 66.36, 20, "TE",
         "光栅 + 19 层介质膜（共 20 层），λ = 1.053，θ = 66.36°，M = 20，TE"),
    Case("kMdgTM", "orders", mdg_stack, 1.053, 66.36, 20, "TM",
         "光栅 + 19 层介质膜（共 20 层），λ = 1.053，θ = 66.36°，M = 20，TM"),
    Case("kMultilayerTE", "orders", multilayer_stack, 0.83, 27.0, 12, "TE",
         "两个光栅层夹一层吸收膜，入射介质 n = 1.2，出射介质 n = 1.5 + 0.03i，λ = 0.83，θ = 27°，M = 12，TE"),
    Case("kMultilayerTM", "orders", multilayer_stack, 0.83, 27.0, 12, "TM",
         "两个光栅层夹一层吸收膜，入射介质 n = 1.2，出射介质 n = 1.5 + 0.03i，λ = 0.83，θ = 27°，M = 12，TM"),
    Case("kRidgesSilver50TE", "orders", ridges_on_silver, 0.8, 23.0, 50, "TE",
         "四脊大周期光栅（周期 6.3），下接吸收膜、介质间隔层和 40 nm 银膜，λ = 0.8，θ = 23°，M = 50，TE"),
    Case("kRidgesSilver50TM", "orders", ridges_on_silver, 0.8, 23.0, 50, "TM",
         "四脊大周期光栅（周期 6.3），下接吸收膜、介质间隔层和 40 nm 银膜，λ = 0.8，θ = 23°，M = 50，TM"),
    Case("kSilverGlassTE", "orders", silver_on_glass, 0.8, 30.0, 15, "TE",
         "银光栅（脊 n = 0.079095 + 5.42875i，槽为空气，厚 0.08）在玻璃（n = 1.5）上，周期 0.6，λ = 0.8，θ = 30°，M = 15，TE"),
    Case("kSilverGlassTM", "orders", silver_on_glass, 0.8, 30.0, 15, "TM",
         "银光栅（脊 n = 0.079095 + 5.42875i，槽为空气，厚 0.08）在玻璃（n = 1.5）上，周期 0.6，λ = 0.8，θ = 30°，M = 15，TM"),
    Case("kSilverBulkTE", "orders", silver_on_silver, 0.8, 30.0, 15, "TE",
         "银光栅（厚 0.10，边界 {−0.1, 0.15, 0.5}）在银衬底上，周期 0.6，λ = 0.8，θ = 30°，M = 15，TE"),
    Case("kSilverBulkTM", "orders", silver_on_silver, 0.8, 30.0, 15, "TM",
         "银光栅（厚 0.10，边界 {−0.1, 0.15, 0.5}）在银衬底上，周期 0.6，λ = 0.8，θ = 30°，M = 15，TM"),
    Case("kRandom3", "full", random_stack_3, 1.53394, 37.5193, 6, "TM", "随机膜系 #3"),
    Case("kRandom12", "full", random_stack_12, 0.69177, 54.4633, 4, "TE", "随机膜系 #12"),
    Case("kRandom19", "full", random_stack_19, 0.57215, 37.1167, 3, "TM", "随机膜系 #19"),
    Case("kRandom41", "full", random_stack_41, 0.99384, 65.3981, 8, "TE", "随机膜系 #41"),
    Case("kRandom45", "full", random_stack_45, 0.99554, 11.0152, 3, "TE", "随机膜系 #45"),
    Case("kRandom55", "full", random_stack_55, 1.46171, -74.5678, 6, "TM", "随机膜系 #55"),
    Case("kRandom62", "full", random_stack_62, 0.93068, 74.7309, 6, "TM", "随机膜系 #62"),
    Case("kRandom71", "full", random_stack_71, 0.77593, -64.3562, 3, "TM", "随机膜系 #71"),
    Case("kRandom76", "full", random_stack_76, 0.47496, 24.135, 5, "TE", "随机膜系 #76"),
]


# ---------------------------------------------------------------------------------------------
# 读 C++ 文件里的表

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def read_cpp_tables(path):
    """返回 {表名: (DEr, DEt)}，两者都是第 −M … +M 级的列表（OrderRef 表里没列出的级次补零）。

    C++ 文件里每张表都必须在 CASES 里有对应的一项，反过来也一样；任何一张表、任何一行没有按预期的格式读出来，
    都抛 ValueError，不会悄悄跳过。
    """
    with open(path, encoding="utf-8") as f:
        src = f.read()
    by_name = {c.name: c for c in CASES}
    out = {}
    # 先数出所有声明（不管写法），读到的表必须一张不少
    declared_orders = re.findall(r"\bOrderRef\s+(k\w+)\s*\[\s*\]", src)
    declared_full = re.findall(r"\bdouble\s+(k\w+(?:Der|Det))\s*\[\s*\]", src)
    parsed_orders, parsed_full = [], []
    for m in re.finditer(r"static const OrderRef (k\w+)\[\] = \{(.*?)\n\};", src, re.S):
        name, body = m.group(1), m.group(2)
        parsed_orders.append(name)
        if name not in by_name or by_name[name].kind != "orders":
            raise ValueError(f"{path}: 表 {name} 不在 CASES 里")
        M = by_name[name].M
        der, det = [0.0] * (2 * M + 1), [0.0] * (2 * M + 1)
        rows = list(re.finditer(rf"\{{\s*([-+]?\d+)\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\}}", body))
        if len(rows) != body.count("{"):
            raise ValueError(f"{path}: 表 {name} 有 {body.count('{')} 行，只读出了 {len(rows)} 行")
        seen = set()
        for row in rows:
            j = int(row.group(1)) + M
            if not 0 <= j <= 2 * M or j in seen:
                raise ValueError(f"{path}: 表 {name} 的级次 {row.group(1)} 超出 ±{M} 或者重复")
            seen.add(j)
            der[j], det[j] = float(row.group(2)), float(row.group(3))
        out[name] = (der, det)
    for m in re.finditer(r"static const double (k\w+)(Der|Det)\[\] = \{[^\n]*\n(.*?)\n\};", src, re.S):
        name, side, body = m.group(1), m.group(2), m.group(3)
        parsed_full.append(name + side)
        if name not in by_name or by_name[name].kind != "full":
            raise ValueError(f"{path}: 表 {name}{side} 不在 CASES 里")
        items = [x.strip() for x in body.split(",") if x.strip()]
        if not all(re.fullmatch(_NUMBER, x) for x in items):
            raise ValueError(f"{path}: 表 {name}{side} 里有读不成数的项")
        values = [float(x) for x in items]
        if len(values) != 2 * by_name[name].M + 1:
            raise ValueError(f"{path}: 表 {name}{side} 有 {len(values)} 个数，应为 {2 * by_name[name].M + 1}")
        der, det = out.get(name, (None, None))
        out[name] = (values, det) if side == "Der" else (der, values)
    if sorted(declared_orders) != sorted(parsed_orders) or sorted(declared_full) != sorted(parsed_full):
        raise ValueError(f"{path}: 有表的写法与预期不同，没有读出来："
                         f"{sorted(set(declared_orders + declared_full) - set(parsed_orders + parsed_full))}")
    missing = sorted(set(by_name) - set(out))
    if missing:
        raise ValueError(f"{path}: 找不到这些表：{', '.join(missing)}")
    for name, (der, det) in out.items():
        if der is None or det is None:
            raise ValueError(f"{path}: 表 {name} 缺少 Der 或 Det")
    return out


def max_deviation(der, det, ref):
    """逐级比较两组效率，返回 max |Δ|（含 NaN 时返回 inf）。"""
    worst = 0.0
    for a, b in zip(list(der) + list(det), list(ref[0]) + list(ref[1])):
        e = abs(float(a) - float(b))
        if not e <= worst:
            worst = e if e == e else float("inf")
    return worst
