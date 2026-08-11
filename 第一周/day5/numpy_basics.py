# -*- coding: utf-8 -*-
r"""
Day 5 零散时间配套：numpy 常用基础速览（跑一遍 = 学会查表）
========================================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第一周\day5"
    python numpy_basics.py

numpy（Numerical Python，数值计算库）= Python 里做数学/矩阵运算的"计算器"，
大模型的所有计算（矩阵乘法、概率分布）底层都用它。今天只学最常用的 6 招。

注意：这不是机器学习脚本（没有模型/数据/损失/优化器），
它是语法速查小抄，练完就能看懂前面几天所有脚本的 numpy 部分。
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

print("=" * 60)
print("招式 1：创建数组（列表 → 数组，和一排格子）")
print("=" * 60)
a = np.array([1, 2, 3, 4])          # 从列表创建一维数组
zero = np.zeros(3)                  # 3 个 0
one = np.ones((2, 3))               # 2行3列全是 1
seq = np.arange(0, 1, 0.25)         # 0 到 1（不含）每 0.25 一个
rng = np.random.default_rng(42)     # 随机数生成器（seed 固定=结果可复现）
r = rng.normal(size=(3, 2))         # 3行2列标准正态分布随机数
print("a =", a, " | np.zeros(3) =", zero)
print("np.ones((2,3)) =\n", one)
print("np.arange(0,1,0.25) =", seq)
print("随机矩阵 r.shape =", r.shape)

print()
print("=" * 60)
print("招式 2：shape 形状（几行几列 / 几维）")
print("=" * 60)
m = np.arange(12).reshape(3, 4)     # 12 个数重新排成 3 行 4 列
print("m =\n", m)
print("m.shape =", m.shape, "  m.ndim =", m.ndim, "维  ｜  m.size =", m.size, "个数")
print("→ 大模型里 (seq, batch, dim) 就是一个 3 维数组，Day 4 学过")

print()
print("=" * 60)
print("招式 3：索引与切片（取某行/某列/某块）")
print("=" * 60)
print("m[0] =", m[0], "（第 0 行）")
print("m[1, 2] =", m[1, 2], "（第 1 行第 2 列）")
print("m[:, 1] =", m[:, 1], "（冒号=全部行，取第 1 列）")
print("m[1:, :2] =\n", m[1:, :2], "（第 1 行起，前 2 列）")

print()
print("=" * 60)
print("招式 4：数学运算（逐元素 + 求和/均值）")
print("=" * 60)
x = np.array([1.0, 2.0, 3.0, 4.0])
print("x+1 =", x + 1, "  x*2 =", x * 2, "  x**2 =", x**2)
print("np.sqrt(x) =", np.sqrt(x))
print("np.sum(x) =", np.sum(x), "  np.mean(x) =", np.mean(x), "  np.max(x) =", np.max(x))

print()
print("=" * 60)
print("招式 5：矩阵乘法（@ 或 np.dot）vs 逐元素乘法（*）")
print("=" * 60)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("A * B（对应位置相乘）=\n", A * B)
print("A @ B（矩阵乘法，行×列）=\n", A @ B)
print("→ 注意区别！注意力里的 Q@Kᵀ 就是矩阵乘法")

print()
print("=" * 60)
print("招式 6：广播（小数组自动扩展成大数组的形状）")
print("=" * 60)
v = np.array([10, 20, 30])
M = np.ones((3, 3))
print("M =\n", M)
print("M + v =\n", M + v, "\n（v 自动变成 3 行，每行都是 [10,20,30]）")

print()
print("=" * 60)
print("招式 7：联系 Day 3 —— 用 numpy 手写 softmax（每行和为 1 的概率分布）")
print("=" * 60)
scores = np.array([[3.0, 1.0, 0.2],
                   [0.5, 2.0, 1.0]])
exp = np.exp(scores - scores.max(axis=1, keepdims=True))   # 减最大值防溢出（数学技巧）
weights = exp / exp.sum(axis=1, keepdims=True)             # 每行归一化
print("scores =\n", scores)
print("softmax 权重（每行和为 1）=\n", np.round(weights, 3))
print("每行和 =", weights.sum(axis=1), "  → 每行是一个概率分布 ✅")

print()
print("=" * 60)
print("小练习（自己动手）：把上面的 softmax 改成 4 个词，跑一遍，确认每行和仍为 1")
print("=" * 60)
