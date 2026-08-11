# -*- coding: utf-8 -*-
r"""
Day 5 配套脚本：预训练 vs 微调 vs LoRA —— 纯数学演示（不下载任何模型）
========================================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第一周\day5"
    python lora_math_demo.py

本脚本用 numpy 数学模拟"预训练 → 微调 → LoRA"全过程，帮你建立认知。
它对应"机器学习的五件套"（对照 Day 4 学过的五件套）：
    模型    = 线性回归（y = X·W，W 是可学习的权重矩阵）
    数据    = 程序随机生成的任务A（"通用任务"）与任务B（"特定任务"）
    损失函数 = 均方误差 MSE（预测值 与 标准答案 差的平方的平均值）
    优化器  = 最小二乘闭式解（正规方程 (XᵀX)⁻¹Xᵀy，一步算出最优权重）
    训练/测试 = 在一批随机数据上"拟合"出权重，再用另一批新数据测"泛化能力"
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------- 工具函数 ----------------
def make_X(n, seed):
    """生成 n 个 8 维样本（每个样本 8 个特征），服从标准正态分布"""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 8))


def task_A_Y(X):
    """任务 A（预训练任务，模拟"通用语言理解"）：每个输出只用前 6 个特征。
    y0 = x0 - x1,  y1 = x2 - x3,  y2 = x4 - x5"""
    W = np.zeros((8, 3))
    for j in range(3):
        W[2 * j, j] = 1.0
        W[2 * j + 1, j] = -1.0
    return X @ W


def task_B_Y(X):
    """任务 B（微调任务，模拟"特定业务任务"）：只用最后 2 个特征 x6、x7。
    y0 = x6 - x7,  y1 = x6 + x7,  y2 = x7 - x6"""
    W = np.zeros((8, 3))
    W[6, 0] = 1.0;  W[7, 0] = -1.0
    W[6, 1] = 1.0;  W[7, 1] = 1.0
    W[6, 2] = -1.0; W[7, 2] = 1.0
    return X @ W


def fit(X, Y):
    """最小二乘闭式解（正规方程）：W = (XᵀX + λI)⁻¹ XᵀY，λ 防奇异"""
    lam = 1e-6 * np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + lam, X.T @ Y)


def mse(X, Y, W):
    """均方误差：衡量权重 W 在数据 (X, Y) 上预测得有多差（越小越好）"""
    return float(np.mean((X @ W - Y) ** 2))


# ================= 第 1 部分：预训练 =================
print("=" * 62)
print("1. 预训练（Pre-training）：在任务 A 上训练出一个通用底座 W0")
print("=" * 62)
Xa = make_X(400, 1)
Ya = task_A_Y(Xa)
Xa_test = make_X(200, 11)
Ya_test = task_A_Y(Xa_test)
Xb = make_X(400, 2)
Yb = task_B_Y(Xb)
Xb_test = make_X(200, 12)
Yb_test = task_B_Y(Xb_test)

W0 = fit(Xa, Ya)
print(f"预训练完成。W0 在任务A（通用任务）上的误差 : {mse(Xa_test, Ya_test, W0):.6f}")
print(f"                    W0 在任务B（特定任务）上的误差 : {mse(Xb_test, Yb_test, W0):.6f}")
print("→ 结论：底座 W0 把任务 A 学得很好（误差≈0），但完全不懂任务 B。")

# ================= 第 2 部分：完整微调 =================
print()
print("=" * 62)
print("2. 完整微调（Full Fine-tuning）：用任务 B 的数据把 W0 所有参数再训练一遍")
print("=" * 62)
Wf = fit(Xb, Yb)
print(f"完整微调后，Wf 在任务A上的误差 : {mse(Xa_test, Ya_test, Wf):.6f}  （预训练时≈0）")
print(f"                 Wf 在任务B上的误差 : {mse(Xb_test, Yb_test, Wf):.6f}  （学会新任务了）")
print("→ 结论：完整微调学会了任务 B，但把任务 A 忘光了——这就是“灾难性遗忘”。")

# ================= 第 3 部分：LoRA 的数学本质 =================
print()
print("=" * 62)
print("3. LoRA 的数学本质：微调其实只改了一个“小增量” ΔW = Wf - W0")
print("=" * 62)
delta = Wf - W0
U, s, Vt = np.linalg.svd(delta, full_matrices=False)
print(f"ΔW 的形状 : {delta.shape}（8 行 × 3 列）")
print(f"ΔW 的奇异值（singular values，衡量每个方向的重要性）: {np.round(s, 4)}")
print("→ 结论：奇异值一共只有 3 个（8×3 的矩阵秩最多为 3），说明 ΔW 本身是低秩的。")

print()
print("LoRA 的做法：不直接改 W0，而是把 ΔW 近似成两个小矩阵的乘积 ΔW ≈ B·A，")
print("其中 B 是 8×r、A 是 r×3，r 远远小于 8（这里 r 取 1、2、3）→ 参数大大变少。")
for r in [1, 2, 3]:
    B = U[:, :r] * s[:r]
    A = Vt[:r, :]
    delta_r = B @ A
    W_lora = W0 + delta_r
    err = np.linalg.norm(delta - delta_r) / np.linalg.norm(delta)
    print(f"  r={r} : LoRA 后 任务A误差={mse(Xa_test, Ya_test, W_lora):.4f} | "
          f"任务B误差={mse(Xb_test, Yb_test, W_lora):.4f} | ΔW近似误差={err:.4f}")
print("→ 结论：r 越小，更新越“温和”（保留旧知识多、学新任务少）；")
print("        r 越大，越接近完整微调（新任务学得好、但忘得也多）。")

# ================= 第 4 部分：真实尺寸模拟“低秩增量” =================
print()
print("=" * 62)
print("4. 真实场景：为什么 ΔW 可以放心地用低秩近似？（用 512×512 的大矩阵模拟）")
print("=" * 62)
d = 512
r_true = 16
rng = np.random.default_rng(7)
W0_big = rng.normal(size=(d, d))
Ur = rng.normal(size=(d, r_true))
Vr = rng.normal(size=(d, r_true))
delta_true = Ur @ Vr.T                     # 人为构造一个"秩只有 16"的增量
Wf_big = W0_big + delta_true
D = Wf_big - W0_big
s_big = np.linalg.svd(D, compute_uv=False)
print("ΔW 的前 20 个奇异值（第 16 个之后几乎全是 0）：")
print("  " + "  ".join(f"{x:.2f}" for x in s_big[:20]))
print("→ 结论：真实微调学到的 ΔW，绝大多数“方向”上几乎不动，只有少数几个方向")
print("        在变。所以用 r=16 的低秩 B·A 就能几乎无损地表示它。")

U16, s16, Vt16 = np.linalg.svd(D, full_matrices=False)
D16 = (U16[:, :16] * s16[:16]) @ Vt16[:16, :]
err16 = np.linalg.norm(D - D16) / np.linalg.norm(D)
print(f"用 r=16 近似 ΔW 的误差 : {err16:.2e}  → 几乎无损！")

# ================= 第 5 部分：LoRA 到底省了多少参数 =================
print()
print("=" * 62)
print("5. 参数账：全量微调 vs LoRA（这就是 LoRA 省显存的原因）")
print("=" * 62)
print("对一个 d×d 的权重矩阵：")
print("  全量微调：训练 d² 个参数；  LoRA：只训练 B( d×r ) + A( r×d ) = 2·d·r 个参数")
print(f"  节省比例 = 1 - (2·d·r)/d² = 1 - 2r/d\n")

for d, r in [(512, 16), (3584, 8)]:
    full = d * d
    lora = 2 * d * r
    save = 1 - lora / full
    print(f"  例：d={d}, r={r} → 全量={full:,} 个参数，LoRA={lora:,} 个参数，节省 {save*100:.2f}%")

print()
print("以 Qwen2.5-7B 为例（隐藏维度 d=3584，28 层，每层有 q/k/v/o 共 4 个投影矩阵）：")
n_layers, proj = 28, 4
d, r = 3584, 8
lora_total = n_layers * proj * 2 * d * r
params_7b = 7.61e9
print(f"  LoRA 只训练 q/k/v/o 的增量：{lora_total:,} 个参数")
print(f"  全模型参数：{params_7b:.2e}（约 76.1 亿）")
print(f"  训练的参数只占全模型的 {lora_total/params_7b*100:.3f}%")
print("→ 这就是面试常说的“LoRA 只训练不到 1% 的参数”。")

print()
print("=" * 62)
print("一句话总结：预训练给底座，微调改底座，LoRA 把“改”限制在一个")
print("小空间（低秩增量 B·A）里——省显存、省时间、还保留底座的大部分旧知识。")
print("=" * 62)
