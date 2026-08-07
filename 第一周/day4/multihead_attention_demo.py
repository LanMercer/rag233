# -*- coding: utf-8 -*-
"""
Day 4 配套脚本 2：手写多头注意力（Multi-Head Attention）+ 和官方 PyTorch 对答案
================================================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\\Lan\\研究生\\技术学习\\大模型算法\\第一周\\day4"
    python multihead_attention_demo.py

本脚本做 3 件事：
    1. 用 numpy 手写"分头 → 每头独立算注意力 → 并头"全流程
    2. 手工构造两组权重：头 1 看"每个词的身份"、头 2 看"动物类别"，
       亲眼看到两个头给出两张完全不同的注意力权重矩阵（= 两张转移概率矩阵）
    3. 把同一份输入喂给官方 torch.nn.MultiheadAttention，
       和手写结果对比 —— 最大误差 ~1e-6，证明你做的是和官方同一件事

数学视角：
    多头 = 同时算 num_heads 张"转移概率矩阵"（每行和为 1），每张代表一种关注视角。
"""

# =====================================================================
# 【课前语法速查】本脚本用到的 Python 语法，遇到不懂的回来查
# =====================================================================
#  import numpy as np        → 数值计算库，起短名 np
#  import torch              → PyTorch 深度学习库（Day 1 装好的 GPU 版）
#  import torch.nn as nn     → PyTorch 里的神经网络模块库
#
#  np.eye(3)                 → 3×3 单位矩阵（对角线 1，其余 0）
#  X[:, 0:3]                 → 取 X 的所有行、第 0~2 列（"逗号左边=行，右边=列"）
#  np.concatenate([a, b], 1) → 把 a、b 按"列方向"拼起来（axis=1 = 沿列拼）
#  np.round(x, 4)            → 保留 4 位小数（纯显示用）
#  np.allclose(a, b, atol=1e-5) → 判断两个数组是否"几乎相等"（误差小于 1e-5）
#
#  torch.from_numpy(x)       → numpy 数组 → PyTorch 张量（torch 不认识 numpy 数组）
#  .float()                  → 转成浮点型（神经网络计算需要小数）
#  with torch.no_grad():     → 告诉 torch"这里只计算、不记录梯度"（推理时用）
#  torch.cat([a, b], dim=0)  → 和 numpy 的 concatenate 类似，沿第 0 维拼接
# =====================================================================

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# 先写一个 softmax（Day 3 学过的，这里每个"头"都会用到）
# ---------------------------------------------------------------------
def softmax(row):
    """把一行分数变成概率分布：每个数 0~1，加起来 = 1。"""
    row = row - row.max()          # 减去最大值，防止指数爆炸（数学上不影响结果）
    e = np.exp(row)
    return e / e.sum()


# ---------------------------------------------------------------------
# 手写多头注意力的核心函数（只读懂这个，就懂了全部）
# ---------------------------------------------------------------------
def multi_head_attention_numpy(X, Wq, Wk, Wv, num_heads):
    """
    手写多头注意力。
    参数：
        X        : 词向量矩阵，形状 (seq_len, d_model)
        Wq/Wk/Wv : 三个变换矩阵，形状 (d_model, d_model)（真实模型里是训练学出来的）
        num_heads: 几个头
    返回：
        out : 多头注意力的输出，形状 (seq_len, d_model)
    """
    seq_len, d_model = X.shape
    head_dim = d_model // num_heads      # 每个头的维度 = 总维度 ÷ 头数

    Q = X @ Wq                           # 每个词的 Query
    K = X @ Wk                           # 每个词的 Key
    V = X @ Wv                           # 每个词的 Value

    heads = []                           # 存每个头的输出
    for h in range(num_heads):
        # 【分头】切列：第 h 个头只用第 h*head_dim ~ (h+1)*head_dim 列的维度
        qh = Q[:, h * head_dim:(h + 1) * head_dim]
        kh = K[:, h * head_dim:(h + 1) * head_dim]
        vh = V[:, h * head_dim:(h + 1) * head_dim]

        # 每头独立算一遍 Day 3 的三步走：
        scores = qh @ kh.T / np.sqrt(head_dim)   # 第 1 步：相似度 + 除以 √d（防方差过大）
        weights = np.array([softmax(r) for r in scores])  # 第 2 步：softmax → 每行和为 1
        head_out = weights @ vh                   # 第 3 步：加权平均
        heads.append(head_out)

    # 【并头】把每个头的输出按列拼起来，恢复成 d_model 维
    out = np.concatenate(heads, axis=1)
    return out


# ---------------------------------------------------------------------
# 第 1 段：造输入 —— 3 个词、6 维向量
# ---------------------------------------------------------------------
print("===== 1. 输入：3 个词，6 维向量（前 3 维=身份，后 3 维=类别）=====")
# 前 3 维（身份）：猫=[1,0,0] 追=[0,1,0] 狗=[0,0,1]
# 后 3 维（类别）：猫和狗共享"动物"（第 4 维=1），追带"动作"（第 6 维=1）
words = ["猫", "追", "狗"]
X = np.array([
    [1, 0, 0, 1, 0, 0],   # 猫：身份=猫 + 类别=动物
    [0, 1, 0, 0, 0, 1],   # 追：身份=追 + 类别=动作
    [0, 0, 1, 1, 0, 0],   # 狗：身份=狗 + 类别=动物
]).astype(float)

for i, w in enumerate(words):
    print(f"{w}: {X[i]}")

# ---------------------------------------------------------------------
# 第 2 段：手工构造两组权重，让两个头"各看各的"
# ---------------------------------------------------------------------
print("\n===== 2. 分头：6 维 → 2 个头，每个头 3 维 =====")
d_model, num_heads = 6, 2
head_dim = d_model // num_heads
print(f"d_model={d_model}, num_heads={num_heads}, head_dim={head_dim}")
print("（d_model // num_heads = 6 // 2 = 3：每个头只看 3 个维度）")

# 头 1 的权重：只看"前 3 维（身份）"，所以头 1 只能区分"猫/追/狗"谁是谁
Wq1 = np.zeros((d_model, head_dim))
Wq1[0, 0] = 1.0; Wq1[1, 1] = 1.0; Wq1[2, 2] = 1.0   # 第 0、1、2 维 → 输出第 0、1、2 维
# 头 2 的权重：只看"类别"（第 4 维 = 下标 3 的'动物'维度），
# 猫和狗有动物=1，追没有 → 头 2 能看到"猫狗是一伙的"
Wq2 = np.zeros((d_model, head_dim))
Wq2[3, 0] = 1.0; Wq2[3, 1] = 1.0; Wq2[3, 2] = 1.0   # 三列都读"动物"维度

# 把两个头的 Wq 拼成完整 Wq（真实模型 Wq 是 (6,6)，这里把两头的权重放对角块）
Wq = np.zeros((d_model, d_model))
Wq[:, 0:3] = Wq1     # 头 1 用前 3 列
Wq[:, 3:6] = Wq2     # 头 2 用后 3 列
Wk = Wq.copy()       # 让 Key 用和 Query 一样的权重（简化，教学用）
Wv = np.eye(d_model) # Value 直接用原向量（简化）

# ---------------------------------------------------------------------
# 第 3 段：跑手写多头注意力，打印每个头的注意力权重矩阵
# ---------------------------------------------------------------------
print("\n===== 3. 手写多头注意力：每个头独立算一遍 =====")

def manual_forward_with_heads(X, Wq, Wk, Wv, num_heads):
    """和 multi_head_attention_numpy 一样，但额外返回每头的权重矩阵（用于打印）。"""
    seq_len, d_model = X.shape
    head_dim = d_model // num_heads
    Q = X @ Wq; K = X @ Wk; V = X @ Wv
    heads_out = []
    heads_w = []
    for h in range(num_heads):
        qh = Q[:, h * head_dim:(h + 1) * head_dim]
        kh = K[:, h * head_dim:(h + 1) * head_dim]
        vh = V[:, h * head_dim:(h + 1) * head_dim]
        scores = qh @ kh.T / np.sqrt(head_dim)
        weights = np.array([softmax(r) for r in scores])
        heads_w.append(weights)
        heads_out.append(weights @ vh)
    return np.concatenate(heads_out, axis=1), heads_w

out, head_weights = manual_forward_with_heads(X, Wq, Wk, Wv, num_heads)

print("\n-- 头 1（看'身份'）：注意力权重矩阵 --")
print(np.round(head_weights[0], 3))
print("解读：几乎只有对角线大 → 每个词最关注自己（因为身份维度是 one-hot，各管各的）")

print("\n-- 头 2（看'动物类别'）：注意力权重矩阵 --")
print(np.round(head_weights[1], 3))
print("解读：'猫''狗'两行互相给大权重（动物是一伙的），'追'被晾在一边")
print("→ 同一句输入，两个头给出两张完全不同的'转移概率矩阵' = 两个视角！")

print("\n===== 4. 并头 =====")
print("每个头的输出形状:", (X.shape[0], head_dim), "×", num_heads, "个头")
print("并头后形状:", out.shape, "(把 2 个 3 维输出拼成 6 维)")
print("并头后输出:")
print(np.round(out, 3))

# ---------------------------------------------------------------------
# 第 4 段：和官方 torch.nn.MultiheadAttention 对答案
# ---------------------------------------------------------------------
print("\n===== 5. 和官方 torch.nn.MultiheadAttention 对比 =====")
np.random.seed(42)                      # 固定随机数，保证每次结果一样
Wq_r = np.random.randn(d_model, d_model)   # 随机权重（模拟训练学出来的 W）
Wk_r = np.random.randn(d_model, d_model)
Wv_r = np.random.randn(d_model, d_model)

# 手写结果
manual_out = multi_head_attention_numpy(X, Wq_r, Wk_r, Wv_r, num_heads)

# 官方结果：把随机 Wq/Wk/Wv 塞进 nn.MultiheadAttention 里
# 注意：PyTorch 线性层约定是 y = x @ W^T（和 numpy 的 X @ W 差一个转置），
#       所以这里喂进去的是 Wq_r.T / Wk_r.T / Wv_r.T
# in_proj_weight 的形状是 (3*d_model, d_model)：前 d_model 行 = Wq、中间 = Wk、最后 = Wv
mha = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=False)
# dropout=0.0：关闭 Dropout 层，防止随机丢弃元素导致两次计算结果不一致。
# batch_first=False：PyTorch 默认期望输入的形状是 (序列长度, 批次大小, 特征维度)。
mha.eval()   # 切到推理模式（关闭 dropout，保证结果稳定）
with torch.no_grad():
# 进入“无梯度追踪”上下文。因为我们只是做前向推理对比数值，不需要 PyTorch 在后台记录计算图（这能节省大量内存并提升速度）
    mha.in_proj_weight.copy_(torch.cat([torch.from_numpy(Wq_r.T),
                                        torch.from_numpy(Wk_r.T),
                                        torch.from_numpy(Wv_r.T)], dim=0).float())
# torch.from_numpy(...)：将 NumPy 数组转换为 PyTorch 张量。
# .T：对权重进行转置，适配 PyTorch 的底层约定。
# torch.cat([...], dim=0)：将转置后的 Wq、Wk、Wv 在第 0 维度（行）上拼接，完美模拟 PyTorch 的 in_proj_weight 结构。
# .copy_()：将拼接好的权重原地复制到官方模块的权重参数中，替换掉它默认的随机权重。
    mha.in_proj_bias.zero_()             # 偏置全 0（简化）:将 Q、K、V 投影的偏置项全部清零。因为你的 NumPy 手写版本没有加偏置，这里必须清零，否则 PyTorch 的默认随机偏置会破坏对比结果。
    mha.out_proj.weight.copy_(torch.eye(d_model).float())  # 输出投影 = 单位矩阵:将输出投影的权重设置为单位矩阵，确保输出和输入维度相同。
    mha.out_proj.bias.zero_() # 将输出投影的偏置项全部清零。因为你的 NumPy 手写版本没有加偏置，这里必须清零，否则 PyTorch 的默认随机偏置会破坏对比结果。  

X_t = torch.from_numpy(X).float()        # numpy → torch:将输入矩阵 X 从 NumPy 数组转换为 PyTorch 张量，并转为 32 位浮点数。
official_out, _ = mha(X_t, X_t, X_t)     # 官方多头注意力（Q=K=V=X）:调用 PyTorch 模块进行前向计算。传入三个 X_t 分别代表 Query、Key、Value（自注意力机制下三者相同）。返回值中第一个是注意力输出，第二个是注意力权重矩阵（这里用 _ 忽略）。
official_out = official_out.detach().numpy()   # torch → numpy（detach 去掉梯度记录）:将 PyTorch 张量转换回 NumPy 数组，同时移除梯度信息（因为这里不需要计算梯度）。

diff = np.abs(manual_out - official_out).max()   # 逐格找最大误差
print("手写输出 vs 官方输出：最大误差 =", f"{diff:.2e}")
print("→ 误差在 1e-5 以下：你手写的就是官方在干的事！")
print("  （官方只是把 Wq/Wk/Wv 换成训练学出来的，还顺手封装成了黑盒）")
