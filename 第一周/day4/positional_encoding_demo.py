# -*- coding: utf-8 -*-
"""
Day 4 配套脚本 1：位置编码可视化 —— "给词编座位号"到底长什么样
================================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\\Lan\\研究生\\技术学习\\大模型算法\\第一周\\day4"
    python positional_encoding_demo.py

本脚本做 3 件事：
    1. 用 sin/cos 公式生成 20 个位置的位置编码，打印前 8 个位置、画出热力图
    2. 把位置编码"加"到词向量上（呼应 Day 2：加，不是拼接）
    3. 关键实验：对比"我爱你 / 你爱我"在 加/不加位置编码 两种情况下的注意力输出，
       证明没有位置编码时，模型分不清词序

数学视角：
    位置编码用"不同频率的正弦/余弦波"给每个位置一个唯一的"频率指纹"，
    不同位置编码不同 → 模型才能知道"我是第几个词"。

只用到 numpy + matplotlib，装 torch 时已自动带上。
"""

# =====================================================================
# 【课前语法速查】本脚本用到的 Python 语法，遇到不懂的回来查
# =====================================================================
#  import numpy as np            → 导入数值计算库 numpy，起短名 np
#
#  np.arange(20)                 → 生成数组 [0, 1, 2, ..., 19]
#  np.zeros((20, 8))             → 生成 20 行 8 列、全 0 的二维数组
#                                 （每行 = 一个位置的编码，共 20 个位置）
#
#  x[:, None]                    → 给一维数组"加一个空轴"，变成列向量
#                                 [1,2,3] → [[1],[2],[3]]（矩阵乘法需要）
#
#  pe[:, 0::2]                   → 切片：pe 的所有行、从第 0 列开始每隔 2 列取一列
#                                 也就是"第 0、2、4、6 ... 列"（偶数维）
#  pe[:, 1::2]                   → 第 1、3、5、7 ... 列（奇数维）
#
#  np.sin / np.cos               → 三角函数，对数组里的每个数逐个计算
#  np.exp(x)                     → e 的 x 次方
#  np.dot(a, b)                  → 两个向量的点积（对应位置相乘再求和）
#
#  with open("xx.png","wb") as f: → 把图片保存成文件（wb = 二进制写入）
#  if ... else ...               → 条件判断
# =====================================================================

import numpy as np

# 关闭 matplotlib 的弹窗模式，图片直接存成文件（避免在部分电脑上没反应）
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 让 matplotlib 支持中文（Windows 一般有微软雅黑；找不到就跳过，标题可能显示成方块但数据不受影响）
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    if _name in _available_fonts:
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False   # 解决坐标轴负号显示成方块


# ---------------------------------------------------------------------
# 第 1 段：用 sin/cos 公式生成位置编码
# ---------------------------------------------------------------------
def make_positional_encoding(max_len, d_model):
    """
    生成位置编码矩阵。
    参数：
        max_len : 最多支持多少个位置（词）
        d_model : 每个词向量的维度
    返回：
        形状 (max_len, d_model) 的数组，第 i 行 = 第 i 个位置的编码
    """
    pe = np.zeros((max_len, d_model))                    # 先铺一块全 0 的画布
    position = np.arange(max_len)[:, None]               # 位置编号 0,1,2,... 排成一列
    # div_term 是一组"递减的频率"。不同维度用不同频率 → 每个位置才能有独特波形
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)            # 偶数维（第0、2、4…列）用 sin
    pe[:, 1::2] = np.cos(position * div_term[:pe[:, 1::2].shape[1]])  # 奇数维用 cos（奇数维比偶数维少一列）
    return pe


# ---------------------------------------------------------------------
# 第 2 段：打印 + 画热力图
# ---------------------------------------------------------------------
print("===== 1. 前 8 个位置的编码（每个位置是一个 8 维向量）=====")
pe = make_positional_encoding(max_len=20, d_model=8)     # 20 个位置、8 维
for pos in range(8):
    print(f"位置 {pos}: {np.round(pe[pos], 3)}")          # np.round 保留 3 位小数

print("\n===== 2. 画热力图并保存 positional_encoding.png =====")
# 热力图：横轴 = 8 个维度，纵轴 = 20 个位置，颜色深浅 = 数值大小
plt.figure(figsize=(9, 5))
plt.imshow(pe.T, aspect="auto", cmap="coolwarm")         # 转置一下让"位置"在纵轴
plt.colorbar(label="编码数值")
plt.xlabel("维度（0~7）")
plt.ylabel("位置（0~19）")
plt.title("位置编码热力图：每个位置一行，互不相同 = 唯一的'座位号'")
plt.tight_layout()     # 自动调整子图参数，确保标题、标签和颜色条不会互相重叠或被边缘裁切。
plt.savefig("positional_encoding.png", dpi=150)          # 存成图片，图片就在 day4 文件夹
print("已保存: positional_encoding.png（打开文件夹就能看到）")

# ---------------------------------------------------------------------
# 第 3 段：关键实验 —— 没有位置编码时，"我爱你"和"你爱我"分不清
# ---------------------------------------------------------------------
print("\n===== 3. 关键实验：不加位置信息，'我爱你' 和 '你爱我' 的注意力输出 =====")

def attention_output(X):
    """极简单头自注意力（Day 3 学过的三步走），返回每个词融合上下文后的输出。"""
    scores = X @ X.T               # 第 1 步：相似度分数（Q=K=V=词向量本身）
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))  # 数值稳定
    weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)     # 第 2 步：softmax
    return weights @ X             # 第 3 步：对 V 加权平均


# 用 one-hot 词向量表示 我 / 爱 / 你（8 维，方便和第 2 段的 8 维位置编码相加）
E = {
    "我": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "爱": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "你": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}

sentence1 = ["我", "爱", "你"]
sentence2 = ["你", "爱", "我"]     # 顺序换一下

X1 = np.stack([E[w] for w in sentence1])   # "我爱你" 的词向量矩阵 (3,3)
X2 = np.stack([E[w] for w in sentence2])   # "你爱我" 的词向量矩阵 (3,3)

out1 = attention_output(X1)
out2 = attention_output(X2)

print("'我爱你' 的输出:")
print(np.round(out1, 4))
print("'你爱我' 的输出:")
print(np.round(out2, 4))

# 检查：把 out2 倒过来是不是和 out1 一模一样
# out2[::-1] = 把"你爱我"的输出倒着排 → 如果和"我爱你"的输出完全相等，就说明模型没区分顺序
same = np.allclose(out1, out2[::-1], atol=1e-6)
print("\n结论：'你爱我' 的输出倒过来 == '我爱你' 的输出？ →", same)
if same:
    print("→ 完全一样！不加位置信息时，模型只看到『我爱你也你爱我』是同一堆词，")
    print("  分不清谁是主语谁是宾语（注意力只认词、不认顺序）。")

# ---------------------------------------------------------------------
# 第 4 段：加上位置编码，再看一次 —— 结果不一样了
# ---------------------------------------------------------------------
print("\n===== 4. 加上位置编码后，两个句子的输出 =====")
pe3 = make_positional_encoding(max_len=3, d_model=8)     # 3 个位置、8 维编码

# 把位置编码"加"到词向量上（Day 2 说过：加，不是拼接）
X1p = np.stack([E[w] + pe3[i] for i, w in enumerate(sentence1)])
X2p = np.stack([E[w] + pe3[i] for i, w in enumerate(sentence2)])

out1p = attention_output(X1p)
out2p = attention_output(X2p)

print("'我爱你'（带位置）的输出:")
print(np.round(out1p, 4))
print("'你爱我'（带位置）的输出:")
print(np.round(out2p, 4))

same_p = np.allclose(out1p, out2p[::-1], atol=1e-6)
print("\n结论：'你爱我' 的输出倒过来 == '我爱你' 的输出？ →", same_p)
if not same_p:
    print("→ 不一样了！位置编码让模型知道了每个词在第几个位置，")
    print("  '我爱你' 和 '你爱我' 不再是同一句话。这就是'编座位号'的作用。")

print("\n===== 5. 数学视角 =====")
print("位置编码 = 用不同频率的 sin/cos 给每个位置一个唯一向量（频率指纹）。")
print("不同位置波形不同 → 加进词向量后，模型才能区分词序。")
