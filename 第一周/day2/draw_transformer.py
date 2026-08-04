# -*- coding: utf-8 -*-
r"""
Day 2 配套脚本：自动绘制 Transformer 整体结构图
================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第一周\day2"
    python draw_transformer.py

输出：当前文件夹下生成 transformer_structure.png（1920x1200）
依赖：matplotlib（未安装的话先执行  pip install matplotlib）

说明：本图依据 Jay Alammar《The Illustrated Transformer》重绘，
用于帮你理解"整体结构 + Encoder/Decoder 布局 + 多头注意力 + 位置编码"的位置关系。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------- 中文字体设置（Windows 用微软雅黑） ----------------
def setup_cjk_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "PingFang SC",
                 "Noto Sans CJK SC", "Arial Unicode MS"]:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break

setup_cjk_font()
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 画布 ----------------
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13.6)
ax.axis("off")

# ---------------- 颜色 ----------------
C_INPUT = ("#d5f5e3", "#1e8449")   # 输入 / 词嵌入：绿色
C_POS   = ("#fef9e7", "#b7950b")   # 位置编码：黄色
C_ENC   = ("#d6eaf8", "#1f618d")   # 编码器模块：蓝色
C_DEC   = ("#fdebd0", "#b9770e")   # 解码器模块：橙色
C_OUT   = ("#e8daef", "#7d3c98")   # 输出：紫色
C_ARROW = "#566573"

# ---------------- 工具函数 ----------------
def box(x, y, w, h, text, colors, fontsize=9, lw=1.4, zorder=3):
    """实心圆角方块"""
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1b2631", zorder=zorder + 1, linespacing=1.6)


def frame_box(x, y, w, h, label):
    """虚线外框（Encoder/Decoder 容器）"""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.2, linestyle="--",
                                edgecolor="#99aab5", facecolor="none", zorder=1))
    ax.text(x + w / 2, y + h - 0.35, label, ha="center", va="center",
            fontsize=11.5, color="#2c3e50", fontweight="bold", zorder=4)


def line(p1, p2, arrow=False, color=C_ARROW, lw=1.8):
    """直线/箭头"""
    style = "-|>" if arrow else "-"
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, zorder=2))


def note(x, y, text, fontsize=8, ha="left", color="#5d6d7e"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, zorder=5, linespacing=1.5)


# ================= 标题 =================
ax.text(5, 13.35, "Transformer 整体结构图（依据 Jay Alammar《The Illustrated Transformer》重绘）",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#1b2631")

# ================= 输入部分（底部） =================
box(0.45, 0.30, 2.70, 0.80, "英文输入句子\nInputs", C_INPUT)
box(0.45, 1.50, 2.70, 0.80, "输入嵌入\nInput Embedding", C_INPUT)
box(0.45, 2.70, 2.70, 0.80, "位置编码\nPositional Encoding", C_POS)

box(7.00, 0.30, 2.40, 0.80, "法文输出（右移一位）\nOutputs (shifted right)", C_INPUT, fontsize=8.5)
box(7.00, 1.50, 2.40, 0.80, "输出嵌入\nOutput Embedding", C_INPUT)
box(7.00, 2.70, 2.40, 0.80, "位置编码\nPositional Encoding", C_POS)

line((1.8, 1.10), (1.8, 1.50), arrow=True)
line((1.8, 2.30), (1.8, 2.70), arrow=True)
line((8.2, 1.10), (8.2, 1.50), arrow=True)
line((8.2, 2.30), (8.2, 2.70), arrow=True)

# ================= 编码器（左） =================
frame_box(0.3, 3.2, 3.0, 7.8, "Encoder × N（N 层相同结构）")

box(0.5, 9.4, 2.6, 1.0, "多头注意力\nMulti-Head Attention", C_ENC)
box(0.5, 8.1, 2.6, 0.7, "Add & Norm", C_ENC)
box(0.5, 6.5, 2.6, 1.0, "前馈网络\nFeed-Forward", C_ENC)
box(0.5, 5.2, 2.6, 0.7, "Add & Norm", C_ENC)

# 输入走左侧边线进入编码器
line((1.8, 3.55), (0.42, 3.55))
line((0.42, 3.55), (0.42, 9.90))
line((0.42, 9.90), (0.50, 9.90), arrow=True)

# 编码器内部逐层流动
line((1.8, 9.40), (1.8, 8.80), arrow=True)
line((1.8, 8.10), (1.8, 7.50), arrow=True)
line((1.8, 6.50), (1.8, 5.90), arrow=True)

# ================= 解码器（右） =================
frame_box(6.9, 3.2, 2.6, 7.8, "Decoder × N（N 层相同结构）")

box(7.1, 9.4, 2.2, 1.0, "带掩码多头注意力\nMasked Multi-Head Attention", C_DEC, fontsize=8.5)
box(7.1, 8.5, 2.2, 0.6, "Add & Norm", C_DEC)
box(7.1, 7.0, 2.2, 1.0, "多头注意力\n（K、V 来自编码器）", C_DEC, fontsize=8.5)
box(7.1, 6.1, 2.2, 0.6, "Add & Norm", C_DEC)
box(7.1, 4.6, 2.2, 1.0, "前馈网络\nFeed-Forward", C_DEC)
box(7.1, 3.7, 2.2, 0.6, "Add & Norm", C_DEC)

# 输入走右侧边线进入解码器
line((8.2, 3.55), (9.55, 3.55))
line((9.55, 3.55), (9.55, 9.90))
line((9.55, 9.90), (9.30, 9.90), arrow=True)

# 解码器内部逐层流动
line((8.2, 9.40), (8.2, 9.10), arrow=True)
line((8.2, 8.50), (8.2, 8.00), arrow=True)
line((8.2, 7.00), (8.2, 6.70), arrow=True)
line((8.2, 6.10), (8.2, 5.60), arrow=True)
line((8.2, 4.60), (8.2, 4.30), arrow=True)

# ================= 编码器输出 -> 解码器（K、V） =================
line((3.1, 5.55), (3.35, 5.55))
line((3.35, 5.55), (7.1, 7.475), arrow=True)

# ================= 解码器输出 -> 线性层 -> Softmax =================
line((8.2, 4.30), (9.45, 4.30))
line((9.45, 4.30), (9.45, 11.55))
line((9.45, 11.55), (9.40, 11.55), arrow=True)

# ================= 输出部分（顶部） =================
box(7.0, 11.2, 2.4, 0.7, "线性层\nLinear", C_OUT)
box(7.0, 12.1, 2.4, 0.7, "Softmax\n（输出概率分布）", C_OUT, fontsize=8.5)
box(7.0, 13.0, 2.4, 0.5, "输出概率\nOutput Probabilities", C_OUT, fontsize=8)

# ================= 解释性标注 =================
note(3.45, 9.90, "Q、K、V 都来自上一层\n（自注意力）")
note(6.75, 9.90, "带掩码：预测当前词时\n不能偷看后面的词", ha="right")
note(6.75, 7.50, "Q 来自解码器\nK、V 来自编码器", ha="right")
note(4.60, 4.90, "编码器输出作为 K、V 传给解码器")
note(3.45, 12.45, "输出：下一个词的概率分布\n（每步只生成一个词）")
note(3.30, 3.10, "词向量 + 位置信息", fontsize=7.5, color="#7d6608")

# ================= 图例 / 脚注 =================
ax.text(5, 2.0, "绿色 = 输入/嵌入   黄色 = 位置编码   蓝色 = 编码器模块   橙色 = 解码器模块   紫色 = 输出",
        ha="center", va="center", fontsize=9, color="#34495e")
ax.text(5, 1.1, "结构参考：Jay Alammar, The Illustrated Transformer (2018) ｜ 原论文：Attention Is All You Need (Vaswani et al., 2017)",
        ha="center", va="center", fontsize=8, color="#85929e")

# ================= 保存 =================
fig.savefig("transformer_structure.png", dpi=120, facecolor="white")
print("已生成 transformer_structure.png（请打开看看效果，对照教程第 5 步）")
