# -*- coding: utf-8 -*-
r"""
Day 5 配套脚本：自动绘制 Transformer 完整结构图（定稿版）
========================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第一周\day5"
    python draw_transformer_full.py

输出：当前文件夹下生成 transformer_structure_full.png（约 1920x2160）
依赖：matplotlib（Day 2 已装）

说明：本图在 Day 2 的"整体结构图"基础上定稿，新增下半部分的
"单层内部解剖图"，补齐 Day 5 的三大件并标注每层作用：
    残差连接（Residual Connection，残差=和标准答案的差距的余量）：
        x 原样加到子层输出上（x + 子层(x)），防深层梯度消失
    LayerNorm（层归一化，把数值拉回均值0方差1的标准化操作）：
        数值稳定，训练更快
    前馈网络（Feed-Forward Network，FFN）：
        每个位置独立加工一遍信息
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

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

# ---------------- 颜色 ----------------
C_INPUT = ("#d5f5e3", "#1e8449")   # 输入 / 词嵌入：绿色
C_POS   = ("#fef9e7", "#b7950b")   # 位置编码：黄色
C_ENC   = ("#d6eaf8", "#1f618d")   # 编码器模块：蓝色
C_DEC   = ("#fdebd0", "#b9770e")   # 解码器模块：橙色
C_OUT   = ("#e8daef", "#7d3c98")   # 输出：紫色
C_RES   = ("#f9ebea", "#c0392b")   # 残差连接：红色系（Day 5 新增重点）
C_ARROW = "#566573"
C_DASH  = "#c0392b"                # 残差跳线：红色虚线


# ---------------- 工具函数（每个都接收一个坐标轴 ax） ----------------
def box(ax, x, y, w, h, text, colors, fontsize=9, lw=1.4, zorder=3):
    """实心圆角方块"""
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1b2631", zorder=zorder + 1, linespacing=1.6)


def frame_box(ax, x, y, w, h, label, fontsize=11.5):
    """虚线外框（Encoder/Decoder 容器）"""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.2, linestyle="--",
                                edgecolor="#99aab5", facecolor="none", zorder=1))
    ax.text(x + w / 2, y + h - 0.35, label, ha="center", va="center",
            fontsize=fontsize, color="#2c3e50", fontweight="bold", zorder=4)


def line(ax, p1, p2, arrow=False, color=C_ARROW, lw=1.8, style="-"):
    """直线/箭头/虚线"""
    s = "-|>" if arrow else "-"
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=s, mutation_scale=16,
                                 color=color, lw=lw, linestyle=style, zorder=2))


def note(ax, x, y, text, fontsize=8, ha="left", color="#5d6d7e"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, zorder=5, linespacing=1.5)


# ================= 上半部分：整体结构图（Day 2 版，微调标注） =================
def draw_overall(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.6)
    ax.axis("off")
    ax.set_title("上半部分：整体结构图（Encoder + Decoder）", fontsize=13,
                 fontweight="bold", color="#1b2631", pad=8)

    # ---------- 输入部分（底部） ----------
    box(ax, 0.45, 0.30, 2.70, 0.80, "英文输入句子\nInputs", C_INPUT)
    box(ax, 0.45, 1.50, 2.70, 0.80, "输入嵌入\nInput Embedding", C_INPUT)
    box(ax, 0.45, 2.70, 2.70, 0.80, "位置编码\nPositional Encoding", C_POS)

    box(ax, 7.00, 0.30, 2.40, 0.80, "法文输出（右移一位）\nOutputs (shifted right)", C_INPUT, fontsize=8.5)
    box(ax, 7.00, 1.50, 2.40, 0.80, "输出嵌入\nOutput Embedding", C_INPUT)
    box(ax, 7.00, 2.70, 2.40, 0.80, "位置编码\nPositional Encoding", C_POS)

    line(ax, (1.8, 1.10), (1.8, 1.50), arrow=True)
    line(ax, (1.8, 2.30), (1.8, 2.70), arrow=True)
    line(ax, (8.2, 1.10), (8.2, 1.50), arrow=True)
    line(ax, (8.2, 2.30), (8.2, 2.70), arrow=True)

    # ---------- 编码器（左） ----------
    frame_box(ax, 0.3, 3.2, 3.0, 7.8, "Encoder × N（N 层相同结构）")

    box(ax, 0.5, 9.4, 2.6, 1.0, "多头注意力\nMulti-Head Attention", C_ENC)
    box(ax, 0.5, 8.1, 2.6, 0.7, "Add & Norm", C_ENC)
    box(ax, 0.5, 6.5, 2.6, 1.0, "前馈网络\nFeed-Forward", C_ENC)
    box(ax, 0.5, 5.2, 2.6, 0.7, "Add & Norm", C_ENC)

    line(ax, (1.8, 3.55), (0.42, 3.55))
    line(ax, (0.42, 3.55), (0.42, 9.90))
    line(ax, (0.42, 9.90), (0.50, 9.90), arrow=True)

    line(ax, (1.8, 9.40), (1.8, 8.80), arrow=True)
    line(ax, (1.8, 8.10), (1.8, 7.50), arrow=True)
    line(ax, (1.8, 6.50), (1.8, 5.90), arrow=True)

    # ---------- 解码器（右） ----------
    frame_box(ax, 6.9, 3.2, 2.6, 7.8, "Decoder × N（N 层相同结构）")

    box(ax, 7.1, 9.4, 2.2, 1.0, "带掩码多头注意力\nMasked Multi-Head Attention", C_DEC, fontsize=8.5)
    box(ax, 7.1, 8.5, 2.2, 0.6, "Add & Norm", C_DEC)
    box(ax, 7.1, 7.0, 2.2, 1.0, "多头注意力\n（K、V 来自编码器）", C_DEC, fontsize=8.5)
    box(ax, 7.1, 6.1, 2.2, 0.6, "Add & Norm", C_DEC)
    box(ax, 7.1, 4.6, 2.2, 1.0, "前馈网络\nFeed-Forward", C_DEC)
    box(ax, 7.1, 3.7, 2.2, 0.6, "Add & Norm", C_DEC)

    line(ax, (8.2, 3.55), (9.55, 3.55))
    line(ax, (9.55, 3.55), (9.55, 9.90))
    line(ax, (9.55, 9.90), (9.30, 9.90), arrow=True)

    line(ax, (8.2, 9.40), (8.2, 9.10), arrow=True)
    line(ax, (8.2, 8.50), (8.2, 8.00), arrow=True)
    line(ax, (8.2, 7.00), (8.2, 6.70), arrow=True)
    line(ax, (8.2, 6.10), (8.2, 5.60), arrow=True)
    line(ax, (8.2, 4.60), (8.2, 4.30), arrow=True)

    # ---------- 编码器输出 -> 解码器（K、V） ----------
    line(ax, (3.1, 5.55), (3.35, 5.55))
    line(ax, (3.35, 5.55), (7.1, 7.475), arrow=True)

    # ---------- 解码器输出 -> 线性层 -> Softmax ----------
    line(ax, (8.2, 4.30), (9.45, 4.30))
    line(ax, (9.45, 4.30), (9.45, 11.55))
    line(ax, (9.45, 11.55), (9.40, 11.55), arrow=True)

    # ---------- 输出部分（顶部） ----------
    box(ax, 7.0, 11.2, 2.4, 0.7, "线性层\nLinear", C_OUT)
    box(ax, 7.0, 12.1, 2.4, 0.7, "Softmax\n（输出概率分布）", C_OUT, fontsize=8.5)
    box(ax, 7.0, 13.0, 2.4, 0.5, "输出概率\nOutput Probabilities", C_OUT, fontsize=8)

    # ---------- 解释性标注 ----------
    note(ax, 3.45, 9.90, "Q、K、V 都来自上一层\n（自注意力）")
    note(ax, 6.75, 9.90, "带掩码：预测当前词时\n不能偷看后面的词", ha="right")
    note(ax, 6.75, 7.50, "Q 来自解码器\nK、V 来自编码器", ha="right")
    note(ax, 4.60, 4.90, "编码器输出作为 K、V 传给解码器")
    note(ax, 3.45, 12.45, "输出：下一个词的概率分布\n（每步只生成一个词）")
    note(ax, 3.30, 3.10, "词向量 + 位置信息", fontsize=7.5, color="#7d6608")

    ax.text(5, 2.0, "绿色=输入/嵌入  黄色=位置编码  蓝色=编码器  橙色=解码器  紫色=输出  ｜  Add & Norm 的内部细节见下图",
            ha="center", va="center", fontsize=9, color="#34495e")


# ================= 下半部分：单层内部解剖图（Day 5 新增） =================
def draw_layer(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.9)
    ax.axis("off")
    ax.set_title("下半部分：Encoder 每一层内部解剖（残差连接 + LayerNorm + 前馈网络）",
                 fontsize=13, fontweight="bold", color="#1b2631", pad=8)

    # ---------- 左侧：单层组件纵向堆叠 ----------
    box(ax, 0.7, 0.3, 2.6, 0.8, "输入 x\n（词向量矩阵）", C_INPUT, fontsize=8.5)
    box(ax, 0.7, 1.7, 2.6, 1.3, "多头注意力\nMulti-Head Attention\n（Day 4 学过）", C_ENC, fontsize=8)
    box(ax, 0.7, 3.6, 2.6, 0.8, "⊕ 残差连接\nx + Attn(x)", C_RES, fontsize=8.5)
    box(ax, 0.7, 5.0, 2.6, 0.8, "LayerNorm\n层归一化", C_ENC, fontsize=8.5)
    box(ax, 0.7, 6.4, 2.6, 1.3, "前馈网络 Feed-Forward\nLinear → ReLU → Linear", C_ENC, fontsize=8)
    box(ax, 0.7, 8.3, 2.6, 0.8, "⊕ 残差连接\nx1 + FFN(x1)", C_RES, fontsize=8.5)
    box(ax, 0.7, 9.7, 2.6, 0.8, "LayerNorm\n层归一化", C_ENC, fontsize=8.5)

    # 层内数据流动
    line(ax, (2.0, 1.1), (2.0, 1.7), arrow=True)
    line(ax, (2.0, 3.0), (2.0, 3.6), arrow=True)
    line(ax, (2.0, 4.4), (2.0, 5.0), arrow=True)
    line(ax, (2.0, 5.8), (2.0, 6.4), arrow=True)
    line(ax, (2.0, 7.7), (2.0, 8.3), arrow=True)
    line(ax, (2.0, 9.1), (2.0, 9.7), arrow=True)
    line(ax, (2.0, 10.5), (2.0, 11.3), arrow=True)

    # 残差跳线（红色虚线）：x 跳过子层，原样加到后面的 ⊕ 上
    line(ax, (0.42, 0.7), (0.42, 4.0), arrow=True, color=C_DASH, lw=1.6, style="--")
    line(ax, (0.42, 5.4), (0.42, 8.7), arrow=True, color=C_DASH, lw=1.6, style="--")
    note(ax, 0.20, 2.2, "x\n跳过\nMHA", fontsize=7, color="#c0392b", ha="center")
    note(ax, 0.20, 7.1, "x1\n跳过\nFFN", fontsize=7, color="#c0392b", ha="center")

    # 整层公式
    note(ax, 4.3, 11.35, "整层公式：x2 = LayerNorm( x1 + FFN(x1) ) ，其中 x1 = LayerNorm( x + Attn(x) )",
         fontsize=9, color="#1b2631")

    # ---------- 右侧：每个组件的作用 ----------
    frame_box(ax, 4.3, 0.3, 5.5, 10.4, "每个组件的作用（今天要背下来）", fontsize=10)

    notes = [
        "① 输入 x：上一个模块传来的特征\n   词向量矩阵，形状 (长度, 批量, 维度)",
        "② 多头注意力 MHA：词与词交换信息，\n   找出“谁该注意谁”（Day 4 学过）",
        "③ ⊕ 残差连接：把 x 原样加到子层输出上\n   公式 x + Attn(x)——“跳过子层的高速公路”，\n   让梯度有捷径直达，防深层梯度消失",
        "④ LayerNorm：把每个位置的特征归一到\n   均值 0、方差 1，再缩放平移（γ、β 可学）\n   ——数值稳定，训练更快",
        "⑤ 前馈网络 FFN：每个位置独立加工一遍\n   公式 W2·ReLU(W1·x+b1)+b2\n   ——注意力负责“交换”，FFN 负责“思考”",
        "⑥ 再一个 ⊕ + LayerNorm：同样的套路\n   整层输出 = LN(x1 + FFN(x1))",
        "标准“子层包装”模式（记这个）：\n   子层 → 残差加法 → LayerNorm",
    ]
    y0 = 9.95
    for i, t in enumerate(notes):
        note(ax, 4.55, y0 - i * 1.42, t, fontsize=7.8, color="#1b2631")


# ================= 主流程 =================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 18),
    gridspec_kw={"height_ratios": [13.6, 11.9]}
)
fig.subplots_adjust(hspace=0.35, top=0.955, bottom=0.015, left=0.01, right=0.99)

fig.suptitle("Transformer 完整结构图（定稿版）—— 上半=整体结构，下半=单层解剖",
             fontsize=17, fontweight="bold", color="#1b2631")

draw_overall(ax1)
draw_layer(ax2)

fig.savefig("transformer_structure_full.png", dpi=120, facecolor="white")
print("[OK] 已生成 transformer_structure_full.png（定稿结构图，请打开查看）")
