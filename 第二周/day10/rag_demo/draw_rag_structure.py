# -*- coding: utf-8 -*-
r"""
Day 10 配套脚本：自动绘制 RAG 全链路结构图
============================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第二周\day10\rag_demo"
    python draw_rag_structure.py

输出：当前文件夹下生成 rag_结构图.png
依赖：matplotlib（未安装的话先执行  pip install matplotlib）

说明：本图把 RAG 全链路画成两条线——
    上排"建库（离线）"：PDF → 切块 → Embedding → 存 Chroma；
    下排"问答（在线）"：提问 → 向量化 → 检索 Top-K → 拼上下文 → Qwen 生成。
    和图下方文字配套，就是 Day 10 验收要交的"RAG 链路结构图"。
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def setup_cjk_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "PingFang SC",
                 "Noto Sans CJK SC", "Arial Unicode MS"]:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break


setup_cjk_font()
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis("off")

# ---------------- 颜色 ----------------
C_DOC     = ("#d5f5e3", "#1e8449")   # 文档 / 输入：绿色
C_SPLIT   = ("#fef9e7", "#b7950b")   # 切块：黄色
C_EMBED   = ("#d6eaf8", "#1f618d")   # Embedding：蓝色
C_CHROMA  = ("#fdebd0", "#b9770e")   # 向量库 Chroma：橙色
C_RETR    = ("#e8daef", "#7d3c98")   # 检索：紫色
C_GEN     = ("#f2d7d5", "#943126")   # 生成 Qwen：深红
C_ARROW   = "#566573"

BOX_W = 1.8
BOX_H = 1.4
X = [0.3, 2.6, 4.9, 7.2, 9.5]     # 5 列 x 坐标
Y_ROW1 = 6.8                       # 上排（建库）底边 y
Y_ROW2 = 3.4                       # 下排（问答）底边 y


def box(x, y, w, h, text, colors, fontsize=10, lw=1.6, zorder=3):
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1b2631", zorder=zorder + 1, linespacing=1.7)


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=22, linewidth=lw,
                                 color=color, zorder=2))


def row_label(x, y, text, color):
    ax.text(x, y, text, fontsize=12, color=color, ha="center",
            fontweight="bold")


# ---------------- 上排：建库（离线）① ② ③ ④ ----------------
box(X[0], Y_ROW1, BOX_W, BOX_H, "① 文档加载\nPyPDF", C_DOC)
box(X[1], Y_ROW1, BOX_W, BOX_H, "② 文本切块\nRecursiveCharacter\nTextSplitter", C_SPLIT)
box(X[2], Y_ROW1, BOX_W, BOX_H, "③ 向量化\nEmbedding\nbge-small-zh", C_EMBED)
box(X[3], Y_ROW1, BOX_W, BOX_H, "④ 存入向量库\nChroma\n（向量+原文）", C_CHROMA)
row_label((X[0] + X[1]) / 2, Y_ROW1 + 1.7, "建库 · 离线 · 一次搞定", "#1e8449")

for i in range(3):
    arrow(X[i] + BOX_W, Y_ROW1 + BOX_H / 2, X[i + 1], Y_ROW1 + BOX_H / 2)

# ---------------- 下排：问答（在线）⑤ ⑥ ----------------
box(X[0], Y_ROW2, BOX_W, BOX_H, "用户提问\n（自然语言）", C_DOC)
box(X[1], Y_ROW2, BOX_W, BOX_H, "⑤ 问题向量化\nEmbedding", C_EMBED)
box(X[2], Y_ROW2, BOX_W, BOX_H, "检索 Top-K\n余弦相似度", C_RETR)
box(X[3], Y_ROW2, BOX_W, BOX_H, "拼上下文\n模板 09\n[资料§N]", C_SPLIT)
box(X[4], Y_ROW2, BOX_W, BOX_H, "⑥ 生成\nQwen", C_GEN)
row_label((X[0] + X[2]) / 2, Y_ROW2 + 1.7, "问答 · 在线 · 每次提问实时跑", "#7d3c98")

for i in range(4):
    arrow(X[i] + BOX_W, Y_ROW2 + BOX_H / 2, X[i + 1], Y_ROW2 + BOX_H / 2)

# ---------------- 检索与向量库的连接 ----------------
# 从 Chroma（上排第 4 列）引一条虚线到"检索 Top-K"（下排第 3 列）
ax.plot([X[3] + BOX_W / 2, X[2] + BOX_W / 2], [Y_ROW1, Y_ROW2 + BOX_H],
        linestyle="--", color="#b9770e", lw=1.8, zorder=1)
ax.text(X[3] + BOX_W / 2 + 0.15, (Y_ROW1 + Y_ROW2 + BOX_H) / 2 + 0.3,
        "读取向量", fontsize=9, color="#b9770e", rotation=-28)

# ---------------- 答案输出 ----------------
box(X[4], 0.9, BOX_W, 1.2, "答案\n带 [资料§N] 出处", C_GEN, fontsize=10)
arrow(X[4] + BOX_W / 2, Y_ROW2, X[4] + BOX_W / 2, 0.9 + 1.2 + 0.02)

# ---------------- 标题 ----------------
ax.text(6, 9.4, "RAG 全链路结构图（Day 10）", fontsize=18, ha="center",
        fontweight="bold", color="#1b2631")

fig.tight_layout()
out = r".\rag_结构图.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"已生成：{out}")
