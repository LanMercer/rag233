# -*- coding: utf-8 -*-
r"""
Day 13 配套脚本：自动绘制"机器人行业垂直领域 RAG 问答系统"项目架构图
========================================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第三周\day13"
    python draw_project_structure.py

输出：当前文件夹下生成 项目架构图.png
依赖：matplotlib（未安装的话先执行  pip install matplotlib）

说明：本图把 Day13 定型的"四层架构"画成一张图——
    ① 检索层（rag_demo/）：知识库（Chroma 向量库）→ 余弦相似度 Top-K 检索；
    ② 答案组织层（模板库 v2）：模板 09 拼上下文 [资料§N]（只依据资料 + 引用出处）；
    ③ 生成层（Qwen2.5-3B-Instruct）：把提示词生成答案；
    ④ 风格/能力层（微调 LoRA adapter，Day12 产物）：W0 + B·A 合入生成层。
    右侧两个侧挂盒子：离线建库（GMR 论文 → 切块 → Embedding → Chroma）与 风格层，
    用虚线箭头表示它们"喂养"主链路。
    参考 day10 的 draw_rag_structure.py 的绘制风格改写。
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

fig, ax = plt.subplots(figsize=(13, 10))
ax.set_xlim(0, 13)
ax.set_ylim(0, 10)
ax.axis("off")

# ---------------- 颜色 ----------------
C_QUERY  = ("#d5f5e3", "#1e8449")   # 用户提问 / 答案：绿色
C_RETR   = ("#d6eaf8", "#1f618d")   # 检索层：蓝色
C_TMPL   = ("#fef9e7", "#b7950b")   # 答案组织层：黄色
C_GEN    = ("#f2d7d5", "#943126")   # 生成层：深红
C_STYLE  = ("#e8daef", "#7d3c98")   # 风格/能力层（微调）：紫色
C_BUILD  = ("#fdebd0", "#b9770e")   # 离线建库：橙色
C_ARROW  = "#566573"

MAIN_W, MAIN_H = 6.0, 1.5          # 主链路盒子尺寸
SIDE_W, SIDE_H = 3.4, 1.7          # 侧挂盒子尺寸
MAIN_X, MAIN_CX = 0.6, 3.6         # 主盒子左下角 x / 中心 x
SIDE_X = 9.0                       # 侧挂盒子左下角 x

def box(x, y, w, h, text, colors, fontsize=10, lw=1.6, zorder=3):
    face, edge = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1b2631", zorder=zorder + 1, linespacing=1.8)


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=24, linewidth=lw,
                                 color=color, zorder=2))


def dashed(x1, y1, x2, y2, color, lw=1.8):
    ax.plot([x1, x2], [y1, y2], linestyle="--", color=color, lw=lw, zorder=1)


# ---------------- 标题 ----------------
ax.text(6.5, 9.55, "机器人行业垂直领域 RAG 问答系统 · 项目架构图（Day 13）",
        fontsize=17, ha="center", fontweight="bold", color="#1b2631")

# ---------------- 顶部：用户提问 ----------------
box(MAIN_X + 1.0, 8.2, MAIN_W - 2.0, 0.9, "用户提问（自然语言）", C_QUERY, fontsize=11)

# ---------------- 主链路：检索层 → 答案组织层 → 生成层 ----------------
# ① 检索层
box(MAIN_X, 6.3, MAIN_W, MAIN_H, "① 检索层 · rag_demo/\n"
    "知识库（Chroma 向量库）→ 余弦相似度 Top-K 检索", C_RETR, fontsize=10)
# ② 答案组织层
box(MAIN_X, 4.1, MAIN_W, MAIN_H, "② 答案组织层 · 模板库 v2\n"
    "模板 09 拼上下文 [资料§N]（只依据资料 + 引用出处）", C_TMPL, fontsize=10)
# ③ 生成层
box(MAIN_X, 1.9, MAIN_W, MAIN_H, "③ 生成层 · Qwen2.5-3B-Instruct\n"
    "生成答案（权重 = W0 原权重）", C_GEN, fontsize=10)

# ---------------- 底部：答案 ----------------
box(MAIN_X + 0.5, 0.35, MAIN_W - 1.0, 0.95, "答案（带 [资料§N] 出处）", C_QUERY, fontsize=11)

# ---------------- 侧挂盒子 ----------------
# 离线建库（喂给检索层）
box(SIDE_X, 6.3, SIDE_W, SIDE_H, "离线建库 · 一次搞定\n"
    "GMR 论文 PDF → 切块\n→ bge Embedding → Chroma", C_BUILD, fontsize=9)
# 风格/能力层（微调 adapter，挂到生成层）
box(SIDE_X, 1.9, SIDE_W, SIDE_H, "④ 风格/能力层 · 微调\n"
    "LoRA adapter（Day12）\nW0 + B·A 增量", C_STYLE, fontsize=9)

# ---------------- 箭头：主链路（纵向） ----------------
arrow(MAIN_CX, 8.2, MAIN_CX, 7.8)                # 用户提问 → 检索层
arrow(MAIN_CX, 6.3, MAIN_CX, 5.6)                # 检索层 → 答案组织层
arrow(MAIN_CX, 4.1, MAIN_CX, 3.4)                # 答案组织层 → 生成层
arrow(MAIN_CX, 1.9, MAIN_CX, 1.3)                # 生成层 → 答案

# ---------------- 箭头：侧挂 → 主链路（虚线） ----------------
dashed(SIDE_X, 7.1, MAIN_X + MAIN_W, 7.0, "#b9770e")   # 离线建库 → 检索层
ax.text(8.0, 7.25, "建库（离线）", fontsize=9, color="#b9770e", ha="center")
dashed(SIDE_X, 2.7, MAIN_X + MAIN_W, 2.6, "#7d3c98")   # 风格层 → 生成层
ax.text(8.0, 2.9, "合入权重 W0+B·A", fontsize=9, color="#7d3c98", ha="center")

# ---------------- 层标签（右侧两盒子之间标"在线链路"） ----------------
ax.text(10.7, 4.95, "在线链路 · 每次问答实时跑", fontsize=10, color="#943126",
        ha="center", fontweight="bold")

fig.tight_layout()
out = r".\项目架构图.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"已生成：{out}")
