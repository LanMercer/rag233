# -*- coding: utf-8 -*-
r"""
plot_compare.py —— "优化前后对比"柱状图（第四周 Day20 第 2 步 · 对应路线图 O3-2）
================================================================================
五件套说明（本脚本在机器学习的哪一环）：
- 模型   ：无（本脚本不加载任何模型，也不连模型服务）
- 数据   ：**只读两个源**——① `第四周\实验日志.md` 的「一、实验记录表」（优化后的轮次）
                      ② 内置常量 `DAY14_BASELINE`（优化前的第三周 day14 首版评测表）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（结果呈现环节）：
    把"同一套 20 题固定评测集"上的**优化前 → 优化后**四组指标画成一张双面板柱状图，
    输出 PNG（默认直接落进 `第四周\发布包\blog\assets\优化前后对比.png`，供博客/简历使用），
    并把数字以 Markdown 表的形式打到终端（方便抄进《优化成果报告》/汇报）。

★ 为什么坚持"从账本读数字"而不是把数字写死在脚本里？（面试会问，也是纪律）
    数字只有一个源（`实验日志.md`）。一旦图上的数字和报告/日志对不上，
    面试官第一个问题就是"你到底以哪个为准"。
    → 所以本脚本**只内置"优化前"那一行**（因为它在第三周 day14，不在第四周日志表里，
      且必须逐字可查），**优化后的行一律从日志表里按 `--optimized-id` 取**；
      取不到就**报错退出**，绝不"猜一个数字画上去"。

用法（llm 环境；本文件所在目录下执行）：
    python plot_compare.py                                    # 默认：day14 基线 vs R3
    python plot_compare.py --optimized-id R6                  # 换成"定稿+引文兜底"那一行
    python plot_compare.py --extra-id R6                      # 加一根"优化后+兜底"作对照
    python plot_compare.py --optimized-id B0r3 --baseline-id B0r3   # 只画基线（自检用）
    python plot_compare.py --out ..\..\_tmp\test.png --no-md  # 不覆盖发布包、不打印表

参数：
    --log            实验日志路径（默认 第四周\实验日志.md）
    --baseline-id    优化前行：`day14`（内置常量）或日志表里的编号（默认 day14）
    --optimized-id   优化后行：日志表里的编号（默认 R3 = 定稿行）
    --extra-id       可选第三条（如 R6），留空则不画
    --out            输出 PNG 路径（默认 第四周\发布包\blog\assets\优化前后对比.png）
    --dpi            清晰度（默认 200；简历用 300）
    --no-md          不打印 Markdown 表
================================================================================
"""

import argparse
import os
import re
import sys

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固（Windows 控制台默认 GBK，打中文/emoji 会崩）
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")  # 不弹窗、直接存文件（服务器/无显示器环境也能跑）
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 第 2 区：路径与常量
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))               # 第四周\day20
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))     # 仓库根
DEFAULT_LOG = os.path.join(REPO_DIR, "第四周", "实验日志.md")
DEFAULT_OUT = os.path.join(REPO_DIR, "第四周", "发布包", "blog", "assets", "优化前后对比.png")

# ★ 优化前（第三周 day14 首版评测表）——**逐字来自 `第三周\day14\首版评测表.md`**，
#   它是"优化前"的唯一出处（那会儿还没有双轨判据、也没有多跑投票）。
#   strict 列写 day14 的"检索命中率"（旧口径 = strict）；loose 列留 None（当时未采集）。
DAY14_BASELINE = {
    "exp_id": "day14",
    "date": "第三周",
    "note": "首版基线（runs=1，旧判据口径）",
    "strict": 20.0,   # 2/10
    "loose": None,    # 当时无此口径
    "gen": 20.0,      # 2/10
    "anti": 60.0,     # 6/10
    "f1": 0.667,      # TP6 FP2 FN4 TN8
    "overall": 40.0,  # 8/20
    "tp_fp_fn_tn": (6, 2, 4, 8),
    "dir": r"第三周\day14",
}

# 四组"百分比"指标（左面板）；F1 单独画（右面板，0~1 量纲不同不能混在一起）
METRIC_KEYS = [
    ("strict", "检索命中率\n(strict)"),
    ("gen", "生成正确率\n(资料内)"),
    ("anti", "防幻觉正确率\n(资料外)"),
    ("overall", "总正确率"),
]
SERIES_NAME = ["优化前", "优化后", "优化后 + 引文兜底"]
SERIES_COLOR = ["#9aa5b1", "#2f7fd1", "#3fa96a"]


# ---------------------------------------------------------------------------
# 第 3 区：解析实验日志的「一、实验记录表」（与 day18\app.py 同一套解析规则）
# ---------------------------------------------------------------------------
def _pct(text):
    """从 '4/10 = 40.0%' 里抠出 40.0；抠不到返回 None。"""
    m = re.search(r"=\s*([0-9.]+)\s*%", str(text))
    if m:
        return float(m.group(1))
    m = re.search(r"([0-9.]+)\s*%", str(text))
    return float(m.group(1)) if m else None


def _f1(text):
    """从 '0.824（TP7 FP0 FN3 TN10）' 里抠出 0.824 与混淆矩阵四元组。"""
    text = str(text)
    m = re.search(r"([01]?\.[0-9]+)", text)
    f1 = float(m.group(1)) if m else None
    m2 = re.search(r"TP\s*(\d+)\s*FP\s*(\d+)\s*FN\s*(\d+)\s*TN\s*(\d+)", text)
    matrix = tuple(int(g) for g in m2.groups()) if m2 else None
    return f1, matrix


def parse_experiment_log(path):
    """读「一、实验记录表」，返回 {编号: 行字典}。解析不了就返回空 dict（不猜）。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    in_table, rows = False, {}
    for line in lines:
        s = line.strip()
        if s.startswith("## 一、实验记录表"):
            in_table = True
            continue
        if in_table and s.startswith("## "):
            break
        if not in_table or not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 11:
            continue
        if set(cells[0]) <= set("-: "):        # 跳过分隔行 |---|---|
            continue
        if cells[1] in ("编号", ""):            # 跳过表头/空行
            continue
        f1, matrix = _f1(cells[8])
        rows[cells[1]] = {
            "date": cells[0],
            "exp_id": cells[1],
            "note": cells[2],
            "config": cells[3],
            "strict": _pct(cells[4]),
            "loose": _pct(cells[5]),
            "gen": _pct(cells[6]),
            "anti": _pct(cells[7]),
            "f1": f1,
            "tp_fp_fn_tn": matrix,
            "overall": _pct(cells[9]),
            "dir": cells[11] if len(cells) > 11 else "",
        }
    return rows


def matrix_from_run_log(result_dir):
    """
    F1 单元格里常常只写 `0.824`（没带矩阵），而**混淆矩阵是 F1 的唯一出处**。
    所以这里回该轮的 `run_log.txt` 里抄 `（TP=7 FP=0 FN=3 TN=10）`——
    这正是 day18 立下的纪律："对外引用的每个 F1 都要回 run_log.txt 对一次矩阵"。
    找不到就返回 None（**不猜**），图上会写"矩阵未记录"。
    """
    if not result_dir:
        return None, None
    for sub in ("day17", "day19", "day16", "day18", ""):
        cand = os.path.join(REPO_DIR, "第四周", sub, result_dir, "run_log.txt") if sub \
            else os.path.join(REPO_DIR, result_dir, "run_log.txt")
        if not os.path.exists(cand):
            continue
        with open(cand, encoding="utf-8", errors="replace") as f:
            text = f.read()
        m = re.search(r"TP=(\d+)\s*FP=(\d+)\s*FN=(\d+)\s*TN=(\d+)", text)
        if m:
            return tuple(int(g) for g in m.groups()), os.path.relpath(cand, REPO_DIR)
    return None, None


def pick_row(rows, exp_id, log_path):
    """按编号取一行；取不到就报错退出（**绝不猜数字**）。"""
    if exp_id == "day14":
        return dict(DAY14_BASELINE)
    if exp_id not in rows:
        ids = "、".join(list(rows.keys())[:20])
        raise SystemExit(
            f"[FAIL] 在 {os.path.relpath(log_path, REPO_DIR)} 的「一、实验记录表」里找不到编号 "
            f"'{exp_id}'。\n       现有编号：{ids}\n"
            f"       请确认该轮实验已用 `--append-log` 写入账本（或把 --optimized-id 换成存在的编号）。"
        )
    return rows[exp_id]


# ---------------------------------------------------------------------------
# 第 4 区：画图
# ---------------------------------------------------------------------------
def setup_font():
    """中文字体：Windows 上按优先级挑一个装了的中文字体，避免方块乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示（用 ASCII 减号）


def draw(series, out_path, dpi):
    """
    series = [(名字, 颜色, 行字典), ...]，第一项=优化前，第二项=优化后，第三项=可选。
    """
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], wspace=0.22)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- 左面板：四组百分比指标（分组柱状图）----
    n = len(series)
    width = 0.8 / n
    x = range(len(METRIC_KEYS))
    for i, (name, color, row) in enumerate(series):
        vals, xs = [], []
        for j, (key, _) in enumerate(METRIC_KEYS):
            v = row.get(key)
            if v is None:
                continue
            vals.append(v)
            xs.append(j + (i - (n - 1) / 2) * width)
        bars = ax1.bar(xs, vals, width=width, label=name, color=color,
                       edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.0f}%",
                     ha="center", va="bottom", fontsize=9, color="#333333")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels([label for _, label in METRIC_KEYS], fontsize=9.5)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("正确率（%）", fontsize=10)
    ax1.set_title("四组指标：优化前 → 优化后", fontsize=12, pad=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=9, frameon=False, loc="upper left")

    # ---- 右面板：防幻觉 F1（0~1，量纲与百分比不同，必须分面板）----
    xs = list(range(n))
    f1s = [row.get("f1") or 0.0 for _, _, row in series]
    bars = ax2.bar(xs, f1s, width=0.55,
                   color=[c for _, c, _ in series], edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, f1s):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10, color="#333333")
    ax2.axhline(0.667, color="#c0392b", linestyle="--", linewidth=1.0)
    ax2.text(n - 0.55, 0.678, "优化前 0.667", fontsize=8, color="#c0392b")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([name for name, _, _ in series], fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("防幻觉 F1（混淆矩阵口径）", fontsize=10)
    ax2.set_title("防幻觉 F1", fontsize=12, pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.set_axisbelow(True)

    # 混淆矩阵写在小字里（F1 同样的值可能来自完全不同的结构 → 必须并列展示）
    lines = []
    for name, _, row in series:
        m = row.get("tp_fp_fn_tn")
        lines.append(f"{name}: {('TP%d FP%d FN%d TN%d' % m) if m else '矩阵未记录'}")
    fig.text(0.795, 0.055, "\n".join(lines), fontsize=7.6, color="#555555", va="bottom")

    # ---- 标题与"可追溯"脚注 ----
    names = " ｜ ".join(f"{name}={row['exp_id']}" for name, _, row in series)
    fig.suptitle("机器人领域 RAG 文档问答系统 · 优化前后对比", fontsize=13.5, y=0.975)
    fig.text(
        0.5, 0.905,
        "同一套 20 题固定评测集（资料内 10 + 资料外 10）｜每题 3 次生成取多数投票（runs=3）｜"
        "k=8 ｜ T=0.2 ｜ 模板 09(v1) ｜ 模型 Qwen2.5-3B-Instruct(+LoRA)",
        ha="center", fontsize=8.6, color="#444444",
    )
    fig.text(
        0.012, 0.012,
        f"数字来源：{names}；优化前行见 第三周\\day14\\首版评测表.md，其余行见 第四周\\实验日志.md。"
        "n=20（95% CI 约 ±0.22）→ 结论看量级与归因，不抠小数点。",
        fontsize=7.6, color="#777777",
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.06, right=0.985)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 第 5 区：把数字打成 Markdown 表（方便抄进报告 / 博客）
# ---------------------------------------------------------------------------
def print_markdown(series):
    print("\n" + "=" * 78)
    print("Markdown 表（可直接粘进《优化成果报告》/ 汇报）")
    print("=" * 78)
    head = ["指标"] + [f"{name}（{row['exp_id']}）" for name, _, row in series]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for key, label in METRIC_KEYS:
        cells = []
        for _, _, row in series:
            v = row.get(key)
            cells.append("——" if v is None else f"{v:.1f}%")
        print(f"| {label.replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
    cells = [("——" if row.get("f1") is None else f"{row['f1']:.3f}") for _, _, row in series]
    print("| 防幻觉 F1（TP/FP/FN/TN） | " + " | ".join(
        f"{c}（{'/'.join(map(str, row['tp_fp_fn_tn'])) if row.get('tp_fp_fn_tn') else '未记录'}）"
        for c, (_, _, row) in zip(cells, series)) + " |")
    print("\n> 口径提醒：优化前那一行是 **runs=1 的旧判据口径**，优化后各行是 **runs=3 + 双轨判据**，"
          "两者**不能严格相减**；要讲得最严谨就用**同口径**的 `B0r3 → R3` 那一对；"
          "而「优化前 40% → 优化后 55%」是**面向 HR 的一句话口径**（已在报告里注明）。")
    print()


# ---------------------------------------------------------------------------
# 第 6 区：入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="优化前后对比柱状图（Day20 O3-2）")
    parser.add_argument("--log", default=DEFAULT_LOG, help="实验日志路径")
    parser.add_argument("--baseline-id", default="day14", help="优化前行编号（day14 或日志表里的编号）")
    parser.add_argument("--optimized-id", default="R3", help="优化后行编号（默认 R3 定稿行）")
    parser.add_argument("--extra-id", default="", help="可选第三行（如 R6）")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出 PNG 路径")
    parser.add_argument("--dpi", type=int, default=200, help="清晰度（简历可用 300）")
    parser.add_argument("--no-md", action="store_true", help="不打印 Markdown 表")
    args = parser.parse_args()

    setup_font()
    rows = parse_experiment_log(args.log)
    if not rows:
        raise SystemExit(f"[FAIL] 没能从 {args.log} 解析出「一、实验记录表」；"
                         f"请确认该小节还在、且表行没有被改成非 Markdown 表格。")

    baseline = pick_row(rows, args.baseline_id, args.log)
    optimized = pick_row(rows, args.optimized_id, args.log)
    series = [(SERIES_NAME[0], SERIES_COLOR[0], baseline),
              (SERIES_NAME[1], SERIES_COLOR[1], optimized)]
    extra = None
    if args.extra_id:
        extra = pick_row(rows, args.extra_id, args.log)
        series.append((SERIES_NAME[2], SERIES_COLOR[2], extra))

    # 补齐混淆矩阵（优先单元格里的；没有就回 run_log.txt 抄）
    for name, _, row in series:
        if row.get("tp_fp_fn_tn") is None:
            matrix, src = matrix_from_run_log(row.get("dir", ""))
            row["tp_fp_fn_tn"] = matrix
            row["matrix_src"] = src or "未找到 run_log.txt"
        else:
            row["matrix_src"] = "内置常量（第三周 day14 首版评测表）"

    print("=" * 78)
    print("优化前后对比（数据源：实验日志 + 第三周 day14 首版评测表）")
    print("=" * 78)
    for name, _, row in series:
        print(f"  {name:<12s} {row['exp_id']:<6s} 日期={row['date']:<6s} "
              f"strict={row['strict']} gen={row['gen']} anti={row['anti']} "
              f"F1={row['f1']} overall={row['overall']}")
        print(f"               说明：{row['note']}")
        print(f"               来源：{row['dir']} ｜ 矩阵来源：{row['matrix_src']}")

    out = draw(series, args.out, args.dpi)
    print(f"\n[OK] 图已保存：{out}")
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"     体积 {size_mb:.2f} MB（发布包预算：<1.5 MB，超了就降 --dpi 或改存 JPG）")
    if not args.no_md:
        print_markdown(series)


if __name__ == "__main__":
    main()
