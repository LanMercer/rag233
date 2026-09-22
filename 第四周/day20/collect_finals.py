# -*- coding: utf-8 -*-
r"""
collect_finals.py —— 从 run_log.txt 抄"定稿数字"并自动校验（第四周 Day20 第 2 步 · O3-1/O3-4）
================================================================================
五件套说明（本脚本在机器学习的哪一环）：
- 模型   ：无（不加载模型、不连服务、不重跑评测——**只读已跑完的产物**）
- 数据   ：**唯一来源 = 各结果目录的 `run_log.txt`「汇总指标」段**（不读截图、不读记忆、不读报告）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（结果呈现 / 交付对账环节）：
    把「定稿需要用到的几轮」的六项指标 + TP/FP/FN/TN + 一致率，**原样抄成一张 Markdown 表**
    （默认落盘 `定稿数字表.md`），并**顺手做三项自动校验**：
        ① F1 与混淆矩阵是否自洽（由 TP/FP/FN/TN 重算 P/R/F1，必须与 log 里的 F1 一致）；
        ② 一致率是否 ≥ 0.93（低于这个值 = 单次数字有随机性，不能对外引用）；
        ③ 该轮是否真的存在（不存在 → 写「待跑」，**绝不猜数字**）。
    这三项正是 day18 那次"把 P 抄成 F1、整列失真"事故的自动化防线。

★ 为什么要有这个脚本？（面试可讲的"工程纪律"）
    day18 的教训：**"多跑投票"防的是随机误差，"照抄混淆矩阵"防的是人为笔误**。
    本项目已经出现过一次"F1 列抄错"（0.889 其实是 P）、一次"同一份报告里三个数字不一致"。
    → 数字只要经过"人手转录"，就会出错。**唯一能自动化的部分：把 log 里的数字机器搬到表里。**
    → 所以本脚本**不做任何计算性改写**（不四舍五入到别的精度、不换算口径），
      只做"抄 + 校验"，且校验失败就**报错退出**（防止把错数字写进定稿）。

用法（llm 环境；本文件所在目录下执行）：
    python collect_finals.py                                  # 跑内置的关键轮次清单
    python collect_finals.py --round R3=day17\result_lora_k8_runs3
    python collect_finals.py --no-write                       # 只打印、不写 定稿数字表.md
    python collect_finals.py --out ..\第四周\定稿数字表.md      # 换输出位置

参数：
    --repo      仓库根目录（默认自动推断到 `大模型算法`）
    --out       输出 Markdown（默认本目录 `定稿数字表.md`）
    --round     追加/覆盖一轮：`编号=相对仓库根的目录`（可重复；覆盖内置同名编号）
    --no-write  只打印
================================================================================
"""

import argparse
import io
import os
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固（Windows 控制台默认 GBK，打中文会崩）
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # 第四周\day20
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))  # 仓库根

# ---------------------------------------------------------------------------
# 第 2 区：内置的"定稿关键轮次"清单
#   为什么是这几轮：它们是**要写进报告/博客/简历**的数字，其余轮次（消融、失败尝试）
#   引用时只需在《优化成果报告》里出现，不必进"定稿表"。
# ---------------------------------------------------------------------------
BUILTIN_ROUNDS = [
    ("R3",   r"第四周\day17\result_lora_k8_runs3",      "定稿行（qi=off + k=8，唯一 FP=0）"),
    ("R6",   r"第四周\day19\result_cite_strip_runs3",   "定稿行 + 引文白名单 strip（零副作用兜底）"),
    ("F3v2", r"第四周\day19\result_lora_v2_runs3",      "F3 负结果：真实 201 条（只作对照引用）"),
    ("F3v3", r"第四周\day20\result_lora_v3_runs3",      "F3 拆变量轮：真实 201 + 合成 99 = 300"),
    ("F4o",  r"第四周\day20\result_original_k8_runs3",  "F4 原模型（未微调）同评测集"),
    ("F4l",  r"第四周\day17\result_lora_k8_runs3",      "F4 微调模型侧（同 R3，检索层应逐位相同）"),
]

# 从 run_log.txt 里抠数字的正则（与 eval_v2.py 打印格式一一对应）
PAT = {
    "head":  re.compile(r"RAG 评测 v2\.1 开始｜profile=(\S+)｜TOP_K=(\d+)｜temp=([\d.]+)｜runs=(\d+)｜template=(\w+)"),
    "strict": re.compile(r"① 检索命中 strict\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%"),
    "loose":  re.compile(r"① 检索命中 loose\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%"),
    "gen":    re.compile(r"② 生成正确率\(in\)\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%"),
    "anti":   re.compile(r"③ 防幻觉正确率\(out\)\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%"),
    "overall": re.compile(r"④ 总正确率\s*:\s*([\d.]+)%（(\d+)/(\d+)）"),
    "f1":     re.compile(r"⑤ 防幻觉F1\s*:\s*P=([\d.]+)\s*R=([\d.]+)\s*F1=([\d.]+)\s*（TP=(\d+)\s*FP=(\d+)\s*FN=(\d+)\s*TN=(\d+)）"),
    "consist": re.compile(r"⑥ 生成一致率\s*:\s*([\d.]+)"),
    "cite_raw": re.compile(r"⑧ 引文合法性\s*:\s*引文\s*(\d+)\s*处（(\d+)\s*题带引文），非法\s*(\d+)\s*处"),
    "cite_deliv": re.compile(r"交付口径\s*:\s*policy=(\w+)｜引文\s*(\d+)\s*处，非法\s*(\d+)\s*处"),
}

CONSIST_FLOOR = 0.93      # 一致率门槛（低于它 = 数字有随机性，不能对外引用）


def parse_run_log(path):
    """读一个 run_log.txt，返回 (六项汇总 dict, 校验问题 list)。解析不到的字段 = None（不猜）。"""
    text = io.open(path, encoding="utf-8", errors="replace").read()
    out, problems = {}, []

    m = PAT["head"].search(text)
    if m:
        out["profile"], out["top_k"], out["temp"], out["runs"], out["template"] = (
            m.group(1), int(m.group(2)), float(m.group(3)), int(m.group(4)), m.group(5))
    else:
        problems.append("没找到抬头的 `RAG 评测 v2.1 开始｜...` 那行（口径列会是空的）")

    for key in ("strict", "loose", "gen", "anti"):
        m = PAT[key].search(text)
        if m:
            out[key] = int(m.group(1))
            out[key + "_den"] = int(m.group(2))
            out[key + "_pct"] = float(m.group(3))
        else:
            out[key] = out[key + "_den"] = out[key + "_pct"] = None
            problems.append(f"缺 `{key}` 汇总行")

    m = PAT["overall"].search(text)
    if m:
        out["overall_pct"], out["overall"], out["overall_den"] = float(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        out["overall_pct"] = out["overall"] = out["overall_den"] = None
        problems.append("缺 `④ 总正确率` 汇总行")

    m = PAT["f1"].search(text)
    if m:
        out["P"], out["R"], out["F1"] = float(m.group(1)), float(m.group(2)), float(m.group(3))
        out["TP"], out["FP"], out["FN"], out["TN"] = (int(m.group(i)) for i in (4, 5, 6, 7))
    else:
        for k in ("P", "R", "F1", "TP", "FP", "FN", "TN"):
            out[k] = None
        problems.append("缺 `⑤ 防幻觉F1` 汇总行（**没有它就不能对外引用 F1**）")

    m = PAT["consist"].search(text)
    out["consist"] = float(m.group(1)) if m else None
    if m is None:
        problems.append("缺 `⑥ 生成一致率` 汇总行（runs=1 的轮次本来就没有，可忽略）")

    m = PAT["cite_raw"].search(text)
    m2 = PAT["cite_deliv"].search(text)
    if m:
        out["cite_n"], out["cite_q"], out["cite_bad"] = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if m2:
        out["deliv_policy"], out["deliv_n"], out["deliv_bad"] = m2.group(1), int(m2.group(2)), int(m2.group(3))

    # ---- 校验①：F1 与混淆矩阵自洽（day18 事故的自动化防线）----
    if out.get("TP") is not None:
        tp, fp, fn, tn = out["TP"], out["FP"], out["FN"], out["TN"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out["_p_calc"], out["_r_calc"], out["_f1_calc"] = p, r, f1
        ok_p = abs(p - out["P"]) <= 0.0015
        ok_r = abs(r - out["R"]) <= 0.0015
        ok_f1 = abs(f1 - out["F1"]) <= 0.0015
        out["_selfcheck"] = bool(ok_p and ok_r and ok_f1)
        if not out["_selfcheck"]:
            problems.append(f"F1 与矩阵**不自洽**：由 TP/FP/FN/TN 重算应为 "
                            f"P={p:.3f} R={r:.3f} F1={f1:.3f}，但 log 写的是 "
                            f"P={out['P']} R={out['R']} F1={out['F1']}（**别抄，先查 log**）")
    else:
        out["_selfcheck"] = None

    # ---- 校验②：一致率门槛 ----
    if out.get("consist") is not None and out["consist"] < CONSIST_FLOOR:
        problems.append(f"一致率 {out['consist']} < {CONSIST_FLOOR} → **该轮数字有随机性，不宜对外引用**")

    return out, problems


def fmt_int(row, key):
    if row.get(key) is None:
        return "待跑"
    return f"{row[key]}/{row[key + '_den']}"


def fmt_pct(row, key):
    if row.get(key + "_pct") is None:
        return "待跑"
    return f"{row[key + '_pct']:.1f}%"


def fmt_f1(row):
    if row.get("F1") is None:
        return "待跑"
    return f"**{row['F1']:.3f}**（{row['TP']}/{row['FP']}/{row['FN']}/{row['TN']}）"


def fmt_matrix_pct(row):
    """把混淆矩阵也写成 4 个百分比（报告里常要"四个格子"的口径）。"""
    if row.get("TP") is None:
        return "待跑"
    total = row["TP"] + row["FP"] + row["FN"] + row["TN"]
    return (f"TP {row['TP']} / FP {row['FP']} / FN {row['FN']} / TN {row['TN']}"
            f"（共 {total} 题 = 资料内 10 + 资料外 10）")


def main():
    ap = argparse.ArgumentParser(description="从 run_log.txt 抄定稿数字 + 自动校验")
    ap.add_argument("--repo", default=REPO_DIR, help="仓库根目录")
    ap.add_argument("--out", default=os.path.join(SCRIPT_DIR, "定稿数字表.md"), help="输出 Markdown 路径")
    ap.add_argument("--round", action="append", default=[], metavar="编号=相对路径",
                    help="追加/覆盖一轮（可重复）")
    ap.add_argument("--no-write", action="store_true", help="只打印、不写文件")
    args = ap.parse_args()

    rounds = list(BUILTIN_ROUNDS)
    for spec in args.round:
        if "=" not in spec:
            raise SystemExit(f"[FAIL] --round 需要写成 `编号=相对路径`，收到：{spec}")
        exp_id, rel = spec.split("=", 1)
        rounds = [(e, r, l) for (e, r, l) in rounds if e != exp_id]
        rounds.append((exp_id.strip(), rel.strip(), "（命令行指定）"))

    print("=" * 88)
    print("定稿数字对账（唯一来源：各结果目录的 run_log.txt「汇总指标」段）")
    print("=" * 88)

    rows, problems_all, missing = [], [], []
    for exp_id, rel, label in rounds:
        log_path = os.path.join(args.repo, rel, "run_log.txt")
        if not os.path.exists(log_path):
            missing.append((exp_id, rel))
            print(f"\n[{exp_id}] ⚠ 待跑：{rel}\\run_log.txt 不存在 → 表里写「待跑」（不猜数字）")
            rows.append({"exp_id": exp_id, "dir": rel, "label": label, "date": "待跑", "todo": True})
            continue

        row, problems = parse_run_log(log_path)
        row.update({"exp_id": exp_id, "dir": rel, "label": label, "todo": False})
        row["date"] = datetime.fromtimestamp(os.path.getmtime(log_path)).strftime("%Y-%m-%d")
        rows.append(row)
        problems_all += [(exp_id, p) for p in problems]

        print(f"\n[{exp_id}] {rel}")
        print(f"    口径：profile={row.get('profile')} k={row.get('top_k')} T={row.get('temp')} "
              f"runs={row.get('runs')} 模板={row.get('template')}｜日期（log 时间戳）={row['date']}")
        print(f"    检索 strict={fmt_int(row, 'strict')} loose={fmt_int(row, 'loose')}｜"
              f"生成={fmt_int(row, 'gen')}｜防幻觉={fmt_int(row, 'anti')}｜"
              f"F1={fmt_f1(row)}｜总={fmt_int(row, 'overall')}｜一致率={row.get('consist')}")
        if row.get("_selfcheck") is True:
            print(f"    ✅ 校验① F1 与矩阵自洽（P={row['_p_calc']:.3f} R={row['_r_calc']:.3f} "
                  f"F1={row['_f1_calc']:.3f}）")
        elif row.get("_selfcheck") is False:
            print("    ❌ 校验① F1 与矩阵**不自洽** → 见下方问题清单")

    print("\n" + "=" * 88)
    print("校验结果汇总")
    print("=" * 88)
    if problems_all:
        for exp_id, p in problems_all:
            print(f"  ❌ [{exp_id}] {p}")
    else:
        print("  ✅ 三项校验全过：① F1↔矩阵自洽 ② 一致率达标 ③ 轮次存在")
    if missing:
        print(f"  ⏳ 待跑的轮次 {len(missing)} 个：" +
              "、".join(f"{e}（{r}）" for e, r in missing))

    # ---- 生成 Markdown ----
    lines = []
    lines.append("# 定稿数字表（Day20 · 机器抄录，唯一来源 run_log.txt）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} ｜ 生成脚本：`第四周\\day20\\collect_finals.py`\n")
    lines.append("> **口径**：同一 20 题固定评测集（资料内 10 + 资料外 10）｜k=8｜`T=0.2`｜`runs=3` 多数投票｜"
                 "双轨检索判据（strict = 召回 `source_chunk`；loose = 期望关键词落在 Top-K 上下文）｜"
                 "模型 Qwen2.5-3B-Instruct(+LoRA)。\n")
    lines.append("> **纪律**：本表由脚本从 `run_log.txt` 抄录并**自动校验 F1↔混淆矩阵自洽**；"
                 "报告/博客/简历里的数字必须与本表一致；**表里写「待跑」的，说明还没跑，绝不用别的数字顶替**。\n")
    lines.append("")
    lines.append("| 轮次 | 说明 | 口径（profile/k/runs/模板） | 日期 | 检索 strict | 检索 loose | 生成(in) | 防幻觉(out) | F1（TP/FP/FN/TN） | 总正确率 | 一致率 | 结果目录 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        if row.get("todo"):
            lines.append(f"| **{row['exp_id']}** | {row['label']} | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | `{row['dir']}` |")
            continue
        cfg = f"{row.get('profile')} / k={row.get('top_k')} / runs={row.get('runs')} / {row.get('template')}"
        lines.append(
            f"| **{row['exp_id']}** | {row['label']} | {cfg} | {row['date']} | "
            f"{fmt_int(row, 'strict')} = {fmt_pct(row, 'strict')} | "
            f"{fmt_int(row, 'loose')} = {fmt_pct(row, 'loose')} | "
            f"{fmt_int(row, 'gen')} = {fmt_pct(row, 'gen')} | "
            f"{fmt_int(row, 'anti')} = {fmt_pct(row, 'anti')} | "
            f"{fmt_f1(row)} | {fmt_int(row, 'overall')} = {fmt_pct(row, 'overall')} | "
            f"{row.get('consist')} | `{row['dir']}` |")

    lines.append("")
    lines.append("## 混淆矩阵细目（报告里的「四个格子」口径）\n")
    lines.append("| 轮次 | 混淆矩阵 | F1 自洽校验 | 一致率门槛（≥0.93） |")
    lines.append("|---|---|---|---|")
    for row in rows:
        if row.get("todo"):
            lines.append(f"| **{row['exp_id']}** | 待跑 | 待跑 | 待跑 |")
            continue
        chk = {True: "✅ 自洽", False: "❌ 不自洽", None: "—（无矩阵）"}[row.get("_selfcheck")]
        cons = row.get("consist")
        cons_txt = "—" if cons is None else (f"{cons} ✅" if cons >= CONSIST_FLOOR else f"{cons} ⚠ 低于门槛")
        lines.append(f"| **{row['exp_id']}** | {fmt_matrix_pct(row)} | {chk} | {cons_txt} |")

    if any(r.get("cite_n") is not None for r in rows):
        lines.append("")
        lines.append("## 引文合法性（O2-G3，`⑧` 行）\n")
        lines.append("| 轮次 | 原始：引文/非法 | 交付：引文/非法 | 兜底策略 |")
        lines.append("|---|---|---|---|")
        for row in rows:
            if row.get("cite_n") is None:
                continue
            if row.get("deliv_n") is not None:
                deliv = f"{row['deliv_n']} 处 / {row['deliv_bad']} 处"
                policy = f"`{row['deliv_policy']}`"
            else:
                deliv = "—（该轮未采集交付口径）"
                policy = "`record`（无兜底）"
            lines.append(
                f"| **{row['exp_id']}** | {row['cite_n']} 处 / {row['cite_bad']} 处 | "
                f"{deliv} | {policy} |")

    lines.append("")
    lines.append("## 校验问题清单\n")
    if problems_all:
        for exp_id, p in problems_all:
            lines.append(f"- ❌ **[{exp_id}]** {p}")
    else:
        lines.append("- ✅ 无（① F1↔矩阵自洽 ② 一致率 ≥0.93 ③ 轮次均存在）")
    if missing:
        lines.append(f"- ⏳ 待跑：{'、'.join(e for e, _ in missing)}"
                     f"（跑完再执行一次本脚本即可自动补进上表）")

    lines.append("")
    lines.append("## 回填去处（数字一变，这几处一起改）\n")
    lines.append("1. `第四周\\day20\\优化成果报告.md`（定稿：表一 / 表一补充 / §六 / §七 / §5.5 收尾清单）")
    lines.append("2. `第四周\\发布包\\blog\\项目展示页.md`（A 线文章里的 `__`）")
    lines.append("3. `第三周\\day15\\README.md`（§三 效果评估的「优化后」表 + §五 评测账）、"
                 "`第三周\\day15\\效果评估报告.md`（§2.2 + §六）、"
                 "`大模型算法学习成果汇报.md`（4.1 效果量化 + 成果表）")
    lines.append("4. 简历项目段（P4 结果句）与 `面试追问自测表.md` / `P线-叙事与复述.md` 里的数字")
    lines.append("5. `第四周\\day20\\week4工作汇报.md`（周汇总的量化结果表）")
    lines.append("")
    lines.append("> ⚠ **不允许**在报告/博客/简历里出现本表之外的数字（尤其「看起来很齐」的整数）；"
                 "口径不同（如 day14 的 runs=1 旧判据）时**必须标注口径**，不能与 runs=3 直接相减。")
    md = "\n".join(lines) + "\n"

    if args.no_write:
        print("\n[--no-write] 未写文件。Markdown 预览：\n")
        print(md)
    else:
        io.open(args.out, "w", encoding="utf-8", newline="").write(md)
        print(f"\n[OK] 已写出：{args.out}")

    # 校验不通过 → 非零退出（防止"错数字"被当成定稿）
    hard = [p for _, p in problems_all if "不自洽" in p]
    if hard:
        print("\n[FAIL] 存在 F1↔矩阵不自洽的轮次 → 请回 run_log.txt 核对后再定稿（本脚本以非零码退出）")
        sys.exit(1)


if __name__ == "__main__":
    main()
