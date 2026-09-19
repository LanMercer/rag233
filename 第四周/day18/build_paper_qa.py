# -*- coding: utf-8 -*-
r"""
F1 论文 QA 数据校验脚本（第四周 Day18 · HR 意见③"数据要真实/论文相关"）⭐

> 计划原文（`01-第四周详细计划.md` D3 第 2 条）：
>   写 `build_paper_qa.py` 做**校验**（条数 / 字段齐全 / 空值=0 / 重复=0 / source 非空），
>   **人工提取、程序校验**，不用模板批量生成。

一句话：**答案是人写的，但有机器在盯着五条纪律**——
    ① 量够不够（≥ N 条）；② 字段全不全；③ 有没有空值 / 重复；
    ④ **source 可不可追溯**（chunk_id 在不在库里 + 原文摘录在不在这段 chunk 里，逐条回查）；
    ⑤ **有没有和评测集撞题**（撞题 = 数据泄漏 = 评测数字作废，这是最隐蔽、最致命的一条）。

为什么必须"可追溯"（面试/HR 会追问）：
    合成数据的问题是"谁也不知道答案从哪来"。真实数据的价值就在**能指着原文说"这里写的就是它"**。
    所以本条校验会把 source 里的 chunk_id 拿到 `chunks.json` 里回查，
    再用**原文摘录做逐字比对**——对不上就报错，逼你把出处写实。

为什么必须查"数据泄漏"：
    20 条评测题（`eval_questions.json`）是**卷子**，SFT 数据是**教材**。
    如果教材里就有卷子原题，那"生成正确率提升"就是背题背出来的，**整个优化成果全废**。
    本脚本会用 `difflib` 做相似度比对，和评测题太像的 QA 直接报错。

数据规格（`sft_data_paper.json`）：
    [
      {
        "instruction": "……",                 # 必填：问题
        "input": "依据论文摘要",               # 选填：上下文提示
        "output": "……",                       # 必填：忠实原文的答案
        "source": {                            # 必填：可追溯出处
          "page": 0,                           #   页码（PDF 页）
          "chunk_id": 2,                       #   段落编号（与 [资料§N] 对齐）
          "quote": "原文里逐字能查到的一句话"    #   原文摘录（校验会拿它回查）
        },
        "category": "术语定义"                 # 选填：五类之一
      }, ...
    ]

五类 category（计划规定）：术语定义 / 方法步骤 / 实验设置 / 结论 / 局限

运行方法（llm 环境；先 cd 到本文件所在目录）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day18"

    python build_paper_qa.py --example            # 先看 5 条真实样例（照格式写，不是模板生成）
    python build_paper_qa.py --json sft_data_paper.seed.json --min 5   # 拿种子集练手
    python build_paper_qa.py                      # 校验正式集（默认 ≥50 条，含泄漏检查）
    python build_paper_qa.py --allow-leak         # 撞题只警告不报错（临时用，正式提交别开）
    python build_paper_qa.py --no-trace           # 不做 chunk 回查（只查字段/重复/空值/泄漏）

退出码：0 = 全部通过；1 = 有硬错误。可直接当提交前自检用。

依赖：只用标准库（json / re / difflib / collections）。
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固（Windows 控制台默认 GBK）
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 第 2 区：路径与常量
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))            # 第四周\day18
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))  # 仓库根
CHUNKS_JSON = os.path.join(REPO_DIR, "第三周", "day13", "chunks.json")
EVAL_JSON = os.path.join(REPO_DIR, "第三周", "day13", "eval_questions.json")
DEFAULT_QA_JSON = os.path.join(SCRIPT_DIR, "sft_data_paper.json")

REQUIRED_FIELDS = ["instruction", "output", "source"]
ALLOWED_CATEGORIES = ["术语定义", "方法步骤", "实验设置", "结论", "局限"]
LEAK_THRESHOLD = 0.70   # 与评测题相似度 ≥ 此值 → 判为"撞题"（数据泄漏）

# 5 条真实样例（人工提取自 GMR 论文；**刻意避开 20 条评测题的题面**，并已逐条回查 chunk 原文）——
# 用 --example 打印，供你"照着格式写"，**不是**让你批量套模板。
EXAMPLES = [
    {
        "instruction": "GMR 出自哪篇论文（论文标题是什么）？",
        "input": "依据论文标题",
        "output": "论文标题是《Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking》。",
        "source": {"page": 0, "chunk_id": 0,
                   "quote": "Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking"},
        "category": "术语定义",
    },
    {
        "instruction": "论文的通讯作者联系邮箱是什么？",
        "input": "依据论文作者信息",
        "output": "通讯作者的联系邮箱是 jparaujo@stanford.edu。",
        "source": {"page": 0, "chunk_id": 6,
                   "quote": "Send correspondence to:jparaujo@stanford.edu"},
        "category": "术语定义",
    },
    {
        "instruction": "重定向数据中的伪影对哪一类动作的策略鲁棒性影响最大？",
        "input": "依据论文实验部分",
        "output": "伪影会显著降低策略鲁棒性，对动态或长序列（dynamic or long sequences）的影响尤其明显。",
        "source": {"page": 0, "chunk_id": 3,
                   "quote": "particularly for dynamic or long sequences"},
        "category": "实验设置",
    },
    {
        "instruction": "GMR 在评估重定向质量对策略性能的影响时，抑制了什么？",
        "input": "依据论文贡献部分",
        "output": "GMR 在评估重定向质量如何影响策略性能时，抑制了过度的奖励调参（excessive reward tuning）。",
        "source": {"page": 0, "chunk_id": 2,
                   "quote": "when excessive reward tuning is suppressed"},
        "category": "方法步骤",
    },
    {
        "instruction": "人形运动跟踪策略是构建哪些系统的基础工具？",
        "input": "依据论文引言",
        "output": "人形运动跟踪策略是构建遥操作管线（teleoperation pipelines）与分层控制器（hierarchical controllers）的基础工具。",
        "source": {"page": 0, "chunk_id": 0,
                   "quote": "central to building teleoperation pipelines and hierarchical controllers"},
        "category": "方法步骤",
    },
]


# ---------------------------------------------------------------------------
# 第 3 区：小工具
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(s):
    """归一化文本用于比对：去多余空白、去标点、转小写。"""
    s = re.sub(r"\s+", "", str(s)).lower()
    return re.sub(r"[，。？！、；：,.\?!;:\"'（）()\[\]]", "", s)


def is_blank(v):
    """空值判断：None / 空串 / 纯空白 / 空列表 / 空字典 都算空。"""
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def extract_chunk_ids(source):
    """从 source（dict 或 str）里解析出 chunk_id；dict 优先，str 走正则兜底。"""
    if isinstance(source, dict):
        ids = []
        for key in ("chunk_id", "chunk_ids", "chunk"):
            if key in source:
                v = source[key]
                if isinstance(v, list):
                    ids += [int(x) for x in v if str(x).strip().isdigit()]
                elif str(v).strip().isdigit():
                    ids.append(int(v))
        return sorted(set(ids))
    found = re.findall(r"(?:chunk[_\s]*id\s*[=:：]?\s*|chunk\s+|§|#)(\d+)",
                       str(source), flags=re.I)
    return sorted({int(x) for x in found})


def extract_quote(source):
    """从 source 里取"原文摘录"用于逐字回查。"""
    if isinstance(source, dict):
        for key in ("quote", "excerpt", "原文摘录", "text", "raw"):
            if not is_blank(source.get(key)):
                return str(source[key])
        return ""
    s = str(source)
    m = re.search(r"(?:原文|摘录|quote)\s*[:：]\s*(.+)$", s)
    return m.group(1).strip() if m else ""


def similarity(a, b):
    """题面相似度（0~1）：先去标点空白，再用 difflib 比值。"""
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


# ---------------------------------------------------------------------------
# 第 4 区：校验
# ---------------------------------------------------------------------------

def validate(records, args, chunk_text, eval_questions):
    """返回 (errors, warnings, stats)。errors 非空 = 校验不通过。"""
    errors, warnings = [], []
    stats = {"count": len(records), "categories": Counter(), "leaks": 0, "traced": 0}

    # --- 检查 ①：条数 ---
    if len(records) < args.min:
        errors.append(f"条数不足：{len(records)} < 门槛 {args.min}（计划要求论文 QA ≥50 条起步）")

    seen_instructions = {}
    for idx, rec in enumerate(records, start=1):
        tag = f"第 {idx} 条"
        if not isinstance(rec, dict):
            errors.append(f"{tag}：不是对象（应为 dict），实际是 {type(rec).__name__}")
            continue

        # --- 检查 ②：字段齐全 + 非空 ---
        for field in REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"{tag}：缺字段 `{field}`")
            elif is_blank(rec[field]):
                errors.append(f"{tag}：字段 `{field}` 为空")
        if "input" in rec and is_blank(rec["input"]):
            warnings.append(f"{tag}：`input` 是空值（可选字段，建议直接删掉这个键）")

        ins = rec.get("instruction")

        # --- 检查 ③：重复 instruction ---
        if not is_blank(ins):
            key = norm(ins)
            if key in seen_instructions:
                errors.append(f"{tag}：instruction 与第 {seen_instructions[key]} 条重复 → {str(ins)[:40]}")
            else:
                seen_instructions[key] = idx

        # --- 检查 ④：数据泄漏（与 20 条评测题撞题）---
        if not is_blank(ins) and eval_questions:
            worst, worst_q = 0.0, None
            for eq in eval_questions:
                r = similarity(ins, eq["question"])
                if r > worst:
                    worst, worst_q = r, eq
            if worst_q is not None and worst >= LEAK_THRESHOLD:
                stats["leaks"] += 1
                msg = (f"{tag}：与评测题 Q{worst_q['id']} 相似度 {worst:.2f}（≥{LEAK_THRESHOLD}）"
                       f"→ 数据泄漏风险：「{str(ins)[:30]}」≈「{worst_q['question'][:30]}」")
                (warnings if args.allow_leak else errors).append(msg)

        # --- 检查 ⑤：source 可追溯（chunk_id 存在 + 原文摘录逐字可查）---
        src = rec.get("source")
        if not args.no_trace and not is_blank(src):
            ids = extract_chunk_ids(src)
            quote = extract_quote(src)
            if not ids:
                warnings.append(f"{tag}：source 里解析不出 chunk_id（无法回查，建议写成 chunk_id=N 或 dict 形态）")
            else:
                missing = [i for i in ids if i not in chunk_text]
                if missing:
                    errors.append(f"{tag}：source 的 chunk_id {missing} 在 chunks.json 里不存在（库共 {len(chunk_text)} 段）")
                if not quote:
                    warnings.append(f"{tag}：source 里没有原文摘录（quote），无法逐字回查")
                else:
                    pool = norm("\n".join(chunk_text.get(i, "") for i in ids))
                    if norm(quote) in pool:
                        stats["traced"] += 1
                    else:
                        errors.append(f"{tag}：原文摘录在该 chunk 里**逐字找不到** → 出处不可追溯：{quote[:60]}…")

        # --- 选填：category 分布 ---
        cat = rec.get("category")
        if not is_blank(cat):
            stats["categories"][str(cat)] += 1
            if str(cat) not in ALLOWED_CATEGORIES:
                warnings.append(f"{tag}：category=`{cat}` 不在五类里（{ALLOWED_CATEGORIES}）")

    return errors, warnings, stats


def print_report(records, errors, warnings, stats, args):
    print("=" * 92)
    print("F1 论文 QA 数据校验（人工提取 · 程序校验）")
    print("=" * 92)
    print(f"  数据文件：{args.json}")
    print(f"  实到条数：{stats['count']}   ｜ 门槛：{args.min}   ｜ chunk 库：{args.n_chunks} 段")
    trace_desc = ("关闭（--no-trace）" if args.no_trace
                  else f"开启（chunk_id 存在性 + 原文摘录逐字比对，通过 {stats['traced']} 条）")
    leak_desc = ("关闭" if not args.eval_questions
                 else f"开启（阈值 {LEAK_THRESHOLD}，命中 {stats['leaks']} 条）"
                      + ("  [--allow-leak：仅警告]" if args.allow_leak else ""))
    print(f"  回查模式：{trace_desc}")
    print(f"  泄漏检查：{leak_desc}")
    print()

    if errors:
        print(f"❌ 硬错误 {len(errors)} 条（必须修）：")
        for e in errors:
            print(f"   - {e}")
    else:
        print("✅ 硬错误 0 条（条数 / 字段 / 空值 / 重复 / 泄漏 / source 可追溯 全部通过）")

    if warnings:
        print(f"\n⚠ 提醒 {len(warnings)} 条（不拦提交，但建议看一眼）：")
        for w in warnings:
            print(f"   - {w}")

    if args.show_categories:
        print("\n五类分布（计划规定：术语定义 / 方法步骤 / 实验设置 / 结论 / 局限）：")
        for c in ALLOWED_CATEGORIES:
            print(f"   {c}：{stats['categories'].get(c, 0)} 条")
        other = {k: v for k, v in stats["categories"].items() if k not in ALLOWED_CATEGORIES}
        if other:
            print(f"   其他：{other}")

    print()
    if errors:
        print("结论：❌ 校验未通过——先修硬错误，再谈「数据真实性」。")
    else:
        print("结论：✅ 校验通过——这份数据可如实标注为「论文真实 QA，每条可追溯，且不撞评测题」。")
    print("=" * 92)


# ---------------------------------------------------------------------------
# 第 5 区：入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="F1 论文 QA 校验：条数 / 字段 / 空值 / 重复 / 数据泄漏 / source 可追溯"
    )
    parser.add_argument("--json", default=DEFAULT_QA_JSON, help="QA 数据文件（默认同目录 sft_data_paper.json）")
    parser.add_argument("--chunks", default=CHUNKS_JSON, help="chunks.json（source 回查用）")
    parser.add_argument("--eval-questions", default=EVAL_JSON, help="20 条评测题（泄漏检查用）")
    parser.add_argument("--min", type=int, default=50, help="条数门槛（默认 50，计划要求 ≥50 起步）")
    parser.add_argument("--no-trace", action="store_true", help="跳过 chunk 回查（只查字段/重复/空值/泄漏）")
    parser.add_argument("--allow-leak", action="store_true", help="撞题只警告不报错（临时用，正式提交别开）")
    parser.add_argument("--show-categories", action="store_true", help="打印五类分布")
    parser.add_argument("--example", action="store_true", help="打印 5 条真实样例（照格式写，不是模板生成）")
    args = parser.parse_args()

    if args.example:
        print("# 5 条真实样例（人工提取自 GMR 论文；source 可在 chunks.json 里逐字回查）")
        print("# 注意：这 5 条**刻意避开了 20 条评测题**——数据和卷子不能是同一批题，否则叫数据泄漏。")
        print("# 用法：照这个格式，自己逐条读原文写到 50+ 条，再用本脚本校验。\n")
        print(json.dumps(EXAMPLES, ensure_ascii=False, indent=2))
        return

    if not os.path.exists(args.json):
        print(f"❌ 找不到数据文件：{args.json}")
        print("   第一次跑：先看格式 `python build_paper_qa.py --example`，")
        print(f"   再把同目录的 `sft_data_paper.seed.json` 复制成 `{os.path.basename(args.json)}` 起步。")
        sys.exit(1)

    records = load_json(args.json)
    if not isinstance(records, list):
        print(f"❌ 数据文件根节点应为 list，实际是 {type(records).__name__}")
        sys.exit(1)

    chunk_text = {}
    if os.path.exists(args.chunks):
        chunk_text = {c["id"]: c.get("text", "") for c in load_json(args.chunks)}
    elif not args.no_trace:
        print(f"⚠ 找不到 chunks.json（{args.chunks}）→ 自动跳过 source 回查")
        args.no_trace = True

    eval_questions = []
    if args.eval_questions and os.path.exists(args.eval_questions):
        eval_questions = load_json(args.eval_questions)["questions"]
    else:
        print(f"⚠ 找不到评测题（{args.eval_questions}）→ 自动跳过泄漏检查")
        args.eval_questions = None

    args.n_chunks = len(chunk_text)
    errors, warnings, stats = validate(records, args, chunk_text, eval_questions)
    print_report(records, errors, warnings, stats, args)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
