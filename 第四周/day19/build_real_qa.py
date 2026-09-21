# -*- coding: utf-8 -*-
"""
build_real_qa.py —— F2 真实行业 QA 校验器（Day19）

与 day18 的 build_paper_qa.py 的分工：
    build_paper_qa.py ：论文 QA（quote 能逐字回查到 chunks.json）
    build_real_qa.py  ：行业 QA（来源在外部 → 校验"来源五要素 + 去重 + 泄漏 + 分布"）

用法：
    python build_real_qa.py --min 100 --show-categories
    python build_real_qa.py --example            # 打印一条样例（照格式写）
    python build_real_qa.py --allow-leak         # 临时调试：撞题只警告不报错

起步：
    copy sft_data_real.seed.json sft_data_real.json
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_QA = os.path.join(BASE_DIR, "sft_data_real.json")
EVAL_JSON = os.path.join(REPO_ROOT, "第三周", "day13", "eval_questions.json")
LEAK_THRESHOLD = 0.70
SOURCE_KEYS = ["org", "title", "url", "date", "quote"]
CATEGORIES = ["公司经营", "产品技术", "行业标准与政策", "市场与产业链", "论文/研报结论"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(s):
    """归一化：去空白 + 去标点 + 转小写（中文标点也去掉）。"""
    s = str(s or "").lower()
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(r"[，。、；：！？“”‘’（）《》【】,.;:!?\"'()\[\]<>·\-—_/\\|]+", "", s)
    return s


def is_blank(v):
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def similarity(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def validate(records, args, eval_questions):
    errors, warnings = [], []
    stats = {"dup": 0, "leaks": 0}

    if len(records) < args.min:
        errors.append(f"条数不足：实到 {len(records)} 条，门槛 {args.min} 条（先补数据，别硬凑）")

    seen, orgs, cats = {}, Counter(), Counter()
    for i, r in enumerate(records, start=1):
        tag = f"第 {i} 条（id={r.get('id', '-')}）"

        for k in ("instruction", "output", "source"):
            if k not in r or is_blank(r[k]):
                errors.append(f"{tag}：缺必填字段 {k}")

        src = r.get("source") or {}
        if isinstance(src, dict):
            for k in SOURCE_KEYS:
                if k not in src or is_blank(src[k]):
                    errors.append(f"{tag}：source 缺 {k}")
            url = str(src.get("url", ""))
            if url and not url.lower().startswith("http"):
                errors.append(f"{tag}：source.url 不是合法链接（{url[:40]}）")
            if len(str(src.get("quote", ""))) > 120:
                warnings.append(f"{tag}：source.quote 超过 120 字（可能转载了整段）")
            if src.get("org"):
                orgs[src["org"]] += 1
        elif src:
            errors.append(f"{tag}：source 必须是对象（含 {'/'.join(SOURCE_KEYS)}）")

        key = norm(r.get("instruction"))
        if key in seen:
            stats["dup"] += 1
            errors.append(f"{tag}：instruction 与第 {seen[key]} 条重复（换个说法问同一件事也要算重复）")
        else:
            seen[key] = i

        cats[r.get("category", "未分类")] += 1

        for q in eval_questions:
            s = similarity(r.get("instruction", ""), q.get("question", ""))
            if s >= LEAK_THRESHOLD:
                stats["leaks"] += 1
                msg = f"{tag}：与评测题 Q{q['id']} 相似度 {s:.2f}（≥{LEAK_THRESHOLD}）= 数据泄漏"
                (warnings if args.allow_leak else errors).append(msg)
                break

    if orgs and records:
        top_org, top_n = orgs.most_common(1)[0]
        if top_n / len(records) > 0.5:
            warnings.append(f"机构分布过于集中：{top_org} 占 {top_n}/{len(records)}（>50%），真实性存疑")

    return errors, warnings, stats, cats, orgs


def print_report(records, errors, warnings, stats, cats, orgs, args):
    print("=" * 68)
    print(f"F2 真实行业 QA 校验 ｜ 文件：{os.path.basename(args.json)}")
    print("=" * 68)
    print(f"① 条数          ：{len(records)} / 门槛 {args.min}")
    print("② 必填字段      ：instruction / output / source")
    print(f"③ source 五要素 ：{' / '.join(SOURCE_KEYS)}（url 必须 http 开头，quote ≤120 字）")
    print(f"④ 重复 instruction：{stats['dup']} 条")
    print(f"⑤ 数据泄漏      ：{stats['leaks']} 条（阈值 {LEAK_THRESHOLD}）")
    if args.show_categories:
        print("⑥ 类目分布      ：" + " / ".join(f"{k} {v}" for k, v in cats.most_common()))
        if orgs:
            print("   机构 Top5    ：" + " / ".join(f"{k} {v}" for k, v in orgs.most_common(5)))
    print("-" * 68)
    if warnings:
        print(f"⚠ 警告 {len(warnings)} 条：")
        for w in warnings[:10]:
            print("  -", w)
    if errors:
        print(f"❌ 错误 {len(errors)} 条：")
        for e in errors[:20]:
            print("  -", e)
        print(f"\n[未通过] 共 {len(errors)} 个硬错误 —— 修完再来。")
        return 1
    print(f"✅ [通过] {len(records)} 条真实行业 QA，来源可追溯、无重复、无泄漏。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="F2 真实行业 QA 校验器（Day19）")
    ap.add_argument("--json", default=DEFAULT_QA, help="数据文件（默认同目录 sft_data_real.json）")
    ap.add_argument("--eval-questions", default=EVAL_JSON, help="20 条评测题（泄漏检查用）")
    ap.add_argument("--min", type=int, default=100, help="条数门槛（计划要求 ≥100）")
    ap.add_argument("--allow-leak", action="store_true", help="撞题只警告不报错（临时调试用）")
    ap.add_argument("--show-categories", action="store_true", help="打印类目/机构分布")
    ap.add_argument("--example", action="store_true", help="打印一条样例格式")
    args = ap.parse_args()

    if args.example:
        print(json.dumps({
            "instruction": "（问题，来自公开资料能回答的内容）",
            "input": "（可选上下文；行业 QA 一般留空）",
            "output": "（忠实来源的答案，不要加自己推断的数字）",
            "source": {"org": "（机构）", "title": "（标题）", "url": "https://...",
                       "date": "（年-月）", "quote": "（回答问题的那一两句原文，≤120 字）"},
            "category": f"（{'/'.join(CATEGORIES)} 之一）",
        }, ensure_ascii=False, indent=2))
        return 0

    if not os.path.exists(args.json):
        print(f"[错误] 找不到 {args.json} —— 先从种子文件复制起步："
              "copy sft_data_real.seed.json sft_data_real.json")
        return 1

    records = load_json(args.json)
    eval_questions = load_json(args.eval_questions)
    if isinstance(eval_questions, dict):          # eval_questions.json = {"meta":..., "questions":[...]}
        eval_questions = eval_questions.get("questions", [])
    if not eval_questions:
        print(f"⚠ 评测题为空（{args.eval_questions}）→ 跳过泄漏检查（正式校验前请确认）")
    errors, warnings, stats, cats, orgs = validate(records, args, eval_questions)
    return print_report(records, errors, warnings, stats, cats, orgs, args)


if __name__ == "__main__":
    sys.exit(main())
