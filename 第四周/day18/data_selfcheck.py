# -*- coding: utf-8 -*-
r"""
O1-R8 评测数据自检脚本（第四周 Day18 · 决策门"几乎没动"侧的第一个动作）⭐

一句话：**在怀疑"检索不行"之前，先确认"卷子有没有出错"**。
    逐题核对 eval_questions.json 里的 `source_chunk`（期望段落）到底是不是真的"含答案的段落"——
    如果标注本身就错，那"检索未命中"就是假失败，后面所有优化归因都会被带偏。

它回答三个问题（对应三种"假失败"）：
    ① 标注的 chunk 到底存不存在？（chunks.json 里有没有这个 id）
    ② 期望关键词能不能在"标注的段落"里找到？（标注自洽性）
    ③ 关键词其实出现在哪些 chunk 里？（如果不在标注段落里，在别处 → 说明"标注错了"，而不是"检索错了"）

判读口径（重要，避免误判）：
    - 「关键词命中标注段落」= ✅ 标注自洽 → 检索没召回它，是真失败，该优化检索；
    - 「关键词不在标注段落、但在别的 chunk」= ⚠ 标注可能错 → 先改题面/标注（改判据要一次性做完并留痕）；
    - 「关键词哪儿都找不到」= ❓ 多为"翻译题"（英文原文 vs 中文关键词，如 骨骼长度/关节），
      程序查不出来，**必须人工读原文核对**（脚本会把这些题单独列出来）。

它不连模型、不连向量库，只读两个 JSON，**几秒钟跑完**。

运行方法（llm 环境；先 cd 到本文件所在目录）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day18"

    python data_selfcheck.py                  # 全部题自检 + 打印表
    python data_selfcheck.py --only-in        # 只看 in_material 10 题（默认就是全看）
    python data_selfcheck.py --dump-chunk 5   # 打印 chunk 5 原文（人工核对用）
    python data_selfcheck.py --dump-chunk 0 6 # 一次看多个
    python data_selfcheck.py --json           # 额外落盘 data_selfcheck.json（供报告引用）

依赖：只用标准库；不需要 langchain / chroma / 模型服务。
"""

import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固（Windows 控制台默认 GBK，打不出部分符号会崩）
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 第 2 区：路径常量（相对本脚本定位，换机器也能跑）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))            # 第四周\day18
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))  # 仓库根
DAY13_DIR = os.path.join(REPO_DIR, "第三周", "day13")
CHUNKS_JSON = os.path.join(DAY13_DIR, "chunks.json")               # 105 段原文（建库时导出）
QUESTIONS_JSON = os.path.join(DAY13_DIR, "eval_questions.json")    # 20 条评测题

# ---------------------------------------------------------------------------
# 第 3 区：判据校准补丁（与 day17\eval_v2.py 的 KEYWORD_PATCH 保持一致）
# ---------------------------------------------------------------------------
# 纪律：判据改动必须 ①一次性做完 ②留痕 ③前后一致。
# 本脚本是"读题验证"，必须用与评测脚本**同一套**关键词口径，否则自检结论与实跑对不上。
KEYWORD_PATCH = {
    5: {"expected_keywords": ["LAFAN1", "LAFAN"]},  # Q5：LAFAN1 -> LAFAN1/LAFAN
}

# 这些题的关键词是"中文概括/方向词"（原文是英文），字面匹配必然找不到 → 归为"需人工核对"，
# 不能让程序把翻译题判成"标注错误"。
SEMANTIC_HINT = {
    2: "关键词为中文概括（embodiment gap 的中文说法），需人工读 chunk 0/5 原文",
    7: "含中文关键词（脚滑动/自穿透/不可行），英文原文 chunk 1 有 foot sliding / self-penetration",
    8: "关键词是方向词（优于/接近/闭源），需人工读 chunk 3 判断",
    10: "关键词是中文术语（骨骼长度/关节/运动范围），英文原文 chunk 5 为 bone length / joint range of motion",
}


# ---------------------------------------------------------------------------
# 第 4 区：加载与判据应用
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_keyword_patch(questions):
    """把 KEYWORD_PATCH 应用到题目上，返回 (题目列表, 应用了哪些补丁的可读说明)。"""
    patched = []
    msgs = []
    for q in questions:
        q = json.loads(json.dumps(q))  # 深拷贝，不动原对象
        patch = KEYWORD_PATCH.get(q["id"])
        if patch:
            for k, v in patch.items():
                old = q.get(k)
                q[k] = v
                msgs.append(f"Q{q['id']} {k}: {old} -> {v}")
        patched.append(q)
    return patched, msgs


def build_chunk_index(chunks):
    """chunk_id -> 文本。chunks.json 形如 [{'id':0,'page':0,'text':'...'}, ...]。"""
    return {c["id"]: c.get("text", "") for c in chunks}


def contains(text, keyword):
    """字面包含判断（大小写不敏感；用于英文专名/直接事实题）。"""
    return keyword.lower() in text.lower()


# ---------------------------------------------------------------------------
# 第 5 区：核心自检
# ---------------------------------------------------------------------------

def check_question(q, chunk_text):
    """
    对一条 in_material 题做三重核对，返回一个结果字典。
    out_of_material（拒答题）不查标注，只确认它确实"不带 source_chunk"。
    """
    rec = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "source_chunk": q.get("source_chunk", []),
        "keywords": q.get("expected_keywords", []),
        "missing_chunks": [],      # 标注了但 chunks.json 里不存在的 id
        "kw_in_anno": {},          # 关键词 -> 是否出现在"标注段落"里
        "kw_chunks": {},           # 关键词 -> 实际出现在哪些 chunk
        "anno_self_consistent": None,
        "verdict": "",
    }

    if q["type"] != "in_material":
        rec["anno_self_consistent"] = None
        rec["verdict"] = "拒答题（out_of_material）：不查标注；应如实答「资料中没有提到」"
        return rec

    anno_ids = rec["source_chunk"]
    rec["missing_chunks"] = [i for i in anno_ids if i not in chunk_text]
    anno_text = "\n".join(chunk_text.get(i, "") for i in anno_ids)

    hit_any = False
    for kw in rec["keywords"]:
        in_anno = contains(anno_text, kw)
        rec["kw_in_anno"][kw] = in_anno
        hit_any = hit_any or in_anno

    # 全库扫描：这个关键词实际上出现在哪些 chunk（用于判断"标注错位"）
    for kw in rec["keywords"]:
        rec["kw_chunks"][kw] = sorted(
            cid for cid, text in chunk_text.items() if contains(text, kw)
        )

    rec["anno_self_consistent"] = hit_any
    if rec["missing_chunks"]:
        rec["verdict"] = f"❌ 标注的 chunk 不存在：{rec['missing_chunks']}（题目/库对不上）"
    elif hit_any:
        rec["verdict"] = "✅ 标注自洽（关键词能在标注段落里找到）→ 检索未召回它属真失败"
    else:
        # 关键词不在标注段落里：再看它在别处有没有出现
        elsewhere = sorted({c for ids in rec["kw_chunks"].values() for c in ids})
        if elsewhere:
            rec["verdict"] = (f"⚠ 关键词不在标注段落，却在 chunk {elsewhere} 出现 → "
                              f"疑『标注错位』，需人工确认是否改题面")
        else:
            rec["verdict"] = ("❓ 关键词全库都找不到 → 多为『英文原文 vs 中文关键词』的翻译题，"
                              "需人工读原文核对（见 SEMANTIC_HINT）")
    return rec


# ---------------------------------------------------------------------------
# 第 6 区：打印
# ---------------------------------------------------------------------------

def print_table(records):
    print("=" * 100)
    print("O1-R8 评测数据自检（逐题核对 source_chunk 是不是真的「含答案」）")
    print("=" * 100)
    print(f"{'id':>3} | {'期望标注':<12} | {'关键词':<34} | {'标注自洽':<6} | 实际含关键词的 chunk")
    print("-" * 100)
    for r in records:
        if r["type"] != "in_material":
            continue
        anno = ",".join(str(i) for i in r["source_chunk"]) or "-"
        kws = "、".join(r["keywords"])[:34]
        ok = {True: "✅", False: "⚠/❓", None: "-"}[r["anno_self_consistent"]]
        actual = "; ".join(
            f"{kw}→{ids if ids else '无'}" for kw, ids in r["kw_chunks"].items()
        )
        print(f"{r['id']:>3} | {anno:<12} | {kws:<34} | {ok:<6} | {actual}")
    print("-" * 100)


def print_details(records):
    print("\n逐题判读：")
    for r in records:
        if r["type"] != "in_material":
            continue
        print(f"\nQ{r['id']}（{r['question']}）")
        print(f"  期望 source_chunk：{r['source_chunk']} ｜ 关键词：{r['keywords']}")
        print(f"  判定：{r['verdict']}")
        if r["id"] in SEMANTIC_HINT:
            print(f"  人工核对提示：{SEMANTIC_HINT[r['id']]}")


def summarize(records):
    in_recs = [r for r in records if r["type"] == "in_material"]
    n_ok = sum(1 for r in in_recs if r["anno_self_consistent"] is True)
    n_bad = sum(1 for r in in_recs if r["missing_chunks"])
    n_else = sum(
        1 for r in in_recs
        if r["anno_self_consistent"] is False and not r["missing_chunks"]
        and any(ids for ids in r["kw_chunks"].values())
    )
    n_manual = sum(
        1 for r in in_recs
        if r["anno_self_consistent"] is False and not r["missing_chunks"]
        and not any(ids for ids in r["kw_chunks"].values())
    )
    print("\n" + "=" * 100)
    print("自检小结")
    print("=" * 100)
    print(f"  ⭐ 标注自洽（关键词命中标注段落）      ：{n_ok}/{len(in_recs)} 题")
    print(f"  ⚠ 疑『标注错位』（关键词在别处）      ：{n_else}/{len(in_recs)} 题")
    print(f"  ❌ 标注 chunk 不存在                  ：{n_bad}/{len(in_recs)} 题")
    print(f"  ❓ 需人工读原文（多为中英翻译题）    ：{n_manual}/{len(in_recs)} 题")
    print()
    print("  结论怎么写（按实测填，别预填）：")
    if n_bad == 0 and n_else == 0:
        print("    → 标注层面没有硬错误：检索未命中是**真失败**，瓶颈在")
        print("      『中文问句 ↔ 英文段落的语义对齐』（假设②），下一步做 O1-R4 HyDE / O1-R5 混合检索。")
    else:
        print("    → 存在标注问题：先把题面/标注修好（一次性改完并留痕），再谈检索优化；")
        print("      否则『检索命中率』这个数字本身就是脏的。")
    print()
    print("  注意：程序只做字面匹配，**翻译题必须人工读原文**才能下结论；")
    print("        人工核对建议逐题抄进《优化成果报告》的『数据自检』小节（可追溯）。")


# ---------------------------------------------------------------------------
# 第 7 区：入口
# ---------------------------------------------------------------------------

def dump_chunks(chunk_text, ids):
    for i in ids:
        print("\n" + "=" * 100)
        print(f"chunk {i}（共 {len(chunk_text.get(i, ''))} 字符）")
        print("=" * 100)
        print(chunk_text.get(i, "（不存在）"))


def scan_chunks(chunks, width):
    """按顺序打印每个 chunk 的前 width 个字符——人工提取论文 QA 时的"选题扫描"用。"""
    print(f"chunk 预览（共 {len(chunks)} 段，每段前 {width} 字符；用它快速找「哪些段落能出题」）")
    print("-" * 100)
    for c in chunks:
        text = " ".join(str(c.get("text", "")).split())
        print(f"[{c['id']:>3}|p{c.get('page')}] {text[:width]}")


def main():
    parser = argparse.ArgumentParser(
        description="O1-R8 评测数据自检：逐题核对 source_chunk 是否真的对应答案段落"
    )
    parser.add_argument("--chunks", default=CHUNKS_JSON, help="chunks.json 路径")
    parser.add_argument("--questions", default=QUESTIONS_JSON, help="eval_questions.json 路径")
    parser.add_argument("--only-in", action="store_true", help="只看 in_material 题（默认全看）")
    parser.add_argument("--dump-chunk", type=int, nargs="*", default=None,
                        help="打印指定 chunk 的完整原文（人工核对用，可给多个 id）")
    parser.add_argument("--scan", type=int, nargs="?", const=150, default=None, metavar="WIDTH",
                        help="按顺序预览所有 chunk 的前 WIDTH 字符（默认 150；人工提取 QA 时选题用）")
    parser.add_argument("--json", action="store_true", help="额外落盘 data_selfcheck.json")
    args = parser.parse_args()

    chunks = load_json(args.chunks)
    chunk_text = build_chunk_index(chunks)
    questions = load_json(args.questions)["questions"]
    questions, patch_msgs = apply_keyword_patch(questions)

    if args.scan is not None:
        scan_chunks(chunks, args.scan)
        return

    print("评测数据自检")
    print(f"  切块文件：{args.chunks}（共 {len(chunks)} 个 chunk）")
    print(f"  题目文件：{args.questions}（共 {len(questions)} 条）")
    print(f"  判据补丁：{'；'.join(patch_msgs) if patch_msgs else '无'}")
    print()

    if args.dump_chunk is not None:
        dump_chunks(chunk_text, args.dump_chunk)
        return

    records = [check_question(q, chunk_text) for q in questions]
    if args.only_in:
        records = [r for r in records if r["type"] == "in_material"]

    print_table(records)
    print_details(records)
    summarize(records)

    if args.json:
        out = os.path.join(SCRIPT_DIR, "data_selfcheck.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 明细已落盘：{out}")


if __name__ == "__main__":
    main()
