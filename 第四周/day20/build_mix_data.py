# -*- coding: utf-8 -*-
r"""
build_mix_data.py —— 生成 F3 "拆变量轮"的训练集：真实 201 + 合成补到 300
================================================================================
五件套说明（本脚本在机器学习的哪一环）：
- 模型   ：无（纯数据处理，不加载模型、不连服务）
- 数据   ：输入 = `第四周\day19\sft_data_v2_real.json`（真实 201 条，origin=real）
                 + `第三周\day11\sft_data.json`（合成 300 条，第三周遗留）
           输出 = `sft_data_v2_mix.json`（**300 条**：真实 201 + 合成 99）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（数据准备环节）：
    把"真实数据比合成数据差"这个**尚未拆开的负结果**（day19 F3v2：F1 0.824→0.333）
    变成**可归因的对照实验**的输入：
        v1 = 合成 300（已有，F1 0.824 = R3）
        v2 = 真实 201（已有，F1 0.333 = F3v2）      ← "来源"和"数量"两个变量同时变了
        v3 = 真实 201 + 合成 99 = 300（本脚本生成）  ← 把"数量"对齐，只看"来源"
    若 v3 回到 R3 水平 → 退化主要来自**样本数量少了 99 条**；
    若 v3 仍然退化   → 才能说"这批真实数据与评测集的分布不匹配"。

★ 为什么要"补短板式"补合成数据，而不是随便抓 99 条？（面试会问）
    随机抓会引入一个新变量（合成的**主题分布**可能和真实数据差很多）。
    → 本脚本的做法：**按真实数据的分布去配比**——用真实 201 条的 `category`（若有）
      作为目标分布，从合成池里"哪个主题缺得多就多补哪个"，让"主题分布"尽量对齐。
      合成池没有 category 字段（第三周的数据只有 instruction/input/output），
      所以退化成"随机 + 固定种子"并在报告里**如实写明这一点**（不假装对齐了）。

★ 三条硬性检查（不通过就退出，绝不产出一个"看起来对"的文件）
    ① 总条数 = --total（默认 300）
    ② `instruction` 归一化后**无重复**（真实池内部、合成池内部、两池之间）
    ③ 空值 = 0；且每条都带 `origin`（real / synth）→ 训练集来源可追溯

用法（llm 环境；本文件所在目录下执行）：
    python build_mix_data.py                         # 生成 sft_data_v2_mix.json（300 条）
    python build_mix_data.py --dry-run               # 只看会怎么配比，不写文件
    python build_mix_data.py --total 300 --seed 42   # 显式指定（默认就是这两个值）
================================================================================
"""

import argparse
import difflib
import json
import os
import random
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_REAL = os.path.join(REPO_DIR, "第四周", "day19", "sft_data_v2_real.json")
DEFAULT_SYNTH = os.path.join(REPO_DIR, "第三周", "day11", "sft_data.json")


# ---------------------------------------------------------------------------
# 第 2 区：小工具
# ---------------------------------------------------------------------------
def norm(text):
    """归一化：去掉所有空白 + 去掉中英文标点 + 转小写 → 用来判"是不是同一道题"。"""
    s = str(text or "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。、；：？！,.;:?!\"'“”‘’（）()\[\]【】《》<>《》\-—_/\\|]+", "", s)
    return s.lower()


def load_json_list(path, label):
    if not os.path.exists(path):
        raise SystemExit(f"[FAIL] 找不到{label}：{path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # 兼容 {"questions": [...]} 与 [...] 两种形态（day19 踩过 "str" object has no attribute get）
    if isinstance(data, dict):
        data = data.get("data") or data.get("questions") or []
    if not isinstance(data, list) or not data:
        raise SystemExit(f"[FAIL] {label}不是非空列表：{path}")
    return data


def check_record(rec, label, idx):
    """单条记录的三字段检查；返回 (是否合格, 说明)。"""
    for key in ("instruction", "output"):
        if not str(rec.get(key, "")).strip():
            return False, f"{label} 第 {idx} 条缺 `{key}`（或为空）"
    rec.setdefault("input", "")          # input 允许为空（计算/解释类题没有材料）
    return True, ""


def near_dup_pairs(real_norms, synth_norms, threshold=0.92, topk=5):
    """
    近似重复检查（只报告、不删）：合成的题和真实的题"长得太像"会把两条都变味。
    先用"首 4 字 + 长度差 40%"做便宜预筛，再对候选算 difflib 相似度。
    """
    hits = []
    for i, sn in enumerate(synth_norms):
        if not sn:
            continue
        for j, rn in enumerate(real_norms):
            if not rn:
                continue
            if abs(len(sn) - len(rn)) > max(len(sn), len(rn)) * 0.4 + 4:
                continue
            if sn[:4] != rn[:4]:
                continue
            ratio = difflib.SequenceMatcher(None, sn, rn).ratio()
            if ratio >= threshold:
                hits.append((i, j, ratio))
    hits.sort(key=lambda t: -t[2])
    return hits[:topk], len(hits)


# ---------------------------------------------------------------------------
# 第 3 区：主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="生成 F3 拆变量轮训练集（真实 + 合成补到 300）")
    parser.add_argument("--real", default=DEFAULT_REAL, help="真实数据（F1+F2 拼好的 201 条）")
    parser.add_argument("--synth", default=DEFAULT_SYNTH, help="合成数据池（第三周 300 条）")
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "sft_data_v2_mix.json"),
                        help="输出文件（默认本目录 sft_data_v2_mix.json）")
    parser.add_argument("--total", type=int, default=300, help="输出的总条数（默认 300，与 v1 对齐）")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子（固定 → 可复现）")
    parser.add_argument("--dry-run", action="store_true", help="只打印方案，不写文件")
    args = parser.parse_args()

    real_raw = load_json_list(args.real, "真实数据")
    synth_raw = load_json_list(args.synth, "合成数据池")

    print("=" * 78)
    print("F3 拆变量轮 · 训练集构建（real + synth = total，只让'来源'这一个变量变）")
    print("=" * 78)
    print(f"  真实池：{os.path.relpath(args.real, REPO_DIR)} → {len(real_raw)} 条")
    print(f"  合成池：{os.path.relpath(args.synth, REPO_DIR)} → {len(synth_raw)} 条")

    # ---- ① 真实数据：去重 + 三字段检查，全部保留 ----
    real_records, real_norms, bad = [], [], []
    for i, rec in enumerate(real_raw, 1):
        ok, why = check_record(rec, "真实", i)
        if not ok:
            bad.append(why)
            continue
        n = norm(rec["instruction"])
        if n in real_norms:
            bad.append(f"真实 第 {i} 条 `instruction` 与前面重复（已跳过）")
            continue
        real_norms.append(n)
        real_records.append({
            "instruction": str(rec["instruction"]).strip(),
            "input": str(rec.get("input", "")).strip(),
            "output": str(rec["output"]).strip(),
            "origin": "real",
            # 来源信息保留下来（训练脚本只读三字段，多出来的字段不影响训练；
            # 但"每条可追溯"这件事必须在数据文件里成立）
            "source": rec.get("source", {}),
            "category": rec.get("category", ""),
        })

    need = args.total - len(real_records)
    if need <= 0:
        raise SystemExit(f"[FAIL] 真实数据已有 {len(real_records)} 条 ≥ --total {args.total}；"
                         f"要么调大 --total，要么不再需要合成补齐（那就直接用真实版）。")

    # ---- ② 合成池：内部去重 + 与真实池去重，然后按固定种子抽样 ----
    synth_pool, synth_norms, dropped = [], [], Counter()
    real_set = set(real_norms)
    for i, rec in enumerate(synth_raw, 1):
        ok, why = check_record(rec, "合成", i)
        if not ok:
            dropped["字段不全"] += 1
            continue
        n = norm(rec["instruction"])
        if n in real_set:
            dropped["与真实数据重复"] += 1
            continue
        if n in synth_norms:
            dropped["合成池内部重复"] += 1
            continue
        synth_norms.append(n)
        synth_pool.append(rec)

    print(f"  合成池可用：{len(synth_pool)} 条（剔除：{dict(dropped) or '无'}）")
    print(f"  需要补：{args.total} − {len(real_records)} = {need} 条")

    if len(synth_pool) < need:
        raise SystemExit(f"[FAIL] 合成池可用条数 {len(synth_pool)} < 需要的 {need} 条。"
                         f"→ 换一个更大的合成池，或把 --total 调小（但那样'数量'就没对齐了）。")

    rng = random.Random(args.seed)               # 固定种子 → 同一份抽样可复现
    picked_idx = sorted(rng.sample(range(len(synth_pool)), need))
    synth_records = [{
        "instruction": str(synth_pool[i]["instruction"]).strip(),
        "input": str(synth_pool[i].get("input", "")).strip(),
        "output": str(synth_pool[i]["output"]).strip(),
        "origin": "synth",
        "source": {"note": "第三周 day11 合成数据（语料池 + 模板生成），非真实来源"},
        "category": "",
    } for i in picked_idx]

    records = real_records + synth_records

    # ---- ③ 硬性检查 ----
    print("-" * 78)
    print("三段硬性检查：")
    ok_all = True

    if len(records) == args.total:
        print(f"  ① 总条数：{len(records)} = --total {args.total}  ✅")
    else:
        ok_all = False
        print(f"  ① 总条数：{len(records)} ≠ --total {args.total}  ❌")

    norms = [norm(r["instruction"]) for r in records]
    dup = [k for k, c in Counter(norms).items() if c > 1]
    if not dup:
        print("  ② instruction 去重：0 条重复  ✅")
    else:
        ok_all = False
        print(f"  ② instruction 去重：{len(dup)} 条重复  ❌  例：{dup[:3]}")

    empty = [i for i, r in enumerate(records, 1) if not r["instruction"] or not r["output"]]
    if not empty:
        print("  ③ 空值：0 条  ✅")
    else:
        ok_all = False
        print(f"  ③ 空值：{len(empty)} 条  ❌  行号：{empty[:5]}")

    cnt = Counter(r["origin"] for r in records)
    print(f"\n  来源分布：real {cnt['real']} 条 ｜ synth {cnt['synth']} 条（合计 {len(records)}）")
    cat = Counter(r["category"] for r in records if r["category"])
    if cat:
        print(f"  真实数据类目分布（合成侧无 category）：{dict(cat)}")

    # ---- ④ 近似重复报告（只提示，不删） ----
    # ⚠ 两侧都要传**归一化后**的字符串：归一化会去掉标点/空白，若只归一化一侧，
    #    `sn[:4] != rn[:4]` 这个预筛就会因为标点位置不同而误判（漏报）。
    hits, total_hits = near_dup_pairs(real_norms, [norm(r["instruction"]) for r in synth_records])
    if total_hits == 0:
        print("  近似重复（相似度 ≥0.92）：0 对  ✅")
    else:
        print(f"  ⚠ 近似重复（相似度 ≥0.92）：{total_hits} 对（**只提示不删**，报告里如实写）")
        for i, j, ratio in hits:
            print(f"      {ratio:.3f}  synth#{i} 「{synth_records[i]['instruction'][:28]}…」"
                  f"  ≈  real#{j} 「{real_records[j]['instruction'][:28]}…」")

    if not ok_all:
        raise SystemExit("\n[FAIL] 硬性检查未通过 → 不写文件（避免产出一个'看起来对'的训练集）。")

    if args.dry_run:
        print(f"\n[--dry-run] 未写文件。正式生成请去掉 --dry-run（会写到 {args.out}）")
        return

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\n[OK] 已写出：{args.out}（{len(records)} 条，{size_kb:.0f} KB）")
    print("     下一步：用它训练 v3 adapter（`train_lora_v3.py`，与 v2 只差 DATA_PATH 等 3 个常量）")
    print("     ⚠ 训练前先停 8000 端口的模型服务（6G 显存一次只能装一个 3B）")


if __name__ == "__main__":
    main()
