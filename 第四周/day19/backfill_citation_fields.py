# -*- coding: utf-8 -*-
"""
Day19 · O2-G3 补算脚本：给**已跑完**的结果目录补上 9/21 新增的两个字段，并重生成《评测表.md》。

为什么需要它：
    R6 / R6b 跑在 9/21 早版 `eval_v2.py` 上（当时还没有 `refused_raw` / `citation_pseudo_n`）。
    这两个字段**只影响报告的"留痕行"**（"拒答状态随兜底翻转"、"抓不到的伪引文"），
    不影响任何成绩指标。重跑一轮会**重掷生成**（T=0.2）→ 已定稿的数字全变；
    所以用**同一条代码、从已存的 `answer_raw` 离线重算**：纯本地、确定性、不调模型。

安全性设计（三道）：
    ① **断言**：补算前后 `summarize()` 的全部指标逐格相同（F1 / TP-FP-FN-TN / 各项正确率 / 一致率），
       不同就**中止且不写盘** —— 保证"补算没动成绩"。
    ② **备份**：原 `eval_results.json` 复制为 `eval_results.pre_backfill.json`。
    ③ **留痕**：结果目录里写一份《补算说明.md》，说明补了什么、依据什么、数字是否变。

用法（在 day19 目录下）：
    python backfill_citation_fields.py                      # 默认补 result_cite_strip_runs3 与 result_cite_retry_runs3
    python backfill_citation_fields.py --dirs a b --dry-run # 只看会补什么，不写盘
"""

import argparse
import json
import os
import shutil
import sys
import types


def _ensure_importable():
    """
    本脚本只用 eval_v2 的**纯函数**（判分 / 引文校验 / 汇总 / 报表），
    不连向量库、不加载嵌入模型。但 eval_v2 顶部会 `import langchain_chroma / langchain_huggingface`，
    若当前环境没装（例如用了非 llm 环境的 python），import 就会失败。
    → 缺哪个就在 sys.modules 里放个占位模块（**只让 import 通过，不发起任何调用**）。
    """
    for name, attrs in (("langchain_chroma", ("Chroma",)),
                        ("langchain_huggingface", ("HuggingFaceEmbeddings",))):
        try:
            __import__(name)
        except ModuleNotFoundError:
            mod = types.ModuleType(name)
            for a in attrs:
                setattr(mod, a, type(a, (), {}))
            sys.modules[name] = mod
            print(f"[提示] 当前环境没有 {name}（本脚本只用纯函数）→ 已用占位模块导入")


# 复用 eval_v2 的函数与配置读取（本脚本不调模型、不连向量库）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ensure_importable()
import eval_v2 as E  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIRS = ["result_cite_strip_runs3", "result_cite_retry_runs3"]

# 每轮的元信息（与首跑时的命令行一字不差，否则重生成的表头会与 run_log 对不上）
RUN_META = {
    "result_cite_strip_runs3": dict(
        exp_id="R6", policy="strip", runs=3,
        note="O2-G3 引文强制白名单：非法编号剪除（只改交付，不改生成）"),
    "result_cite_retry_runs3": dict(
        exp_id="R6b", policy="retry", runs=3,
        note="O2-G3 retry：点名非法编号后重答一次（含提示诱导效应）"),
}


def metric_fingerprint(s):
    """把 summarize() 里所有**成绩指标**压成一个可比较的元组（用于"补算没动成绩"的断言）。"""
    return (
        s["retrieval_hit_strict"], s["retrieval_hit_loose"],
        s["gen_correct"], s["anti_correct"],
        s["confusion"], round(s["precision"], 6), round(s["recall"], 6), round(s["f1"], 6),
        round(s["overall"], 6),
        None if s["avg_consistency"] is None else round(s["avg_consistency"], 6),
        # 引文侧的"原始口径"与兜底动作也必须不变（只有新加的两项允许变）
        s["citation"]["total"], s["citation"]["bad"], s["citation"]["bad_qs"],
        s["citation"]["total_delivered"], s["citation"]["bad_delivered"],
        s["citation"]["bad_qs_delivered"], s["citation"]["removed"],
        s["citation"]["retried_qs"], s["citation"]["only_bad_qs"],
        s["citation"]["guard_changed"],
    )


def load_meta():
    data = E.load_json(E.DEFAULT_EVAL_JSON)
    return data["meta"]


def backfill_one(dirname, dry_run=False):
    d = os.path.join(SCRIPT_DIR, dirname)
    json_path = os.path.join(d, "eval_results.json")
    if not os.path.exists(json_path):
        print(f"[跳过] 找不到 {json_path}")
        return False

    results = E.load_json(json_path)
    cfg, _ = E.load_config()
    profile = cfg["profiles"][cfg.get("default_profile", "lora")]
    meta = load_meta()
    info = RUN_META.get(dirname, dict(exp_id="?", policy="record", runs=1, note=""))

    args = argparse.Namespace(
        profile=cfg.get("default_profile", "lora"), top_k=8, temperature=0.2,
        runs=info["runs"], template="v1", limit=0, dry_run=False, manual_fix="on",
        citation_policy=info["policy"], rewrite_cache="", rewrite_mode="term",
        query_instruction="off", keyword_patch="on", note=info["note"],
        exp_id=info["exp_id"], out_dir=d)

    before = metric_fingerprint(E.summarize(results))

    # ---- 补算（只加两个字段；runs 内所有答案一起算）----
    added = {"refused_raw": 0, "citation_pseudo_n": 0}
    for r in results:
        if "refused_raw" not in r:
            r["refused_raw"] = E.is_refusal(r.get("answer_raw") or r["answer"])
            added["refused_raw"] += 1
        if "citation_pseudo_n" not in r:
            r["citation_pseudo_n"] = E.count_pseudo_citations(r["answer"])
            added["citation_pseudo_n"] += 1

    after = metric_fingerprint(E.summarize(results))
    if before != after:
        print(f"[❌ 中止] {dirname}：补算前后指标不一致 → 不写盘（这是保护，不是 bug）")
        for i, (a, b) in enumerate(zip(before, after)):
            if a != b:
                print(f"    第 {i} 项：{a!r} → {b!r}")
        return False

    s = E.summarize(results)
    cc = s["citation"]
    print(f"[{dirname}] exp={info['exp_id']} policy={info['policy']}")
    print(f"  补 refused_raw {added['refused_raw']} 题、citation_pseudo_n {added['citation_pseudo_n']} 题"
          + (f"（跳过已有）" if added["refused_raw"] == 0 else ""))
    print(f"  成绩指标：补算前后**逐格相同** ✅  F1={s['f1']:.3f} "
          f"（TP{s['confusion']['tp']} FP{s['confusion']['fp']} "
          f"FN{s['confusion']['fn']} TN{s['confusion']['tn']}）")
    print(f"  新露出的信息：拒答翻转 {cc['guard_refused_changed']}｜伪引文 {cc['pseudo_n']} 处")

    if dry_run:
        print("  [dry-run] 不写盘。")
        return True

    # 备份 + 写回 + 重生成评测表
    bak = os.path.join(d, "eval_results.pre_backfill.json")
    if not os.path.exists(bak):
        shutil.copy2(json_path, bak)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    report = E.build_report(meta, results, s, args, profile)
    report += (
        "\n---\n\n> ⚠ **补算说明**：本表由 `backfill_citation_fields.py` 重生成 —— R6 / R6b 首跑于 9/21 早版脚本，"
        "当时还没有 `refused_raw` / `citation_pseudo_n` 两个字段。补算方式是**同一条代码、从已存的 `answer_raw` "
        "离线重算**（不调模型、不连向量库），脚本**断言了补算前后全部成绩指标逐格相同**；"
        "原始 json 备份在 `eval_results.pre_backfill.json`。\n")
    with open(os.path.join(d, "评测表.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  已写回 {os.path.relpath(json_path, SCRIPT_DIR)}、重生成 评测表.md（原 json 已备份）")
    return True


def main():
    ap = argparse.ArgumentParser(description="Day19 O2-G3 补算：给旧结果目录补 refused_raw / citation_pseudo_n")
    ap.add_argument("--dirs", nargs="*", default=DEFAULT_DIRS, help="结果目录名（默认两个 O2-G3 目录）")
    ap.add_argument("--dry-run", action="store_true", help="只报会补什么，不写盘")
    a = ap.parse_args()

    ok = all([backfill_one(d, a.dry_run) for d in a.dirs])
    print("\n[完成] 全部通过 ✅" if ok else "\n[完成] 有目录未通过（见上）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
