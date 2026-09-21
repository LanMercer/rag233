# -*- coding: utf-8 -*-
"""
rewrite_queries.py —— O1-R4 查询改写（Day19 核心产出）

作用：把 20 条评测题的中文问句，改写成"更贴英文论文语料"的检索查询，落盘成缓存文件。
      缓存下来有两个好处：① 只跑一次，省时间；② 可人工复核（改写质量本身要检查）。

设计纪律（Day19 第 1.2 节）：
    改写【只给检索层用】。生成层的问题仍是原中文问句 → 保证"只改一个变量"，
    也保证答案按原关键词判分。

用法：
    python rewrite_queries.py                    # 三档全跑（term / hyde / both）
    python rewrite_queries.py --modes term       # 只跑术语档
    python rewrite_queries.py --dry-run          # 不调模型，只写 plain（先验证流程）
    python rewrite_queries.py --show 3           # 打印第 3 题的改写结果，人工抽查

前置：8000 端口的模型服务已启动（第三周\\day13\\local_api_lora.py），日志出现「模型已就绪」。
"""
import argparse
import json
import os
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
QUESTIONS = os.path.join(REPO_ROOT, "第三周", "day13", "eval_questions.json")
DEFAULT_OUT = os.path.join(BASE_DIR, "queries_rewritten.json")
API_URL = "http://127.0.0.1:8000/v1/chat/completions"

SYSTEM_TERM = """你是检索查询改写助手。把用户的中文问题改写成适合检索英文技术论文的英文检索词。
要求：
1. 只输出英文关键词与术语，用英文逗号分隔，最多 8 个；
2. 保留问题里的实体（人名、机构、数据集、方法名、仓库名的原文拼写）；
3. 不要输出解释、不要输出句子、不要加引号或前缀。"""

SYSTEM_HYDE = """你是检索查询改写助手。请先假设一段英文论文里"回答了该问题"的文字，然后把它作为检索查询输出。
要求：
1. 只输出这段英文文字（1~2 句），不要解释、不要复述问题；
2. 尽量使用论文语体（如 "We evaluate on the LAFAN1 dataset..."）；
3. 如果不确定事实，只写"可能出现的表述"，不要编造具体数字。"""


def load_questions(path):
    """评测题文件是 {"meta":..., "questions":[...]}；也兼容直接给一个 list。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("questions", [])
    return data


def call_model(system, user, temperature=0.01, max_tokens=160, timeout=120):
    """调本地 OpenAI 兼容服务（就是评测用的那个 8000 端口，不额外占显存）。"""
    payload = {
        "model": "Qwen2.5-3B-Instruct-LoRA",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(API_URL, json=payload, timeout=timeout)
    if r.status_code != 200:
        # FastAPI 把真实异常放在响应体的 detail 里（服务端是 `raise HTTPException(500, f"生成失败：{e}")`）；
        # 只用 raise_for_status() 的话，终端只剩一句 "500 Server Error"，真正的原因会被吞掉。
        raise RuntimeError(f"服务端返回 {r.status_code}：{r.text[:500]}")
    # 常见 500 原因：temperature 必须 > 0（transformers 的 TemperatureLogitsWarper 校验，
    # 因服务端写死 do_sample=True，所以 temperature=0 会直接抛 ValueError）。
    return r.json()["choices"][0]["message"]["content"]


def clean(text):
    """去掉模型爱加的前后缀（'改写：' / 引号 / markdown 代码块 / 多余换行）。"""
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()      # 去代码块围栏
    t = re.sub(r"^(改写|查询|检索查询|查询改写)\s*[:：]\s*", "", t).strip()
    t = t.strip("\"'“”‘’ ")
    t = re.sub(r"\s*\n+\s*", " ", t).strip()                    # 合成单行
    return t


def rewrite_one(question, modes, dry_run=False):
    """返回 {"plain":..., "term":..., "hyde":..., "both":...}（只填要求跑且成功的档）。"""
    rec = {"plain": question}
    if dry_run:
        return rec
    if "term" in modes:
        rec["term"] = clean(call_model(SYSTEM_TERM, f"问题：{question}", max_tokens=96))
    if "hyde" in modes:
        rec["hyde"] = clean(call_model(SYSTEM_HYDE, f"问题：{question}", max_tokens=200))
    if "both" in modes:
        term = rec.get("term") or clean(call_model(SYSTEM_TERM, f"问题：{question}", max_tokens=96))
        hyde = rec.get("hyde") or clean(call_model(SYSTEM_HYDE, f"问题：{question}", max_tokens=200))
        rec["both"] = f"{term} | {hyde}"
    return rec


def main():
    ap = argparse.ArgumentParser(description="O1-R4 查询改写（Day19）")
    ap.add_argument("--questions", default=QUESTIONS, help="评测题 json")
    ap.add_argument("--out", default=DEFAULT_OUT, help="改写缓存输出路径")
    ap.add_argument("--modes", default="term,hyde,both", help="要跑的档位（逗号分隔）")
    ap.add_argument("--dry-run", action="store_true", help="不调模型，只写 plain")
    ap.add_argument("--show", type=int, default=None, help="打印某题的改写结果（人工抽查）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    if args.dry_run:
        print("⚠ dry-run：不调模型，只写 plain（用来先验证文件读写与后续 eval 的接线）")

    result = {}
    t0 = time.time()
    for i, q in enumerate(questions, start=1):
        print(f"[{i:>2}/{len(questions)}] id={q['id']:>2} {q['question']}")
        result[str(q["id"])] = rewrite_one(q["question"], modes, dry_run=args.dry_run)
    result["_meta"] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen2.5-3B-Instruct-LoRA",
        "modes": modes,
        "api_url": API_URL,
        "prompt_version": "day19-v1",
        "elapsed_sec": round(time.time() - t0, 1),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[落盘] {args.out}（共用时 {result['_meta']['elapsed_sec']} 秒）")

    if args.show:
        rec = result.get(str(args.show), {})
        print(f"\n=== 抽查 id={args.show} ===")
        for k in ("plain", "term", "hyde", "both"):
            if rec.get(k):
                print(f"[{k:>5}] {rec[k]}")
    print("\n下一步：人工抽查几条（重点看有没有编造术语），再跑 eval_v2.py --rewrite-cache")


if __name__ == "__main__":
    main()
