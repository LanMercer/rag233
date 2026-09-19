# -*- coding: utf-8 -*-
r"""
RAG 评测脚本 v2 · 参数化版（第四周 Day16 · O0 尺子校准前半）⭐ 本周全部优化实验的"尺子"

它和第三周 day14 的 eval.py 是什么关系？
    eval.py   = 一次性尺子：模型/Top-K/温度全写死在代码里，只出"首版基线"那一组数字；
    eval_v2.py = 可配置尺子：同一份代码 + 不同命令行参数 = 一轮受控实验，
                 结果按参数分目录落盘，互不覆盖，且可随时复现（这就是 HR 要的"可归因优化"）。

五件套说明（这个脚本在机器学习的哪一环）：
- 模型   ：生成层 = 由 --profile 决定——
             lora     → Qwen2.5-3B-Instruct-LoRA（第三周 day12 微调，day13 local_api_lora.py 起服务，默认）
             original → Qwen2.5-3B-Instruct（原模型，第二周 day9 local_api.py 起服务）
           检索层 = bge-small-zh-v1.5（把问题向量化，去 Chroma 里找相似段落）
- 数据   ：第三周 day13 定好的 20 条评测题（eval_questions.json：in_material 10 + out_of_material 10）
           + 第三周 day13 建好的 GMR 论文向量库（chroma_db）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试/评估环节）：对每条题
           ① 检索：问题向量化 → Chroma 余弦相似度 Top-K（记下召回了哪些 chunk）
           ② 组织：按 --template 选模板（v1 = 第三周模板 09；v2 = 强化约束草稿）拼上下文
           ③ 生成：POST 本地 OpenAI 兼容接口拿答案（同一题重复 --runs 次）
           ④ 判分：规则判分（in 看期望关键词 / out 看是否如实拒答）
                   + 每题多跑时"多数投票" + 记录一致率（看生成稳不稳）
                   + 检索命中判据"双轨"：strict（Top-K ∩ source_chunk）+ loose（关键词是否落在 Top-K 上下文里）
           ⑤ 汇总：检索命中 / 生成正确 / 防幻觉 / 防幻觉 F1 / 总正确率 → 写评测表 + 明细 json
    ↑ 属于机器学习的【评估（Evaluation）】环节：不训练、不动权重，只"打分"。

配置从哪来（不重复造轮子）：
    复用 第三周\day15\config.json 的 profile（api_url / model_name / top_k / temperature）、
    embed_model、persist_dir（其相对路径以 config.json 所在目录为基准解析）。
    想换模型名/端口/embedding，只改那一份 config.json，本脚本不用动。

运行前置（顺序别乱）：
    1. 向量库在：第三周\day13\chroma_db（没有就先跑 day13 的 build_index.py）；
    2. 与 --profile 对应的服务已在 8000 端口运行（6G 显存一次只能跑一个 3B）：
         lora     → cd 第三周\day13 ; python -m uvicorn local_api_lora:app --host 127.0.0.1 --port 8000
         original → cd 第二周\day9  ; python -m uvicorn local_api:app --host 127.0.0.1 --port 8000
       等日志出现"[OK] 模型已就绪：…"再跑本脚本。

运行方法（llm 环境；先 cd 到本文件所在目录）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day16"

    # 看全部参数
    python eval_v2.py --help

    # D1 最小冒烟：只跑前 2 条题、1 次生成（不投票），确认"参数化"真的生效
    python eval_v2.py --top-k 4 --runs 1 --limit 2

    # D2 复跑基线（期望复现 检索 2/10、生成 2/10、防幻觉 6/10、F1≈0.667）
    python eval_v2.py --profile lora --top-k 4 --temperature 0.2 --runs 1 --note "baseline 复跑"

    # D2 首轮检索实验（一次只改一个变量）
    python eval_v2.py --top-k 8 --note "TOP_K 4->8"

    # 定稿数字用投票（耗时 ×3，只在关键轮次用）
    python eval_v2.py --top-k 8 --runs 3 --note "定稿：k=8 + 3 次投票"

参数一览：
    --profile      lora | original            默认取 config.json 的 default_profile
    --top-k        检索返回条数                默认 4
    --temperature  生成温度                    默认 0.2
    --runs         每题重复生成次数（≥3 取多数投票）默认 1
    --template     v1 | v2                    默认 v1（v2 为强化约束草稿，D4 再做并排消融）
    --out          输出目录                    默认 result_<profile>_k<K>[_temp<t>][_<template>][_runs<n>]
    --limit        只跑前 N 条（冒烟用，0=全部）  默认 0
    --max-tokens   单条答案上限                 默认 512
    --manual-fix   on | off                   默认 on（沿用 day14 的人工改判表，复现基线必须 on）
    --append-log   跑完把一行结果追加到 第四周\实验日志.md
    --note         本轮改了什么变量（写进日志行）

运行后输出（都在 --out 目录里）：
    评测表.md        —— 指标汇总 + 双轨检索命中 + 逐题结果（贴进报告/日志用）
    eval_results.json —— 全部明细（每题检索、答案、逐次判定、一致率）
    run_log.txt      —— 本轮完整运行日志（脚本自己用 UTF-8 写，避免 shell 重定向存成乱码）
"""

import argparse
import json
import os
import sys
from collections import Counter

import requests

# 镜像兜底（与本项目其它脚本保持一致）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固 + 导入 langchain 组件
# ---------------------------------------------------------------------------
# Windows 控制台默认 GBK，打不出 emoji / 部分符号会崩，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain_chroma import Chroma                          # 连向量库
from langchain_huggingface import HuggingFaceEmbeddings      # 问题向量化

# ---------------------------------------------------------------------------
# 第 2 区：路径常量（相对本脚本定位，换机器也能跑）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # 第四周\day16
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))   # 仓库根（大模型算法）

# 复用第三周 day15 的 config.json（profile / embed_model / persist_dir 都读它）
CONFIG_PATH = os.path.join(REPO_DIR, "第三周", "day15", "config.json")
# 默认评测题（第三周 day13 定）
DEFAULT_EVAL_JSON = os.path.join(REPO_DIR, "第三周", "day13", "eval_questions.json")
# 实验日志（第四周\实验日志.md）
DEFAULT_LOG_MD = os.path.join(REPO_DIR, "第四周", "实验日志.md")

# 拒答词表：命中任一即认为模型"如实拒答"（防幻觉成功的信号），与 day14 一字不差
REFUSAL_MARKERS = [
    "资料中没有提到", "资料中未提到", "资料中没有提及", "资料中未提及",
    "资料里没有", "资料中并未", "资料中并没有", "材料中没有", "文中没有",
    "没有提到", "没有提及", "未提到", "未提及", "未涉及",
    "没有相关内容", "无法从资料", "资料未提供", "没有找到相关内容",
]

# 模板库（答案组织层）：
#   v1 = 第三周模板 09（基线口径，和 day14 一字不差，复现基线必须用它）
#   v2 = 强化约束草稿（D1 先占位，day19 的 O2-G2 再做 v1/v2 并排消融）
#        改动点：① 弱化"无内容就直说"的触发条件 → 治"过拒答 FP"
#                ② 加"只能引用出现过的编号、禁止编造数字与引用" → 治"编造引用 FN"
PROMPT_TEMPLATES = {
    "v1": """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}""",
    "v2": """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

【规则】
1. 只输出答案本身，不要复述题目、不要输出无关解释；
2. 只有当上面资料确实找不到相关信息时，才回答「资料中没有提到」；资料里有依据就照实回答；
3. 只能引用资料中出现过的段落编号，格式为 [资料§N]；禁止编造编号、禁止编造数字或统计量；
4. 答案不超过 3 句。

【问题】
{question}""",
}

# 人工复核修正表（沿用第三周 day14，--manual-fix off 可关闭）：
#   自动规则会被"表面命中"骗过，跑完人工看一遍明细再在这里覆盖。
MANUAL_FIXES = {
    9: {"correct": False, "detail": "答的是 ProtoMotions 的仓库（NVLabs/ProtoMotions），不是 GMR 论文代码（YanjieZe/GMR），github 关键词碰巧命中，人工改判错"},
    18: {"correct": False, "detail": "先编造一串引用统计数字（425/679/248 等），末尾才补「资料中没有提到」——拒答词命中但内容已编造，属规则漏网，人工改判错"},
}

# ---------------------------------------------------------------------------
# 第 3 区：配置读取（复用 day15 config.json）
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p, base_dir):
    """相对路径按 base_dir 解析成绝对路径；已是绝对路径则原样返回。"""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def load_config():
    """读 day15 的 config.json，返回 (配置字典, 配置所在目录)。"""
    cfg = load_json(CONFIG_PATH)
    return cfg, os.path.dirname(CONFIG_PATH)


# ---------------------------------------------------------------------------
# 第 4 区：服务检查（活着 + 是不是我们想要的那个模型）
# ---------------------------------------------------------------------------

def api_base(api_url):
    """从 .../v1/chat/completions 截出服务根地址（用于拼 /health、/v1/models）。"""
    return api_url.rsplit("/v1/chat/completions", 1)[0]


def check_service(profile):
    """
    两查（沿用 day15 ask.py 的经验，防"服务在跑 ≠ 服务正确"）：
      ① GET /health      → 服务活着、模型加载完（model_ready=True）
      ② GET /v1/models   → 服务端真实模型名与 profile 一致
    返回 (是否通过, 提示文本)。
    """
    base = api_base(profile["api_url"])
    try:
        r = requests.get(base + "/health", timeout=10)
        r.raise_for_status()
        if r.json().get("model_ready") is not True:
            return False, "服务已启动但模型还没加载完（model_ready=False），等日志出现「模型已就绪」再跑。"
    except Exception as e:
        return False, f"连不上服务（{e}）。请先启动与 --profile 对应的服务再跑。"

    try:
        served = requests.get(base + "/v1/models", timeout=10).json()["data"][0]["id"]
    except Exception as e:
        return False, f"服务活着但读 /v1/models 失败（{e}）。"

    if served != profile["model_name"]:
        return False, (f"⚠ 服务端模型 = {served}，但 profile 要求 = {profile['model_name']}。"
                       "请切换服务（6G 显存一次只能跑一个 3B）后重跑。")
    return True, f"服务检查通过：{served} 已就绪，且与 profile 一致 ✅"


# ---------------------------------------------------------------------------
# 第 5 区：检索 / 生成 / 判分
# ---------------------------------------------------------------------------

def load_store(persist_dir, embed_model):
    """连接 Chroma 向量库；embedding 必须与建库时是同一个模型。"""
    embedder = HuggingFaceEmbeddings(model_name=embed_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)


def retrieve(store, question, top_k):
    """
    检索层：question 向量化 → 余弦相似度 Top-K。
    返回 (docs, ids)：docs = [(Document, 相似度), ...]，ids = [chunk_id, ...]。
    """
    docs = store.similarity_search_with_relevance_scores(question, k=top_k)
    ids = []
    for d, _ in docs:
        try:
            ids.append(int(d.metadata.get("chunk_id")))
        except (TypeError, ValueError):
            ids.append(None)
    return docs, ids


def build_context(docs):
    """把检索到的片段拼成上下文文本（模板里 {context} 的内容）。"""
    return "\n".join(f"[{d.metadata.get('chunk_id')}] {d.page_content}" for d, _ in docs)


def generate_answer(profile, template, question, context, temperature, max_tokens, dry_run=False):
    """生成层：套模板 → POST 本地接口 → 返回答案文本。dry_run=True 时用桩答案（不连模型服务）。"""
    if dry_run:
        return "【dry-run 桩答案】资料中没有提到。"   # 只验证流程，数字无意义
    prompt = template.format(context=context, question=question)
    payload = {
        "model": profile["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(profile["api_url"], json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def is_refusal(text):
    """判分规则①：模型是否"如实拒答"（说资料里没有）。"""
    return any(m in text for m in REFUSAL_MARKERS)


def hit_keywords(answer, keywords):
    """判分规则②：期望关键词是否命中答案。返回 (bool|None, 命中列表)。"""
    if not keywords:
        return None, []
    low = answer.lower()
    hit = [kw for kw in keywords if kw.lower() in low]
    return bool(hit), hit


def judge_once(q, answer):
    """对"单次生成"判分，返回 (correct|None, detail)。口径与 day14 一致。"""
    if q["type"] == "in_material":
        hit, matched = hit_keywords(answer, q.get("expected_keywords", []))
        if hit is True:
            return True, "命中关键词：" + "、".join(matched)
        if hit is None:
            return None, "未配关键词，需人工判分"
        if is_refusal(answer):
            return False, "该答未答（资料里有答案却拒答）"
        return False, "未命中任何期望关键词"
    # out_of_material / 反事实：如实拒答才算防幻觉成功
    ok = is_refusal(answer)
    return ok, ("如实拒答「资料中没有提到」" if ok else "疑似幻觉：未拒答、自行编造或臆测")


def judge_question(q, top_ids, context, answers):
    """
    对一条题的全部重复生成做判定 + 多数投票。
    answers = [第1次答案, 第2次答案, ...]（--runs 次）
    返回一条结果记录（含 strict/loose 双轨检索命中、投票结果、一致率）。
    """
    per_run = []
    for a in answers:
        ok, detail = judge_once(q, a)
        per_run.append({"correct": ok, "detail": detail, "answer": a})

    # --- 多数投票：取判定众数；平票时取第一次的判定（保守，不做额外猜测）---
    valid = [p["correct"] for p in per_run if p["correct"] is not None]
    if valid:
        counter = Counter(valid)
        top_count = max(counter.values())
        winners = [k for k, v in counter.items() if v == top_count]
        final_correct = per_run[0]["correct"] if len(winners) > 1 else winners[0]
        consistency = top_count / len(valid)
    else:
        final_correct, consistency = None, 0.0

    # 代表的答案：取与最终判定相同判定的第一次答案（便于贴表）
    repr_answer = next((p["answer"] for p in per_run if p["correct"] == final_correct), per_run[0]["answer"])
    repr_detail = next((p["detail"] for p in per_run if p["correct"] == final_correct), per_run[0]["detail"])

    rec = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "note": q.get("note", ""),
        "top_ids": top_ids,
        "source_chunk": q.get("source_chunk", []),
        "answer": repr_answer,
        "refused": is_refusal(repr_answer),
        "correct": final_correct,
        "detail": repr_detail,
        "runs": len(answers),
        "consistency": round(consistency, 3),
        "per_run": per_run,   # 每次生成的答案 + 规则原判（--manual-fix off 时用它还原规则判定）
    }

    # --- 检索命中：双轨判据（口径透明，两项并列报）---
    src = set(rec["source_chunk"])
    if q["type"] == "in_material":
        # strict：Top-K 里是否真出现 source_chunk 中任一编号（day14 旧口径，偏严）
        rec["retrieval_hit_strict"] = bool(src) and bool(src & set(top_ids))
        # loose：期望关键词是否落在 Top-K 的上下文里（治 Q4 那种"没召回该段但答对了"的反例）
        kws = q.get("expected_keywords", [])
        if kws:
            low_ctx = context.lower()
            rec["retrieval_hit_loose"] = any(kw.lower() in low_ctx for kw in kws)
        else:
            rec["retrieval_hit_loose"] = None
        # 兼容旧字段名（方便和 day14 结果对照）
        rec["retrieval_hit"] = rec["retrieval_hit_strict"]
    else:
        rec["retrieval_hit_strict"] = None
        rec["retrieval_hit_loose"] = None
        rec["retrieval_hit"] = None

    # --- 人工复核修正（--manual-fix on 时生效，复现基线必须 on）---
    fix = MANUAL_FIXES.get(q["id"])
    if fix is not None:
        if "correct" in fix:
            rec["correct"] = fix["correct"]
        if "detail" in fix:
            rec["detail"] = fix["detail"]
        rec["manual_fixed"] = True
    else:
        rec["manual_fixed"] = False
    return rec


# ---------------------------------------------------------------------------
# 第 6 区：指标汇总
# ---------------------------------------------------------------------------

def summarize(results):
    """把结果汇总成指标字典（三组正确率 + 双轨检索命中 + 防幻觉 F1 + 总正确率 + 投票一致率）。"""
    in_mat = [r for r in results if r["type"] == "in_material"]
    out_mat = [r for r in results if r["type"] == "out_of_material"]

    # ① 检索层命中率（双轨，只看带 source_chunk 的 in_material 题）
    hit_den = [r for r in in_mat if r.get("source_chunk")]
    strict_num = sum(1 for r in hit_den if r.get("retrieval_hit_strict"))
    loose_den = [r for r in hit_den if r.get("retrieval_hit_loose") is not None]
    loose_num = sum(1 for r in loose_den if r.get("retrieval_hit_loose"))

    # ② 生成层正确率（in_material）
    gen_den = [r for r in in_mat if r["correct"] is not None]
    gen_num = sum(1 for r in gen_den if r["correct"])

    # ③ 防幻觉正确率（out_of_material）
    anti_den = [r for r in out_mat if r["correct"] is not None]
    anti_num = sum(1 for r in anti_den if r["correct"])

    # ④ 防幻觉 F1（二分类口径，与 day14 完全一致）
    tp = anti_num
    fp = sum(1 for r in in_mat if r["refused"])   # 资料里有答案却拒答
    fn = len(out_mat) - tp                         # 该拒没拒（幻觉漏网）
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0

    # ⑤ 总正确率（宏观平均）
    judged = [r for r in results if r["correct"] is not None]
    overall = sum(1 for r in judged if r["correct"]) / len(judged) if judged else 0.0

    # ⑥ 投票一致率（--runs ≥ 2 才有意义）：看生成"稳不稳"
    cons = [r["consistency"] for r in results if r.get("runs", 1) > 1]

    return {
        "total": len(results),
        "in_material": len(in_mat),
        "out_of_material": len(out_mat),
        "retrieval_hit_strict": (strict_num, len(hit_den)),
        "retrieval_hit_loose": (loose_num, len(loose_den)),
        "gen_correct": (gen_num, len(gen_den)),
        "anti_correct": (anti_num, len(anti_den)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": len(in_mat) - fp},
        "precision": p, "recall": rec, "f1": f1,
        "overall": overall,
        "avg_consistency": (sum(cons) / len(cons)) if cons else None,
    }


# ---------------------------------------------------------------------------
# 第 7 区：写评测表 + 追加实验日志
# ---------------------------------------------------------------------------

def preview(text, n=80):
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[:n] + "…"


class Tee:
    """
    把输出同时写给"屏幕 + 文件"的小工具。
    为什么要它：PowerShell 的 `>` / `Tee-Object` 重定向会按控制台编码解码子进程输出，
    中文路径/中文输出下容易存成乱码（实测踩过）。让 Python 自己用 UTF-8 写日志，最稳。
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def pct(n, d):
    return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else "—"


def build_report(meta, results, s, args, profile):
    """把结果拼成《评测表.md》文本。"""
    lines = []
    lines.append(f"# RAG 评测表 v2（Day16 · profile={args.profile} · k={args.top_k} · "
                 f"temp={args.temperature} · runs={args.runs} · template={args.template}）\n")
    lines.append(f"- 项目：{meta['project']}")
    lines.append(f"- 知识库：{meta['knowledge_base']}")
    lines.append(f"- 评测题：{meta['total']} 条（in_material {meta['in_material']} + out_of_material {meta['out_of_material']}）")
    if args.limit:
        lines.append(f"- ⚠ 本次为**冒烟子集**：只跑了前 {args.limit} 条题（--limit {args.limit}），指标不代表完整 20 题")
    if args.dry_run:
        lines.append("- ⚠ 本次为 **dry-run**（生成用桩答案、未连模型服务）：只验证脚本流程，**指标数字无意义**")
    lines.append(f"- 评测模型：{profile['model_name']}｜TOP_K={args.top_k}｜temperature={args.temperature}｜"
                 f"模板={args.template}｜--runs={args.runs}｜人工改判={'on' if args.manual_fix else 'off'}")
    if args.note:
        lines.append(f"- 本轮改动：{args.note}")
    lines.append("")

    lines.append("## 一、汇总指标\n")
    sn, sd = s["retrieval_hit_strict"]
    ln, ld = s["retrieval_hit_loose"]
    gn, gd = s["gen_correct"]
    an, ad = s["anti_correct"]
    c = s["confusion"]
    lines.append("| 指标 | 数值 | 含义 |")
    lines.append("|------|------|------|")
    lines.append(f"| 检索层命中率 · strict | {pct(sn, sd)} | Top-K 与 source_chunk 有交集（day14 旧口径，偏严） |")
    lines.append(f"| 检索层命中率 · loose | {pct(ln, ld)} | 期望关键词是否落在 Top-K 上下文里（新口径，治 Q4 反例） |")
    lines.append(f"| 生成层正确率（in_material） | {pct(gn, gd)} | 资料内有答案的题，期望关键词是否命中答案 |")
    lines.append(f"| 防幻觉正确率（out_of_material） | {pct(an, ad)} | 资料外的题，是否如实答「资料中没有提到」 |")
    lines.append(f"| 总正确率（全部题） | {pct(sum(1 for r in results if r['correct'] is True), len([r for r in results if r['correct'] is not None]))} | 宏观平均 |")
    if s["avg_consistency"] is not None:
        lines.append(f"| 生成一致率（同题多跑多数占比） | {s['avg_consistency']:.3f} | 越大说明生成越稳；小说明单次数字有随机性 |")
    lines.append("")
    lines.append("### 防幻觉 F1（二分类口径：判'拒答'这事做得好不好）\n")
    lines.append("| 混淆矩阵 | 真实：资料外（out，应拒答） | 真实：资料内（in，应作答） |")
    lines.append("|----------|------------------------------|------------------------------|")
    lines.append(f"| 模型拒答 | TP = {c['tp']}（真防住幻觉） | FP = {c['fp']}（该答没答） |")
    lines.append(f"| 模型没拒答 | FN = {c['fn']}（幻觉漏网） | TN = {c['tn']}（正常作答） |")
    lines.append("")
    lines.append(f"- 精确率 = TP/(TP+FP) = {s['precision']:.3f}（拒答的时候，拒得对不对）")
    lines.append(f"- 召回率 = TP/(TP+FN) = {s['recall']:.3f}（该拒的题里，拒住了多少）")
    lines.append(f"- **F1 = 2·P·R/(P+R) = {s['f1']:.3f}**")
    lines.append("")

    lines.append("## 二、逐题结果\n")
    lines.append("| id | 类型 | 检索 strict | 检索 loose | 生成判定 | 一致率 | 判定说明 | 答案摘要 |")
    lines.append("|----|------|-------------|------------|----------|--------|----------|----------|")
    for r in results:
        if r["type"] == "in_material":
            st = f"✅ {r['source_chunk']}∩{r['top_ids']}" if r.get("retrieval_hit_strict") else f"❌ top={r['top_ids']}"
            lo = "✅" if r.get("retrieval_hit_loose") else "❌"
        else:
            st, lo = "—", "—"
        mark = "✅" if r["correct"] else ("❌" if r["correct"] is False else "❓")
        cons = f"{r['consistency']:.2f}" if r.get("runs", 1) > 1 else "—"
        lines.append(f"| {r['id']} | {r['type']} | {st} | {lo} | {mark} | {cons} | {r['detail']} | {preview(r['answer'], 60)} |")

    lines.append("\n## 三、附录：每题完整答案（人工复核用）\n")
    for r in results:
        lines.append(f"### Q{r['id']}（{r['type']}）{r['question']}")
        lines.append(f"- 判定：{'✅' if r['correct'] else ('❌' if r['correct'] is False else '❓')} ｜ {r['detail']}"
                     + ("（人工改判）" if r.get("manual_fixed") else ""))
        if r["type"] == "in_material":
            lines.append(f"- 检索 Top-K：{r['top_ids']}（期望 source_chunk：{r['source_chunk']}）")
        lines.append(f"- 备注（出题依据）：{r['note']}")
        lines.append(f"- 答案：{r['answer']}\n")
    return "\n".join(lines)


def append_log(args, s, profile):
    """把本轮结果追加成一行到 第四周\实验日志.md（--append-log 时）"""
    sn, sd = s["retrieval_hit_strict"]
    ln, ld = s["retrieval_hit_loose"]
    gn, gd = s["gen_correct"]
    an, ad = s["anti_correct"]
    row = (f"| {args.date} | {args.exp_id} | {args.note or '—'} | "
           f"{profile['model_name']} / k={args.top_k} / T={args.temperature} / runs={args.runs} / {args.template} | "
           f"{pct(sn, sd)} | {pct(ln, ld)} | {pct(gn, gd)} | {pct(an, ad)} | {s['f1']:.3f} | "
           f"{pct(sum(1 for r in _last_results if r['correct'] is True), len([r for r in _last_results if r['correct'] is not None]))} | "
           f"{args.conclusion or '（待填）'} | {os.path.basename(args.out_dir)} |")
    header_needed = not os.path.exists(DEFAULT_LOG_MD)
    with open(DEFAULT_LOG_MD, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# 第四周实验日志（每轮只改 1 个变量：改了啥 → 数字 → 结论 → 下一步）\n\n")
        f.write(row + "\n")
    return DEFAULT_LOG_MD


# ---------------------------------------------------------------------------
# 第 8 区：主流程
# ---------------------------------------------------------------------------
_last_results = []   # 给 append_log 用的最近一次结果（避免多传一层参数）


def main():
    parser = argparse.ArgumentParser(
        description="RAG 评测脚本 v2（参数化 + 多数投票 + 双轨判据）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--profile", default=None, help="lora | original（默认取 config.json 的 default_profile）")
    parser.add_argument("--top-k", type=int, default=4, help="检索返回条数（默认 4）")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度（默认 0.2）")
    parser.add_argument("--runs", type=int, default=1, help="每题重复生成次数，≥3 时取多数投票（默认 1）")
    parser.add_argument("--template", default="v1", choices=list(PROMPT_TEMPLATES), help="提示词模板版本（默认 v1）")
    parser.add_argument("--out", default=None, help="输出目录（默认 result_<profile>_k<K>...）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用；0=全部）")
    parser.add_argument("--max-tokens", type=int, default=512, help="单条答案上限（默认 512）")
    parser.add_argument("--manual-fix", default="on", choices=["on", "off"], help="是否应用人工改判表（默认 on）")
    parser.add_argument("--append-log", action="store_true", help="把本轮结果追加一行到 第四周\\实验日志.md")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连模型服务，生成用桩答案（只验证脚本流程；数字无意义）")
    parser.add_argument("--note", default="", help="本轮改了哪个变量（写进评测表与日志）")
    parser.add_argument("--exp-id", default="", help="实验编号（如 R1），写日志用")
    parser.add_argument("--date", default="", help="实验日期，写日志用（如 9/18）")
    parser.add_argument("--conclusion", default="", help="本轮结论（写日志用）")
    args = parser.parse_args()

    cfg, cfg_dir = load_config()
    profile_name = args.profile or cfg.get("default_profile", "lora")
    if profile_name not in cfg["profiles"]:
        print(f"❌ 没有这个 profile：{profile_name}。可选：{list(cfg['profiles'])}")
        sys.exit(1)
    args.profile = profile_name
    profile = cfg["profiles"][profile_name]

    # 输出目录命名：默认 result_<profile>_k<K>；非默认旋钮加后缀，保证每轮互不覆盖
    if args.out:
        out_dir = args.out if os.path.isabs(args.out) else os.path.join(SCRIPT_DIR, args.out)
    else:
        name = f"result_{profile_name}_k{args.top_k}"
        if args.temperature != 0.2:
            name += f"_temp{args.temperature}"
        if args.template != "v1":
            name += f"_{args.template}"
        if args.runs > 1:
            name += f"_runs{args.runs}"
        if args.dry_run:
            name += "_dryrun"
        out_dir = os.path.join(SCRIPT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir

    # 本轮运行日志（屏幕 + 文件双写；Python 自己用 UTF-8 写，避免 shell 重定向乱码）
    log_path = os.path.join(out_dir, "run_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout

    eval_json = DEFAULT_EVAL_JSON
    if not os.path.exists(eval_json):
        print(f"❌ 找不到评测题文件：{eval_json}")
        log_file.close()
        sys.exit(1)
    data = load_json(eval_json)
    meta, questions = data["meta"], data["questions"]
    if args.limit > 0:
        questions = questions[:args.limit]

    template = PROMPT_TEMPLATES[args.template]
    persist_dir = resolve_path(cfg["persist_dir"], cfg_dir)
    embed_model = cfg["embed_model"]

    print("=" * 72)
    print(f"RAG 评测 v2 开始｜profile={profile_name}｜TOP_K={args.top_k}｜temp={args.temperature}｜"
          f"runs={args.runs}｜template={args.template}")
    print(f"评测题 {len(questions)} 条（本次运行）｜输出目录：{os.path.relpath(out_dir, SCRIPT_DIR)}")
    if args.note:
        print(f"本轮改动：{args.note}")
    print("=" * 72)

    # 1) 服务两查（dry-run 跳过）
    if args.dry_run:
        print("[dry-run] 跳过服务检查与真实生成：答案用桩文本，只验证脚本流程，指标数字无意义。")
    else:
        ok, msg = check_service(profile)
        print(msg)
        if not ok:
            print("→ 先启动与 profile 匹配的服务，再重跑本脚本。")
            sys.exit(1)

    # 2) 加载向量库
    print(f"[OK] 加载向量库：{persist_dir}")
    store = load_store(persist_dir, embed_model)

    # 3) 逐题评测（每题检索一次、生成 runs 次）
    results = []
    out_json = os.path.join(out_dir, "eval_results.json")
    try:
        for i, q in enumerate(questions, start=1):
            print("-" * 72)
            print(f"[{i:>2}/{len(questions)}] id={q['id']:>2} {q['type']}｜{q['question']}")
            docs, ids = retrieve(store, q["question"], args.top_k)
            context = build_context(docs)

            answers = []
            for r in range(args.runs):
                answers.append(generate_answer(profile, template, q["question"], context,
                                               args.temperature, args.max_tokens, dry_run=args.dry_run))

            rec = judge_question(q, ids, context, answers)
            if not args.manual_fix:
                # 关掉人工改判时，把判定还原成"规则原判"（per_run[0] 的规则结果）
                rec["correct"] = rec["per_run"][0]["correct"]
                rec["detail"] = "（规则原判）" + rec["per_run"][0]["detail"]
                rec["manual_fixed"] = False
            results.append(rec)

            mark = "✅" if rec["correct"] else ("❌" if rec["correct"] is False else "❓")
            print(f"   Top-K: {ids}")
            print(f"   检索  : strict={'✅' if rec['retrieval_hit_strict'] else ('❌' if rec['retrieval_hit_strict'] is False else '—')}"
                  f"  loose={'✅' if rec['retrieval_hit_loose'] else ('❌' if rec['retrieval_hit_loose'] is False else '—')}")
            if args.runs > 1:
                print(f"   一致率: {rec['consistency']:.2f}（{args.runs} 次生成）")
            print(f"   判定  : {mark} ｜ {rec['detail']}")
            print(f"   A: {preview(rec['answer'], 160)}")
            # 逐题落盘：中断也不丢已跑的题
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    except KeyboardInterrupt:
        print("\n[中断] 已保存已完成的题目到 eval_results.json。")
    except Exception as e:
        print(f"\n[出错] {e}（已跑的题已保存）")

    if not results:
        print("没有任何结果，退出。")
        log_file.close()
        return

    # 4) 汇总 + 写表
    global _last_results
    _last_results = results
    s = summarize(results)
    report_path = os.path.join(out_dir, "评测表.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report(meta, results, s, args, profile))

    print("\n" + "=" * 72)
    print("汇总指标")
    print("=" * 72)
    print(f"① 检索命中 strict : {pct(*s['retrieval_hit_strict'])}")
    print(f"① 检索命中 loose  : {pct(*s['retrieval_hit_loose'])}")
    print(f"② 生成正确率(in)  : {pct(*s['gen_correct'])}")
    print(f"③ 防幻觉正确率(out): {pct(*s['anti_correct'])}")
    print(f"④ 总正确率        : {100.0 * s['overall']:.1f}%"
          f"（{sum(1 for r in results if r['correct'] is True)}/{len([r for r in results if r['correct'] is not None])}）")
    c = s["confusion"]
    print(f"⑤ 防幻觉F1        : P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
          f"（TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}）")
    if s["avg_consistency"] is not None:
        print(f"⑥ 生成一致率      : {s['avg_consistency']:.3f}")
    print("=" * 72)
    print(f"[完成] 已生成：\n  {report_path}\n  {out_json}")

    # 5) 实验日志（可选）
    if args.append_log:
        log_path = append_log(args, s, profile)
        print(f"[日志] 已追加一行到：{log_path}")
    else:
        print("\n[日志行] 想记进实验日志，可复制下面这行（或加 --append-log 自动追加）：")
        sn, sd = s["retrieval_hit_strict"]
        ln, ld = s["retrieval_hit_loose"]
        gn, gd = s["gen_correct"]
        an, ad = s["anti_correct"]
        print(f"| {args.date or '日期'} | {args.exp_id or '编号'} | {args.note or '改动'} | k={args.top_k} T={args.temperature} "
              f"runs={args.runs} {args.template} | {pct(sn, sd)} | {pct(ln, ld)} | {pct(gn, gd)} | {pct(an, ad)} | "
              f"F1={s['f1']:.3f} | | {os.path.basename(out_dir)} |")

    print(f"[日志文件] 本轮完整运行日志：{log_path}")
    log_file.flush()
    log_file.close()


if __name__ == "__main__":
    main()
