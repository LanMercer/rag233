# -*- coding: utf-8 -*-
r"""
RAG 效果量化评测脚本（Day14）⭐ 出《首版评测表》

五件套说明（脚本在机器学习的哪一环）：
- 模型   ：生成层 = Qwen2.5-3B-Instruct-LoRA（4bit 底座 + Day12 微调的 LoRA adapter，
           通过 day13 的 local_api_lora.py 接口调用，OpenAI 兼容 /v1/chat/completions）；
           检索层 = bge-small-zh-v1.5（把问题变成向量，去 Chroma 里找相似段落）
- 数据   ：day13 定好的 20 条评测题（eval_questions.json，in_material 10 + out_of_material 10）
           + day13 建好的 GMR 论文向量库（..\day13\chroma_db）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试/评估环节）：
    for 每条评测题：
      ① 检索层：问题向量化 → Chroma 余弦相似度 Top-K 检索（记下命中哪些 chunk）
      ② 组织层：模板 09 把检索结果拼成上下文
      ③ 生成层：POST 本地微调接口 → 拿答案
      ④ 判分层：规则自动判分（in_material 看期望关键词是否命中；
                               out_of_material 看是否如实拒答"资料中没有提到"）
    最后统计三组指标（检索命中率 / 生成正确率 / 防幻觉正确率）+ 防幻觉 F1，
    输出《首版评测表.md》+ eval_results.json（明细留档）。
    ↑ 这属于机器学习的【评估（Evaluation）】环节——不训练、不动权重，只"打分"。

运行前置（顺序别乱）：
  1. day13 向量库已建好：..\day13\chroma_db 存在（没有就先跑 day13 的 build_index.py）；
  2. day13 微调接口服务已在另一个终端启动（别关）：
       cd "D:\Lan\研究生\技术学习\大模型算法\第三周\day13"
       python -m uvicorn local_api_lora:app --host 127.0.0.1 --port 8000
     等启动日志出现 "[OK] 模型已就绪：Qwen2.5-3B-Instruct-LoRA" 再跑本脚本；
  3. 本脚本会先 GET /health 检查服务，不通就直接退出并提示。

运行方法（llm 环境）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第三周\day14"
    python eval.py

运行后本文件夹新增：
    首版评测表.md     —— 每题一行 + 汇总指标（Day15 README / 效果评估报告直接引用）
    eval_results.json —— 全部明细（每题检索结果 / 答案 / 判定），复查与二次分析用
"""

import os
import sys
import json
import requests

# 镜像兜底（与本项目其它脚本保持一致）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# 第 0 区：配置（冲刺计划：本轮只出基线数字，默认配置，不做调优）
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 本文件所在目录（第三周\day14）

API_URL = "http://127.0.0.1:8000/v1/chat/completions"   # day13 微调接口地址
HEALTH_URL = "http://127.0.0.1:8000/health"             # 服务健康检查
MODEL_NAME = "Qwen2.5-3B-Instruct-LoRA"                 # 评测用模型（微调版）

EVAL_JSON = os.path.join(BASE_DIR, "..", "day13", "eval_questions.json")  # 20 条评测题（day13 定）
PERSIST_DIR = os.path.join(BASE_DIR, "..", "day13", "chroma_db")          # GMR 向量库（day13 建）
EMBED_MODEL = r"D:\Lan\研究生\技术学习\大模型算法\download\bge-small-zh-v1.5"  # 必须与建库时一致

TOP_K = 4            # 默认配置（冲刺计划：先出基线，调优放完结后 P1）
TEMPERATURE = 0.2    # RAG 评测用低温：压熵求稳，让回答可复现
MAX_TOKENS = 512     # 答案上限（模板 09 限 3 句，512 够用）

OUT_MD = os.path.join(BASE_DIR, "首版评测表.md")          # 首版评测表（本轮核心产出）
OUT_JSON = os.path.join(BASE_DIR, "eval_results.json")    # 每题明细（人工复核 + Day15 报告用）

# 模板 09（答案组织层，与 day10 ask.py 一字不差）
PROMPT_TEMPLATE = """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}"""

# 拒答词表：命中任一即认为模型"如实拒答"（防幻觉成功的信号）
# 注意顺序：长词在前、短词在后没关系（用 in 子串匹配，互不冲突）
REFUSAL_MARKERS = [
    "资料中没有提到", "资料中未提到", "资料中没有提及", "资料中未提及",
    "资料里没有", "资料中并未", "资料中并没有", "材料中没有", "文中没有",
    "没有提到", "没有提及", "未提到", "未提及", "未涉及",
    "没有相关内容", "无法从资料", "资料未提供", "没有找到相关内容",
]

# 人工复核修正表（跑完先人工看一遍明细，个别规则误判就在这里覆盖）：
#   例：MANUAL_FIXES = {13: {"correct": True, "detail": "答 arxiv 放宽算对"}}
#   没发现问题就保持空字典 {}

MANUAL_FIXES = {
    9: {"correct": False, "detail": "答的是 ProtoMotions 的仓库（NVLabs/ProtoMotions），不是 GMR 论文代码（YanjieZe/GMR），github 关键词碰巧命中，人工改判错"},
    18: {"correct": False, "detail": "先编造一串引用统计数字（425/679/248 等），末尾才补「资料中没有提到」——拒答词命中但内容已编造，属规则漏网，人工改判错"},
}
# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固 + 导入库
# ---------------------------------------------------------------------------
# Windows 控制台默认 GBK，打不出 emoji / 部分符号会崩，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain_chroma import Chroma                          # 连向量库
from langchain_huggingface import HuggingFaceEmbeddings      # 问题向量化

# ---------------------------------------------------------------------------
# 第 2 区：四个小函数（对应 检索 / 生成 / 两个判分规则）
# ---------------------------------------------------------------------------

def load_questions():
    """读 day13 的 20 条评测题，返回 (meta, questions)。"""
    with open(EVAL_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data["meta"], data["questions"]


def check_service():
    """GET /health，确认微调接口活着且模型已就绪。"""
    try:
        r = requests.get(HEALTH_URL, timeout=10)
        return r.status_code == 200 and r.json().get("model_ready") is True
    except Exception:
        return False


def load_store():
    """连接 day13 建好的 Chroma 向量库（embedding 必须与建库时同一个模型）。"""
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)


def retrieve(store, question):
    """
    检索层：question 向量化 → 余弦相似度 Top-K。
    返回 (docs, ids)：docs = [(Document, 相似度), ...]，ids = [chunk_id, ...]。
    """
    docs = store.similarity_search_with_relevance_scores(question, k=TOP_K)
    ids = []
    for d, _ in docs:
        try:
            ids.append(int(d.metadata.get("chunk_id")))
        except (TypeError, ValueError):
            ids.append(None)
    return docs, ids


def generate_answer(question, context):
    """生成层：套模板 09，POST 到本地微调接口，返回答案文本。"""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    resp = requests.post(API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def is_refusal(text):
    """判分规则①：模型是否"如实拒答"（说资料里没有）。"""
    return any(m in text for m in REFUSAL_MARKERS)


def hit_keywords(answer, keywords):
    """
    判分规则②：期望关键词是否命中答案。
    返回 (bool|None, 命中列表)。没给关键词 → 返回 (None, [])，留给人工判。
    """
    if not keywords:
        return None, []
    low = answer.lower()
    hit = [kw for kw in keywords if kw.lower() in low]
    return bool(hit), hit


def judge_one(q, top_ids, answer):
    """按题目类型自动判分，组装一条结果记录。"""
    qid = q["id"]
    rec = {
        "id": qid,
        "type": q["type"],
        "question": q["question"],
        "note": q.get("note", ""),
        "top_ids": top_ids,
        "source_chunk": q.get("source_chunk", []),
        "answer": answer,
        "refused": is_refusal(answer),
        "correct": None,   # True=对 / False=错 / None=无法自动判
        "detail": "",
    }

    if q["type"] == "in_material":
        # 资料内有答案：期望关键词命中即对
        hit, matched = hit_keywords(answer, q.get("expected_keywords", []))
        rec["correct"] = hit
        if hit is True:
            rec["detail"] = "命中关键词：" + "、".join(matched)
        elif hit is None:
            rec["detail"] = "未配关键词，需人工判分"
        elif rec["refused"]:
            rec["detail"] = "该答未答（资料里有答案却拒答）"
        else:
            rec["detail"] = "未命中任何期望关键词"
        # 检索命中：Top-K 里是否出现了 source_chunk 中的任一编号
        src = set(rec["source_chunk"])
        rec["retrieval_hit"] = bool(src) and bool(src & set(top_ids))
    else:
        # 资料外/反事实：如实拒答"资料中没有提到"才算防幻觉成功
        rec["correct"] = rec["refused"]
        rec["retrieval_hit"] = None  # out_of_material 没有"正确答案在哪个 chunk"的标准
        rec["detail"] = "如实拒答「资料中没有提到」" if rec["refused"] else "疑似幻觉：未拒答、自行编造或臆测"

    # 人工复核修正：自动规则误判时在这里覆盖
    fix = MANUAL_FIXES.get(qid)
    if fix is not None:
        if "correct" in fix:
            rec["correct"] = fix["correct"]
        if "detail" in fix:
            rec["detail"] = fix["detail"]
    return rec


# ---------------------------------------------------------------------------
# 第 3 区：指标统计 + 写《首版评测表.md》
# ---------------------------------------------------------------------------

def summarize(results):
    """把 20 条结果汇总成指标字典（三类正确率 + 防幻觉 F1 + 总正确率）。"""
    in_mat = [r for r in results if r["type"] == "in_material"]
    out_mat = [r for r in results if r["type"] == "out_of_material"]

    # ① 检索层命中率：只看带 source_chunk 的 in_material 题（Top-K 是否召回"应命中的段落"）
    hit_den = [r for r in in_mat if r.get("source_chunk")]
    hit_num = sum(1 for r in hit_den if r.get("retrieval_hit"))

    # ② 生成层正确率（in_material）：关键词命中率
    gen_den = [r for r in in_mat if r["correct"] is not None]
    gen_num = sum(1 for r in gen_den if r["correct"])

    # ③ 防幻觉正确率（out_of_material）：如实拒答率
    anti_den = [r for r in out_mat if r["correct"] is not None]
    anti_num = sum(1 for r in anti_den if r["correct"])

    # ④ 防幻觉 F1（二分类口径）：
    #    真实阳性 = out_of_material（资料里确实没有，应当拒答）
    #    模型判"阳性" = 拒答。TP=正确拒答 / FP=该答没答 / FN=该拒没拒（幻觉漏网）
    tp = anti_num
    fp = sum(1 for r in in_mat if r["refused"])       # 资料里有答案却拒答 → 错误拒答
    fn = len(out_mat) - tp                             # 资料里没有却不拒答 → 幻觉漏网
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0

    # ⑤ 总正确率（20 题宏观平均）
    judged = [r for r in results if r["correct"] is not None]
    overall = sum(1 for r in judged if r["correct"]) / len(judged) if judged else 0.0

    return {
        "total": len(results),
        "in_material": len(in_mat),
        "out_of_material": len(out_mat),
        "retrieval_hit": (hit_num, len(hit_den)),
        "gen_correct": (gen_num, len(gen_den)),
        "anti_correct": (anti_num, len(anti_den)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": len(in_mat) - fp},
        "precision": p, "recall": rec, "f1": f1,
        "overall": overall,
    }


def preview(text, n=80):
    """答案摘要：换行压成空格 + 截断，用于表格单元格。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[:n] + "…"


def build_report(meta, results, s):
    """把结果拼成《首版评测表.md》文本。"""
    lines = []
    lines.append("# 首版评测表（Day14 · 量化评测基线）\n")
    lines.append(f"- 项目：{meta['project']}")
    lines.append(f"- 知识库：{meta['knowledge_base']}")
    lines.append(f"- 评测题：{meta['total']} 条（in_material {meta['in_material']} + out_of_material {meta['out_of_material']}）")
    lines.append(f"- 评测模型：{MODEL_NAME}｜TOP_K={TOP_K}｜temperature={TEMPERATURE}｜模板 09")
    lines.append(f"- 评测题版本：{meta.get('version', 'v1')}（{meta.get('created_date', 'day13')} 定）\n")

    lines.append("## 一、逐题结果\n")
    lines.append("| id | 类型 | 检索命中（Top-K 召回 source_chunk） | 生成判定 | 判定说明 | 答案摘要 |")
    lines.append("|----|------|----------------------------------|----------|----------|----------|")
    for r in results:
        if r["type"] == "in_material":
            if r.get("retrieval_hit"):
                hit_cell = f"✅ {r.get('source_chunk')}∩{r['top_ids']}"
            else:
                hit_cell = f"❌ top={r['top_ids']}"
        else:
            hit_cell = "—（无标准）"
        mark = "✅" if r["correct"] else ("❌" if r["correct"] is False else "❓")
        lines.append(
            f"| {r['id']} | {r['type']} | {hit_cell} | {mark} | {r['detail']} | {preview(r['answer'], 60)} |"
        )

    lines.append("\n## 二、汇总指标\n")
    h_num, h_den = s["retrieval_hit"]
    g_num, g_den = s["gen_correct"]
    a_num, a_den = s["anti_correct"]
    c = s["confusion"]

    def pct(n, d):
        return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else "—"

    lines.append("| 指标 | 数值 | 含义 |")
    lines.append("|------|------|------|")
    lines.append(f"| 检索层命中率 | {pct(h_num, h_den)} | 应命中的段落是否被 Top-K 召回（只看 in_material 10 题） |")
    lines.append(f"| 生成层正确率（in_material） | {pct(g_num, g_den)} | 资料内有答案的题，期望关键词是否命中答案 |")
    lines.append(f"| 防幻觉正确率（out_of_material） | {pct(a_num, a_den)} | 资料外的题，是否如实答「资料中没有提到」 |")
    lines.append(f"| 总正确率（20 题） | {pct(sum(1 for r in results if r['correct'] is True), len([r for r in results if r['correct'] is not None]))} | 宏观平均 |")
    lines.append("")
    lines.append("### 防幻觉 F1（二分类口径：判'拒答'这事做得好不好）")
    lines.append("")
    lines.append("| 混淆矩阵 | 真实：资料外（out，应拒答） | 真实：资料内（in，应作答） |")
    lines.append("|----------|------------------------------|------------------------------|")
    lines.append(f"| 模型拒答 | TP = {c['tp']}（真防住幻觉） | FP = {c['fp']}（该答没答） |")
    lines.append(f"| 模型没拒答 | FN = {c['fn']}（幻觉漏网） | TN = {c['tn']}（正常作答） |")
    lines.append("")
    lines.append(f"- 精确率 = TP/(TP+FP) = {s['precision']:.3f}（拒答的时候，拒得对不对）")
    lines.append(f"- 召回率 = TP/(TP+FN) = {s['recall']:.3f}（该拒的题里，拒住了多少）")
    lines.append(f"- **F1 = 2·P·R/(P+R) = {s['f1']:.3f}**")
    lines.append("")

    lines.append("## 三、附录：问答明细（人工复核用）\n")
    for r in results:
        lines.append(f"### Q{r['id']}（{r['type']}）{r['question']}")
        lines.append(f"- 判定：{'✅' if r['correct'] else ('❌' if r['correct'] is False else '❓')} ｜ {r['detail']}")
        if r["type"] == "in_material":
            lines.append(f"- 检索 Top-K：{r['top_ids']}（期望 source_chunk：{r['source_chunk']}）")
        lines.append(f"- 备注（出题依据）：{r['note']}")
        lines.append(f"- 答案：{r['answer']}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 第 4 区：主流程
# ---------------------------------------------------------------------------

def main():
    meta, questions = load_questions()
    print("=" * 68)
    print(f"RAG 量化评测开始｜评测题 {meta['total']} 条（in {meta['in_material']} / out {meta['out_of_material']}）")
    print(f"模型 {MODEL_NAME}｜TOP_K={TOP_K}｜temperature={TEMPERATURE}｜模板 09")
    print("=" * 68)

    # 1) 服务检查
    if not check_service():
        print("[失败] 微调接口不通（/health 非 200 或 model_ready=False）。")
        print("  请先在 day13 目录启动服务：python -m uvicorn local_api_lora:app --host 127.0.0.1 --port 8000")
        print("  并等日志出现「[OK] 模型已就绪：Qwen2.5-3B-Instruct-LoRA」后重试。")
        sys.exit(1)
    print("[OK] 服务检查通过：model_ready=True\n")

    # 2) 加载向量库
    print("[OK] 加载向量库（bge 模型首次会慢一点）……")
    store = load_store()

    # 3) 逐题评测
    results = []
    try:
        for i, q in enumerate(questions, start=1):
            print("-" * 68)
            print(f"[{i:>2}/{len(questions)}] id={q['id']:>2} {q['type']}")
            print(f"   Q: {q['question']}")

            docs, ids = retrieve(store, q["question"])
            context = "\n".join(
                f"[{d.metadata.get('chunk_id')}] {d.page_content}" for d, _ in docs
            )
            answer = generate_answer(q["question"], context)
            rec = judge_one(q, ids, answer)
            results.append(rec)

            mark = "✅" if rec["correct"] else ("❌" if rec["correct"] is False else "❓")
            print(f"   Top-K: {ids}")
            print(f"   判定  : {mark} ｜ {rec['detail']}")
            print(f"   A: {preview(answer, 160)}")
            # 逐题落盘中间结果：即使中断也不丢已跑的题
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    except KeyboardInterrupt:
        print("\n[中断] 已保存已完成的题目到 eval_results.json，可继续跑或用它复查。")
    except Exception as e:
        print(f"\n[出错] {e}（已跑的题已保存到 eval_results.json）")

    if not results:
        print("没有任何结果，退出。")
        return

    # 4) 汇总 + 写报告
    s = summarize(results)
    report = build_report(meta, results, s)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 68)
    print("汇总指标")
    print("=" * 68)
    h_num, h_den = s["retrieval_hit"]
    g_num, g_den = s["gen_correct"]
    a_num, a_den = s["anti_correct"]
    if h_den:
        print(f"① 检索层命中率      : {h_num}/{h_den} = {100.0 * h_num / h_den:.1f}%")
    if g_den:
        print(f"② 生成层正确率(in)  : {g_num}/{g_den} = {100.0 * g_num / g_den:.1f}%")
    if a_den:
        print(f"③ 防幻觉正确率(out) : {a_num}/{a_den} = {100.0 * a_num / a_den:.1f}%")
    print(f"④ 总正确率(20题)    : {100.0 * s['overall']:.1f}%")
    c = s["confusion"]
    print(f"⑤ 防幻觉F1         : P={s['precision']:.3f} R={s['recall']:.3f} "
          f"F1={s['f1']:.3f}  （混淆: TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}）")
    print("=" * 68)
    print(f"[完成] 已生成：\n  {OUT_MD}\n  {OUT_JSON}")


if __name__ == "__main__":
    main()
