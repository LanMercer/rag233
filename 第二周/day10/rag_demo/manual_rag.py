# -*- coding: utf-8 -*-
r"""
手写简化版 RAG（配合 Day10 教程第 4 步使用）⭐ 不用 LangChain 检索组件，自己写"原理"

五件套说明（脚本在机器学习的哪一环）：
- 模型   ：bge-small-zh-v1.5（Embedding，CPU）+ Qwen2.5-3B-Instruct（生成，走 Day9 接口）
- 数据   ：build_index.py 生成的 chunks.json（同一批切块文本，保证和 LangChain 版可比）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事：问题向量化 → 自己算余弦相似度 → 自己取 Top-K → 拼模板 09 → 调本地接口
    ↑ 和 ask.py（LangChain 版）唯一的区别：检索部分全部自己写，生成部分仍是调 Day9 服务。
      跑完和 ask.py 对照，你就彻底看懂"检索"到底发生了什么。

前置条件（和 ask.py 一样）：
1. 已用 build_index.py 建库（至少 chunks.json 已生成）；
2. 已启动 Day9 服务（uvicorn local_api:app --host 127.0.0.1 --port 8000）。

运行方法：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第二周\day10\rag_demo"
    python manual_rag.py
"""

import os
import json

import numpy as np
import requests

# 终端编码加固：Windows 控制台默认 GBK，强制 UTF-8 输出防 UnicodeEncodeError / 乱码
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 第 0 区：配置
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CHUNKS_JSON = r".\chunks.json"                  # build_index.py 导出的切块文本
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "Qwen2.5-3B-Instruct"
EMBED_MODEL = r"D:\Lan\研究生\技术学习\大模型算法\download\bge-small-zh-v1.5"
TOP_K = 4
TEMPERATURE = 0.2

# bge 官方建议：检索时给"问题"加一个固定前缀，能提升检索效果；库内文档文本不需要加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

PROMPT_TEMPLATE = """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}"""

# ---------------------------------------------------------------------------
# 第 1 区：手写检索三件套（向量化 → 余弦相似度 → Top-K）
# ---------------------------------------------------------------------------

def load_chunks():
    """读 build_index.py 导出的切块文本（每个元素：{"id": N, "page": p, "text": "..."}）。"""
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts):
    """向量化：把一批文本各变成一个 512 维向量。返回 (数量, 512) 的 numpy 矩阵。"""
    from sentence_transformers import SentenceTransformer   # 延迟 import，只在运行时加载
    model = SentenceTransformer(EMBED_MODEL)
    # normalize_embeddings=True：让每个向量的长度=1。
    # 归一化之后，"点积"就等于"余弦相似度"（因为 |a||b| = 1），省一次除法。
    return model.encode(texts, normalize_embeddings=True)


def cosine_topk(query_vec, doc_vecs, chunk_ids, k):
    """
    手写"余弦相似度 + Top-K"：
      余弦相似度 cos(a,b) = (a·b) / (|a||b|)。
      向量都已归一化（长度=1），所以 cos = a·b = 点积。一行 `doc_vecs @ query_vec` 全算完。
      然后 argsort 从大到小取前 k 个 —— 这就是"打分 → 取 Top-K"。
    """
    scores = doc_vecs @ query_vec                     # (N,) 每个 chunk 与问题的相似度
    order = np.argsort(scores)[::-1][:k]              # 降序排列，取前 k 个下标
    return [(int(chunk_ids[i]), float(scores[i])) for i in order]


def build_prompt(chunks, top_k, question):
    """按模板 09 拼提示词：检索到的片段 → [N] 文本 → 填进【资料】。"""
    lines = [f"[{idx}] {chunks_by_id[idx]['text']}" for idx, _ in top_k]
    context = "\n".join(lines)
    return PROMPT_TEMPLATE.format(context=context, question=question)


# ---------------------------------------------------------------------------
# 第 2 区：调用 Day9 生成
# ---------------------------------------------------------------------------

def call_generate(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 512,
    }
    resp = requests.post(API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 第 3 区：主流程
# ---------------------------------------------------------------------------

chunks_by_id = {}   # 全局查表：id -> chunk（避免每次都用 list 下标）

def main():
    global chunks_by_id
    chunks = load_chunks()
    chunks_by_id = {c["id"]: c for c in chunks}
    print(f"读取 chunks.json：{len(chunks)} 个 chunk")

    # 1) 库内所有 chunk 向量化（CPU 约需 10~30 秒，只做一次）
    print("向量化全部 chunk（CPU 约 10~30 秒）...")
    doc_texts = [c["text"] for c in chunks]
    doc_vecs = embed_texts(doc_texts)
    chunk_ids = [c["id"] for c in chunks]

    questions = [
        "注意力机制的核心思想是什么？",
        "这篇论文提出的模型叫什么名字？",
        "位置编码在模型中的作用是什么？",
    ]

    for q in questions:
        # 2) 问题向量化（bge 检索给问题加前缀），拿到的向量形状 (1, 512)
        q_vec = embed_texts([QUERY_INSTRUCTION + q])
        # 3) 手写余弦相似度 + Top-K
        top_k = cosine_topk(q_vec[0], doc_vecs, chunk_ids, TOP_K)
        # 4) 拼模板 09
        prompt = build_prompt(chunks_by_id, top_k, q)
        # 5) 调 Day9 接口生成
        answer = call_generate(prompt)

        print("=" * 72)
        print(f"【问题】{q}")
        print("-" * 72)
        print("【手写版检索到的 Top-K（余弦相似度）】")
        for idx, score in top_k:
            preview = chunks_by_id[idx]["text"][:48].replace("\n", " ")
            print(f"  [资料§{idx}] 余弦相似度={score:.3f} | {preview}...")
        print("-" * 72)
        print(f"【答案】{answer}")


if __name__ == "__main__":
    main()
