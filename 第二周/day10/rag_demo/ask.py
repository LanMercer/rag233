# -*- coding: utf-8 -*-
r"""
RAG 问答脚本（配合 Day10 教程第 3 步使用）⭐ 全链路第 ⑤⑥ 步

五件套说明（脚本在机器学习的哪一环）：
- 模型   ：Qwen2.5-3B-Instruct（生成层，通过 Day9 的 /v1/chat/completions 接口调用）
- 数据   ：build_index.py 建好的 Chroma 向量库（检索来源）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事：把问题向量化 → 去 Chroma 库检索 Top-K 最相似的片段 → 按模板 09 拼提示词
              → 调 Day9 本地接口生成答案 → 打印带 [资料§N] 出处的回答
    ↑ 这就是 RAG 的"问答"阶段（在线，每次提问实时跑）。

前置条件（重要）：
1. 已用 build_index.py 建好库（.\\chroma_db 存在）；
2. 已启动 Day9 服务（另一个终端里跑着，别关）：
       uvicorn local_api:app --host 127.0.0.1 --port 8000

运行方法（新开一个终端）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第二周\day10\rag_demo"
    python ask.py
"""

import os
import requests

# ---------------------------------------------------------------------------
# 第 0 区：配置
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

API_URL = "http://127.0.0.1:8000/v1/chat/completions"   # Day9 本地服务地址
MODEL_NAME = "Qwen2.5-3B-Instruct"                     # 模型名（要跟 Day9 接口里的对上）
PERSIST_DIR = r".\chroma_db"                           # 向量库位置（build_index.py 建的）
EMBED_MODEL = r"D:\Lan\研究生\技术学习\大模型算法\download\bge-small-zh-v1.5"  # 向量模型必须和建库时一致！
TOP_K = 4          # 检索 Top-K：取最相似的 4 段（周计划避坑表：答非所问就 4→8）
TEMPERATURE = 0.2  # RAG 问答要"稳"，用低温（0.2）压熵；Day9 默认 0.7 对开放聊天合适

# 模板 09（Day 8 打磨好的 RAG 标准版）：{context} 是拼好的资料，{question} 是用户问题
PROMPT_TEMPLATE = """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}"""

# ---------------------------------------------------------------------------
# 第 1 区：导入库
# ---------------------------------------------------------------------------

from langchain_chroma import Chroma                          # 连向量库
from langchain_huggingface import HuggingFaceEmbeddings      # 问题也要向量化

# 终端编码加固：Windows 控制台默认 GBK，强制 UTF-8 输出防 UnicodeEncodeError / 乱码
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 第 2 区：加载向量库 + 一问一答
# ---------------------------------------------------------------------------

def load_store():
    """连接 build_index.py 建好的向量库（embedding 函数必须和建库时同一个模型）。"""
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)


def ask(store, question: str):
    """
    一次完整的 RAG 问答 = 检索（第 ⑤ 步）+ 生成（第 ⑥ 步）。

    第 ⑤ 步检索：store.similarity_search_with_relevance_scores()
      - 内部把 question 向量化，与库中每个 chunk 的向量算相似度（建库时指定了余弦空间），
        返回分数最高的 Top-K 个 chunk + 相似度分数；
      - 分数约在 0~1，越接近 1 说明和问题越相关。
    """
    docs = store.similarity_search_with_relevance_scores(question, k=TOP_K)

    # 拼上下文：每段变成 "[N] 文本"（N 就是建库时编的 chunk_id，正好对应模板 09 的 [资料§N]）
    lines = [f"[{doc.metadata.get('chunk_id', 0)}] {doc.page_content}" for doc, _ in docs]
    context = "\n".join(lines)

    # 第 ⑥ 步生成：套模板 09，POST 到 Day9 的 OpenAI 兼容接口
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 512,     # 多点答案可能超过 Day9 默认 200，放宽到 512
    }
    resp = requests.post(API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]

    # 打印：问题 → 检索到的 Top-K（含分数）→ 最终答案（含出处）
    print("=" * 72)
    print(f"【问题】{question}")
    print("-" * 72)
    print("【检索到的 Top-K 片段】")
    for doc, score in docs:
        idx = doc.metadata.get("chunk_id", 0)
        preview = doc.page_content[:48].replace("\n", " ")
        print(f"  [资料§{idx}] 相似度={score:.3f} | {preview}...")
    print("-" * 72)
    print(f"【答案】{answer}")


def main():
    # 内置 3 个基于默认 PDF（Attention 论文中文版）的问题；验收时换成你自己的 PDF 和问题
    questions = [
        "GMR的原理是什么？",
        "GMR的优点是什么？",
        "GMR的缺点是什么？",
    ]
    store = load_store()
    for q in questions:
        ask(store, q)


if __name__ == "__main__":
    main()
