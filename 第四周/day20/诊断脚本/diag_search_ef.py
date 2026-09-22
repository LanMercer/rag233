# -*- coding: utf-8 -*-
"""临时实验（用完即删）：把 hnsw:search_ef 提上去，能否让在线库的 Top-8 与精确余弦逐题一致。

背景：chromadb 0.5.15 的 segment/impl/vector/hnsw_params.py:58
      self.search_ef = int(metadata.get("hnsw:search_ef", 10))
      → 默认 ef=10，105 段语料的贪心图搜索会漏掉边界邻居（实测 8/20 题各差 1 条）。
"""
import io
import json
import os
import sys
import uuid

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"D:\Lan\研究生\技术学习\大模型算法"
SPACE = os.path.join(REPO, "第四周", "发布包", "space_demo")
sys.path.insert(0, SPACE)
import app as online

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

raw = json.load(io.open(os.path.join(SPACE, "chunks.json"), encoding="utf-8"))
E = HuggingFaceEmbeddings(model_name=online.EMBED_MODEL)
docs = [Document(page_content=c["text"], metadata={"chunk_id": c["id"]}) for c in raw]

D = np.asarray(E.embed_documents([c["text"] for c in raw]), dtype=np.float32)
D /= np.linalg.norm(D, axis=1, keepdims=True)

records = json.load(io.open(os.path.join(REPO, "第四周", "day20",
                                        "result_lora_v3_runs3", "eval_results.json"),
                            encoding="utf-8"))
qs = [r["question"] for r in records]
Q = np.asarray(E.embed_documents(qs), dtype=np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
exact = [[int(x) for x in row] for row in np.argsort(-(Q @ D.T), axis=1)[:, :8]]

CONFIGS = {
    "① 只 hnsw:space（现状）": {"hnsw:space": "cosine"},
    "② + search_ef=200": {"hnsw:space": "cosine", "hnsw:search_ef": 200},
    "③ + search_ef=200 & M=32 & construction_ef=400":
        {"hnsw:space": "cosine", "hnsw:search_ef": 200,
         "hnsw:M": 32, "hnsw:construction_ef": 400},
}

for name, meta in CONFIGS.items():
    store = Chroma.from_documents(
        documents=docs, embedding=E,
        collection_name="exp_" + uuid.uuid4().hex[:10],   # 必须唯一，否则会往同一 collection 追加
        collection_metadata=meta,
    )
    tot_local = tot_exact = exact_match = 0
    for i, r in enumerate(records):
        B = [c for c, _ in online.retrieve(store, r["question"], 8)]
        A = list(r["top_ids"])[:8]
        C = exact[i]
        tot_local += len(set(A) & set(B))
        tot_exact += len(set(B) & set(C))
        exact_match += (B == C)
    n = len(records)
    print(f"{name}")
    print(f"   与本地落盘 A 重合 {tot_local/n:.2f}/8 ｜ 与精确 C 重合 {tot_exact/n:.2f}/8"
          f" ｜ Top-8 逐题完全等于精确的题数 {exact_match}/{n}")
