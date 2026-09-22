# -*- coding: utf-8 -*-
"""临时诊断（用完即删）：Top-8 差异是"我的库建错了"还是"HNSW 近似检索的固有抖动"。

三方对照（同一批 105 段、同一个 bge、都取 Top-8、都不加 instruction、都用原始问题）：
  A = 本地 F3v3 评测真实落盘的 top_ids（本地 Chroma，k=8）
  B = 在线默认库 Chroma（k=8）
  C = 精确余弦暴力检索（numpy，无近似，作为"真值"参照）
判读：C 更像谁，就说明谁更接近真实排名；A/B 之间的小差异则属近似检索抖动。
"""
import io
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"D:\Lan\研究生\技术学习\大模型算法"
SPACE = os.path.join(REPO, "第四周", "发布包", "space_demo")
sys.path.insert(0, SPACE)
import app as online

from langchain_huggingface import HuggingFaceEmbeddings

raw = json.load(io.open(os.path.join(SPACE, "chunks.json"), encoding="utf-8"))
E = HuggingFaceEmbeddings(model_name=online.EMBED_MODEL)

# 段落向量（顺序 = chunk_id 0..104）
D = np.asarray(E.embed_documents([c["text"] for c in raw]), dtype=np.float32)
D /= np.linalg.norm(D, axis=1, keepdims=True)

store, n = online.build_default_store()
print("内置库条数 :", store._collection.count(), "（期望 105，说明没有重复追加）")

records = json.load(io.open(os.path.join(REPO, "第四周", "day20",
                                        "result_lora_v3_runs3", "eval_results.json"),
                            encoding="utf-8"))
qs = [r["question"] for r in records]
Q = np.asarray(E.embed_documents(qs), dtype=np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)

exact = np.argsort(-(Q @ D.T), axis=1)[:, :8]   # 精确 Top-8

hit_ab = hit_ac = hit_bc = 0
print(f"\n{'题':<3}{'A∩C':<6}{'B∩C':<6}{'A∩B':<6}  A=本地落盘  B=在线  C=精确")
for i, r in enumerate(records):
    A = list(r["top_ids"])[:8]
    B = [c for c, _ in online.retrieve(store, r["question"], 8)]
    C = [int(x) for x in exact[i]]
    a_c, b_c, a_b = len(set(A) & set(C)), len(set(B) & set(C)), len(set(A) & set(B))
    hit_ac += a_c
    hit_bc += b_c
    hit_ab += a_b
    flag = "  ← A/B 不一致" if A != B else ""
    print(f"{r['id']:<3}{a_c:<6}{b_c:<6}{a_b:<6}{flag}")

n_q = len(records)
print(f"\n平均重合数（满分 8）：A(本地)∩C(精确) = {hit_ac/n_q:.2f}"
      f"  ｜  B(在线)∩C(精确) = {hit_bc/n_q:.2f}  ｜  A∩B = {hit_ab/n_q:.2f}")
print("\n判读：谁的 ∩C 更高，谁就更接近精确排名；A∩B 高但不为 8，说明差异只在边界处抖动。")
