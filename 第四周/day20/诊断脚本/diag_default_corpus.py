# -*- coding: utf-8 -*-
"""临时诊断（用完即删）：① 重复文档/重复 collection ② chunk 36/4 为何在线榜上消失。"""
import io
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"D:\Lan\研究生\技术学习\大模型算法"
SPACE = os.path.join(REPO, "第四周", "发布包", "space_demo")
sys.path.insert(0, SPACE)
import app as online

raw = json.load(io.open(os.path.join(SPACE, "chunks.json"), encoding="utf-8"))
ids = [c["id"] for c in raw]
texts = [c["text"] for c in raw]
print("chunks.json 条数 :", len(raw))
print("  id 是否有重复 :", len(ids) != len(set(ids)),
      "| 重复 id :", [k for k, v in __import__("collections").Counter(ids).items() if v > 1])
print("  text 是否有重复:", len(texts) != len(set(texts)),
      "| 重复 text 的 id :",
      [c["id"] for c in raw if __import__("collections").Counter(texts)[c["text"]] > 1][:10])

print("\n--- 重复建库会不会往同一个 collection 追加？ ---")
s1, n1 = online.build_default_store()
print("  第 1 次建库：返回", n1, "段 | collection.count() =", s1._collection.count())
s2, n2 = online.build_default_store()
print("  第 2 次建库：返回", n2, "段 | collection.count() =", s2._collection.count())
print("  两个 store 是不是同一个 collection :", s1._collection.name, "/", s2._collection.name)

q = "论文的实验在哪个数据集上进行？"
print("\n--- 重复建库后同一问题的 Top-4 ---")
print("  第 1 个 store :", [c for c, _ in online.retrieve(s1, q, 4)])
print("  第 2 个 store :", [c for c, _ in online.retrieve(s2, q, 4)])

print("\n--- chunk 36 / 4 在库里吗？自查询自己的排名 ---")
for target in (36, 4):
    got = s1._collection.get(where={"chunk_id": target}, include=["metadatas", "documents"])
    print(f"  chunk_id={target} 命中条数 :", len(got["ids"]),
          "| 文本前 60 字 :", (got["documents"][0][:60].replace("\n", " ") if got["documents"] else "无"))
    txt = next(c["text"] for c in raw if c["id"] == target)
    hits = s1.similarity_search_with_relevance_scores(txt, k=5)
    print("     用它自己的原文去查，Top-5 :",
          [(d.metadata["chunk_id"], round(sc, 4)) for d, sc in hits])
