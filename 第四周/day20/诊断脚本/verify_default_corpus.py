# -*- coding: utf-8 -*-
"""临时校验脚本（用完即删）：验证 space_demo 的内置默认语料与项目离线库口径是否一致。

对照物不是"重新跑一遍本地库"（那是循环论证），而是**本地 F3v3 评测那轮真实落盘的
`top_ids`**（result_lora_v3_runs3/eval_results.json，TOP_K=8 / instruction off / 改写 off）。
做法：在线默认库用**原始问题、不加 instruction、k=8** 再检索一次，逐题比对 8 个 chunk_id。

另含：
  ① 两个模型路径（HF 名 vs 本地目录）是不是同一套权重
  ③ answer_question 不上传 PDF 时能否自动走内置库（generate 换假函数，不烧 API 额度）
"""
import io
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"D:\Lan\研究生\技术学习\大模型算法"
SPACE = os.path.join(REPO, "第四周", "发布包", "space_demo")
sys.path.insert(0, SPACE)

CFG = json.load(io.open(os.path.join(REPO, "第三周", "day15", "config.json"),
                        encoding="utf-8-sig"))
MODEL_LOCAL = CFG["embed_model"]
EVAL_JSON = os.path.join(REPO, "第四周", "day20", "result_lora_v3_runs3", "eval_results.json")

from langchain_huggingface import HuggingFaceEmbeddings

import app as online

print("=" * 78)
print("① 两个模型路径是否同权重（线上用 HF 名，项目用本地目录）")
print("   线上 HF 名 :", online.EMBED_MODEL)
print("   项目 本地  :", MODEL_LOCAL)
e_hf = HuggingFaceEmbeddings(model_name=online.EMBED_MODEL)
e_local = HuggingFaceEmbeddings(model_name=MODEL_LOCAL)
q0 = CFG["demo_questions"][1]
v_hf, v_local = e_hf.embed_query(q0), e_local.embed_query(q0)
print("   向量维度 :", len(v_hf), "| 逐元素完全相同 :", v_hf == v_local)

print("=" * 78)
print("② 在线默认库 Top-8  vs  本地 F3v3 评测真实落盘的 top_ids")
store, n = online.build_default_store()
print("   内置语料建库 :", n, "段（期望 105）")

records = json.load(io.open(EVAL_JSON, encoding="utf-8"))
same_n, diff = 0, []
for r in records:
    want = list(r["top_ids"])[:8]
    got = [cid for cid, _ in online.retrieve(store, r["question"], 8)]
    if want == got:
        same_n += 1
    else:
        diff.append((r["id"], r["question"], want, got))

print(f"   逐题完全一致 : {same_n}/{len(records)}")
for qid, q, want, got in diff:
    print(f"\n   ❌ Q{qid} {q}")
    print(f"        本地 : {want}")
    print(f"        在线 : {got}")
if diff:
    # 再看是不是"集合相同、仅序不同"（HNSW 近似 + 浮点边界抖动会造成这种情况）
    from collections import Counter
    set_same = sum(1 for _, _, w, g in diff if Counter(w) == Counter(g))
    print(f"\n   ↳ 其中『集合相同、仅顺序不同』的有 {set_same}/{len(diff)} 题")

print("=" * 78)
print("③ 不上传 PDF 直接提问（generate 已替换为假函数，不烧 API 额度）")
online.generate = lambda question, context: f"[假生成] 上下文 {len(context)} 字；问题：{question}"


class FakeProgress:
    def __call__(self, *a, **kw):
        pass


st, ans, srcs, snips = online.answer_question(None, q0, 4, FakeProgress())
print("   首次提问后 store 已回写 State :", st is not None)
print("   回答 :", ans.replace("\n", " ")[:120])
print("   出处 :", srcs)
print("   召回片段条数 :", snips.count("**["))

st2, ans2, _, _ = online.answer_question(st, CFG["demo_questions"][0], 4, FakeProgress())
print("   第二次提问复用同一个库（不再打\"本次使用内置语料\"前缀）:",
      st2 is st and not ans2.startswith("（本次使用"))

print("=" * 78)
print("结论：", "口径一致 ✅" if not diff else f"有 {len(diff)} 题不一致，见上 ⚠")
