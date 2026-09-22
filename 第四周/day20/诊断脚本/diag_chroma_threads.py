# -*- coding: utf-8 -*-
"""临时实验（用完即删）：chromadb 的库能不能跨线程用？

线上 Demo 的真实形态：Gradio 用线程池跑 handler，而预热/建库在**另一个线程**。
组合测：
  A 建库线程 == 查询线程（内存库）
  B 建库线程 != 查询线程（内存库）
  C 建库线程 != 查询线程（**落盘**库，persist_directory）
  D 多线程池（模拟 Gradio worker 复用）分别查上面两种库
"""
import os
import tempfile
import sys
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPACE = r"D:\Lan\研究生\技术学习\大模型算法\第四周\发布包\space_demo"
sys.path.insert(0, SPACE)
import app  # noqa: E402

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import io, json, uuid  # noqa: E402
raw = json.load(io.open(os.path.join(SPACE, "chunks.json"), encoding="utf-8"))
E = HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)
DOCS = [Document(page_content=c["text"], metadata={"chunk_id": c["id"]}) for c in raw]
Q = "论文的实验在哪个数据集上进行？"


def make_store(persist_dir=None):
    kw = dict(collection_name="t_" + uuid.uuid4().hex[:8], collection_metadata=app.HNSW_METADATA)
    if persist_dir:
        kw["persist_directory"] = persist_dir
    return Chroma.from_documents(documents=DOCS, embedding=E, **kw)


def probe(store, tag):
    try:
        hits = [c for c, _ in app.retrieve(store, Q, 4)]
        return f"✅ {tag} → {hits}"
    except Exception as e:
        return f"❌ {tag} → {type(e).__name__}: {e}"


print("A 同线程（内存库）")
s = make_store()
print("  ", probe(s, "建库线程内查询"))

print("B 跨线程（内存库）：工作线程建库 → 主线程查")
box = {}


def worker_build():
    box["s"] = make_store()


import threading
t = threading.Thread(target=worker_build)
t.start()
t.join()
print("  ", probe(box["s"], "主线程查工作线程建的库"))

print("C 跨线程（落盘库）：工作线程建库 → 主线程查")
pdir = os.path.join(tempfile.gettempdir(), "chroma_smoke_" + uuid.uuid4().hex[:6])
box2 = {}
threading.Thread(target=lambda: box2.update(s=make_store(pdir))).start()
t.join()
print("  ", probe(box2["s"], "主线程查落盘库"))

print("D 线程池（模拟 Gradio worker 复用），每个库连查 8 次")
with ThreadPoolExecutor(max_workers=4) as ex:
    for tag, st in [("内存库", s), ("落盘库", box2["s"])]:
        res = list(ex.map(lambda _: probe(st, tag), range(8)))
        ok = sum(1 for r in res if r.startswith("✅"))
        print(f"   {tag}: 成功 {ok}/8")
        for r in res:
            if r.startswith("❌"):
                print("      ", r)
                break
