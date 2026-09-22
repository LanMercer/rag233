# -*- coding: utf-8 -*-
"""临时实验（用完即删）：验证"建库线程退出后，另一个线程还能不能查"。

假设（读 chromadb/db/impl/sqlite_pool.py 得到）：
  内存库用 LockPool，连接存在 threading.local()，_connections 只存**弱引用**；
  建库线程一退出，连接被 GC → file::memory:?cache=shared 这个库随之销毁
  → 别的线程新建连接只能看到一个空库 → `no such table: collections`。
  落盘库用 PerThreadPool + 磁盘文件 → 表在文件里，与线程生死无关。

场景 E：短命线程建库（落盘，退出）→ 新线程查  → 期望 ✅
场景 F：短命线程建库（内存，退出）→ 新线程查  → 期望 ❌（复现线上那次报错）
"""
import io
import json
import os
import sys
import tempfile
import threading
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPACE = r"D:\Lan\研究生\技术学习\大模型算法\第四周\发布包\space_demo"
sys.path.insert(0, SPACE)
import app  # noqa: E402

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

raw = json.load(io.open(os.path.join(SPACE, "chunks.json"), encoding="utf-8"))
E = HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)
DOCS = [Document(page_content=c["text"], metadata={"chunk_id": c["id"]}) for c in raw]
Q = "论文的实验在哪个数据集上进行？"


def build_in_short_lived_thread(persist_dir=None):
    """在一次性线程里建库，线程随即结束（模拟预热线程）。"""
    box = {}

    def _job():
        kw = dict(collection_name="t_" + uuid.uuid4().hex[:8],
                  collection_metadata=app.HNSW_METADATA)
        if persist_dir:
            kw["persist_directory"] = persist_dir
        box["s"] = Chroma.from_documents(documents=DOCS, embedding=E, **kw)

    t = threading.Thread(target=_job)
    t.start()
    t.join()
    return box["s"]


def query_in_new_thread(store):
    """在一个全新的线程里查（模拟 Gradio worker 线程）。"""
    box = {}

    def _job():
        try:
            box["r"] = [c for c, _ in app.retrieve(store, Q, 4)]
        except Exception as e:
            box["r"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_job)
    t.start()
    t.join()
    return box["r"]


print("E 短命线程建【落盘】库 → 线程退出 → 新线程查")
pdir = os.path.join(tempfile.gettempdir(), "chroma_disk_" + uuid.uuid4().hex[:6])
print("   ", query_in_new_thread(build_in_short_lived_thread(pdir)))

print("F 短命线程建【内存】库 → 线程退出 → 新线程查（预期复现线上报错）")
print("   ", query_in_new_thread(build_in_short_lived_thread(None)))

print("\nG 多轮重复查落盘库（模拟多个访客 / worker 复用），看是否稳定")
store = build_in_short_lived_thread(pdir)
oks = 0
for i in range(6):
    r = query_in_new_thread(store)
    oks += not isinstance(r, str)
print(f"    成功 {oks}/6，示例返回：{query_in_new_thread(store)}")
