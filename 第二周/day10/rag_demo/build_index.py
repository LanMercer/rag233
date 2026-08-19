# -*- coding: utf-8 -*-
r"""
RAG 建库脚本（配合 Day10 教程第 3 步使用）⭐ 全链路第 ①~④ 步

五件套说明（脚本在机器学习的哪一环）：
- 模型   ：bge-small-zh-v1.5（智源开源的中文 Embedding 向量化模型，CPU 即可跑）
- 数据   ：一个 PDF（默认指向第一周 day2 的《Attention Is All You Need》中文版，可改成你自己的 PDF）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事：读取 PDF → 文本切块 → 向量化（Embedding）→ 存入 Chroma 持久化向量库
    ↑ 这是 RAG 的"建库"阶段（离线，跑一次就够了）。后面 ask.py 的"问答"阶段会来查这个库。

运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第二周\day10\rag_demo"
    python build_index.py

运行后本文件夹会新增：
    .\chroma_db\   —— 向量数据库（核心产物，问答应答都查它）
    .\chunks.json  —— 切块后的纯文本，供手写版 manual_rag.py 复用（可删除，不影响库）
"""

import os
import json
import shutil   # 目录清理：重跑建库时清空旧向量库，防止向同一个 collection 追加导致数据重复

# ---------------------------------------------------------------------------
# 第 0 区：配置（日常改这里就够）
# ---------------------------------------------------------------------------

# 兜底镜像：仅当 EMBED_MODEL 是在线 ID（如 BAAI/bge-small-zh-v1.5）且需要联网下载时才用到。
# 本机方案是 ModelScope 下载到本地路径加载，不会触发联网，这一行可保留可删。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ① 要建库的 PDF 路径（改成你自己的 PDF：论文 / 研报 / 教材章节都行）
PDF_PATH = r"D:\Lan\研究生\技术学习\gmr\2510.02252v1.pdf"

# ② 切块参数（Day 9 周计划避坑表里写过的"中文切块标准配置"）
CHUNK_SIZE = 500          # 每块最多 500 字符
CHUNK_OVERLAP = 50        # 相邻两块重叠 50 字符，防止语义被从中间截断
# ③ Embedding 模型：优先用 ModelScope 下载到本地的 bge-small-zh-v1.5（下载命令见教程第 0.5 步；本地加载免网络）
EMBED_MODEL = r"D:\Lan\研究生\技术学习\大模型算法\download\bge-small-zh-v1.5"

PERSIST_DIR = r".\chroma_db"   # 向量库落盘目录
CHUNKS_JSON = r".\chunks.json" # 切块文本导出（给手写版用）

# ---------------------------------------------------------------------------
# 第 1 区：导入库
# ---------------------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader   # 读 PDF（langchain_community 已标记废弃但可用，新包 langchain-pypdf 的镜像源暂无）
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 切块（langchain 1.x 起拆分到独立包）
from langchain_huggingface import HuggingFaceEmbeddings        # 向量化
from langchain_chroma import Chroma                            # 向量数据库

# 终端编码加固：Windows 控制台默认 GBK，打不出 emoji/部分符号会崩（UnicodeEncodeError）。
# 强制用 UTF-8 输出，配合终端 chcp 65001 显示中文不乱码。
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 第 2 区：三个小函数（对应全链路第 ① 文档加载 / ② 文本切块 / ③④ 向量化入库）
# ---------------------------------------------------------------------------

def load_pdf(path: str):
    """第 ① 步：文档加载。PyPDFLoader 把 PDF 每页读成一个 Document。"""
    print(f"[① 文档加载] 读取：{path}")
    loader = PyPDFLoader(path)
    docs = loader.load()   # list[Document]，每个含 page_content（页文本）和 metadata（页码等）
    print(f"             共 {len(docs)} 页，文本已提取")
    return docs


def split_chunks(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """第 ② 步：文本切块。RecursiveCharacterTextSplitter 按优先级递归切分。"""
    print(f"[② 文本切块] chunk_size={chunk_size}, chunk_overlap={chunk_overlap}，分隔符加入中文标点")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 分隔符优先级从高到低：先按段落、再按行、再按中文句号/感叹号/问号/分号……最后按字硬切。
        # 加入中文标点是关键——否则它默认只认英文句点，会把中文句子从中间切碎（第二周避坑表第 4 条）。
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "! ", "? ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # 给每个 chunk 编上号 [N]，与 Day 8 模板 09 要求的 [资料§N] 出处对齐
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    print(f"             共 {len(chunks)} 个 chunk")
    return chunks


def save_chunks_json(chunks):
    """把切块后的纯文本导出成 JSON，供手写版 manual_rag.py 复用（保证两个版本用同一批数据）。"""
    data = [
        {"id": c.metadata["chunk_id"], "page": c.metadata.get("page"), "text": c.page_content}
        for c in chunks
    ]
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"             chunk 文本已存到 {CHUNKS_JSON}（手写版会读它）")


def build_index(chunks):
    """第 ③④ 步：Embedding + 存入 Chroma。"""
    print(f"[③ Embedding] 加载向量模型 {EMBED_MODEL}（首次约需下载 ~100MB，之后有本地缓存）")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # bge 系列模型输出 512 维向量：一句话 → 512 个浮点数

    # 重跑建库前先清空旧库：Chroma.from_documents 默认往同一个 collection 追加，
    # 不清空会导致库里有两份相同数据、检索结果重复（之前踩过这个坑）。
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print("             检测到旧库，已清空（重新建库避免数据重复）")

    print(f"[④ 存入 Chroma] persist_directory={PERSIST_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=PERSIST_DIR,
        # 关键：指定向量空间用"余弦"度量。Chroma 默认用 L2 距离，指定 cosine 后
        # "余弦相似度检索"才名副其实（教程第 2.5 节讲原理，这里落成配置）。
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"             建库完成：{len(chunks)} 个 chunk 的向量 + 原文已落盘")


def main():
    print("=" * 60)
    print("RAG 建库开始（全链路 ①~④ 步）")
    print("=" * 60)
    docs = load_pdf(PDF_PATH)
    chunks = split_chunks(docs)
    save_chunks_json(chunks)
    build_index(chunks)
    print("=" * 60)
    print("[完成] 建库完成！现在可以跑 ask.py 问问题了。")
    print("=" * 60)


if __name__ == "__main__":
    main()
