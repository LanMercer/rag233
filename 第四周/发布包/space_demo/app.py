# -*- coding: utf-8 -*-
r"""
机器人领域 RAG 文档问答系统 · 在线版 Demo（HuggingFace Spaces 用）

五件套说明（本脚本在机器学习的哪一环）：
- 模型   ：检索层 = bge-small-zh-v1.5（中文 Embedding，HF 在线自动下载，CPU 可跑）；
           生成层 = DeepSeek API（deepseek-chat，OpenAI 兼容；免费 CPU 跑不动 3B，故线上用 DeepSeek 代生成）
- 数据   ：用户上传的 PDF（自动切块建库）；也可用内置示例问题
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（推理/应用环节）：
    上传 PDF → PyPDF 加载 → 中文切块 → bge 向量化 → Chroma（余弦）建库
    → 提问 → Top-K 检索 → 套模板（只依据资料 + 强制出处 + 防幻觉）
    → 调 DeepSeek API 生成 → 返回答案 + 召回片段
    ↑ 属于机器学习【推理 / 应用】环节——不训练、不动权重。

★ 重要标注（必须保留在页面上，不得删除）：
    本 Demo 的**检索层是本项目真实实现**（bge-small-zh + Chroma 余弦 Top-K + 模板 09）；
    但**生成层用的是 DeepSeek API，不是本项目的微调模型**。本地版本为
    QLoRA 微调过的 Qwen2.5-3B-Instruct 4bit（只训 0.48% 参数，LoRA adapter）。
    DeepSeek 更强、且**没有加载本项目的 adapter**——因此本页面只演示"检索链路 + 交互形态"，
    评测数字（检索命中率 / 生成正确率 / 防幻觉 F1）**只归属本地 Qwen 版本，不代表本页结果**。
    微调效果与优化前后对比见项目博客 / 离线评测与演示 GIF。

环境变量（在 Space → Settings → Variables and secrets 配置）：
- API_BASE_URL ：https://api.deepseek.com/v1/chat/completions
- API_KEY      ：DeepSeek 密钥（只放 Secrets，绝不写进代码）
- MODEL_NAME   ：deepseek-chat

本地运行：
    pip install -r requirements.txt
    python app.py
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import requests

# ---------------------------------------------------------------------------
# 第 0 区：配置（全部从环境变量读，绝不硬编码密钥）
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get(
    "API_BASE_URL", "https://api.deepseek.com/v1/chat/completions"
).strip()
API_KEY = os.environ.get("API_KEY", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-chat").strip()

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")  # 在线下载

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 4
TEMPERATURE = 0.2
MAX_TOKENS = 512

# 模板 09（与本地项目一字不差）：只依据资料 + 强制出处 + 防幻觉
PROMPT_TEMPLATE = """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}"""

DEMO_NOTE = (
    "> **⚠️ 诚实标注（必读）**：本页面**检索层为项目真实实现**"
    "（bge-small-zh + Chroma 余弦 Top-K + 模板 09 带出处约束）；"
    "但**生成层使用的是 DeepSeek API（deepseek-chat），不是本项目的微调模型**"
    "——免费 CPU 环境跑不动 3B，故用 DeepSeek 代生成以省算力。"
    "本地版本为 **QLoRA 微调过的 Qwen2.5-3B-Instruct 4bit**（只训 0.48% 参数）。"
    "DeepSeek 更强且**未加载本项目 LoRA adapter**，故本页只演示检索链路与交互形态；"
    "**评测数字（检索命中率 / 生成正确率 / F1）只归属本地 Qwen 版本，不代表本页 DeepSeek 的结果**。"
    "微调效果与优化前后对比见项目博客 / 离线评测与演示 GIF。\n\n"
)

# ---------------------------------------------------------------------------
# 第 1 区：检索层（加载 PDF → 切块 → 向量化 → Chroma）
# ---------------------------------------------------------------------------

def build_store(pdf_path: str, progress=gr.Progress()):
    """把上传的 PDF 建成一个内存向量库，返回 (store, chunk_count)。"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    progress(0.1, desc="读取 PDF……")
    docs = PyPDFLoader(pdf_path).load()

    progress(0.35, desc="中文切块……")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 分隔符加入中文标点，避免把中文句子从中间切碎
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "! ", "? ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    progress(0.6, desc="加载向量模型（首次约 100MB，稍等）……")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    progress(0.85, desc="建立向量库……")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_metadata={"hnsw:space": "cosine"},  # 余弦相似度检索
    )
    return store, len(chunks)


def retrieve(store, question: str, top_k: int):
    """余弦相似度 Top-K：返回 [(chunk_id, 片段文本), ...]。"""
    docs = store.similarity_search(question, k=top_k)
    return [(d.metadata.get("chunk_id"), d.page_content) for d in docs]


# ---------------------------------------------------------------------------
# 第 2 区：生成层（调 DeepSeek API，OpenAI 兼容）
# ---------------------------------------------------------------------------

def generate(question: str, context: str) -> str:
    """套模板 09 → POST 云端接口 → 返回答案文本。"""
    if not (API_BASE_URL and API_KEY):
        return ("[未配置 DeepSeek API] 请在 Space 的 Settings → Secrets 配置 "
                "API_BASE_URL（https://api.deepseek.com/v1/chat/completions）与 API_KEY，"
                "Variables 配 MODEL_NAME=deepseek-chat；或改用本地版本。")

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(API_BASE_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:  # 网络/额度等问题不要让页面崩
        return f"[生成失败] {type(e).__name__}: {e}"


def answer_question(store, question: str, top_k: int):
    """完整链路：检索 → 拼上下文 → 生成 → 组织展示文本。"""
    if store is None:
        return "请先上传一个 PDF 并点击「建立知识库」。", "", ""
    if not question or not question.strip():
        return "请输入问题。", "", ""

    hits = retrieve(store, question, top_k)
    context = "\n".join(f"[{cid}] {text}" for cid, text in hits)
    ans = generate(question, context)

    sources = "、".join(f"[资料§{cid}]" for cid, _ in hits)
    snippets = "\n\n".join(f"**[{cid}]** {text[:300]}{'…' if len(text) > 300 else ''}"
                           for cid, text in hits)
    return ans, sources, snippets


# ---------------------------------------------------------------------------
# 第 3 区：Gradio 界面
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="机器人领域 RAG 文档问答系统 Demo") as demo:
        gr.Markdown("# 🤖 机器人领域 RAG 文档问答系统 · 在线 Demo")
        gr.Markdown(DEMO_NOTE)

        store_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(label="① 上传领域 PDF（论文/研报/技术文档）", file_types=[".pdf"])
                build_btn = gr.Button("建立知识库", variant="primary")
                build_msg = gr.Markdown("")
                top_k = gr.Slider(1, 10, value=DEFAULT_TOP_K, step=1, label="② 检索 Top-K")
            with gr.Column(scale=2):
                question = gr.Textbox(label="③ 输入问题", lines=2,
                                      placeholder="例如：论文的实验在哪个数据集上进行？")
                ask_btn = gr.Button("提问", variant="primary")
                answer = gr.Textbox(label="回答（带出处）", lines=6)
                sources = gr.Textbox(label="引用片段编号")
                with gr.Accordion("召回片段（检索 Top-K 原文）", open=False):
                    snippets = gr.Markdown("")

        def _build(pdf, k, progress=gr.Progress()):
            if pdf is None:
                return None, "请先上传 PDF。", ""
            # gr.File 在不同版本返回"路径字符串"或"临时文件对象"，两种都兼容
            src = pdf.name if hasattr(pdf, "name") else pdf
            # 复制到固定临时文件（上传文件名可能含中文/空格）
            tmp = Path(tempfile.gettempdir()) / "uploaded.pdf"
            tmp.write_bytes(Path(src).read_bytes())
            store, n = build_store(str(tmp), progress)
            return store, f"✅ 建库完成：共 {n} 个文本块。现在可以提问了。", ""

        build_btn.click(_build, [pdf_file, top_k], [store_state, build_msg, answer], show_progress=True)
        ask_btn.click(answer_question, [store_state, question, top_k], [answer, sources, snippets])
        question.submit(answer_question, [store_state, question, top_k], [answer, sources, snippets])

        gr.Markdown(DEMO_NOTE)
    return demo


if __name__ == "__main__":
    build_ui().launch()
