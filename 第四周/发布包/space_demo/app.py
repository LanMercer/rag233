# -*- coding: utf-8 -*-
r"""
机器人领域 RAG 文档问答系统 · 在线版 Demo（魔搭创空间 / HuggingFace Spaces 通用）

五件套说明（本脚本在机器学习的哪一环）：
- 模型   ：检索层 = bge-small-zh-v1.5（中文 Embedding，HF 在线自动下载，CPU 可跑）；
           生成层 = DeepSeek API（deepseek-chat，OpenAI 兼容；免费 CPU 跑不动 3B，故线上用 DeepSeek 代生成）
- 数据   ：★ **默认**用内置示例语料 `chunks.json`（GMR 论文 105 段，与项目离线评测同一批切块，
           故在线检索口径与项目一致）；也可上传自己的 PDF（会自动覆盖内置语料）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（推理/应用环节）：
    ① 未上传时：读 `chunks.json` → bge 向量化 → Chroma（余弦）建库（**启动后台预热**，见下）
    ② 上传时  ：上传 PDF → PyPDF 加载 → 中文切块 → bge 向量化 → Chroma（余弦）建库
    ③ 提问 → Top-K 检索 → 套模板（只依据资料 + 强制出处 + 防幻觉）
            → 调 DeepSeek API 生成 → 返回答案 + 召回片段
    ↑ 属于机器学习【推理 / 应用】环节——不训练、不动权重。

★ 内置语料"就是评测那一批"的证据（2026-09-22 本地实测，20 题 × Top-8 逐题比对）：
    在线默认库检索出的 chunk_id 序列 与 离线评测落盘的 top_ids **20/20 完全一致**
    （对照物：第四周/day20/result_lora_v3_runs3/eval_results.json，TOP_K=8 / instruction off / 改写 off）。
    所以页面上给出的 `[资料§N]` 可以直接拿去对照离线评测表，不会"线上线下一套编号"。
    ⚠ 但这条结论**依赖 HNSW_METADATA 里的 search_ef=200**：用 chromadb 默认的 ef=10 时，
      同一批语料两次建库的 Top-8 只有 7/20 题与精确检索一致（近似检索随索引图抖动，不可复现）。

★ 内置库为什么"后台预热 + 懒加载兜底"（两边都要）：
    建默认库要先加载 bge-small-zh-v1.5 权重（约 95MB，首次还要联网下载），实测首次约 25s。
    · 不能放在"启动路径上同步做"：进程迟迟不 bind 端口，平台健康检查会判失败
      → 页面「Could not load this space」。
    · 也不能只在"第一次提问时"才做：第一个访客要干等半分钟，体感像页面坏了。
    ⇒ 在 __main__ 里起一个**后台守护线程**预热（端口照常绑定），同时保留懒加载兜底
      （预热失败/还没好时，首问自己建）。实测：预热完成后首问 2~3s，未预热时 31s。

★ 重要标注（必须保留在页面上，不得删除）：
    本 Demo 的**检索层是本项目真实实现**（bge-small-zh + Chroma 余弦 Top-K + 模板 09）；
    但**生成层用的是 DeepSeek API，不是本项目的微调模型**。本地版本为
    QLoRA 微调过的 Qwen2.5-3B-Instruct 4bit（只训 0.48% 参数，LoRA adapter）。
    DeepSeek 更强、且**没有加载本项目的 adapter**——因此本页面只演示"检索链路 + 交互形态"，
    评测数字（检索命中率 / 生成正确率 / 防幻觉 F1）**只归属本地 Qwen 版本，不代表本页结果**。
    微调效果与优化前后对比见项目博客 / 离线评测与演示 GIF。

环境变量（在平台的「设置 / Settings」里配置）：
- API_BASE_URL ：https://api.deepseek.com/v1/chat/completions
- API_KEY      ：DeepSeek 密钥（只放 Secrets/环境变量，绝不写进代码）
- MODEL_NAME   ：deepseek-chat
- EMBED_MODEL  ：（可选）默认 BAAI/bge-small-zh-v1.5
- HF_ENDPOINT  ：（可选）https://hf-mirror.com —— 国内平台拉 HF 权重必设
- GRADIO_ANALYTICS_ENABLED：（可选）False，关掉 gradio 的版本自检外呼

★ 部署硬性要求（踩过坑，别删）：
1. 必须监听 0.0.0.0:7860。gradio 默认绑 127.0.0.1，平台的反向代理
   从容器外连不进来 → 页面报「Could not load this space」/ 网关 502。
2. 平台的 Gradio SDK 版本必须与 requirements.txt 的 gradio==4.44.0 一致，
   否则 pip 依赖解析冲突会直接让容器起不来（见 requirements.txt 顶部说明）。

本地运行：
    pip install -r requirements.txt
    python app.py
"""

import os
import shutil
import tempfile
import threading
import uuid
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

# 内置示例语料：GMR 论文（arXiv:2510.02252）的切块文本，105 段。
# 来源：项目 `第三周\day13\chunks.json`（与离线评测用的向量库同源，故 chunk_id 与评测表一致）。
DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "chunks.json"
DEFAULT_CORPUS_NAME = "内置示例语料（GMR 论文，arXiv:2510.02252，105 段）"

# ★ 为什么每个库都要给一个随机 collection_name（2026-09-22 线上实测踩到的坑）：
#   `Chroma.from_documents(...)` 不传 collection_name 时会用默认名 "langchain"，
#   而 chromadb 在同一进程里**共享同一个 in-memory system** —— 于是第二次建库不是
#   "建一个新的"，而是**往同一个 collection 后面追加**（实测 105 → 210 条）。
#   后果：库里有 2 份相同 chunk，检索 Top-4 会返回 [103, 103, 4, 4] 这种重复编号
#   （有效命中从 4 段掉到 2 段，出处看着像坏了）；每个新访客各建一次就再叠一份。
#   第三周 day13/build_index.py 用 `shutil.rmtree` 绕过的是同一个坑，这里用随机名解决。
def _new_collection_name(tag: str) -> str:
    return f"{tag}_{uuid.uuid4().hex[:12]}"


def _store_dir(tag: str) -> str:
    """给某个用途（default / pdf）一个固定的落盘目录（见 CHROMA_ROOT 说明）。"""
    d = CHROMA_ROOT / tag
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


# ★ 为什么必须显式给 hnsw:search_ef 一个"够大"的值（2026-09-22 线上实测）：
#   chromadb 0.5.15 的 hnsw_params.py:58 是 `self.search_ef = int(metadata.get("hnsw:search_ef", 10))`
#   —— 默认 ef=10，而 ef 是"贪心图搜索时保留的候选数"：候选池比库还小，就会漏掉真正的近邻。
#   本语料只有 105 段、且同一篇论文的段落彼此高度相似（余弦分挤在 0.8~1.0 的窄带里），
#   于是排名靠 ef 之外的段落会被挤掉、且**随建库环境（索引图构建）抖动**：
#       实测 ef=10 时，同一批语料两次建库的 Top-8 与离线落盘结果只重合 6.95/8 和 7.25/8，
#       20 题里只有 7 题与"精确余弦"完全一致；
#       把 ef 提到 200（> 段数 105）后，贪心搜索退化为**穷尽搜索** →
#       在线 Top-8 与离线评测落盘的 top_ids **20/20 逐题完全一致**，且不再受建库环境随机性影响。
#   代价可忽略：105 段的穷尽搜索仍是毫秒级；上传 PDF 的路径同样给 200，
#   对上千段的库也是"召回优先、延迟可接受"的常见取舍。
HNSW_METADATA = {"hnsw:space": "cosine", "hnsw:search_ef": 200}

# ★ 为什么向量库要"落盘到临时目录"，而不是用 chromadb 默认的内存库（2026-09-22 线上实测）：
#   chromadb 0.5.15 的内存库用 `file::memory:?cache=shared` + LockPool，而 LockPool 把连接放在
#   **threading.local()** 里、`_connections` 只存**弱引用**（见 chromadb/db/impl/sqlite_pool.py）。
#   于是"库还能不能查"取决于**建库那个线程是否还活着**：
#       预热线程建完库就退出 → 该线程的连接没了 → 内存库可能整体销毁
#       → Gradio 的 worker 线程新建连接只看到一个空库 → `sqlite3.OperationalError: no such table: collections`
#   这正是本页面第一次冒烟时的报错（页面 500、用户看到「服务异常」）。
#   Gradio 的 worker 线程会被回收、预热线程必然退出 —— 这个坑迟早踩到，且是**间歇性**的（最难查）。
#   落盘库走的是 PerThreadPool + 磁盘文件：表在文件里，与线程生死无关，实测连查 6/6 稳定。
#   目录用系统临时目录，不污染项目/发布包；容器重建时自然清空，不需要额外清理。
CHROMA_ROOT = Path(tempfile.gettempdir()) / "rag233_demo_chroma"

# 内置库在**进程内只建一次**并复用（见 ensure_store）：
#   ① 不重复建 → 不会触发上面那个"追加"坑；
#   ② 所有访客共用一份 → 第一个访客之后，提问都是秒回；
#   ③ 内存有上界（105×512 浮点 ≈ 0.2MB），不会随访客数增长。
# 加锁只为防并发首问时同时建库（最多多建一份，不会串数据）。
_DEFAULT_STORE = None
_DEFAULT_STORE_LOCK = threading.Lock()

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

def _tick(progress, frac: float, desc: str):
    """安全地报一次进度。

    ⚠ 两个都实测踩过的坑：
      ① **判空必须写 `progress is not None`，不能写 `if progress:`**。
         gradio 4.44 的 `Progress.__len__` 实现是 `return self.iterables[-1].length`
         （helpers.py:672）—— 一次进度都还没报过时 `iterables` 是空列表，
         于是 `if progress:` 会直接抛 `IndexError: list index out of range`。
         本函数的调用点在建库之前，正好命中这个状态（2026-09-22 线上冒烟实测）。
      ② 不同 gradio 版本 Progress 的签名/行为有差异：进度条不该有能力把建库搞崩，
         所以这里兜住异常（与 day18/app.py 的 step() 同款防御）。
    """
    if progress is None:
        return
    try:
        progress(frac, desc=desc)
    except Exception:
        pass


def build_store(pdf_path: str, progress=gr.Progress()):
    """把上传的 PDF 建成一个向量库（落盘到临时目录，见 CHROMA_ROOT 说明），返回 (store, chunk_count)。"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    _tick(progress, 0.1, "读取 PDF……")
    docs = PyPDFLoader(pdf_path).load()

    _tick(progress, 0.35, "中文切块……")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 分隔符加入中文标点，避免把中文句子从中间切碎
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "! ", "? ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    _tick(progress, 0.6, "加载向量模型（首次约 100MB，稍等）……")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    _tick(progress, 0.85, "建立向量库……")
    # ⚠ collection_name 必须唯一：否则再点一次「建立知识库」会往同一个 collection 追加，
    #   库里出现两份相同 chunk → 检索结果重复（见文件头 _new_collection_name 说明）
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=_new_collection_name("pdf"),
        collection_metadata=HNSW_METADATA,  # 余弦 + 大 ef（见 HNSW_METADATA 说明）
        persist_directory=_store_dir("pdf"),  # 落盘（见 CHROMA_ROOT 说明）
    )
    return store, len(chunks)


def _is_default_store(store) -> bool:
    """判断某个 store 是不是所有访客共享的内置库（共享库不能删）。"""
    return _DEFAULT_STORE is not None and store is _DEFAULT_STORE[0]


def _dispose_pdf_store(store):
    """释放上一次上传 PDF 建的库（避免反复建库把临时目录撑大）。

    ⚠ 绝不动内置库：它被所有访客共享，删了会让别人正在问的问题报错。
    """
    if store is None or _is_default_store(store):
        return
    try:
        store.delete_collection()
    except Exception:
        pass  # 清不掉就算了，只是多占一点临时空间


def build_default_store(progress=None):
    """把内置示例语料（chunks.json，GMR 论文 105 段）建成向量库（落盘到临时目录，见 CHROMA_ROOT 说明）。

    返回 (store, chunk_count)。**进程内只建一次**（结果缓存到 _DEFAULT_STORE，见上方说明）。
    """
    global _DEFAULT_STORE
    if _DEFAULT_STORE is not None:
        return _DEFAULT_STORE

    import json
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    with _DEFAULT_STORE_LOCK:          # 防并发首问同时建库
        if _DEFAULT_STORE is not None:  # 双检：等锁期间别人已建好就直接用
            return _DEFAULT_STORE

        _tick(progress, 0.3, f"加载{DEFAULT_CORPUS_NAME}……")

        items = json.loads(DEFAULT_CORPUS_PATH.read_text(encoding="utf-8"))
        # ★ 必须把 json 里的 id 写回 metadata["chunk_id"]：
        #   检索层靠它输出 [资料§N]，模板 09 要求答案标注出处编号。
        #   映射正确，在线 Demo 的出处编号才与项目评测表（[36]、[76]…）对得上。
        docs = [
            Document(
                page_content=it["text"],
                metadata={"chunk_id": it["id"], "page": it.get("page")},
            )
            for it in items
        ]

        _tick(progress, 0.7, "加载向量模型（首次约 100MB，稍等）……")
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        # ⚠ collection_name 必须唯一，否则会往默认 collection 追加（见文件头 _new_collection_name 说明）
        store = Chroma.from_documents(
            documents=docs,
            embedding=embedder,
            collection_name=_new_collection_name("default"),
            collection_metadata=HNSW_METADATA,    # 余弦 + 大 ef（见 HNSW_METADATA 说明）
            persist_directory=_store_dir("default"),  # 落盘（见 CHROMA_ROOT 说明）
        )
        _DEFAULT_STORE = (store, len(docs))
        return _DEFAULT_STORE


def ensure_store(store, progress=None):
    """取用现有库；若为空则尝试加载内置示例语料。

    返回 (store, err_msg, loaded_default)：
      - 已有库     → (store, None, False)
      - 建了默认库 → (store, None, True)
      - 默认库失败 → (None, 给用户看的错误文案, False)
    """
    if store is not None:
        return store, None, False
    try:
        store, _n = build_default_store(progress)
        return store, None, True
    except Exception as e:  # 语料缺失/网络不通/依赖问题都不要让页面崩
        return None, (
            "请先上传一个 PDF 并点击「建立知识库」。"
            f"（{DEFAULT_CORPUS_NAME}加载失败：`{type(e).__name__}: {e}`"
            "——常见原因是容器拉不到向量模型，请确认环境变量 `HF_ENDPOINT=https://hf-mirror.com` 已设置）"
        ), False


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


def answer_question(store, question: str, top_k: int, progress=gr.Progress()):
    """完整链路：检索 → 拼上下文 → 生成 → 组织展示文本。

    返回 4 元组（顺序必须与 outputs 一致）：store, 回答, 引用片段编号, 召回片段。
    ★ 第一个值是"回写 State"：第一次提问会懒加载内置示例语料，
      回写后后续提问直接复用，不会每次重建库。
    """
    if not question or not question.strip():
        return store, "请输入问题。", "", ""

    store, err, loaded_default = ensure_store(store, progress)
    if store is None:
        return None, err, "", ""

    hits = retrieve(store, question, top_k)
    context = "\n".join(f"[{cid}] {text}" for cid, text in hits)
    ans = generate(question, context)

    if loaded_default:
        # 明确告诉用户这批答案的依据来自哪个库，避免"没上传却答出来了"的困惑
        ans = f"（本次使用{DEFAULT_CORPUS_NAME}回答；上传自己的 PDF 可覆盖）\n\n{ans}"

    sources = "、".join(f"[资料§{cid}]" for cid, _ in hits)
    snippets = "\n\n".join(f"**[{cid}]** {text[:300]}{'…' if len(text) > 300 else ''}"
                           for cid, text in hits)
    return store, ans, sources, snippets


# ---------------------------------------------------------------------------
# 第 3 区：Gradio 界面
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="机器人领域 RAG 文档问答系统 Demo") as demo:
        gr.Markdown("# 🤖 机器人领域 RAG 文档问答系统 · 在线 Demo")
        gr.Markdown(DEMO_NOTE)
        gr.Markdown(
            "> 📚 **不上传也能直接提问**：未上传 PDF 时，系统自动使用"
            f"**{DEFAULT_CORPUS_NAME}**"
            "（*Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking*，"
            "arXiv:2510.02252；与项目离线评测用的是同一批切块，故出处编号 `[资料§N]` 与评测表一致）。"
            "**上传自己的 PDF 会覆盖它**（论文 / 研报 / 技术文档均可）。"
        )

        store_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(label="① 上传领域 PDF（可选：不传就用内置 GMR 论文语料）", file_types=[".pdf"])
                build_btn = gr.Button("建立知识库（覆盖内置语料）", variant="primary")
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

        def _build(store, pdf, k, progress=gr.Progress()):
            if pdf is None:
                # ★ 不要返回 None：那会把已有的库（含懒加载出来的内置库）清掉
                return store, "请先上传 PDF。", ""
            # gr.File 在不同版本返回"路径字符串"或"临时文件对象"，两种都兼容
            src = pdf.name if hasattr(pdf, "name") else pdf
            # 复制到固定临时文件（上传文件名可能含中文/空格）
            tmp = Path(tempfile.gettempdir()) / "uploaded.pdf"
            tmp.write_bytes(Path(src).read_bytes())
            new_store, n = build_store(str(tmp), progress)
            _dispose_pdf_store(store)  # 建好新的再清旧的，避免中途失败后无库可用
            return new_store, f"✅ 建库完成：共 {n} 个文本块（已覆盖{DEFAULT_CORPUS_NAME}）。现在可以提问了。", ""

        build_btn.click(_build, [store_state, pdf_file, top_k], [store_state, build_msg, answer], show_progress=True)
        # ⚠ outputs 必须含 store_state：第一次提问会把懒加载出来的内置库回写，
        #   后续提问直接复用（否则每问一次都要重新向量化 105 段 + 加载模型）
        ask_btn.click(answer_question, [store_state, question, top_k],
                      [store_state, answer, sources, snippets], show_progress=True)
        question.submit(answer_question, [store_state, question, top_k],
                        [store_state, answer, sources, snippets], show_progress=True)

        gr.Markdown(DEMO_NOTE)
    return demo


if __name__ == "__main__":
    # ★ 后台预热内置库：既不阻塞端口绑定（见文件头"懒加载"说明），又让"第一个提问的人"不用等。
    #   实测：不预热时首问要 31s（现场加载 bge + 向量化 105 段）；预热后首问 2~3s。
    #   预热失败也不影响启动——ensure_store() 会在首问时再试一次，并给出可读的错误文案。
    threading.Thread(target=build_default_store, daemon=True,
                     name="warmup-default-store").start()

    # ★ 必须显式绑 0.0.0.0:7860：
    #   gradio 默认 server_name="127.0.0.1"（见 gradio/http_server.py 的 LOCALHOST_NAME），
    #   只监听容器回环地址 → 平台反向代理连不上 → 页面「Could not load this space」+ 网关 502。
    #   server_port 固定 7860：魔搭/HF 的反代都指向这个端口，不要改动。
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
