# -*- coding: utf-8 -*-
r"""
RAG 问答工作台（第四周 Day18 · D 线"可交付产物"完成版）⭐

↑ 一句话：把 day16 的**界面骨架**补完，变成"**打开就能用**"的工作台——
   上传 PDF 自动建库 → 提问 → 答案带 [资料§N] 出处 → 展开看召回片段
   → 顶部展示"优化前 vs 优化后"指标对比（读实验日志）→ 一键切换原模型/微调模型。

五件套说明（这个脚本在机器学习的哪一环）：
- 模型   ：生成层**不在本进程里加载**！本界面只是 HTTP 客户端——
             lora     → 第三周 day12 微调模型（day13 local_api_lora.py 起服务，默认）
             original → 原模型（第二周 day9 local_api.py 起服务）
           检索层 = bge-small-zh-v1.5（本地 CPU 跑，把问题向量化，从 Chroma 找相似段落）
- 数据   ：① 默认用第三周 day13 建好的 GMR 论文向量库（chroma_db）；
           ② 也可以在页面**上传你自己的 PDF**，现场切块建库（内存库，不落盘、不污染原库）
- 损失   ：无（不训练）｜ 优化器：无（不训练）
- 本脚本做的事（应用/推理阶段）：检索 → 拼模板 09 → 调本地 OpenAI 兼容接口 → 展示。

★ Day18 相对 day16 骨架的 5 处升级（这就是"从骨架到能交付"的差距）：
    ① 上传 PDF → 自动建库（带进度条 + 建库日志，不用再手动跑 build_index.py）
    ② 顶部"优化前 / 优化后"指标对比：**直接读 `第四周\实验日志.md`**，不硬编码数字
       （口径纪律：数字只有一个源，页面不许自己写死，否则迟早和报告对不上）
    ③ 召回片段可视化：显示相似度分数 + 折叠展开（证明答案有依据，面试可讲）
    ④ 检索 Top-K 可调（页面旋钮，对应 eval_v2.py 的 --top-k；方便现场演示"多取几条会怎样"）
    ⑤ 双模型切换 + 服务身份检查（沿用 day16/day15 的"服务在跑 ≠ 服务正确"防呆）

★ 关键设计：为什么界面里不加载生成模型？（面试会问）
    本机 6G 显存，一个 Qwen2.5-3B（4bit）约 1.9~2.0GB，训练与服务同跑会 OOM（第三周踩过）。
    所以架构上把"模型"和"界面"拆成两个进程：
        [模型服务进程]  ←HTTP→  [Web 界面进程]
    界面只做轻活，显存全留给模型服务；换模型只需换一个服务，界面不用改也不用重启。
    （**例外**：检索用的 bge 是 CPU 上的小模型，留在界面进程里没问题。）

⚠ 三条红线（沿用 day16）：
    ① 密钥绝不进代码/仓库；② 线上版必须标注"生成层 ≠ 本项目微调模型"；③ 评测数字只归属本地 Qwen。

运行方法（llm 环境；先 cd 到本文件所在目录）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day18"

    python app.py --check     # 只构建界面、不启动服务（确认环境和代码没问题）
    python app.py             # 启动网页，浏览器打开 http://127.0.0.1:7860

    前置：与所选 profile 对应的模型服务已在 8000 端口运行：
        lora     → cd 第三周\day13 ; python -m uvicorn local_api_lora:app --host 127.0.0.1 --port 8000
        original → cd 第二周\day9  ; python -m uvicorn local_api:app --host 127.0.0.1 --port 8000
        （或直接 `python ..\day17\start.py --start-service`）

参数（可选）：
    --port 7860   网页端口（被占用时换一个，如 7861）
    --check       只构建界面不启动（冒烟自检）
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固（Windows 控制台默认 GBK）
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr  # Gradio = 用 Python 几行起一个网页界面（无需写前端）

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# 第 2 区：路径与配置（复用第三周 day15 的 config.json）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # 第四周\day18
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))    # 仓库根
CONFIG_PATH = os.path.join(REPO_DIR, "第三周", "day15", "config.json")
EXPERIMENT_LOG = os.path.join(REPO_DIR, "第四周", "实验日志.md")       # 指标对比的唯一数据源

# 建库参数（与 第三周\day13\build_index.py 的中文切块标准配置保持一致）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TOKENS = 512

# 模板 09（与 day10 / day14 / day15 / eval_v2.py 一字不差）
PROMPT_TEMPLATE = """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}"""


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p, base_dir):
    """相对路径按 base_dir 解析成绝对路径。"""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


CFG = load_config()
CFG_DIR = os.path.dirname(CONFIG_PATH)
PERSIST_DIR = resolve_path(CFG["persist_dir"], CFG_DIR)
EMBED_MODEL = CFG["embed_model"]

# 向量库只在这里加载一次（进程内复用，避免每次提问都重载 bge）
_STORE = None


def get_default_store():
    """默认知识库：第三周 day13 的 GMR 论文库（懒加载，只加载一次）。"""
    global _STORE
    if _STORE is None:
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _STORE = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
    return _STORE


def build_store_from_pdf(pdf_path, progress=None):
    """
    上传的 PDF → 切块 → 向量化 → 内存向量库（不落盘，不污染 day13 的原库）。
    返回 (store, 建库日志字符串)。
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logs = []

    def step(frac, msg):
        logs.append(msg)
        if progress is not None:
            try:
                progress(frac, desc=msg)
            except Exception:
                pass  # 不同 Gradio 版本 Progress 签名有差异，不能让进度条把建库搞崩

    step(0.10, "① 读取 PDF……")
    docs = PyPDFLoader(pdf_path).load()
    logs.append(f"   共 {len(docs)} 页")

    step(0.35, f"② 中文切块（chunk_size={CHUNK_SIZE} / overlap={CHUNK_OVERLAP}）……")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 分隔符加入中文标点，避免把中文句子从中间切碎（第二周避坑表第 4 条）
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "! ", "? ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    logs.append(f"   共 {len(chunks)} 个 chunk（每个都编了号，对应答案里的 [资料§N]）")

    step(0.60, "③ 加载向量模型 bge-small-zh（首次约几秒）……")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    step(0.85, "④ 建立内存向量库（余弦度量）……")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_metadata={"hnsw:space": "cosine"},
    )
    logs.append("   建库完成 ✅（内存库，关闭页面即释放）")
    return store, "\n".join(logs)


# ---------------------------------------------------------------------------
# 第 3 区：指标对比（读实验日志——数字只有一个源）
# ---------------------------------------------------------------------------

def _pct(text):
    """从 '8/20 = 40.0%' 抠出百分比数值；抠不到返回 None。"""
    m = re.search(r"=\s*([0-9.]+)\s*%", str(text))
    if m:
        return float(m.group(1))
    m = re.search(r"([0-9.]+)\s*%", str(text))
    return float(m.group(1)) if m else None


def _num(text):
    """从单元格里抠出普通小数（F1 是 0~1 的小数，不带 %）。"""
    m = re.search(r"([0-9]*\.?[0-9]+)", str(text))
    return float(m.group(1)) if m else None


def parse_experiment_log():
    """
    解析 `第四周\\实验日志.md` 的「一、实验记录表」，返回 (baseline_row, best_row)。
    容错优先：日志是"人写的账本"，格式可能有手改；解析失败就返回 (None, None)，
    页面降级显示提示，**绝不硬编码数字**。
    """
    try:
        with open(EXPERIMENT_LOG, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None, None

    in_table = False
    rows = []
    for line in lines:
        s = line.strip()
        if s.startswith("## 一、实验记录表"):
            in_table = True
            continue
        if in_table and s.startswith("## "):
            break
        if not in_table or not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 11 or set(cells[0]) <= set("-: "):   # 跳过分隔行
            continue
        if cells[1] in ("编号", ""):                          # 跳过表头/空行
            continue
        rows.append({
            "date": cells[0], "exp_id": cells[1], "note": cells[2], "config": cells[3],
            "strict": _pct(cells[4]), "loose": _pct(cells[5]), "gen": _pct(cells[6]),
            "anti": _pct(cells[7]), "f1": _num(cells[8]), "overall": _pct(cells[9]),
            "conclusion": cells[10], "dir": cells[11] if len(cells) > 11 else "",
        })
    if not rows:
        return None, None

    # 基线优先取"校准后基线 B0r3"（runs=3，可对外引用）；没有才退回 B0。
    # 为什么重要：B0 是 runs=1，与 runs=3 的优化行**不可直接相减**（day17 报告已明确）。
    baseline = next((r for r in rows if r["exp_id"] == "B0r3"), None) \
        or next((r for r in rows if r["exp_id"] == "B0"), rows[0])
    cands = [r for r in rows if r["overall"] is not None and r is not baseline]
    best = max(cands, key=lambda r: r["overall"]) if cands else None
    return baseline, best


def metrics_markdown():
    """把实验日志里的基线/最好成绩渲染成页面顶部的对比表。"""
    baseline, best = parse_experiment_log()
    if baseline is None:
        return ("### 优化前 / 优化后（指标对比）\n\n"
                f"> 未能从 `{os.path.relpath(EXPERIMENT_LOG, REPO_DIR)}` 解析出实验记录表。\n"
                "> 请确认「一、实验记录表」还在、且行格式没被改动。**页面不硬编码数字**，解析不到就如实显示。")

    def fmt(metric, as_percent=True):
        b = baseline.get(metric)
        if best is None or best.get(metric) is None or b is None:
            return "—"
        if as_percent:
            delta = best[metric] - b
            arrow = "⬆" if delta > 0.05 else ("⬇" if delta < -0.05 else "→")
            return f"**{b:.1f}% → {best[metric]:.1f}%**（{arrow} {delta:+.1f} 个百分点）"
        delta = best[metric] - b
        arrow = "⬆" if delta > 0.0005 else ("⬇" if delta < -0.0005 else "→")
        return f"**{b:.3f} → {best[metric]:.3f}**（{arrow} {delta:+.3f}）"

    warn = ""
    if best is not None and best["dir"] != baseline["dir"]:
        # runs 口径不同会让数字不可直接相减 —— 页面必须把这一点说清楚，别误导面试官/HR
        b_runs = re.search(r"runs=(\d+)", baseline["config"])
        w_runs = re.search(r"runs=(\d+)", best["config"])
        if b_runs and w_runs and b_runs.group(1) != w_runs.group(1):
            warn = (f"\n> ⚠ **口径提醒**：基线是 `runs={b_runs.group(1)}`、最好成绩是 `runs={w_runs.group(1)}`，"
                    f"多跑投票本身就会改数字，两者**不能严格相减**。严格可比的对照请看法同口径的那几行。\n")

    return (
        "### 优化前 / 优化后（同 20 题固定评测集 · 数字直接读 `第四周\\实验日志.md`）\n\n"
        "| 指标 | 优化前 → 优化后 |\n|---|---|\n"
        f"| 检索命中率 · strict | {fmt('strict')} |\n"
        f"| 检索命中率 · loose | {fmt('loose')} |\n"
        f"| 生成正确率（in） | {fmt('gen')} |\n"
        f"| 防幻觉正确率（out） | {fmt('anti')} |\n"
        f"| 防幻觉 F1 | {fmt('f1', as_percent=False)} |\n"
        f"| 总正确率 | {fmt('overall')} |\n"
        f"{warn}\n"
        f"> 基线 = `{baseline['exp_id']}`（{baseline['date']}，{baseline['note']}）"
        f" ｜ 当前最好 = `{best['exp_id']}`（{best['date']}，{best['note']}）\n"
        f">\n"
        f"> **数字归属**：以上只代表本地 **Qwen2.5-3B-Instruct-LoRA** 版本；"
        f"线上 Demo（DeepSeek API 代生成）**不适用**这些数字。"
    )


# ---------------------------------------------------------------------------
# 第 4 区：核心问答逻辑（检索 → 拼模板 → 调接口）
# ---------------------------------------------------------------------------

def api_base(api_url):
    """从 .../v1/chat/completions 截出服务根地址（用于拼 /v1/models）。"""
    return api_url.rsplit("/v1/chat/completions", 1)[0]


def server_model_name(profile):
    """GET /v1/models：拿服务端真实加载的模型名（判"服务在跑 ≠ 服务正确"）。"""
    r = requests.get(api_base(profile["api_url"]) + "/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def resolve_store(store):
    """
    解析用哪个知识库：页面里建的库优先，没有就回落到**默认 GMR 库**。
    这样"什么都不上传、直接提问"也能用——默认库是开箱即用的。
    """
    if store is not None:
        return store
    return get_default_store()


def ask(store, profile_name, question, top_k):
    """
    一次完整问答，返回 (答案 markdown, 召回片段 markdown)。
    出错不崩页面：把错误信息当作"答案"返回，方便排查。
    """
    profile = CFG["profiles"][profile_name]
    question = (question or "").strip()
    if not question:
        return "请先输入问题。", ""
    # 没上传、没点建库也没关系 → 自动用默认 GMR 论文库
    try:
        store = resolve_store(store)
    except Exception as e:
        return (f"⚠ **默认知识库加载失败**：{e}\n\n"
                f"请确认 `{PERSIST_DIR}` 存在（第三周 day13 建的 chroma_db）。"), ""

    # ① 服务身份检查（防跑错模型；Day14 踩过的坑）
    try:
        served = server_model_name(profile)
    except Exception as e:
        return (f"⚠ **连不上模型服务**：`{profile['service_note']}`\n\n报错：{e}"), ""
    if served != profile["model_name"]:
        return (f"⚠ **服务端模型 = {served}**，但当前 profile 要求 = {profile['model_name']}。\n\n"
                f"请先停掉旧服务、启动与 profile 匹配的服务（6G 显存一次只能跑一个 3B）。"), ""

    # ② 检索层：问题向量化 → 余弦 Top-K
    docs = store.similarity_search_with_relevance_scores(question, k=int(top_k))
    context = "\n".join(f"[{d.metadata.get('chunk_id')}] {d.page_content}" for d, _ in docs)

    # ③ 生成层：套模板 → POST 本地接口
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    payload = {
        "model": profile["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": profile["temperature"],
        "max_tokens": MAX_TOKENS,
    }
    try:
        resp = requests.post(profile["api_url"], json=payload, timeout=300)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠ 生成失败：{e}", ""

    # ④ 召回片段（折叠展示，证明答案有依据）
    chunk_md = "\n\n".join(
        f"**[资料§{d.metadata.get('chunk_id')}]**（余弦相似度 {score:.3f}）\n\n"
        f"{d.page_content[:400]}{'…' if len(d.page_content) > 400 else ''}"
        for d, score in docs
    )
    return answer, chunk_md


# ---------------------------------------------------------------------------
# 第 5 区：搭界面（Gradio Blocks）
# ---------------------------------------------------------------------------

def _on_build(pdf, progress=gr.Progress()):
    """建库按钮：上传了 PDF 就用它建内存库；没上传就回到默认 GMR 库。"""
    global _STORE
    if pdf is None:
        logs = "未上传 PDF → 使用**默认知识库**（GMR 论文，第三周 day13 建的 chroma_db）。"
        try:
            store = get_default_store()
            return (logs + "\n\n✅ 默认知识库就绪，可以直接提问。", "", store)
        except Exception as e:
            return f"⚠ 默认知识库加载失败：{e}", "", None

    src = pdf.name if hasattr(pdf, "name") else pdf
    # 复制到固定临时文件（上传文件名可能含中文/空格，Day16 踩过）
    tmp = Path(tempfile.gettempdir()) / "uploaded_day18.pdf"
    tmp.write_bytes(Path(src).read_bytes())

    store, logs = build_store_from_pdf(str(tmp), progress)
    _STORE = store
    return logs, "", store


def build_ui():
    # 说明：Gradio 6.x 起 theme 等外观参数要传给 launch()，不再放 Blocks() 构造里
    with gr.Blocks(title="机器人领域 RAG 文档问答工作台") as demo:
        gr.Markdown("# 🤖 机器人领域 RAG 文档问答工作台")
        gr.Markdown(
            "上传领域 PDF 建库 → 提问先检索资料片段 → 由 Qwen2.5-3B 组织**带 [资料§N] 出处**的答案。\n"
            "支持**原模型 / 微调模型**一键切换、检索 Top-K 可调、召回片段可视化。"
        )
        gr.Markdown(metrics_markdown())

        store_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ① 知识库")
                gr.Markdown(
                    "**不选文件也能直接用**：右下角「提问」会自动用默认 GMR 论文库。\n"
                    "想换成自己的资料，再上传 PDF 并点下面的建库按钮。"
                )
                pdf_file = gr.File(label="上传领域 PDF（论文/研报/技术文档）", file_types=[".pdf"])
                build_btn = gr.Button("建立知识库", variant="primary")
                build_log = gr.Markdown("_当前状态：未建库 → 提问时自动使用默认 GMR 论文库。_")
            with gr.Column(scale=2):
                gr.Markdown("### ② 提问")
                with gr.Row():
                    profile_dd = gr.Dropdown(
                        choices=list(CFG["profiles"].keys()),
                        value=CFG.get("default_profile", "lora"),
                        label="选择模型（需先启动对应服务）",
                    )
                    top_k_sl = gr.Slider(1, 10, value=CFG["profiles"][CFG.get("default_profile", "lora")]["top_k"],
                                         step=1, label="检索 Top-K（对照 eval_v2.py 的 --top-k）")
                question_tb = gr.Textbox(
                    label="你的问题",
                    placeholder="例如：论文中 GMR 与哪些方法进行了对比？",
                    lines=2,
                )
                ask_btn = gr.Button("提问", variant="primary")
                answer_md = gr.Markdown(value="_答案会显示在这里（带 [资料§N] 出处）_")
                with gr.Accordion("查看召回的资料片段（证明答案有依据）", open=False):
                    chunks_md = gr.Markdown(value="")

        gr.Examples(examples=[[q] for q in CFG.get("demo_questions", [])],
                    inputs=[question_tb], label="示例问题（点一下自动填入）")

        build_btn.click(_on_build, inputs=[pdf_file], outputs=[build_log, answer_md, store_state])
        ask_btn.click(ask, inputs=[store_state, profile_dd, question_tb, top_k_sl],
                      outputs=[answer_md, chunks_md])
        question_tb.submit(ask, inputs=[store_state, profile_dd, question_tb, top_k_sl],
                           outputs=[answer_md, chunks_md])

    return demo


# ---------------------------------------------------------------------------
# 第 6 区：入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG 问答工作台（Day18 完成版）")
    parser.add_argument("--port", type=int, default=7860, help="网页端口（默认 7860）")
    parser.add_argument("--check", action="store_true", help="只构建界面不启动（冒烟自检）")
    args = parser.parse_args()

    print("=" * 72)
    print("RAG 问答工作台启动中……")
    print(f"配置文件：{CONFIG_PATH}")
    print(f"默认知识库：{PERSIST_DIR}")
    print(f"指标来源：{EXPERIMENT_LOG}")
    print(f"可选模型：{list(CFG['profiles'].keys())}（先启动对应服务再提问）")
    print("=" * 72)

    demo = build_ui()
    if args.check:
        print("[check] 界面构建成功（未启动服务）。去掉 --check 即可打开网页。")
        return
    # Gradio 6.x：theme 从 launch() 传（放 Blocks 构造里会警告）
    demo.launch(server_name="127.0.0.1", server_port=args.port, theme=gr.themes.Soft())
    print(f"[OK] 界面已启动：http://127.0.0.1:{args.port}")


if __name__ == "__main__":
    main()
