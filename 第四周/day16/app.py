# -*- coding: utf-8 -*-
r"""
RAG 问答 Web 界面（第四周 Day16 · D 线可交付产物 · 最小可跑骨架）
↑ 一句话：把原来"命令行 python ask.py"的问答，搬到一个**浏览器页面**上——
  面试官/HR 打开网页就能用，这就是 HR 说的"要有一个成果，可以来使用"。

五件套说明（这个脚本在机器学习的哪一环）：
- 模型   ：生成层**不在本进程里加载**！本界面只是 HTTP 客户端——
             lora     → 第三周 day12 微调模型（day13 local_api_lora.py 起服务，默认）
             original → 原模型（第二周 day9 local_api.py 起服务）
           检索层 = bge-small-zh-v1.5（本地 CPU 跑，把问题向量化，从 Chroma 找相似段落）
- 数据   ：第三周 day13 建好的 GMR 论文向量库（chroma_db）
- 损失   ：无（不训练）｜ 优化器：无（不训练）
- 本脚本做的事（推理/服务阶段，在线运行）：
            浏览器里输入问题 → 后端检索 Top-K → 按模板 09 拼提示词
            → POST 到本地 OpenAI 兼容接口 → 把"答案 + 召回片段 + 出处"返回给页面显示。

★ 关键设计：为什么界面里不加载模型？（Day16 必须搞懂，面试会问）
    本机是 6G 显存，一个 Qwen2.5-3B（4bit）就要约 1.9~2.0GB，
    而"训练"和"模型服务"同时跑会 OOM（第三周踩过 os error 1455 / 崩溃）。
    所以架构上把"模型"和"界面"拆成两个进程：
        [模型服务进程]  ←HTTP→  [Web 界面进程]
    界面进程只做轻活（检索 + 发请求），显存完全留给模型服务；
    想换模型，只要换一个服务（停一个起一个），界面不用改、也不用重启。

与 day15 ask.py 的关系：
    检索 + 模板 09 + 生成调用三段逻辑同源，只是"输出目标"从终端打印换成了网页组件。
    配置也复用同一份 第三周\day15\config.json（profile / 路径 / embed_model 全读它）。

与"线上版"（本周新增 A/B 发布通道）的关系：
    本文件 = **本地版**：生成层指向本机 8000 端口的 Qwen 服务，供自己演示/录屏/彩排用。
    线上版 = 第四周\发布包\space_demo\app.py：同一套界面逻辑"派生"一份，
            生成层换成 DeepSeek API（免费 CPU 跑不动 3B），密钥走环境变量；
              页面必须标注"生成层 ≠ 本项目微调模型"，评测数字只归属本地 Qwen 版本。
    发布不归 Day16：Day18 派生骨架、Day20 上线（详见详细计划第七节 / 发布包说明.md）。
    ⚠ 三条红线：① 密钥绝不进代码/仓库；② 线上版必须标注生成层；③ 数字只归属本地 Qwen。

运行方法（新开一个终端；llm 环境）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day16"

    python app.py --check     # 只构建界面、不启动服务（用来确认环境和代码没问题）
    python app.py             # 启动网页，浏览器打开 http://127.0.0.1:7860

    前置：与所选 profile 对应的模型服务已在 8000 端口运行（见上面"模型"一行）。

参数（可选）：
    --port 7860   网页端口（被占用时换一个，如 7861）
    --check       只构建界面不启动（冒烟自检）
"""

import argparse
import json
import os
import sys

import requests

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固（Windows 控制台默认 GBK，中文/emoji 容易崩）
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr   # Gradio = 用 Python 几行代码起一个网页界面的库（无需写前端）

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# 第 2 区：路径与配置（复用第三周 day15 的 config.json）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # 第四周\day16
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))    # 仓库根
CONFIG_PATH = os.path.join(REPO_DIR, "第三周", "day15", "config.json")

# 第三周 day14 跑出的"优化前基线"（页面顶部先展示它，优化后数字 Day20 再回填）
BASELINE = "检索命中 20%（2/10）｜生成正确 20%（2/10）｜防幻觉 60%（6/10）｜防幻觉 F1 0.667"

# 模板 09（与 day10 / day14 / day15 一字不差；Day19 的 v2 消融在 eval_v2.py 里做）
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

# 向量库只在这里加载一次（进程内复用，避免每次提问都重载 bge 模型）
_STORE = None


def get_store():
    global _STORE
    if _STORE is None:
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _STORE = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
    return _STORE


# ---------------------------------------------------------------------------
# 第 3 区：核心问答逻辑（检索 → 拼模板 → 调接口）
# ---------------------------------------------------------------------------

def api_base(api_url):
    """从 .../v1/chat/completions 截出服务根地址（用于拼 /v1/models）。"""
    return api_url.rsplit("/v1/chat/completions", 1)[0]


def server_model_name(profile):
    """GET /v1/models：拿服务端真实加载的模型名（判"服务在跑 ≠ 服务正确"）。"""
    r = requests.get(api_base(profile["api_url"]) + "/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def ask(profile_name, question):
    """
    一次完整问答，返回 (答案 markdown, 召回片段 markdown)。
    出错不崩页面：把错误信息当作"答案"返回，方便排查。
    """
    profile = CFG["profiles"][profile_name]
    question = (question or "").strip()
    if not question:
        return "请先输入问题。", ""

    # ① 服务身份检查（防跑错模型；Day14 踩过的坑）
    try:
        served = server_model_name(profile)
    except Exception as e:
        return (f"⚠ **连不上模型服务**：`{profile['service_note']}`\n\n"
                f"报错：{e}"), ""
    if served != profile["model_name"]:
        return (f"⚠ **服务端模型 = {served}**，但当前 profile 要求 = {profile['model_name']}。\n\n"
                f"请先停掉旧服务、启动与 profile 匹配的服务（6G 显存一次只能跑一个 3B）。"), ""

    # ② 检索层：问题向量化 → 余弦 Top-K
    store = get_store()
    docs = store.similarity_search_with_relevance_scores(question, k=profile["top_k"])
    context = "\n".join(f"[{d.metadata.get('chunk_id')}] {d.page_content}" for d, _ in docs)

    # ③ 生成层：套模板 → POST 本地接口
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    payload = {
        "model": profile["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": profile["temperature"],
        "max_tokens": 512,
    }
    try:
        resp = requests.post(profile["api_url"], json=payload, timeout=300)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠ 生成失败：{e}", ""

    # ④ 召回片段（给页面折叠展示，证明答案有依据）
    chunk_md = "\n\n".join(
        f"**[资料§{d.metadata.get('chunk_id')}]**（相似度 {score:.3f}）\n\n"
        f"{d.page_content[:300]}…"
        for d, score in docs
    )
    return answer, chunk_md


# ---------------------------------------------------------------------------
# 第 4 区：搭界面（Gradio Blocks）
#   布局：顶部 = 项目名 + 优化前基线；中间 = 模型选择 + 问题 + 按钮；下方 = 答案 + 召回片段
# ---------------------------------------------------------------------------

def build_ui():
    # 说明：Gradio 6.x 起，theme 等外观参数要传给 launch()，不再放 Blocks() 构造里（放这里会有警告）
    with gr.Blocks(title="机器人领域 RAG 文档问答系统") as demo:
        # --- 顶部：项目一句话 + 优化前基线（优化后数字 Day20 回填）---
        gr.Markdown(
            "# 机器人领域 RAG 文档问答系统\n"
            "上传/固定领域资料建库 → 提问先检索资料片段 → 由 Qwen2.5-3B 组织**带出处的答案**。\n\n"
            f"**优化前基线（20 题固定评测集）**：{BASELINE}\n\n"
            "> 优化后数字待第四周实验跑完后回填（D5 定稿）。"
        )

        with gr.Row():
            profile_dd = gr.Dropdown(
                choices=list(CFG["profiles"].keys()),
                value=CFG.get("default_profile", "lora"),
                label="选择模型（需先启动对应服务）",
            )

        question_tb = gr.Textbox(
            label="你的问题",
            placeholder="例如：论文中 GMR 与哪些方法进行了对比？",
            lines=2,
        )
        ask_btn = gr.Button("提问", variant="primary")

        answer_md = gr.Markdown(label="答案", value="_答案会显示在这里（带 [资料§N] 出处）_")
        with gr.Accordion("查看召回的资料片段（证明答案有依据）", open=False):
            chunks_md = gr.Markdown(value="")

        # 点击 / 回车 都触发问答
        ask_btn.click(ask, inputs=[profile_dd, question_tb], outputs=[answer_md, chunks_md])
        question_tb.submit(ask, inputs=[profile_dd, question_tb], outputs=[answer_md, chunks_md])

        # 示例问题（来自 config.json，开箱即用）
        gr.Examples(examples=[[q] for q in CFG.get("demo_questions", [])],
                    inputs=[question_tb], label="示例问题（点一下自动填入）")
    return demo


# ---------------------------------------------------------------------------
# 第 5 区：入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG 问答 Web 界面（Gradio 最小可跑版）")
    parser.add_argument("--port", type=int, default=7860, help="网页端口（默认 7860）")
    parser.add_argument("--check", action="store_true", help="只构建界面不启动（冒烟自检）")
    args = parser.parse_args()

    print("=" * 72)
    print("RAG Web 界面启动中……")
    print(f"配置文件：{CONFIG_PATH}")
    print(f"向量库  ：{PERSIST_DIR}")
    print(f"可选模型：{list(CFG['profiles'].keys())}（先启动对应服务再提问）")
    print("=" * 72)

    demo = build_ui()
    if args.check:
        print("[check] 界面构建成功（未启动服务）。去掉 --check 即可打开网页。")
        return
    # 注意：Gradio 6.x 的 launch() 没有 show_api 参数；theme 也改从这里传（Blocks 构造里传会警告）
    demo.launch(server_name="127.0.0.1", server_port=args.port, theme=gr.themes.Soft())
    print(f"[OK] 界面已启动：http://127.0.0.1:{args.port}")


if __name__ == "__main__":
    main()
