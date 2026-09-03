# -*- coding: utf-8 -*-
r"""
RAG 问答脚本 · MVP 配置化版（第三周 Day15 项目完结日的核心整合产物）
↑ 多行字符串 docstring = 给人看的说明书（Python 会跳过，不执行）。

五件套说明（这个脚本在机器学习的哪一环）：
- 模型   ：生成层 = config.json 里 profiles 决定——
             original → Qwen2.5-3B-Instruct（原模型，Day9 local_api.py 起的服务）
             lora     → Qwen2.5-3B-Instruct-LoRA（微调模型，Day13 local_api_lora.py 起的服务，默认）
           检索层 = bge-small-zh-v1.5（本地向量模型，必须和建库时是同一个，否则向量空间对不上）
- 数据   ：day13 建好的 GMR 论文 Chroma 向量库（config.json 的 persist_dir 指向）
- 损失   ：无（不训练）｜ 优化器：无（不训练）
- 本脚本做的事（推理/问答阶段，在线运行）：把问题向量化 → 去向量库检索 Top-K →
            按模板 09 拼提示词 → 调本地 OpenAI 兼容接口（/v1/chat/completions）生成 →
            打印带 [资料§N] 出处的答案。

和 day10 rag_demo/ask.py 的区别（★ Day15 的改动，也是本项目"MVP 一键切换"的关键）：
  1. API_URL / MODEL_NAME / TOP_K / TEMPERATURE 不再写死在代码里，全部抽到同目录 config.json；
  2. config.json 里建了 original / lora 两个 profile，命令行 --profile 一键切换"用不用微调模型"；
  3. 开跑前自动 GET 一次 /v1/models，服务端模型名和 profile 对不上就停下提醒——
     防止"服务在跑但跑的是另一个模型"（Day14 踩过的 10048/27212 坑）。

运行方法（新开一个终端）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第三周\day15"
    python ask.py                            # 用 default_profile（lora）+ demo_questions 3 条
    python ask.py --profile original         # 一键切到原模型（需先起 day9 原模型服务）
    python ask.py "GMR 的英文全称是什么？"    # 自定义问题（可跟多个；带空格的问题用引号包起来）

前置条件：
  1. day13 的 chroma_db 在（不用重建）；
  2. 与 --profile 对应的服务已在 8000 端口运行：
       original → 在 第二周/day9 目录：python -m uvicorn local_api:app --host 127.0.0.1 --port 8000
       lora     → 在 第三周/day13 目录：python -m uvicorn local_api_lora:app --host 127.0.0.1 --port 8000
     （6G 显存同一时刻只能跑一个 3B 服务；切换 profile 前先停掉旧服务）
"""

# ===========================================================================
# 第 1 区：导入库 + 终端编码加固
# ===========================================================================

import argparse   # 命令行参数解析：让 python ask.py --profile xxx 这种写法能用
import json       # 读写 JSON 配置（config.json 靠它）
import os         # 操作系统工具：拼路径、判断路径是否存在
import sys        # 系统工具：下面用来给标准输出设 UTF-8 编码

import requests   # HTTP 客户端：把请求发给本地模型服务

from langchain_chroma import Chroma                     # 连向量库（和 day10 一样）
from langchain_huggingface import HuggingFaceEmbeddings  # 把"问题"也向量化（和建库同一个模型）

# Windows 控制台默认 GBK，强制 UTF-8 输出防中文乱码 / UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ===========================================================================
# 第 2 区：路径与配置读取（★ Day15 新增：配置外置）
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ↑ 本脚本所在的目录（第三周/day15）。用绝对路径定位，换台机器/挪文件夹都能跑。
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
# ↑ 配置文件路径：和 ask.py 同目录，叫 config.json。


def load_config():
    """读 config.json，返回整个字典。"""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)
    # encoding="utf-8"：config.json 里有中文（label 等），必须指定编码读，否则 Windows 下可能乱码。


def resolve_path(p):
    """把配置里的相对路径（相对本脚本目录）解析成绝对路径。"""
    if os.path.isabs(p):           # 已经是绝对路径（盘符开头）就直接用
        return p
    return os.path.normpath(os.path.join(SCRIPT_DIR, p))
    # os.path.join 拼出 第三周/day15 + ../day13/chroma_db = 第三周/day13/chroma_db
    # os.path.normpath 把 .. 规范化成真正路径，可读性好、避免双斜杠问题。

# ===========================================================================
# 第 3 区：模板 09（答案组织层，Day8 打磨好的 RAG 标准版，和 day10 完全一致）
# ===========================================================================

PROMPT_TEMPLATE = """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}"""

# ===========================================================================
# 第 4 区：连接向量库（检索层）
# ===========================================================================

def load_store(persist_dir, embed_model):
    """连接 Chroma 向量库；embedding 必须和建库时同一个模型。"""
    embedder = HuggingFaceEmbeddings(model_name=embed_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)


# ===========================================================================
# 第 5 区：服务身份检查（★ Day15 新增：防止"跑错模型"）
# ===========================================================================

def check_server(api_url, expected_model):
    """
    开跑前先 GET 一次 /v1/models，确认端口上跑的是我们想要的模型。
    返回 (是否匹配, 提示文本)。
    Day14 的教训：端口被占用 ≠ 服务正确——上次是残留的原模型服务占着 8000，
    评测却以为是微调模型。现在让脚本自己先验一下身份。
    """
    base = api_url.rsplit("/v1/chat/completions", 1)[0]   # 从接口地址去掉尾巴，得到服务根地址
    try:
        r = requests.get(base + "/v1/models", timeout=10) # GET 列出服务端实际加载的模型
        r.raise_for_status()
        served = r.json()["data"][0]["id"]                 # 服务端真实模型名
    except Exception as e:
        return False, f"连不上服务（{e}）。请先启动对应服务再跑。"
    if served == expected_model:
        return True, f"服务端模型 = {served} ✅ 与 profile 一致"
    return False, f"⚠ 服务端模型 = {served}，但 profile 要求 = {expected_model}。请切换服务（6G 一次只能跑一个 3B）。"


# ===========================================================================
# 第 6 区：一次完整的 RAG 问答（检索 + 生成），和 day10 ask() 同源
# ===========================================================================

def ask_one(store, profile, question: str):
    """
    检索：store.similarity_search_with_relevance_scores(question, k=top_k)
      → 内部把 question 用 bge 向量化，与库里每个 chunk 算相似度，取最像的 top_k 段 + 分数。
    生成：按模板 09 把检索结果拼成提示词，POST 给本地 OpenAI 兼容接口拿答案。
    """
    top_k = profile["top_k"]
    docs = store.similarity_search_with_relevance_scores(question, k=top_k)

    lines = [f"[{doc.metadata.get('chunk_id', 0)}] {doc.page_content}" for doc, _ in docs]
    context = "\n".join(lines)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    payload = {
        "model": profile["model_name"],              # 请求里写哪个模型（服务端回显用）
        "messages": [{"role": "user", "content": prompt}],
        "temperature": profile["temperature"],       # RAG 问答用低温，输出"稳"
        "max_tokens": 512,                           # 放宽答案长度上限，防中途截断
    }
    resp = requests.post(profile["api_url"], json=payload, timeout=300)
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]

    # 打印：问题 → 检索到的 Top-K（含分数与片段预览）→ 答案
    print("=" * 72)
    print(f"【问题】{question}")
    print("-" * 72)
    print(f"【检索到的 Top-K 片段】（k={top_k}）")
    for doc, score in docs:
        idx = doc.metadata.get("chunk_id", 0)
        preview = doc.page_content[:48].replace("\n", " ")
        print(f"  [资料§{idx}] 相似度={score:.3f} | {preview}...")
    print("-" * 72)
    print(f"【答案】{answer}")


# ===========================================================================
# 第 7 区：main —— 命令行入口
# ===========================================================================

def main():
    # 命令行参数：--profile 选模型配置；位置参数当自定义问题
    parser = argparse.ArgumentParser(description="RAG 问答 · MVP 配置化版（一键切换原模型 / 微调模型）")
    parser.add_argument("--profile", default=None,
                        help="模型配置名：original（原模型）或 lora（微调模型）。不填则用 config.json 的 default_profile")
    parser.add_argument("questions", nargs="*",
                        help="自定义问题（可跟多个）。不填则跑 config.json 的 demo_questions")
    args = parser.parse_args()

    cfg = load_config()

    # 1) 确定用哪个 profile
    profile_name = args.profile or cfg.get("default_profile", "lora")
    if profile_name not in cfg["profiles"]:
        print(f"❌ 没有这个 profile：{profile_name}。可选：{list(cfg['profiles'])}")
        sys.exit(1)
    profile = cfg["profiles"][profile_name]

    print("=" * 72)
    print(f"RAG MVP · profile = {profile_name}｜{profile['label']}")
    print(f"service_note：{profile['service_note']}")
    print("=" * 72)

    # 2) 服务身份检查（防跑错模型）
    ok, msg = check_server(profile["api_url"], profile["model_name"])
    print(msg)
    if not ok:
        print("→ 先停掉旧服务、启动与 profile 匹配的服务（6G 显存一次只能跑一个 3B），再重跑。")
        sys.exit(1)

    # 3) 连向量库，逐个问
    store = load_store(resolve_path(cfg["persist_dir"]), cfg["embed_model"])
    questions = args.questions or cfg["demo_questions"]
    for q in questions:
        ask_one(store, profile, q)

    print("=" * 72)
    print("✅ 本轮全部问题跑完。")


if __name__ == "__main__":
    main()
