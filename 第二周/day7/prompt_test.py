# -*- coding: utf-8 -*-
"""
提示词模板实测脚本（配合 Day7 教程第 3 步使用）

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：Qwen2.5-3B-Instruct，4bit 量化（NF4）加载（复用 Day6 的加载配置）
- 数据   ：10 个手工编写的提示词模板（总结/翻译/改写/抽取/问答 各 2 个，中英文各半）+ 各自测试输入
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    ① 用 BitsAndBytesConfig 以 4bit 方式加载本地模型；
    ② 依次用 10 个模板生成输出（temperature=0.7, top_p=0.9）；
    ③ 打印每个模板的 模板名 -> 测试输入 -> 模型输出，供填入《提示词模板库.md》v1。

用法：
    python prompt_test.py            # 跑全部 10 个模板
    python prompt_test.py 1 4 7      # 只跑编号为 1、4、7 的模板
"""

import os
import sys

# 如果走 hf-mirror 下载方案，加载前也设一下镜像，避免自动联网找模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 模型所在本地路径（Day6 download_qwen.py 下载的位置；如果实际路径不同请修改这里）
MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"

# 生成参数（Day6 实验过的最稳组合：温度 0.7 + top_p 0.9）
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 150

# ---------------------------------------------------------------------------
# 10 个提示词模板：template 里的 {input} 占位符会被各自 test_input 替换
# 加新模板：在列表末尾复制一个字典，改 id/name/category/lang/template/test_input 即可
# ---------------------------------------------------------------------------
TEMPLATES = [
    {
        "id": 1,
        "name": "中文要点总结",
        "category": "总结",
        "lang": "中",
        "template": (
            "请把下面这段文字总结成 3 个要点，每个要点不超过 20 个字，"
            "用「-」开头逐条列出：\n\n{input}"
        ),
        "test_input": (
            "LangChain 是一个用于构建大模型应用的框架，它把模型调用、提示词管理、"
            "向量检索、工具调用等能力封装成标准组件，让开发者能快速拼装出完整的 AI 应用。"
            "它支持多种大模型、多种向量数据库，是目前 RAG 应用开发最常用的框架之一。"
        ),
    },
    {
        "id": 2,
        "name": "English Summarize",
        "category": "总结",
        "lang": "英",
        "template": (
            "Summarize the following text in ONE sentence (no more than 30 words). "
            "Output only the summary:\n\n{input}"
        ),
        "test_input": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines a "
            "retrieval system with a text generator. Given a question, the system "
            "first searches a knowledge base for relevant passages, then feeds those "
            "passages to the language model as context. This allows the model to answer "
            "questions about private or recent documents that it has never seen during training."
        ),
    },
    {
        "id": 3,
        "name": "中译英",
        "category": "翻译",
        "lang": "中→英",
        "template": (
            "请把下面的中文翻译成英语，要求准确、自然、忠实原意，只输出译文：\n\n{input}"
        ),
        "test_input": (
            "机器学习是一门前沿学科，它让计算机从数据中自动学习规律，"
            "而无需人工编写每一条规则。"
        ),
    },
    {
        "id": 4,
        "name": "英译中",
        "category": "翻译",
        "lang": "英→中",
        "template": (
            "Please translate the following English into natural Chinese. "
            "Output only the translation:\n\n{input}"
        ),
        "test_input": (
            "Prompt engineering is the practice of designing input text that guides "
            "a language model to produce the desired output. Well-designed prompts "
            "can dramatically improve answer quality without retraining the model."
        ),
    },
    {
        "id": 5,
        "name": "正式化改写",
        "category": "改写",
        "lang": "中",
        "template": (
            "请把下面这句话改写成更正式、更适合书面汇报的风格，保持原意不变，只输出改写结果：\n\n{input}"
        ),
        "test_input": (
            "这个模型跑起来挺快的，显存占得也不多，用来做问答 demo 完全够用。"
        ),
    },
    {
        "id": 6,
        "name": "Professional Rewrite",
        "category": "改写",
        "lang": "英",
        "template": (
            "Rewrite the following sentence in a more professional tone while keeping "
            "the original meaning. Output only the rewritten sentence:\n\n{input}"
        ),
        "test_input": (
            "This model is pretty fast and doesn't take much memory, so it's totally "
            "fine for a QA demo."
        ),
    },
    {
        "id": 7,
        "name": "信息抽取 → JSON",
        "category": "抽取",
        "lang": "中",
        "template": (
            "请从下面的文本中抽取「人物 / 时间 / 地点 / 金额」四类信息。"
            "只输出 JSON 对象，字段为 person / date / location / amount，"
            "没有的信息填 null，不要输出任何解释：\n\n{input}"
        ),
        "test_input": (
            "3月15日，王小明在上海浦东发展银行办理了一笔10万元的定期存款业务，"
            "理财经理李芳为其推荐了两款稳健型理财产品。"
        ),
    },
    {
        "id": 8,
        "name": "Extract → JSON",
        "category": "抽取",
        "lang": "英",
        "template": (
            "Extract the person, date, location and amount from the text below. "
            "Output ONLY a JSON object with keys person / date / location / amount. "
            "Use null when a field is missing. No explanations:\n\n{input}"
        ),
        "test_input": (
            "On May 6, Li Hua purchased a laptop for 7,999 yuan at the Suning store "
            "in Nanjing. The receipt was issued by the cashier Wang Wei."
        ),
    },
    {
        "id": 9,
        "name": "基于资料回答",
        "category": "问答",
        "lang": "中",
        "template": (
            "请只依据下面给定的资料回答问题。如果资料里没有相关内容，"
            "请直接说「资料中没有提到」，不要编造：\n\n【资料】\n{input}"
        ),
        "test_input": (
            "4bit 量化把模型权重从 16 位压缩到 4 位，显存占用大约降到原来的四分之一。"
            "Qwen2.5-3B-Instruct 用 4bit 加载后，在 6GB 显存的笔记本上实测占用约 1.92GB。"
            "推理时 temperature 控制输出的随机程度，top_p 控制候选词的范围。\n\n"
            "【问题】\n4bit 量化大概能省多少显存？"
        ),
    },
    {
        "id": 10,
        "name": "Context-based QA",
        "category": "问答",
        "lang": "英",
        "template": (
            "Answer the question using ONLY the given context. If the context does not "
            "contain the answer, say \"Not mentioned in the context.\" Do not make up facts.\n\n"
            "Context:\n{input}"
        ),
        "test_input": (
            "RAG stands for Retrieval-Augmented Generation. It retrieves relevant passages "
            "from a knowledge base before generating an answer, which helps the model "
            "handle private or newly-updated documents. A typical RAG pipeline has five "
            "stages: document loading, text splitting, embedding, retrieval, and generation.\n\n"
            "Question: Why does RAG help with private documents?"
        ),
    },
]


def build_chat_prompt(system_text: str, user_text: str) -> str:
    """把 system / user 文字包装成 Qwen 使用的对话格式（ChatML）。"""
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate(user_text: str) -> str:
    """让模型生成一段文本（temperature=0.7, top_p=0.9）。"""
    prompt = build_chat_prompt(
        "你是一个乐于助人、严谨可靠的中文助手。请严格按用户提示词的要求输出。",
        user_text,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # 只取模型新生成的部分（去掉输入提示），并去掉对话标记
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    global model, tokenizer

    print("=" * 60)
    print("第 1 步：4bit 量化加载 Qwen2.5-3B-Instruct")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[X] 找不到模型文件夹：{MODEL_PATH}")
        print("[X] 请先运行第二周\\day6\\download_qwen.py 下载模型，或修改脚本里的 MODEL_PATH。")
        sys.exit(1)

    print("[OK] GPU 可用：", torch.cuda.is_available())
    print("[OK] 显卡：", torch.cuda.get_device_name(0))

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("[OK] 正在加载分词器（Tokenizer）……")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("[OK] 正在加载模型（4bit 量化），首次约需 1~3 分钟……")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # 解析命令行参数：python prompt_test.py 1 4 7 -> 只跑 1、4、7 号模板
    if len(sys.argv) > 1:
        selected_ids = [int(a) for a in sys.argv[1:]]
        templates = [t for t in TEMPLATES if t["id"] in selected_ids]
    else:
        templates = TEMPLATES

    print("=" * 60)
    print(f"第 2 步：开始实测 {len(templates)} 个模板（共 {len(TEMPLATES)} 个）")
    print("=" * 60)

    for tpl in templates:
        print("\n" + "#" * 60)
        print(f"模板 {tpl['id']:02d}｜{tpl['name']}（{tpl['category']} / {tpl['lang']}）")
        print("#" * 60)
        print("[测试输入]")
        print(tpl["test_input"])
        print("[模型输出]")
        print(generate(tpl["template"].format(input=tpl["test_input"])))

    print("=" * 60)
    print("实测结束。请把每个模板的[模型输出]复制进《提示词模板库.md》v1 的实测记录栏，")
    print("终端输出可整体截图作为 Day7 验收材料。")
    print("=" * 60)


if __name__ == "__main__":
    main()
