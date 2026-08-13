# -*- coding: utf-8 -*-
"""
提示词模板实测脚本 v2（配合 Day8 教程第 5 步使用）

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：Qwen2.5-3B-Instruct，4bit 量化（NF4）加载（复用 Day6 的加载配置）
- 数据   ：20 个手工编写的提示词模板（5 大分类：通用助手 A / 文本处理 B / 抽取结构化 C / 推理问答 D / 代码 E）
           + 各自测试输入（10 个继承 Day7 的 v1 模板，10 个为 Day8 新增）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    ① 用 BitsAndBytesConfig 以 4bit 方式加载本地模型；
    ② 依次用选中的模板生成输出（temperature 默认 0.7 / top_p 0.9；模板可单独设温度）；
    ③ 打印每个模板的 模板名 -> 测试输入 -> 模型输出，供填入《提示词模板库.md》v2。

用法：
    python prompt_test_v2.py          # 跑全部 20 个模板
    python prompt_test_v2.py 7 9 16   # 只跑编号为 7、9、16 的模板

注释开关（同一模板的不同变体：想切换时互换行首的 # 即可，不用改任何代码逻辑）：
    模板 01：zero-shot <-> few-shot（含 1 个示例）
    模板 11：风控专家 <-> 普通助手（角色扮演对比实验，教程 1.4）——⚠ 切换时同步改
            上部 TPL11_NAME 变量（角色名与 template 保持一致）
    模板 18：思维链   <-> 直接回答（思维链对比实验，呼应 Day7 加餐 B）
"""

import os
import sys

# 如果走 hf-mirror 下载方案，加载前也设一下镜像，避免自动联网找模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 模型所在本地路径（Day6 download_qwen.py 下载的位置；如果实际路径不同请修改这里）
MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"

# 生成参数默认值（Day6 实验过的最稳组合：温度 0.7 + top_p 0.9）
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 200

# ---------------------------------------------------------------------------
# 20 个提示词模板（v2，5 大分类）：
#   A 通用助手：11~14     B 文本处理：01~06     C 抽取结构化：07、08、16、17
#   D 推理问答：09、10、18、19                   E 代码：20、21
#
# template 里的 {input} 占位符会被各自 test_input 替换。
# 注意：模板正文里若出现 JSON 示例等字面花括号，必须写成 {{ }} 转义，
#       否则 .format() 会把它当成占位符报错。
# 每个模板可单独指定 temperature（没有就默认 TEMPERATURE）。
# 加新模板：在列表末尾复制一个字典，改 id/name/category/lang/template/test_input 即可。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ★ 模板 11 角色名开关 ★（与下方模板 11 的"角色切换开关"同步）
#   当前生效：角色B（普通助手）——与 template 当前生效版本一致。
#   切换角色时：把下面两行行首的 # 互换，name 与 template 保持一致。
# ---------------------------------------------------------------------------
# TPL11_NAME = "角色扮演·风控专家"   # 角色A（风控专家）
TPL11_NAME = "角色扮演·普通助手"    # 角色B（普通助手）

TEMPLATES = [
    # ---------------- B. 文本处理（继承 v1） ----------------
    {
        "id": 1,
        "name": "中文要点总结",
        "category": "B 文本处理",
        "lang": "中",
        # ============================================================
        # ★ few-shot 切换开关 ★（Day7 教程 3.5 的对比实验）
        #   当前生效：版本A（zero-shot，直接下命令）。
        #   想加示例对比：把版本A两行行首加 # 注释掉，
        #   再把版本B三行行首的 # 去掉，保存重跑即可。
        # ============================================================
        "template": (
            # ---- 版本A（默认）：zero-shot ----
            "请把下面这段文字总结成 3 个要点，每个要点不超过 20 个字，"
            "用「-」开头逐条列出：\n\n{input}"
            # ---- 版本B（备用）：few-shot（含 1 个示例）----
            # "请把下面这段文字总结成 3 个要点，每个要点不超过 20 个字，用「-」开头逐条列出。\n\n"
            # "示例：\n原文：今天天气很好，阳光明媚，我出门买了菜，回家做了一顿晚饭。\n"
            # "输出：\n- 天气很好\n- 出门买菜\n- 回家做饭\n\n现在请总结：\n{input}"
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
        "category": "B 文本处理",
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
        "category": "B 文本处理",
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
        "category": "B 文本处理",
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
        "category": "B 文本处理",
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
        "category": "B 文本处理",
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

    # ---------------- C. 抽取结构化 ----------------
    {
        # v3 迭代版：在 v2（约束+低温+单人示例）基础上，针对 Day8 实测暴露的两个新问题再迭代一轮：
        #   v2 实测问题：person 抽成次要人物"李芳"、amount 抽成 null。
        #   v3 改进：① few-shot 示例换成【与测试文本同构的双人文本】（客户+客户经理，教模型 person 只抽客户主体）；
        #            ② 约束写死"person 抽取办理业务的客户主体、金额按原文数字提取"。
        #   版本线：v1（乱码/合并）→ v2（约束+低温+单人示例）→ v3（双人同构示例+约束写死）
        "id": 7,
        "name": "信息抽取→JSON（v3 迭代版）",
        "category": "C 抽取结构化",
        "lang": "中",
        "temperature": 0.2,
        "template": (
            "请从下面的文本中抽取「人物 / 时间 / 地点 / 金额」四类信息。\n"
            "只输出 JSON 对象，字段为 person / date / location / amount，"
            "没有的信息填 null，不要输出任何解释。\n"
            "注意：\n"
            "- date 字段要提取完整且可读的日期；\n"
            "- person 字段抽取【办理业务的客户主体】，不抽业务员/经理等次要人物；\n"
            "- amount 字段按原文中的金额数字提取（如\"10万元\"提取为\"100000\"）。\n\n"
            "示例：\n"
            "文本：3月1日，张伟在中国银行存入5万元，客户经理刘洋为其办理。\n"
            "输出：{{\"person\": \"张伟\", \"date\": \"3月1日\", "
            "\"location\": \"中国银行\", \"amount\": \"50000\"}}\n\n"
            "现在请抽取：\n{input}"
        ),
        "test_input": (
            "3月15日，王小明在上海浦东发展银行办理了一笔10万元的定期存款业务，"
            "理财经理李芳为其推荐了两款稳健型理财产品。"
        ),
    },
    {
        "id": 8,
        "name": "Extract → JSON",
        "category": "C 抽取结构化",
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
        "id": 16,
        "name": "表格输出 → Markdown",
        "category": "C 抽取结构化",
        "lang": "中",
        "template": (
            "请把下面这段文字里的信息整理成 Markdown 表格，"
            "列名用给定的字段：模型名称 | 参数量 | 显存占用(4bit) | 适用场景。\n"
            "只输出表格本身，不要任何解释文字：\n\n{input}"
        ),
        "test_input": (
            "Qwen2.5-3B-Instruct 有 30 亿参数，4bit 量化后约占 2GB 显存，"
            "适合在 6GB 显卡上做问答 demo。Qwen2.5-7B-Instruct 有 70 亿参数，"
            "4bit 量化后约占 5GB 显存，适合质量要求更高的任务。Qwen2.5-14B-Instruct "
            "有 140 亿参数，4bit 量化后约占 10GB 显存，通常需要更大显存才能跑。"
        ),
    },
    {
        "id": 17,
        "name": "关键词提取",
        "category": "C 抽取结构化",
        "lang": "中",
        "template": (
            "请从下面这段话中提取 5 个最重要的关键词，用逗号分隔输出，不要解释：\n\n{input}"
        ),
        "test_input": (
            "RAG（检索增强生成）通过先检索相关资料再生成答案，解决了大模型不知道私有资料、"
            "回答过时等问题。它的核心流程包括文档加载、文本切块、向量化、相似度检索和答案生成，"
            "是目前企业知识库问答的主流方案。"
        ),
    },

    # ---------------- D. 推理问答 ----------------
    {
        # v2 升级版（RAG 标准版）：五件套齐全 + 资料带段落编号 + 引用出处 + 长度约束
        "id": 9,
        "name": "基于资料回答 RAG（v2）",
        "category": "D 推理问答",
        "lang": "中",
        "temperature": 0.3,
        "template": (
            "你是一名严谨的资料问答助手。\n"
            "请只依据下面给定的资料回答用户问题。\n\n"
            "【资料】\n{input}\n\n"
            "只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；"
            "每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。"
        ),
        "test_input": (
            "[1] 4bit 量化把模型权重从 16 位压缩到 4 位，显存占用大约降到原来的四分之一。\n"
            "[2] Qwen2.5-3B-Instruct 用 4bit 加载后，在 6GB 显存的笔记本上实测占用约 1.92GB。\n"
            "[3] 推理时 temperature 控制输出的随机程度，top_p 控制候选词的范围。\n\n"
            "【问题】\n4bit 量化大概能省多少显存？它在本地部署里有什么意义？"
        ),
    },
    {
        "id": 10,
        "name": "Context-based QA",
        "category": "D 推理问答",
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
    {
        "id": 18,
        "name": "思维链·数学推理",
        "category": "D 推理问答",
        "lang": "中",
        # ============================================================
        # ★ 思维链切换开关 ★（呼应 Day7 加餐实验 B）
        #   当前生效：版本A（思维链，一步步思考）。
        #   想对比"直接回答"：把版本A两行行首加 # 注释掉，
        #   再把版本B行首的 # 去掉，保存重跑即可。
        # ============================================================
        "template": (
            # ---- 版本A（默认）：思维链 ----
            "请一步步思考下面这道数学题，把计算过程完整写出来，最后再给出答案。\n"
            "最终答案以「答案：」开头，只写数字。\n\n{input}"
            # ---- 版本B（备用）：直接回答 ----
            # "请直接回答下面这道数学题，只输出最终答案数字。\n\n{input}"
        ),
        "test_input": (
            "某银行理财产品年化收益率为 4%，小明投入 50000 元，"
            "一年后连本带息一共是多少元？（不考虑税费）"
        ),
    },
    {
        "id": 19,
        "name": "多条件逻辑判断",
        "category": "D 推理问答",
        "lang": "中",
        "template": (
            "请根据下面的规则对案例做判断。请先逐条列出每个条件是否满足，再给出结论。\n\n"
            "规则：当且仅当同时满足「月收入不低于 2 万元」且「征信无逾期记录」时，才批准贷款。\n\n"
            "案例：\n{input}"
        ),
        "test_input": (
            "张某月收入 2.5 万元，但上个月有一笔信用卡还款逾期 2 天。"
        ),
    },

    # ---------------- A. 通用助手（新增） ----------------
    {
        "id": 11,
        "name": TPL11_NAME,
        "category": "A 通用助手",
        "lang": "中",
        # ============================================================
        # ★ 角色切换开关 ★（教程 1.4 的对比实验用）
        #   当前生效：角色B（普通助手）。
        #   想对比"风控专家"：把角色B两行行首加 # 注释掉，
        #   再把角色A两行行首的 # 去掉，保存重跑即可。
        #   ⚠ 同步操作：切换后记得把文件上部的 TPL11_NAME 变量
        #     也一起换注释（名字和角色保持一致）。
        # ============================================================
        "template": (
            # ---- 角色A（备用）：风控专家 ----
            # "你是一名有 10 年经验的银行资深风控分析师，擅长信贷审批，说话严谨专业。\n"
            # "请从风险角度分析下面这笔贷款申请，指出主要风险点，并给出是否批准的建议。\n"
            # ---- 角色B（默认）：普通助手 ----
            "你是一个乐于助人的助手，回答尽量通俗、友好、全面。\n"
            "请回答下面这个贷款申请的问题，并给出你的看法和建议。\n"
            "要求：分 3 点列出，每点不超过 40 字。\n\n{input}"
        ),
        "test_input": (
            "客户李某，28 岁，在互联网公司工作 2 年，月薪 1.8 万元，"
            "申请信用贷款 30 万元，用途为购买新能源汽车，名下无房贷，"
            "近半年征信查询记录 6 次。"
        ),
    },
    {
        "id": 12,
        "name": "头脑风暴",
        "category": "A 通用助手",
        "lang": "中",
        "template": (
            "你是一名擅长发散思考的创意顾问。请围绕下面的主题，给出 8 个不同的思路或点子。\n"
            "要求：每条用一句话说清楚，编号列出，尽量覆盖不同角度。\n\n主题：{input}"
        ),
        "test_input": (
            "如何用大模型技术给银行网点降本增效？"
        ),
    },
    {
        "id": 13,
        "name": "费曼讲解",
        "category": "A 通用助手",
        "lang": "中",
        "template": (
            "你是一名擅长把复杂概念讲清楚的老师。请用费曼学习法解释下面的概念：\n"
            "①先用最通俗的大白话讲一遍（给生活类比）；②再用数学/专业语言准确描述；"
            "③指出新手最容易误解的地方。\n\n概念：{input}"
        ),
        "test_input": (
            "什么是注意力机制（Attention）？"
        ),
    },
    {
        "id": 14,
        "name": "决策分析·加权打分",
        "category": "A 通用助手",
        "lang": "中",
        "template": (
            "你是一名理性的决策分析师。下面是需要评估的方案、评估维度和各维度权重（权重和为 1）。\n"
            "请给每个方案在各维度上打 1~10 分，计算加权总分，并给出你的推荐及理由。\n"
            "只输出 Markdown 表格和一句推荐结论。\n\n{input}"
        ),
        "test_input": (
            "方案：A=国企数据分析岗（稳定、成长中、薪资中），"
            "B=互联网大厂算法实习（成长快、不稳定、薪资高），"
            "C=继续读博（成长慢、不确定性大、长期上限高）。\n"
            "维度与权重：稳定 0.4，薪资 0.3，成长 0.3。"
        ),
    },

    # ---------------- E. 代码（新增） ----------------
    {
        "id": 20,
        "name": "代码生成",
        "category": "E 代码",
        "lang": "中→Python",
        "template": (
            "请用 Python 写一个函数实现下面的功能，只输出代码，不要解释。\n"
            "要求：包含函数定义和 docstring，用中文注释关键步骤。\n\n功能：{input}"
        ),
        "test_input": (
            "写一个函数 fibonacci(n)，返回第 n 个斐波那契数"
            "（n 从 1 开始计数，斐波那契数列为 1, 1, 2, 3, 5...）。"
        ),
    },
    {
        "id": 21,
        "name": "代码解释",
        "category": "E 代码",
        "lang": "Python→中",
        "template": (
            "请用大白话解释下面这段 Python 代码在做什么，并指出关键一行的作用。\n"
            "结构：①整体功能一句话；②逐行解释；③复杂度分析。\n\n{input}"
        ),
        "test_input": (
            "def merge(a, b):\n"
            "    i = j = 0\n"
            "    res = []\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            res.append(a[i]); i += 1\n"
            "        else:\n"
            "            res.append(b[j]); j += 1\n"
            "    res.extend(a[i:]); res.extend(b[j:])\n"
            "    return res"
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


def generate(user_text: str, temperature: float) -> str:
    """让模型生成一段文本（top_p 默认 0.9，temperature 由模板决定）。"""
    prompt = build_chat_prompt(
        "你是一个乐于助人、严谨可靠的中文助手。请严格按用户提示词的要求输出。",
        user_text,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
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

    # 解析命令行参数：python prompt_test_v2.py 7 9 16 -> 只跑 7、9、16 号模板
    if len(sys.argv) > 1:
        selected_ids = [int(a) for a in sys.argv[1:]]
        templates = [t for t in TEMPLATES if t["id"] in selected_ids]
    else:
        templates = TEMPLATES

    print("=" * 60)
    print(f"第 2 步：开始实测 {len(templates)} 个模板（库中共 {len(TEMPLATES)} 个）")
    print("=" * 60)

    for tpl in templates:
        print("\n" + "#" * 60)
        print(f"模板 {tpl['id']:02d}｜{tpl['name']}（{tpl['category']} / {tpl['lang']}）")
        print("#" * 60)
        print("[测试输入]")
        print(tpl["test_input"])
        print("[模型输出]")
        print(generate(tpl["template"].format(input=tpl["test_input"]), tpl.get("temperature", TEMPERATURE)))

    print("=" * 60)
    print("实测结束。请把每个模板的[模型输出]复制进《提示词模板库.md》v2 的实测记录栏，")
    print("终端输出可整体截图作为 Day8 验收材料。")
    print("=" * 60)


if __name__ == "__main__":
    main()
