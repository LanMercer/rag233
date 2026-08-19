# -*- coding: utf-8 -*-
# ↑ 这一行叫"编码声明"：# 开头 = 注释，Python 解释器会跳过不执行。
#   它的作用是告诉解释器"这个文件是用 UTF-8 编码保存的"（里面有中文）。
#   其实 Python 3 默认就按 UTF-8 读源码，所以这行属于"历史遗留惯例"，写上没坏处。
"""
本地模型 OpenAI 兼容接口服务（配合 Day9 教程使用）
↑ 这是一个"多行字符串注释"（用一对三个双引号包起来的 docstring）。
  它不是代码，是给人和 AI 看的说明书——不读代码也能知道这脚本在干嘛。

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：Qwen2.5-3B-Instruct，4bit 量化（NF4）加载（复用 Day6 的加载配置）
- 数据   ：通过 HTTP 请求传入的对话消息（OpenAI 格式的 messages 数组）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    ① 服务启动时把模型加载进显存（FastAPI lifespan 启动钩子，进程内只加载一次）；
    ② 提供 OpenAI 兼容接口：
         GET  /v1/models           -> 列出可用模型
         POST /v1/chat/completions -> 对话补全（与 OpenAI 官方接口同格式）
    ③ 收到请求后：messages 拼成 ChatML 提示词 -> 模型生成 -> 按 OpenAI 格式返回 JSON。
    ↑ 上面这几行把"这个脚本在机器学习的哪一环"写清楚：这是【推理/部署】环节，
      不是训练——所以没有损失函数、没有优化器。这也是总纲规定的"脚本五件套"。

运行（先装依赖，再启动）：
    pip install fastapi "openai>=1.0" langchain-openai
    uvicorn local_api:app --host 127.0.0.1 --port 8000

注意：
- Windows 下 uvicorn 命令要在 (llm) conda 环境里执行；若提示找不到 uvicorn，
  改用  python -m uvicorn local_api:app --host 127.0.0.1 --port 8000
- 模型加载约 1~3 分钟，看到日志里出现"模型已就绪"才算启动成功。
- 别加 --reload：它会在你每次保存代码时重新执行 lifespan、重新加载 3B 模型。
"""
# ↑ 上面的多行字符串注释到这里结束。它一整块都会被解释器跳过。

# ===========================================================================
# 第 1 区：导入库（把要用的工具包"借"进当前文件）
#   import = "引入"。引入之后，下面就能用 "包名.功能()" 的形式调用工具。
# ===========================================================================

import os
# ↑ import os：引入 Python 自带的"操作系统工具包"。
#   后面会用 os.environ（环境变量）和 os.path.exists（检查文件路径是否存在）。

import time
# ↑ import time：引入"时间工具包"。后面接口返回的 created（时间戳）靠它生成。

import uuid
# ↑ import uuid：引入"唯一 ID 生成器"（UUID = Universally Unique Identifier，
#   通用唯一标识符）。后面给每次响应生成"快递单号" id 就靠它。

from contextlib import asynccontextmanager
# ↑ from 包 import 名字：只引入包里的某一个名字。
#   这里引入 asynccontextmanager（异步上下文管理器装饰器），
#   它用来写 lifespan 生命周期函数（见"第 6 区"）——新手先不用深究，记住"是件外套"即可。

# 如果走 hf-mirror 下载方案，加载前也设一下镜像，避免自动联网找模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# ↑ os.environ 是"环境变量字典"（系统给程序的一堆配置项）。
#   setdefault(key, value) = "如果字典里没有这个 key，就设成这个值"（有就不动）。
#   效果：加载模型时优先走 hf-mirror.com 镜像站，不直连 HuggingFace 官网，
#   避免在国内网络环境下下载/访问失败或极慢。

import torch
# ↑ import torch：引入 PyTorch（深度学习框架）。模型推理的底层引擎，
#   后面判断有没有 GPU、控制显存、调用模型生成全都要用到它。

from fastapi import FastAPI, HTTPException
# ↑ 从 fastapi 包引入两个名字：
#   FastAPI —— Web 框架本体（用来创建应用对象 app）；
#   HTTPException —— 报错工具（抛一个带状态码的 HTTP 错误返回给调用方）。

from pydantic import BaseModel, Field
# ↑ 从 pydantic 包引入两个名字（pydantic = Python 数据校验库）：
#   BaseModel —— 数据模型基类（"第 5 区"的请求体模型继承它）；
#   Field —— 字段工具（用来标记"必填"等约束）。

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# ↑ 从 transformers（HuggingFace 的模型库）引入三件套：
#   AutoModelForCausalLM —— "自动加载对话生成模型"的类；
#   AutoTokenizer        —— "自动加载分词器"的类；
#   BitsAndBytesConfig   —— 4bit 量化的配置类（Day 6 就见过）。

# ===========================================================================
# 第 2 区：常量 + 全局变量（整个脚本的"参数设置区"）
#   常量 = 全大写命名、以后不改的固定值。日常想调参，基本只改这个区。
# ===========================================================================

# ---------------------------------------------------------------------------
# 常量：模型路径 / 对外模型名 / 默认生成参数（与 Day8 的 prompt_test_v2.py 保持一致）
# ---------------------------------------------------------------------------
MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"
# ↑ 模型在硬盘上的位置（Day 6 下载的地方）。
#   r"..." 里的 r 是 raw（原始字符串）前缀：反斜杠 \ 会被原样保留，不用写 \\。
#   没有 r 的话，\D 会被 Python 当成"转义字符"而报语法错误——所以路径必须加 r。

MODEL_NAME = "Qwen2.5-3B-Instruct"  # 对外暴露的模型名（OpenAI 请求里的 model 字段）
# ↑ 对外"招牌名字"。调用方在请求里写 "model": "Qwen2.5-3B-Instruct" 就会被认出来。
#   行尾的 # 是"行内注释"：只对这一行后面做说明，不影响代码。

DEFAULT_SYSTEM = "你是一个乐于助人、严谨可靠的中文助手。请严格按用户提示词的要求输出。"
# ↑ 默认的 system（系统）消息文本。如果调用方没传 system 消息，
#   接口会自动补上这句（见"第 8 区"接口 3 里 insert 那行），保证模型永远有一个"人设"。

DEFAULT_TEMPERATURE = 0.7
# ↑ 默认温度 0.7（Day 6 实验过的最稳组合）。
#   温度控制生成分布的"形状"：越高越发散、越低越确定。调用方不传就用这个默认值。

DEFAULT_TOP_P = 0.9
# ↑ 默认 top_p = 0.9：只从"累计概率前 90% 的候选词"里采样，控制候选词范围。

DEFAULT_MAX_TOKENS = 200
# ↑ 默认最多生成 200 个新 token（token ≈ 字/词的切分单位）。
#   超过就截断，此时返回的 finish_reason 会是 "length"。

# 全局：进程内只加载一次模型（启动时由 lifespan 赋值）
chat_model = None
# ↑ 全局变量，先占位 = None（什么都没有）。
#   等"第 6 区"里真正把模型加载进来后，这里才会被赋成"模型对象"。
#   之所以放最外面，是为了让文件里所有函数都能访问同一个模型。

tokenizer = None
# ↑ 全局变量，同上：占位 = None，"第 6 区"会赋成"分词器对象"。

model_ready = False
# ↑ 全局"开关"：False = 模型还没加载完，True = 可以接客了。
#   接口（"第 8 区"接口 3 的第一行 if）靠判断它来决定要不要返回 503。

# ===========================================================================
# 第 3 区：函数一 build_chatml() —— "翻译官"
#   把 OpenAI 格式的 messages 列表，翻译成 Qwen 认识的 ChatML 文本。
# ===========================================================================

# ---------------------------------------------------------------------------
# 对话消息列表 -> ChatML 提示词（Qwen 的对话格式，多轮也能拼）
# ---------------------------------------------------------------------------
def build_chatml(messages) -> str:
    # ↑ 定义一个函数。函数名 build_chatml = "构建 ChatML 文本"。
    #   messages 是入参（一堆对话消息）；-> str 是类型标注"返回字符串"（仅提示用，不强制）。

    parts = []
    # ↑ 建一个空列表 parts，用来收集"翻译好的每一段"。
    #   列表 = 一个能装多个东西的"抽屉柜"。

    for m in messages:
        # ↑ for ... in ... 是"循环"：把 messages 里每条消息依次取出来，
        #   每次取出的那条叫 m。循环体会对每条消息执行一遍。

        if m.role == "system":
            # ↑ if = "如果"。m.role 是这条消息的角色字段（system/user/assistant）。
            #   如果角色是 system（系统指令）……

            parts.append(f"<|im_start|>system\n{m.content}<|im_end|>")
            # ↑ 把这段文本塞进 parts 列表（append = 往列表末尾添加）。
            #   f"..." 是 f-string（格式化字符串）：花括号 {m.content} 会被替换成真实内容。
            #   \n 是换行符。所以拼出来是：<|im_start|>system<换行>具体内容<|im_end|>。
            #   <|im_start|> 和 <|im_end|> 是 ChatML 的"特殊标记"：标明这段话是谁说的、从哪开始到哪结束。

        elif m.role == "user":
            # ↑ elif = "否则如果"。上一条不成立、且这条角色是 user（用户）……

            parts.append(f"<|im_start|>user\n{m.content}<|im_end|>")
            # ↑ 同样方式拼成"用户发言"的 ChatML 片段。

        elif m.role == "assistant":
            # ↑ 否则如果角色是 assistant（助手的历史发言）……

            parts.append(f"<|im_start|>assistant\n{m.content}<|im_end|>")
            # ↑ 拼成"助手发言"的 ChatML 片段。
            #   注意：历史里的 assistant 发言也要拼进去，模型才能"记得"之前聊过什么（多轮对话）。



    # 最后补一个 assistant 开始标记，让模型从这里接下去生成
    parts.append("<|im_start|>assistant\n")
    # ↑ 循环结束后，追加一个"只有开始标记、没有内容"的 assistant 片段。
    #   这是"开口指令"：告诉模型"轮到你了，从这里开始写答案"。
    #   没有这一行，模型不知道自己要接话，输出会是空的或异常。

    return "\n".join(parts)
    # ↑ return = 返回结果。join 是"拼接"：把 parts 里所有片段，
    #   用换行符 \n 当"胶水"粘成一大段文本。
    #   "A".join([x, y, z]) 的结果是 xAyAz —— 这里 A 就是换行符。

# ===========================================================================
# 第 4 区：函数二 generate() —— "真正干活的机器"
#   调用模型生成文本，并算出 token 用量和结束原因。
# ===========================================================================

# ---------------------------------------------------------------------------
# 生成函数：输入 messages -> 输出 (文本, 结束原因, prompt_tokens, completion_tokens)
# ---------------------------------------------------------------------------
def generate(messages, temperature: float, top_p: float, max_tokens: int):
    # ↑ 定义函数 generate（生成）。需要 4 个入参：
    #   messages —— 消息列表；temperature / top_p —— 采样参数；max_tokens —— 最大生成长度。
    #   temperature: float 里的 : float 同样是"类型标注"，纯提示。

    prompt = build_chatml(messages)
    # ↑ 先调用"第 3 区"的翻译官，把消息列表变成 ChatML 文本，存进变量 prompt（提示词）。

    inputs = tokenizer(prompt, return_tensors="pt").to(chat_model.device)
    # ↑ 分词（tokenize）：把文字 prompt 变成"数字编号序列"（token ID），模型只认数字。
    #   return_tensors="pt"：让结果变成 PyTorch 的张量（张量 ≈ 多维数组，类似数学向量/矩阵）。
    #   .to(chat_model.device)：把张量搬到模型所在的设备（GPU 或 CPU）上。
    #   这一步相当于"把题目翻译成模型的语言"。

    gen_kwargs = dict(
        # ↑ 用一个字典（键值对集合）打包"生成参数"。dict(...) = 创建字典。
        #   后面一次性传给 model.generate()。

        max_new_tokens=max_tokens,
        # ↑ 最多生成 max_tokens 个新 token。

        do_sample=True,
        # ↑ 开启"采样"（按概率随机挑词），而不是"贪心"（每次选概率最高的词）。
        #   这是 temperature 能起作用的前提：采样才有随机性，贪心没有。

        temperature=temperature,
        # ↑ 温度参数（控制分布的"尖/平"）。

        top_p=top_p,
        # ↑ top_p 参数（控制候选词范围）。

        pad_token_id=tokenizer.eos_token_id,
        # ↑ pad_token_id = 填充符的 id。这里借用了"结束符"（eos）当填充符，
        #   避免某些张量长度不齐时报错——这是 transformers 的常规做法。
    )
    with torch.no_grad():
        # ↑ with ... 是"上下文管理器"：进入时打开、离开时自动关闭。
        #   torch.no_grad() = "这一段不计算梯度"。
        #   梯度是训练（反向传播更新参数）才需要的；推理时关掉能省显存、跑得更快。

        outputs = chat_model.generate(**inputs, **gen_kwargs)
        # ↑ 真正调用模型生成！** 是"展开字典"：把 inputs 和 gen_kwargs
        #   里的键值对展开成参数传给 generate()。
        #   outputs 里包含"输入的提示 + 模型新生成的 token"（连在一起的一大串）。

    # 只取模型新生成的部分（去掉输入提示词）
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    # ↑ 把"模型新写的那部分"切出来。拆开看：
    #   outputs[0]            —— 第一个（也是唯一一个）批次的结果；
    #   inputs["input_ids"].shape[1] —— 输入提示词的长度（token 个数）；
    #   [长度:]               —— 切片语法"从第 长度 个元素取到末尾"。
    #   合起来 = "从提示词结束的位置往后" = 只留模型新生成的 token 编号。
    #   （模型生成的 token 接在输入后面，所以输出总长一定 ≥ 输入长，切片不会切空。）

    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # ↑ decode（解码）：分词的反向操作——把 token 编号翻译回"人能读的文字"。
    #   skip_special_tokens=True：丢掉 <|im_end|> 这类特殊标记，只留干净的答案文本。

    # 判断结束原因：撞上 EOS 是自然结束 stop，否则是达到长度上限 length
    if len(new_tokens) > 0 and new_tokens[-1] == tokenizer.eos_token_id:
        # ↑ 判断"模型是自然说完还是被截断"。
        #   len(new_tokens) > 0：先生成内容，避免空列表取最后一位时报错；
        #   new_tokens[-1]：最后一个 token 编号（[-1] = 取最后一个）；
        #   它如果等于 eos_token_id（结束符编号），说明模型自己说了句号/结束。

        finish_reason = "stop"
        # ↑ 自然结束：正常说完，返回 "stop"。

    else:
        # ↑ 否则（最后一个 token 不是结束符）……

        finish_reason = "length"
        # ↑ 说明是撞上 max_tokens 上限被强行截断，返回 "length"。

    return text, finish_reason, len(inputs["input_ids"][0]), len(new_tokens)
    # ↑ 一次返回 4 个值（Python 会自动打包成元组）：
    #   text                —— 生成的回答文本；
    #   finish_reason       —— 结束原因（stop / length）；
    #   len(inputs["input_ids"][0]) —— 输入的 token 数（prompt_tokens）；
    #   len(new_tokens)     —— 输出的 token 数（completion_tokens）。

# ===========================================================================
# 第 5 区：请求体模型（用 pydantic 定义"合同"，FastAPI 自动校验）
# ===========================================================================

# ---------------------------------------------------------------------------
# 请求体模型（用 pydantic 定义，FastAPI 会自动做参数校验）
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    # ↑ 定义一个类（class）= 定义一种"数据模板"。继承 BaseModel 就有了自动校验能力。
    #   这个类是"单条消息"的模板。

    role: str   # system / user / assistant
    # ↑ 属性 role：必须是字符串（: str 是类型标注），取值是 system / user / assistant 之一。
    #   行尾 # 后面是行内注释。

    content: str
    # ↑ 属性 content：必须是字符串，装这条消息的具体内容。


class ChatRequest(BaseModel):
    # ↑ 第二个类：整个请求体的模板（所有字段都要按这里规定的类型和默认值来）。

    model: str = MODEL_NAME
    # ↑ model 字段：字符串，默认值是 MODEL_NAME。调用方不传就用默认（本地模型名）。

    messages: list[ChatMessage] = Field(...)
    # ↑ messages 字段：必须是"ChatMessage 的列表"。
    #   Field(...) 里的 ... 是 pydantic 的"必填标记"：这个字段必须提供，缺了直接报 400 错。
    #   注意 FastAPI 收到请求后会自动按这个模板校验：类型不对、字段缺失都会被拦下。

    temperature: float = DEFAULT_TEMPERATURE
    # ↑ 温度字段：浮点数，默认 0.7（没传就用全局默认）。

    top_p: float = DEFAULT_TOP_P
    # ↑ top_p 字段：浮点数，默认 0.9。

    max_tokens: int = DEFAULT_MAX_TOKENS
    # ↑ max_tokens 字段：整数，默认 200。

    stream: bool = False
    # ↑ stream 字段：布尔值（True/False），默认 False。
    #   流式输出开关——今天不实现，True 会被"第 8 区"接口 3 拒绝（501）。

# ===========================================================================
# 第 6 区：lifespan —— 服务的"出生与死亡"
#   yield 之前 = 启动时执行（加载模型）；yield 之后 = 关闭时执行（释放显存）。
# ===========================================================================

# ---------------------------------------------------------------------------
# 服务生命周期：启动时加载模型，关闭时释放显存
# ---------------------------------------------------------------------------
@asynccontextmanager
# ↑ 装饰器（@ = "给函数穿外套"）：把下面的 lifespan 包装成"上下文管理器"。
#   上下文管理器的规则：yield 上面的代码"进入时"跑，yield 下面的代码"退出时"跑。
#   新手不用深究实现，记住"这个模式让服务启动/关闭时能自动做事"即可。

async def lifespan(app: FastAPI):
    # ↑ 定义生命周期函数。async def = 异步函数（FastAPI 生命周期要求的写法，先照着写）。
    #   app: FastAPI 是入参——FastAPI 会自动把应用对象传进来。

    global chat_model, tokenizer, model_ready
    # ↑ global 声明：告诉 Python"下面我要修改的是文件最外层那 3 个全局变量"。
    #   不加这一行的话，函数里给同名变量赋值会创建"局部变量"，外面那个就不会被改到——这是 Python 的一个坑。

    print("=" * 60)
    # ↑ print = 往终端打印一行字。"=" * 60 = 把等号重复 60 次，打印一条分隔线，好看。

    print("[1/2] 加载分词器与模型（4bit 量化），首次约需 1~3 分钟……")
    # ↑ 打印进度提示，让你知道服务在干嘛、大概要等多久。

    print("=" * 60)
    # ↑ 再打一条分隔线。

    if not os.path.exists(MODEL_PATH):
        # ↑ 防御检查：os.path.exists(路径) 返回"这个路径存不存在"。
        #   not 表示取反，所以这一行 = "如果模型文件夹不存在"。

        raise RuntimeError(f"找不到模型文件夹：{MODEL_PATH}，请先下载模型或修改 MODEL_PATH。")
        # ↑ raise = 主动抛出一个错误。RuntimeError = 运行时错误。
        #   f"...{MODEL_PATH}..." 会把真实路径拼进报错信息，方便你排查。
        #   作用：防止"没下载模型就直接启动"时，报一个看不懂的错。

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # ↑ 从本地路径加载分词器（from_pretrained = "从预训练文件读取"），
    #   结果赋值给全局变量 tokenizer（这就是"第 2 区"那个 None 被"填上"的时刻）。

    quant_config = BitsAndBytesConfig(
        # ↑ 创建 4bit 量化配置对象（和 Day 6 完全相同的参数）。

        load_in_4bit=True,
        # ↑ 用 4bit 精度加载权重（省显存的核心开关）。

        bnb_4bit_quant_type="nf4",
        # ↑ 量化算法用 NF4（NormalFloat4，一种针对正态分布权重优化的 4bit 量化法）。

        bnb_4bit_compute_dtype=torch.float16,
        # ↑ 计算时用 float16（半精度），省显存、加快推理。

        bnb_4bit_use_double_quant=True,
        # ↑ 开启"双重量化"（对量化参数再压一次），再省一点显存。
    )
    chat_model = AutoModelForCausalLM.from_pretrained(
        # ↑ 真正加载模型本体（AutoModelForCausalLM = 自动加载"对话/续写"类模型）。

        MODEL_PATH,
        # ↑ 第一个参数：模型在本地磁盘的路径。

        quantization_config=quant_config,
        # ↑ 套用上面的 4bit 量化配置。

        device_map="auto",
        # ↑ 自动分配设备（有 GPU 用 GPU，没有就 CPU）。

        torch_dtype=torch.float16,
        # ↑ 模型权重用半精度存储。
    )
    chat_model.eval()
    # ↑ 把模型切到"评估（推理）模式"：关闭训练相关的随机行为（如 dropout）。
    #   只做推理的服务必须调用它，否则输出可能不稳定。

    model_ready = True
    # ↑ 打开"接客"开关：把全局变量 model_ready 设为 True，
    #   从此接口可以正常处理请求（见"第 8 区"接口 3 的开头）。

    if torch.cuda.is_available():
        # ↑ 检测有没有可用 GPU。torch.cuda.is_available() 返回 True/False。

        used_gb = torch.cuda.memory_allocated() / 1024 ** 3
        # ↑ 计算当前已分配的显存（单位换算）：
        #   torch.cuda.memory_allocated() 返回"已占用字节数"；
        #   1024 ** 3 = 1024 的三次方 = 1 GB 的字节数；
        #   两者相除 = 把字节换算成 GB。

        print(f"[OK] 模型已就绪：{MODEL_NAME}，显存占用约 {used_gb:.2f} GB")
        # ↑ 打印就绪信息。{used_gb:.2f} 是格式控制：保留 2 位小数。
        #   你实测应该约 1.9x GB——正好可以和 Day 6 的"部署账"对一遍。

    else:
        # ↑ 否则（没有 GPU）……

        print(f"[OK] 模型已就绪：{MODEL_NAME}（CPU 模式，速度较慢）")
        # ↑ 提示当前跑在 CPU 上，速度会明显慢。

    yield  # 服务运行期
    # ↑ 分界点：程序在这里"停住"，进入服务运行期。
    #   之后所有 HTTP 请求都由 FastAPI 处理（它们调用"第 8 区"的接口函数）。
    #   直到服务被关闭（Ctrl+C），程序才从这一行之后继续执行。

    # 关闭：释放显存
    del chat_model, tokenizer
    # ↑ 删除模型和分词器对象（del = delete）。
    #   让 Python 的垃圾回收可以释放它们占用的内存/显存。

    if torch.cuda.is_available():
        # ↑ 如果还有 GPU……

        torch.cuda.empty_cache()
        # ↑ 清空 PyTorch 的显存缓存，把显存真正归还给系统。

    print("[OK] 模型已释放，服务关闭。")
    # ↑ 打印收尾提示，告诉你资源已清理、服务可以安全关闭了。

# ===========================================================================
# 第 7 区：创建 FastAPI 应用对象（这个文件的"门面"）
# ===========================================================================

app = FastAPI(title="本地 Qwen OpenAI 兼容接口", lifespan=lifespan)
# ↑ 创建 FastAPI 应用对象，赋值给变量 app。
#   title="..."：给服务起名（会显示在 FastAPI 自动生成的文档里）。
#   lifespan=lifespan：把"第 6 区"那个生命周期函数"挂"到应用上——服务启动/关闭时自动执行它。
#   ⚠ uvicorn 的命令 "uvicorn local_api:app" 要找的正是这个 app 对象：
#   local_api = 文件名，app = 这一行创建的变量名。名字对不上就启动失败。

# ===========================================================================
# 第 8 区：三个接口（路由）
#   @app.get / @app.post 是"路由装饰器"：登记"某个 HTTP 请求来了就调用下面的函数"。
# ===========================================================================

# ---------------------------------------------------------------------------
# 接口 1：健康检查（教程第 3 步快速验证用）
# ---------------------------------------------------------------------------
@app.get("/health")
# ↑ 路由装饰器：当有人用 GET 请求访问 /health 这个地址时，执行下面的 health() 函数。
#   GET = 只读取、不改动的请求方法（浏览器输入网址回车就是 GET）。

def health():
    # ↑ 定义处理函数。注意它没有参数——这个接口不需要调用方传任何东西。

    return {"status": "ok", "model_ready": model_ready}
    # ↑ 返回一个字典。FastAPI 会自动把它转成 JSON 发给调用方。
    #   "status": "ok" —— 表示服务活着；
    #   "model_ready": model_ready —— 把全局开关的值报出去（True/False）。
    #   调用方 curl 一下就知道"服务活着吗？模型加载完了吗？"——相当于"体检报告"。


# ---------------------------------------------------------------------------
# 接口 2：列出模型（OpenAI 的 GET /v1/models 同款）
# ---------------------------------------------------------------------------
@app.get("/v1/models")
# ↑ 路由装饰器：GET /v1/models —— 这是 OpenAI 官方就有的接口，我们照做一份，
#   让任何"会调 OpenAI 的程序"也能列模型。

def list_models():
    # ↑ 定义处理函数。同样不需要参数。

    return {
        # ↑ 开始返回一个字典（会被自动转成 JSON）。

        "object": "list",
        # ↑ 顶层类型标识：这是"一个列表类型的响应"。

        "data": [
            # ↑ data 是一个列表，里面装"每一个可用的模型"。

            {
                # ↑ 每个模型的信息是一个小字典。

                "id": MODEL_NAME,
                # ↑ 模型 ID：用全局常量 MODEL_NAME（Qwen2.5-3B-Instruct）。

                "object": "model",
                # ↑ 元素类型标识：这是一个"模型"对象。

                "created": int(time.time()),
                # ↑ 创建时间戳。time.time() 返回当前秒数（浮点），int() 转成整数。

                "owned_by": "local",
                # ↑ "归属者：本地"——说明这个模型是本地自部署的。
            }
        ],
    }
    # ↑ 整个响应就绪。OpenAI 官方调用方看到这个结构就能认。


# ---------------------------------------------------------------------------
# 接口 3：对话补全（OpenAI 的 POST /v1/chat/completions 同款）★ 核心接口
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
# ↑ 路由装饰器：POST /v1/chat/completions —— 核心接口。
#   POST = 提交数据（通常带请求体 body），适合"发一段对话让模型续写"。
#   这就是 OpenAI 官方最常用的聊天接口，我们做了个一模一样的。

def chat_completions(request: ChatRequest):
    # ↑ 定义处理函数。入参 request 的类型标注是 ChatRequest —— 妙处在这：
    #   FastAPI 看到参数类型，会自动把请求体 JSON 解析出来、按"第 5 区"那个模板校验，
    #   校验通过后封装成 request 对象传进来。调用方参数不合法，根本进不了函数就会被拦下。

    if not model_ready:
        # ↑ if not = "如果不是"：如果模型还没加载完成……

        raise HTTPException(status_code=503, detail="模型还没加载完成，请稍后再试")
        # ↑ 抛一个 HTTP 错误：503（Service Unavailable，服务暂时不可用）。
        #   HTTP 状态码记忆：200 成功 / 400 请求格式错 / 500 服务器内部错 / 503 暂时不可用 / 501 未实现。

    if request.stream:
        # ↑ 如果调用方要求流式输出（stream=True）……

        raise HTTPException(status_code=501, detail="stream 暂未实现，请用非流式请求")
        # ↑ 返回 501（Not Implemented，功能未实现）。诚实拒绝，比假装支持好。

    msgs = list(request.messages)
    # ↑ 把 request.messages 复制一份（list() = 拷贝成新列表）。
    #   为什么要复制？因为下面可能要往列表里插入 system 消息，不想污染原始数据。

    if not msgs:
        # ↑ if not msgs = "如果消息列表是空的"（空列表的布尔值是 False）……

        raise HTTPException(status_code=400, detail="messages 不能为空")
        # ↑ 返回 400（Bad Request，请求参数错误）：你没传消息，没法聊天。

    # 没有 system 消息就补一个默认的
    if msgs[0].role != "system":
        # ↑ msgs[0] = 列表里第一条消息。如果第一条的角色不是 system……

        msgs.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM))
        # ↑ insert(0, x) = 把 x 插入到位置 0（最前面）。
        #   补上默认的 system 人设（"第 2 区"DEFAULT_SYSTEM 那句），保证模型"知道自己是助手"。
        #   注意这里直接调用了"第 5 区"的 ChatMessage 类来构造一条合法消息对象。

    try:
        # ↑ try = "尝试执行下面的代码"。配合 except 使用，组成"异常处理"。

        content, finish_reason, prompt_tokens, completion_tokens = generate(
            msgs, request.temperature, request.top_p, request.max_tokens
        )
        # ↑ 调用"第 4 区"的 generate() 函数，一次性接收它返回的 4 个值
        #   （文本 / 结束原因 / 输入 token 数 / 输出 token 数），分别存进 4 个变量。
        #   传入的生成参数都来自调用方请求（request.temperature 等），没传则用默认值。

    except Exception as e:
        # ↑ except = "如果上面出错了"，把错误对象命名为 e。
        #   Exception = Python 里所有错误的"总基类"，等于"捕获任何错误"。

        raise HTTPException(status_code=500, detail=f"生成失败：{e}")
        # ↑ 返回 500（Internal Server Error，服务器内部错误），并把具体错误信息拼进去。
        #   这样即使模型生成时出问题，服务也不会崩，而是把错误"翻译"成规范的 HTTP 响应。

    # 按 OpenAI 官方返回格式组织 JSON
    return {
        # ↑ 开始组装返回字典——这就是你 curl 时看到的那一大段 JSON 的出处。

        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        # ↑ 生成唯一 ID（"快递单号"）：
        #   uuid.uuid4() 产生一个随机 UUID；.hex 转成十六进制字符串；
        #   [:24] 只取前 24 个字符；前缀 chatcmpl- 是 OpenAI 的习惯写法。

        "object": "chat.completion",
        # ↑ 对象类型标识：这是一个"聊天补全"结果。

        "created": int(time.time()),
        # ↑ 响应生成时间戳（当前秒数转整数）。

        "model": request.model,
        # ↑ 回显调用方请求里写的模型名（可能是默认值 MODEL_NAME）。

        "choices": [
            # ↑ choices 是"候选回答列表"（数组）。OpenAI 支持一次生成多个候选
            #   （参数 n>1），我们只生成 1 个，所以列表里只有一个元素。

            {
                # ↑ 第 0 个（也是唯一一个）候选的详情。

                "index": 0,
                # ↑ 这个候选的序号（第 0 个）。多个候选时依次 0、1、2……

                "message": {"role": "assistant", "content": content},
                # ↑ 回答本体：role=assistant（角色：助手）+ content=模型生成的文本。
                #   调用方真正想要的就是 content。可以把这个 message 原样塞回下一轮请求，
                #   实现多轮对话。

                "finish_reason": finish_reason,
                # ↑ 结束原因（stop 自然结束 / length 被截断）——由 generate() 算好传回来的。
            }
        ],
        "usage": {
            # ↑ token 用量账本：OpenAI 靠它计费，本地模型靠它做性能统计。

            "prompt_tokens": prompt_tokens,
            # ↑ 输入（提示词）消耗的 token 数。

            "completion_tokens": completion_tokens,
            # ↑ 输出（模型生成）消耗的 token 数。

            "total_tokens": prompt_tokens + completion_tokens,
            # ↑ 两者之和 = 这次请求总消耗。
        },
    }
