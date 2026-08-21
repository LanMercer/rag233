# -*- coding: utf-8 -*-
# ↑ 这一行叫"编码声明"：# 开头 = 注释，Python 解释器会跳过不执行。
#   它的作用是告诉解释器"这个文件是用 UTF-8 编码保存的"（里面有中文）。
#   其实 Python 3 默认就按 UTF-8 读源码，这行属于"惯例"，写上没坏处。

"""
train_lora.py —— QLoRA 微调训练脚本（第三周 Day12 第 3 步，本日核心产出）
========================================================================
【五件套】（脚本在机器学习里的哪一环 = 训练环节）
- 模型   ：Qwen2.5-3B-Instruct，4bit 量化（NF4）底座 + LoRA 适配器（r=8, alpha=16）
- 数据   ：Day11 自制的 sft_data.json（300 条机器人行业公告/研报 三字段数据），
          90/10 切分训练/验证集（验证集只看 eval_loss 不参与梯度，防过拟合）
- 损失   ：交叉熵（LM loss）：只对"标准答案 output 部分"算交叉熵，
          即最小化 -E[log P(output token | 前面所有 token)]——模型答得越差 loss 越大
- 优化器 ：AdamW，学习率 2e-4（周计划 1e-4~2e-4 区间取上沿），warmup + cosine 调度
- 训练   ：3 个 epoch，per_device batch=2 + 梯度累积 8（等效 batch=16），
          用 transformers.Trainer 托管（自动做前向/反向/参数更新/loss 日志/验证）
【只存 adapter】训练完用 model.save_pretrained() 只保存 LoRA 低秩增量
（adapter_config.json + adapter_model.safetensors，几 MB~几十 MB），
不保存几 GB 的全量模型——这就是"参数高效微调"。
【运行前】先跑 day11 的 check_env.py 确认五件套全绿、显存 ≥4GB；
关掉浏览器/游戏等占显存程序。本脚本不下载新模型（复用 Day6 的本地权重）。

【怎么看懂这个脚本（先读这里）】
一句话：把 Qwen 模型读进显存 → 挂上"小补丁"（LoRA）→ 喂 300 条数据
       → 更新补丁参数 → 保存补丁。
整个脚本只有两个函数：mem_report（小工具）和 main（主流程）。
main() 从上到下顺序执行，像一条流水线，所以【阅读顺序 = 代码顺序】。
四步流水线：
  第 1/4 步：加载 4bit 底座模型        （第 84~105 行）
  第 2/4 步：挂 LoRA 适配器            （第 107~121 行）
  第 3/4 步：加载数据 + 转 token        （第 123~158 行，最需要仔细看）
  第 4/4 步：开始训练 + 保存 adapter    （第 160~205 行）
最后一行 if __name__ == "__main__": 是"入口"：直接运行本文件时才执行 main()。
========================================================================
"""
import os
# ↑ import os：引入 Python 自带的"操作系统工具包"。
#   后面用 os.makedirs（建文件夹）、os.listdir（列文件）、os.path.join（拼路径）等。

import sys
# ↑ import sys：引入"解释器工具包"。后面用它强制控制台输出 UTF-8，防中文乱码。

# ---------------------------------------------------------------------------
# 防中文乱码（Windows 控制台默认 GBK，打中文会乱）
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    # ↑ hasattr(对象, "属性名") = "判断对象有没有这个功能"。
    #   先判断再调用，旧版 Python 没有 reconfigure 也不会报错（加固写法）。
    sys.stdout.reconfigure(encoding="utf-8")
    # ↑ 把控制台输出流改成 UTF-8 编码，中文就能正常打印了。

import torch
# ↑ import torch：引入 PyTorch（深度学习框架）。
#   模型加载、GPU 控制、显存查看全都靠它。

from datasets import load_dataset
# ↑ from 包 import 名字：只引入包里的某一个名字，直接用 名字() 调用。
#   load_dataset：读取 JSON 数据的函数（Day11 的 check_sft_data.py 干跑过同款）。

from transformers import (
    # ↑ 从 transformers（HuggingFace 模型库）一次引入 6 个名字。括号折行是标准写法。
    AutoModelForCausalLM,
    # ↑ "自动加载对话/续写类模型"的类：自动识别模型类型并加载。
    AutoTokenizer,
    # ↑ "自动加载分词器"的类：负责"文字 ↔ 数字编号"互转。
    BitsAndBytesConfig,
    # ↑ 4bit 量化配置类（Day6/Day9 就见过，让 3B 模型只占 ~1.9GB 显存）。
    DataCollatorForSeq2Seq,
    # ↑ "数据收集器"：把 batch 内长度不齐的序列自动补齐（padding）成矩阵。
    Trainer,
    # ↑ "训练管家"：前向、反向、梯度更新、loss 打印、epoch 末验证全部自动做。
    TrainingArguments,
    # ↑ "训练参数配置单"：只描述参数，不行动。
)
from peft import LoraConfig, get_peft_model
# ↑ from peft 引入两个名字：
#   LoraConfig    —— LoRA 的"配置单"：r（秩）/ alpha（缩放）/ dropout / 挂哪几层；
#   get_peft_model—— 把底座包成"只训练 adapter"的外壳（底座冻结，B·A 可训练）。

# ===========================================================================
# 第 1 区：配置区（五件套全部写在这里，想调参只改这个区）
# ===========================================================================
# 全大写命名 = 常量（约定：值以后不改）。读代码先读这一区，就知道脚本要干嘛。

# —— 第一件：模型 ——
MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"
# ↑ 模型在硬盘上的位置（Day6 下载的）。
#   r"..." 里的 r = raw（原始字符串）：反斜杠 \ 原样保留，不用写 \\。

R = 8            # LoRA 秩：增量矩阵 B·A 的"宽度"，r 越小改动越温和（Day5 的"旋钮"）
ALPHA = 16       # 缩放系数：实际缩放 = alpha / r = 2（经验常用 2 倍）
DROPOUT = 0.05   # LoRA 层 dropout，防过拟合
# 给哪些线性层挂 LoRA：Qwen 的注意力四件套 + MLP 三件套，全挂效果最好
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
# ↑ 这是一个【列表】（用 [ ] 包起来），装着 7 个要挂 LoRA 的层名。
#   q/k/v/o 是注意力四件套，gate/up/down 是 MLP 三件套——挂得越多可学参数越多。

# —— 第二件：数据 ——
DATA_PATH = "sft_data.json"   # Day11 生成的三字段 SFT 数据（已复制到本文件夹）
VAL_RATIO = 0.1               # 10% 当验证集（只看 loss，不参与训练）
MAX_SEQ_LEN = 512             # 单条数据最长 token 数（超出截断）

# —— 第三件：损失 ——
# 交叉熵（LM loss）由 Trainer 自动算：只对 labels 中不是 -100 的位置算，
# 也就是只对"标准答案 output"算，指令部分不参与（见第 3/4 步的标签掩码）。

# —— 第四件：优化器 ——
LEARNING_RATE = 2e-4          # 1e-4~2e-4 区间，QLoRA 常用；太大容易灾难性遗忘
WARMUP_STEPS = 2              # 起步热身：前 2 步线性升温（≈总步数 51 步的 3%，新版已弃用 warmup_ratio）
SCHEDULER = "cosine"          # 余弦退火：学习率先平稳后衰减

# —— 第五件：训练 ——
EPOCHS = 3                    # 把 300 条数据完整过 3 遍
BATCH_SIZE = 2                # 每步 2 条（6G 显存小，batch 不能大）
GRAD_ACCUM = 8                # 梯度累积 8 步，等效 batch = 2 × 8 = 16
OUTPUT_ADAPTER = "lora_adapter"   # 只保存适配器到这里（不进 git）
LOG_DIR = "./train_logs"      # 训练日志目录
SEED = 42                     # 随机种子：固定了，每次跑结果可复现

# ===========================================================================
# 函数 mem_report —— 小工具：打印当前显存占用（全程盯显存用）
# ===========================================================================
def mem_report(tag: str):
    # ↑ def 函数名(参数): = 【定义函数】 = 打包一段可反复调用的代码。
    #   tag: str 是"类型标注"：提示 tag 应该是字符串（只是提示，不强制）。
    """打印当前显存占用（字节 -> GB），全程盯显存用"""
    if torch.cuda.is_available():
        # ↑ if 条件: = 【条件分支】。torch.cuda.is_available() = 有没有可用 GPU。
        #   只有有 GPU 才看显存（没 GPU 就不打印）。
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        # ↑ torch.cuda.memory_allocated() 返回"当前已占用显存（字节数）"。
        #   / 1024 ** 3：字节 ÷ 1024³ = 换算成 GB（1024³ ≈ 10.7 亿）。
        print(f"[显存] {tag}：allocated={alloc:.2f} GB")
        # ↑ f"..." 是 f-string（格式化字符串）：{tag} 会被替换成真实值。
        #   :.2f = 保留 2 位小数。

# ===========================================================================
# 函数 main —— 主流程（四步流水线）
# ===========================================================================
def main():
    print("=" * 60)
    # ↑ print = 往终端打印一行。"=" * 60 = 把等号重复 60 次 = 一条分隔线。

    print("第 1/4 步：加载 4bit 底座模型（Qwen2.5-3B-Instruct）")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 第 1/4 步：加载 4bit 底座模型
    # 目标：把 Qwen 从硬盘读进显存，用 4bit 量化省显存（6G 能装下的核心原因）。
    # -----------------------------------------------------------------------

    # 复用 Day6/Day9 完全同款的 4bit 量化配置（nf4 + 双重量化）
    quant_config = BitsAndBytesConfig(
        # ↑ 创建 4bit 量化配置对象（和 Day6/Day9 部署完全一样）：
        #   基座加载方式必须与训练时一致，否则 Day13 加载 adapter 会报错。
        load_in_4bit=True,
        # ↑ 4bit 加载总开关：权重压成 4bit 精度再进显存（省显存核心）。
        bnb_4bit_quant_type="nf4",
        # ↑ 量化算法用 NF4（NormalFloat4，针对正态分布权重优化）。
        bnb_4bit_compute_dtype=torch.float16,
        # ↑ 计算时用 float16（半精度），省显存、加快训练。
        bnb_4bit_use_double_quant=True,
        # ↑ 开启"双重量化"（对量化参数再压一次），再省一点显存。
    )

    model = AutoModelForCausalLM.from_pretrained(
        # ↑ from_pretrained(路径) = 从本地磁盘把模型读进内存/显存。
        #   它会自动识别模型类型（这里识别出 Qwen），返回模型对象。
        MODEL_PATH,                     # 模型在硬盘的路径
        quantization_config=quant_config,  # 套上 4bit 量化配置
        device_map="auto",              # 自动决定放 GPU 还是 CPU（有 GPU 用 GPU）
        dtype=torch.float16,            # 模型权重用半精度存储（transformers 5.x 新签名，老版本叫 torch_dtype）
    )

    model.config.use_cache = False      # 训练时关 KV cache，省显存
    # ↑ use_cache：推理时缓存历史计算结果来加速；训练时用不上，关掉省显存。

    mem_report("加载 4bit 底座后")      # 调 mem_report：打印此时显存占用

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # ↑ 加载分词器（"文字 ↔ 数字编号"互转的工具）。

    if tokenizer.pad_token is None:
        # ↑ pad_token = "填充符"（padding 用的占位符）。
        #   is None = "是空"：如果模型没定义填充符……
        tokenizer.pad_token = tokenizer.eos_token
        # ↑ 就借用 eos_token（结束符）当填充符。
        #   原因：batch 里数据长度不齐，需要 padding 对齐；没有 pad_token 会报错。
        #   这是 transformers 的常规补丁写法。

    print("=" * 60)
    print("第 2/4 步：挂 LoRA 适配器（LoraConfig + get_peft_model）")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 第 2/4 步：挂 LoRA 适配器
    # 目标：把底座"冻结"，在指定层上挂可训练的小矩阵 B·A（就是"小补丁"）。
    # -----------------------------------------------------------------------

    lora_config = LoraConfig(
        # ↑ LoRA 的"配置单"：只描述 LoRA 长什么样，不行动。
        r=R,                            # 秩 = 补丁的"宽度"（8）
        lora_alpha=ALPHA,               # 缩放系数（16，实际缩放 = 16/8 = 2 倍）
        lora_dropout=DROPOUT,           # LoRA 层 dropout，防过拟合
        target_modules=TARGET_MODULES,  # 挂在哪 7 个层上
        bias="none",                    # 不训练 bias（偏置），只训练 B·A
        task_type="CAUSAL_LM",          # 任务类型：自回归语言模型
    )

    model = get_peft_model(model, lora_config)
    # ↑ get_peft_model(原模型, 配置单) = 真正动手：
    #   把底座冻结（参数不更新），给指定层挂上可训练小矩阵 B·A。
    #   返回的 model 变成 PEFT 模型，绝大多数参数冻结、只有补丁可训练。

    model.print_trainable_parameters()
    # ↑ 打印参数账：可训练参数 / 总参数 / 占比（会看到 trainable%: 0.38 之类）。
    #   这就是"LoRA 只训不到 1% 参数"的官方出处。

    model.enable_input_require_grads()
    # ↑ 让输入侧也保留梯度。梯度检查点（下面一行）要求输入侧有梯度才能反传。

    model.gradient_checkpointing_enable()
    # ↑ 梯度检查点：前向时不留存中间结果、反向时重新算一遍——
    #   用"多算一遍"换"少占显存"，6G 跑 3B 的关键招。

    print("=" * 60)
    print("第 3/4 步：加载数据 + 切验证集 + 转 token（ChatML + 标签掩码）")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 第 3/4 步：数据加载 + 转 token（本日最需要仔细看的部分）
    # 目标：把 300 条 JSON 数据 → 切分训练/验证 → 拼成 ChatML → 转数字编号
    #       → 给"题干部分"打上 -100 掩码（让损失只对答案打分）。
    # -----------------------------------------------------------------------

    raw = load_dataset("json", data_files=DATA_PATH, split="train")
    # ↑ 读 JSON 数据（Day11 干跑过同款）。raw 是一个 Dataset 对象，类似"表格"。

    split = raw.train_test_split(test_size=VAL_RATIO, seed=SEED)
    # ↑ 切成 90% 训练 + 10% 验证。seed=42 保证每次切分结果一样（可复现）。
    #   split 是一个字典：split["train"] = 270 条训练，split["test"] = 30 条验证。

    print(f"训练集 {len(split['train'])} 条，验证集 {len(split['test'])} 条")
    # ↑ len(数据集) = 取条数。f-string 把数字替换进字符串。

    def tokenize_batch(batch):
        # ↑ 嵌套函数：Python 允许在函数里再定义函数。
        #   这个函数是"数据转换器"：把一批原始数据转成模型认识的数字编号。
        #   它本身不被直接调用，而是传给下面 .map(函数) 时由库自动调用。
        prompts, full_texts = [], []
        # ↑ 创建两个空列表：
        #   prompts   = 只装"题干"（ChatML 格式，不含答案）；
        #   full_texts = 装"题干 + 答案 + 结束符"（完整文本）。

        for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            # ↑ for a, b, c in zip(x, y, z): 循环。
            #   zip(三个列表) = 按位置打包成对：把第 1 条的
            #   (instruction, input, output) 依次给 ins/inp/out，然后第 2 条……
            #   相当于一行一行地读数据。

            # input 非空则拼成"指令 + 材料"，否则只有指令（解释/计算类）
            user_content = f"{ins}\n\n材料：{inp}" if inp else ins
            # ↑ 三元表达式 值A if 条件 else 值B：
            #   如果 input 非空 → 拼成"指令 + 材料"；
            #   否则（解释/计算类题目没有材料）→ 只有指令。

            prompt = tokenizer.apply_chat_template(
                # ↑ 自动把对话消息拼成 Qwen 认识的 ChatML 格式：
                #   <|im_start|>user\n...<|im_end|><|im_start|>assistant\n
                [{"role": "user", "content": user_content}],
                # ↑ 一个列表装一个字典，表示"一条用户消息"（Day9 我们手拼过，这里交给库）。
                tokenize=False,
                # ↑ 只拼文本、先不转数字（tokenize=False = 不要立即分词）。
                add_generation_prompt=True,
                # ↑ 拼上"assistant 开始标记"（<|im_start|>assistant\n），
                #   让模型知道"轮到你了，从这里开始写答案"。
            )

            prompts.append(prompt)                            # 收集题干
            # ↑ .append(x) = 往列表末尾添加元素。

            full_texts.append(prompt + out + tokenizer.eos_token)
            # ↑ 字符串拼接：题干 + 标准答案 + 结束符。
            #   tokenizer.eos_token = 结束符文本（句子写完了，模型停下来的标记）。

        # 分别编码：题干（不含答案）和完整文本（题干 + 答案）
        enc_p = tokenizer(prompts, truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False)
        # ↑ 直接调用 tokenizer(文本列表, ...) = 把文字转成数字编号（token ID）。
        #   truncation=True + max_length=512：超过 512 个 token 就截断。
        #   add_special_tokens=False：不额外加特殊符号（ChatML 已经拼好了）。
        #   enc_p 里只有"题干"的编号。

        enc_f = tokenizer(full_texts, truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False)
        # ↑ 同上，但编码的是"题干 + 答案"完整文本。enc_f = 完整编号。

        labels = []
        # ↑ 建一个空列表，用来装"标签"（告诉模型：哪些位置要打分）。

        for ids, pids in zip(enc_f["input_ids"], enc_p["input_ids"]):
            # ↑ 循环：把"完整编号"和"题干编号"按条配对。
            #   ids  = 第 n 条完整文本的编号；pids = 第 n 条题干编号。
            lab = ids.copy()
            # ↑ x.copy() = 复制列表（副本）。改 lab 不会动到原始数据。

            mask = min(len(pids), len(ids))
            # ↑ min(a, b) = 取较小的。题干编号长度（正常情况下就是前 mask 个）。

            lab[:mask] = [-100] * mask
            # ↑ 标签掩码（本脚本最核心的几行）：
            #   lab[:mask] = 切片"取前 mask 个元素"（就是题干那些位置）；
            #   [-100] * mask = 造一个"全是 -100、长度 mask"的列表；
            #   合起来 = 把题干所有位置替换成 -100。
            #   -100 是 PyTorch 交叉熵的"忽略标记"：这些位置不算 loss——
            #   模型只对"答案部分"负责（Day11 讲的"只算 output"）。

            labels.append(lab)          # 收集这条数据的标签

        return {
            # ↑ return = 返回结果（一个字典 = 键值对集合）。
            #   模型训练需要三样东西：
            "input_ids": enc_f["input_ids"],
            # ↑ 完整文本的数字编号（题干 + 答案）。
            "attention_mask": enc_f["attention_mask"],
            # ↑ 注意力掩码：标记哪些位置是有效文字（1），哪些是 padding（0）。
            "labels": labels,
            # ↑ 标签：答案位置的编号 + 题干位置的 -100。
        }

    train_ds = split["train"].map(tokenize_batch, batched=True, remove_columns=raw.column_names)
    # ↑ .map(函数) = 把函数套到每一条数据上（执行 tokenize_batch）。
    #   batched=True = 一次处理一批（更快）。
    #   remove_columns=... = 转完就把原 instruction/input/output 三列删掉，只留新字段。
    eval_ds = split["test"].map(tokenize_batch, batched=True, remove_columns=raw.column_names)
    # ↑ 验证集做同样处理（验证集只看 loss，不参与训练）。

    print("=" * 60)
    print("第 4/4 步：开始训练（Trainer 托管）")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 第 4/4 步：训练 + 保存 adapter
    # -----------------------------------------------------------------------

    args = TrainingArguments(
        # ↑ 训练参数配置单（只描述，不行动）。和配置区的常量一一对应。
        output_dir=LOG_DIR,              # 日志存到哪
        num_train_epochs=EPOCHS,         # 300 条数据刷几遍（3）
        per_device_train_batch_size=BATCH_SIZE,  # 每步喂几条（2）
        gradient_accumulation_steps=GRAD_ACCUM,  # 攒几步再更新（8，等效 batch = 2×8 = 16）
        learning_rate=LEARNING_RATE,     # 步幅（2e-4）
        warmup_steps=WARMUP_STEPS,       # 起步热身：前 2 步线性升温
        lr_scheduler_type=SCHEDULER,     # cosine 余弦退火
        fp16=True,                       # 混合精度：计算用半精度，省显存加速
        logging_steps=5,                 # 每 5 步打印一次 loss
        eval_strategy="epoch",           # 每刷完一遍用验证集考一次（算 eval_loss）
        save_strategy="no",              # 不存中间 checkpoint（省空间）
        seed=SEED,                       # 随机种子
        report_to=[],                    # 不上报 wandb/tensorboard
    )

    trainer = Trainer(
        # ↑ 训练管家：把"模型 + 参数 + 数据"全交给它，
        #   前向传播、反向传播、梯度更新、loss 打印、epoch 末验证全都自动做——
        #   我们不用手写训练循环（这是新手最头疼的部分）。
        model=model,                     # 带 LoRA 补丁的模型
        args=args,                       # 上面的训练参数
        train_dataset=train_ds,          # 训练数据（270 条）
        eval_dataset=eval_ds,            # 验证数据（30 条）
        processing_class=tokenizer,      # 分词器（transformers 5.x 新签名，老版本叫 tokenizer）
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        # ↑ 数据收集器：把 batch 内长度不齐的序列自动补齐（padding）成矩阵。
    )

    mem_report("训练开始前")             # 训练前看一眼显存
    trainer.train()
    # ↑ 开始训练！这一行会一直跑到结束，中途打印 loss 日志。
    mem_report("训练结束后")             # 训练后再看一眼显存（确认没爆）

    metrics = trainer.evaluate()
    # ↑ 训练完在验证集上算分，返回一个字典（含 eval_loss）。

    print(f"[验证] 最终 eval_loss = {metrics.get('eval_loss'):.4f}")
    # ↑ metrics.get("eval_loss") = 从字典里取出 eval_loss 那个数。
    #   :.4f = 保留 4 位小数。

    # 只存 adapter（不存全量模型！几 MB 就是全部成果）
    print("=" * 60)
    print("保存 LoRA 适配器（只存增量，不存全量模型）")
    print("=" * 60)

    os.makedirs(OUTPUT_ADAPTER, exist_ok=True)
    # ↑ os.makedirs(路径, exist_ok=True) = 创建文件夹。
    #   exist_ok=True = 文件夹已存在也不报错。

    model.save_pretrained(OUTPUT_ADAPTER)
    # ↑ 保存模型。因为 model 是 PEFT 包过的，save 的就只有 LoRA 补丁 B·A + 配置，
    #   不是几 GB 的全量模型——这就是"参数高效微调"的核心。

    tokenizer.save_pretrained(OUTPUT_ADAPTER)
    # ↑ 顺便把分词器也存一份（Day13 加载 adapter 时要用同一个分词器）。

    for fname in sorted(os.listdir(OUTPUT_ADAPTER)):
        # ↑ for 循环遍历文件夹里的每个文件名。
        #   os.listdir(目录) = 列出目录里的所有文件名；
        #   sorted(...) = 按名字排序（打印整齐）。
        size_mb = os.path.getsize(os.path.join(OUTPUT_ADAPTER, fname)) / 1024 / 1024
        # ↑ os.path.join(路径, 文件名) = 拼出文件的完整路径；
        #   os.path.getsize(完整路径) = 拿文件字节数；÷1024÷1024 = 转成 MB。
        print(f"  {fname:<40s} {size_mb:.2f} MB")
        # ↑ f"{fname:<40s}" = 左对齐、占 40 个字符宽度（对齐好看）。

    print("=" * 60)
    print(f"[完成] adapter 已保存到 {OUTPUT_ADAPTER}/，接下来运行 compare_lora.py 做效果对比。")


# ===========================================================================
# 入口：直接运行本文件时，才执行 main()
# ===========================================================================
if __name__ == "__main__":
    # ↑ 这是所有 Python 脚本的标准入口写法。
    #   __name__ 是 Python 内置变量：
    #   - 当执行 `python train_lora.py` 时，__name__ = "__main__" → 执行 main()；
    #   - 当被别的脚本 `import train_lora` 时，__name__ = 模块名 → 不自动跑。
    #   这样既能直接运行，又能被别的脚本安全引用（不会一 import 就开始训练）。
    main()
