# -*- coding: utf-8 -*-
"""
Qwen2.5-3B-Instruct 4bit 量化加载 + 推理 + 显存实测 + 采样参数对比脚本（配合 Day6 教程第 3、4 步使用）

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：Qwen2.5-3B-Instruct，用 4bit 量化（NF4 格式）加载到 6G 显存显卡
- 数据   ：手工输入的几个测试问题（纯推理，无训练数据）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    ① 用 BitsAndBytesConfig 以 4bit 方式加载模型，验证 6G 显存装得下；
    ② 打印模型实际显存占用（torch.cuda.memory_allocated），与第一周预算对比；
    ③ 让模型生成一句自我介绍（验证能"开口说话"）；
    ④ 对比 temperature=0 / 0.7 / 1.2 的输出差异（温度控制输出多样性）；
    ⑤ 对比 top_p=0.5 / 0.9 / 1.0 的输出差异（核采样截断）。
"""

import os

# 如果走 hf-mirror 下载方案，加载前也设一下镜像，避免自动联网找模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 模型所在本地路径（download_qwen.py 下载的位置；如果实际路径不同请修改这里）
MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"

# 采样参数（第一周 day5 教程"三件套"里 temperature / top_p 的落地实验）
TEMPERATURES = [0.0, 0.7, 1.2]   # 温度：0=最保守，1.2=更多样
TOP_P_LIST = [0.5, 0.9, 1.0]      # 核采样阈值：越小越保守
MAX_NEW_TOKENS = 100              # 每次最多生成 100 个新词


def build_chat_prompt(user_text: str) -> str:
    """把用户问题包装成 Qwen 使用的对话格式（ChatML）。"""
    return (
        "<|im_start|>system\n你是一个乐于助人、严谨可靠的中文助手。"
        "回答请用中文，尽量简洁。<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate(text: str, temperature: float = 0.7, top_p: float = 0.9) -> str:
    """让模型生成一段文本。temperature=0 时用贪心解码（每次都选最可能的词）。"""
    prompt = build_chat_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # 只取模型新生成的部分（去掉输入提示），并去掉对话标记
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def print_memory(label: str):
    """打印当前显存占用（GB），并给出剩余显存提示。"""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[显存] {label}：已分配 {allocated:.2f} GB，缓存 {reserved:.2f} GB")
    return allocated


if __name__ == "__main__":
    print("=" * 60)
    print("第 1 步：4bit 量化加载 Qwen2.5-3B-Instruct")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[X] 找不到模型文件夹：{MODEL_PATH}")
        print("[X] 请先运行 python download_qwen.py 下载模型，或修改脚本里的 MODEL_PATH。")
        raise SystemExit(1)

    print("[OK] GPU 可用：", torch.cuda.is_available())
    print("[OK] 显卡：", torch.cuda.get_device_name(0))

    # 4bit 量化配置（NF4 格式 + 双重量化，进一步省显存）
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,             # 用 4bit 加载，显存降到约 1/4
        bnb_4bit_quant_type="nf4",     # NF4 = NormalFloat4，专为权重设计的 4bit 格式
        bnb_4bit_compute_dtype=torch.float16,  # 计算时用 fp16，保持精度
        bnb_4bit_use_double_quant=True,        # 双重量化：再省一层显存
    )

    print("[OK] 正在加载分词器（Tokenizer）……")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("[OK] 正在加载模型（4bit 量化），首次约需 1~3 分钟……")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",            # 自动分配 GPU/CPU
        torch_dtype=torch.float16,
    )
    model.eval()

    # 计算模型参数总量（兆=百万，说明 4bit 只占 fp16 的约一半）
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[OK] 模型参数量：约 {num_params:.0f} 百万（3B ≈ 30 亿参数）")

    print("=" * 60)
    print("第 2 步：显存实测（与第一周预算 ≈3.0GB 对比）")
    print("=" * 60)
    mem = print_memory("加载模型后")
    print(f"[结论] 显存实测 {mem:.2f} GB；第一周预算 3.0GB，请对照 week1 工作汇报核对误差。")

    print("=" * 60)
    print("第 3 步：生成测试（验证能开口说话）")
    print("=" * 60)
    answer = generate("用一句话介绍一下你自己。", temperature=0.7, top_p=0.9)
    print("[模型回答]", answer)

    print("=" * 60)
    print("第 4 步：temperature 对比实验（同一个问题，3 个温度）")
    print("=" * 60)
    question = "请写一句关于秋天的句子。"
    print(f"[提问] {question}")
    for t in TEMPERATURES:
        label = "贪心解码（最保守）" if t == 0 else f"温度={t}"
        print(f"\n----- temperature={t}（{label}）-----")
        print("[模型回答]", generate(question, temperature=t, top_p=0.9))

    print("=" * 60)
    print("第 5 步：top_p 对比实验（固定 temperature=0.8）")
    print("=" * 60)
    question2 = "我对象和我打电话面无表情还脸红红的是为什么。"
    print(f"[提问] {question2}")
    for p in TOP_P_LIST:
        print(f"\n----- top_p={p} -----")
        print("[模型回答]", generate(question2, temperature=0.8, top_p=p))

    print("=" * 60)
    print("实验结束。请把本脚本输出保存/截图，作为 Day6 验收材料。")
    print("=" * 60)
