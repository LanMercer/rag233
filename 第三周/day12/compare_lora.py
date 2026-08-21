# -*- coding: utf-8 -*-
"""
compare_lora.py —— 微调前后效果对比脚本（第三周 Day12 第 4 步）
========================================================================
【五件套】（脚本在机器学习里的哪一环 = 测试/推理环节）
- 模型   ：原模型（Qwen2.5-3B-Instruct 4bit）vs 原模型 + lora_adapter
          （6G 显存放不下两个模型，所以**依次加载**：先原模型跑完，释放，再挂 adapter）
- 数据   ：同一组 5 个测试问题（机器人行业：抽取 / 计算 / 概念 / 总结 / 合规）
- 损失   ：无（纯推理对比，不训练）
- 优化器 ：无
- 测试   ：贪心解码（do_sample=False，温度恒等）保证两次输出可复现、只比微调差异
【产出】终端并排打印两段输出，并自动生成《微调前后效果对比表.md》（差异分析列手动补）。
========================================================================
"""
import datetime
import os
import sys

# Windows 控制台默认 GBK 打不出部分字符，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"
ADAPTER_PATH = "lora_adapter"          # train_lora.py 保存的 adapter
OUTPUT_MD = "微调前后效果对比表.md"
MAX_NEW_TOKENS = 300                   # 最多生成 300 个新 token

# 5 个测试问题：覆盖 Day11 SFT 数据里的 5 类任务
QUESTIONS = [
    {
        "类别": "公告信息抽取",
        "问题": "请从以下埃斯顿的年度报告中，提取营业收入的数据。",
        "材料": "埃斯顿发布年度报告。报告期内公司营业收入为 69.5 亿元，同比增长3.2%。截至2023年末，公司订单金额达到 556.0 亿元。",
    },
    {
        "类别": "数据计算",
        "问题": "汇川技术去年净利润 28.26 亿元，今年净利润 25.77 亿元，请计算同比增长率。",
        "材料": "",
    },
    {
        "类别": "概念解释",
        "问题": "请用通俗的语言解释什么是谐波减速器？",
        "材料": "",
    },
    {
        "类别": "公告要点总结",
        "问题": "请总结以下绿的谐波公告的核心要点。",
        "材料": "绿的谐波今日发布业绩预告。公司全年研发投入实现 105.37 亿元，同比增长15.6%；报告期内公司新增 7 项发明专利，完成 6 项重大业务布局。公司表示将持续加大研发投入，推进国产替代。",
    },
    {
        "类别": "合规判断",
        "问题": "优必选拟向特定地区出口工业机器人，请问是否合规？",
        "材料": "",
    },
]


def build_quant_config():
    """与训练脚本完全同款的 4bit 量化配置（基座必须与训练时保持一致）"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(with_adapter: bool):
    """加载原模型（with_adapter=False）或 原模型 + adapter（with_adapter=True）"""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=build_quant_config(),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if with_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, item):
    """对单个问题生成回答（贪心解码，保证可复现）"""
    content = item["材料"] if item["材料"] else item["问题"]
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def write_md(base_answers, ft_answers):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 微调前后效果对比表（Day12）",
        "",
        f"> 模型：Qwen2.5-3B-Instruct（4bit 底座） vs 原模型 + `lora_adapter`（r=8, alpha=16）",
        f"> 解码策略：贪心（do_sample=False，保证可复现）｜ 脚本：`compare_lora.py` ｜ 生成时间：{now}",
        "",
        "| # | 类别 | 测试问题 | 原模型回答 | 微调后回答 | 差异分析 |",
        "|---|------|---------|-----------|-----------|---------|",
    ]
    for i, (item, base, ft) in enumerate(zip(QUESTIONS, base_answers, ft_answers), 1):
        q = item["材料"] or item["问题"]
        q_cell = q.replace("|", "/")
        base_cell = base.replace("\n", " ").replace("|", "/")
        ft_cell = ft.replace("\n", " ").replace("|", "/")
        lines.append(f"| {i} | {item['类别']} | {q_cell} | {base_cell} | {ft_cell} | （手动填写） |")
    lines += [
        "",
        "> 填写说明：差异分析列手动补。对比看 4 个维度——",
        "> ① 格式是否学会 ①②③ 分点；② 是否用上行话（谐波减速器/减速比/PID/产能利用率）；",
        "> ③ 计算题是否学会\"先列式再代入\"；④ 口吻是否更\"研报风\"（评级/风险提示）。",
    ]
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[完成] 对比表已生成：{OUTPUT_MD}（打开补\"差异分析\"列）")


def main():
    if not os.path.isdir(ADAPTER_PATH):
        print(f"❌ 找不到 {ADAPTER_PATH}/，请先运行 train_lora.py 训练出 adapter 再对比。")
        sys.exit(1)

    # 第一轮：原模型
    print("=" * 60)
    print("加载原模型（4bit，不含 adapter）……")
    print("=" * 60)
    model, tokenizer = load_model(with_adapter=False)
    base_answers = []
    for i, item in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] {item['类别']}｜原模型：")
        ans = ask(model, tokenizer, item)
        base_answers.append(ans)
        print("  " + ans.replace("\n", "\n  "))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()          # 释放显存，给第二轮腾地方

    # 第二轮：原模型 + adapter
    print("\n" + "=" * 60)
    print("加载原模型 + lora_adapter（微调后）……")
    print("=" * 60)
    model, tokenizer = load_model(with_adapter=True)
    ft_answers = []
    for i, item in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] {item['类别']}｜微调后：")
        ans = ask(model, tokenizer, item)
        ft_answers.append(ans)
        print("  " + ans.replace("\n", "\n  "))

    write_md(base_answers, ft_answers)
    print("\n对照上面两轮输出，肉眼找差异：格式 / 行话 / 计算步骤 / 口吻。")


if __name__ == "__main__":
    main()
