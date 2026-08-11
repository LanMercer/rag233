# -*- coding: utf-8 -*-
r"""
Day 5 配套脚本：Qwen 本地部署硬件方案测算（6G 显存够不够？）
========================================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第一周\day5"
    python qwen_memory_calculator.py

本脚本不做训练、不做预测，它只是"算账"（乘法加法）——帮你决定第 6 天
用哪个 Qwen 模型、用多少位量化。对应"机器学习五件套"它一个都没有，
因为它根本不是一个学习/预测脚本，而是一张"预算表"。

关键名词先认识：
    参数（Parameter）= 模型里可学习的数字，越多模型越"聪明"，也越占显存
    量化（Quantization）= 把每个参数从 32/16 位压缩到 8/4 位，占显存变小，
                          精度略降但能用（第 6 天用 bitsandbytes 实现 4bit）
    显存（VRAM）= 显卡自带的内存，你的显卡有 6GB
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------- 数据 ----------------
# 模型参数数量（官方数字，含全部权重）
models = {
    "Qwen2.5-0.5B-Instruct": 0.49e9,
    "Qwen2.5-1.5B-Instruct": 1.54e9,
    "Qwen2.5-3B-Instruct":   3.09e9,
    "Qwen2.5-7B-Instruct":   7.61e9,
}
# 每种精度下，一个参数占多少字节（Byte）
precisions = {
    "fp32 (默认)": 4.0,
    "fp16/bf16":   2.0,
    "int8 (8bit)": 1.0,
    "4bit (NF4)":  0.5,
}
# 推理运行开销（经验估算）：CUDA 上下文 + KV缓存 + 激活值，单位 GB
# 这是粗略估算，第 6 天用 torch.cuda.memory_allocated() 实测为准
overhead = {
    "Qwen2.5-0.5B-Instruct": 1.0,
    "Qwen2.5-1.5B-Instruct": 1.2,
    "Qwen2.5-3B-Instruct":   1.4,
    "Qwen2.5-7B-Instruct":   1.8,
}


def verdict(total, budget):
    if total <= budget * 0.75:
        return "✅ 舒适"
    if total <= budget:
        return "⚠️ 勉强"
    return "❌ 不够"


# ================= 输出 =================
print("=" * 78)
print("Qwen 本地部署显存测算表（你：RTX 3060 Laptop 6GB 显存）")
print("=" * 78)
print(f"{'模型':<22}{'精度':<12}{'权重GB':>9}{'开销GB':>8}{'合计GB':>9}  6G显卡   8G显卡")
print("-" * 78)

for name, params in models.items():
    ov = overhead[name]
    for pname, pbytes in precisions.items():
        wgb = params * pbytes / 1e9
        total = wgb + ov
        print(f"{name:<22}{pname:<12}{wgb:>9.2f}{ov:>8.1f}{total:>9.2f}  "
              f"{verdict(total, 6):<8}{verdict(total, 8)}")

print()
print("=" * 78)
print("结论（针对你的 6G 显存）")
print("=" * 78)
print("① 首选：Qwen2.5-3B-Instruct + 4bit 量化（合计约 3.0GB）→ 6G 显卡舒适运行")
print("② 次选：Qwen2.5-1.5B-Instruct + fp16（合计约 4.3GB）→ 也稳")
print("③ 想试 7B：Qwen2.5-7B + 4bit（合计约 5.6GB）→ 6G 显卡“勉强”，")
print("   大概率要开 CPU 卸载（device_map='auto'，牺牲速度换显存），不推荐第一步就上")
print("④ 绝不推荐：7B 的 fp16（15GB）——6G 显卡连门都进不去")
print()
print("周计划写的“8G+ 显存跑 7B 4bit”是成立的（约 5.6GB ≤ 8GB）；")
print("我们这台 6G 机器，第 6 天按方案①部署 Qwen2.5-3B-Instruct 4bit 即可。")
print()
print("第 6 天实操验证方法（跑通后打印真实显存占用）：")
print("    python -c \"import torch; from transformers import AutoModelForCausalLM; ...\"")
print("    # 加载模型后执行：print(torch.cuda.memory_allocated()/1024**3, 'GB')")
print("=" * 78)
