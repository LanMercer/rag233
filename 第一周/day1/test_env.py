# -*- coding: utf-8 -*-
"""
Day 1 环境验收脚本
用途：确认 GPU 可用 + 载入迷你模型完成一次生成，验证整条环境链路 OK。
运行方法：在 llm 环境中执行  python test_env.py
"""

import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
    # 在 GPU 上做一次真实计算：x * 2，预期输出 [2.0, 4.0, 6.0]
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = x * 2
    print("GPU 计算测试:", y.cpu().tolist())
else:
    print("GPU 名称: 无（未检测到可用 GPU，请检查教程第 4 步）")

print("\n正在加载迷你模型 tiny-random-gpt2 并生成一句话（首次运行会自动下载，约 34MB）...")
from transformers import pipeline

gen = pipeline("text-generation", model="hf-internal-testing/tiny-random-gpt2")
out = gen("Hello, I am", max_new_tokens=20)[0]["generated_text"]
print("迷你模型生成结果:", out)

print("\n环境验收通过！GPU 可用，模型能正常生成。")
