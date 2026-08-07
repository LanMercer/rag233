# -*- coding: utf-8 -*-
"""
Day 4 配套脚本 3：摸清 torch.nn.Transformer 的输入 / 输出形状
============================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\\Lan\\研究生\\技术学习\\大模型算法\\第一周\\day4"
    python transformer_shape_demo.py

本脚本做 2 件事：
    1. 构造一个超小的官方 nn.Transformer（6 维、2 层）
    2. 喂入假数据，把每一步的张量形状打印出来，让你"亲眼看到"数据怎么流动

配套：想直接看官方源码，可在终端运行
    python -c "import inspect, torch.nn as nn; print(inspect.getsource(nn.Transformer))"

关键结论（背下来）：
    规律 1：PyTorch 默认 batch_first=False → 张量形状是 (序列长度, 批量, 维度)
    规律 2：Encoder/Decoder 只改内容，不改形状：(长, 批, 维) 进 → (长, 批, 维) 出
    规律 3：最后 Linear 把 维度 → 词表大小，配合 softmax 就是"下一个词的概率分布"
"""

import torch
import torch.nn as nn

print("===== 1. 构造一个 6 维的小 Transformer（2 层）=====")
d_model = 6          # 每个词的向量维度（真实模型是 512/768 等，这里用 6 方便手算）
nhead = 2            # 多头数量（必须能整除 d_model：6 ÷ 2 = 3）
num_layers = 2       # 编码器 / 解码器各堆几层
vocab_size = 12      # 词表大小（假设一共 12 种"词"）

# 词嵌入：把"词的编号"变成"向量"
embed = nn.Embedding(vocab_size, d_model)
# 官方 Transformer：Encoder(2 层) + Decoder(2 层)
transformer = nn.Transformer(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    dim_feedforward=32,     # 前馈网络的隐藏维度（真实模型 2048）
    dropout=0.0,            # 关闭随机失活，保证结果稳定
)
# 输出层：把 d_model 维 → 词表大小的分数
fc = nn.Linear(d_model, vocab_size)

# 看一下模型整体长什么样（你会看到 Encoder 和 Decoder 各自 N 层）
print(transformer)

print("\n===== 2. 喂入假数据并逐层打印形状 =====")
# 假数据：
#   src = 源句子里的词的编号，形状 (src_len=5, batch=3)：5 个词的 3 个句子
#   tgt = 目标句子里的词的编号，形状 (tgt_len=4, batch=3)：4 个词的 3 个句子
src = torch.randint(0, vocab_size, (5, 3))   # (序列长度=5, 批量=3)
tgt = torch.randint(0, vocab_size, (4, 3))   # (序列长度=4, 批量=3)
print(f"源句子 src 的形状 : {tuple(src.shape)}   ← (序列长度=5, 批量=3)")
print(f"目标句子 tgt 形状 : {tuple(tgt.shape)}   ← (序列长度=4, 批量=3)")

# 第 1 步：词嵌入 —— 每个词的编号 → 6 维向量
src_e = embed(src)   # (5, 3, 6)
tgt_e = embed(tgt)   # (4, 3, 6)
print(f"词嵌入之后      : {tuple(src_e.shape)}   ← (5, 3, 6) = 序列长度×批量×维度")

# 第 2 步：进官方 Transformer —— 编码器读 src，解码器参考编码器结果并处理 tgt
#   注意：真实代码里还要加位置编码（今天第 1 步的"座位号"），这里为了看形状先不加
memory = transformer.encoder(src_e)            # 编码器输出：每个源词"理解上下文后"的表示
print(f"编码器输出      : {tuple(memory.shape)}   ← (5, 3, 6)，长度和维度都不变")
dec_out = transformer.decoder(tgt_e, memory)   # 解码器输出：每个目标位置的计算结果
print(f"解码器输出      : {tuple(dec_out.shape)}   ← (4, 3, 6)，跟着目标串长度走")

# 第 3 步：输出层 —— 6 维 → 12 个词的分数
logits = fc(dec_out)
print(f"输出层(Linear)  : {tuple(logits.shape)}   ← (4, 3, 12) = 每个位置 12 个词的分数")

# 第 4 步：softmax → 概率分布（Day 2 的"输出 = 概率分布"在这里落下来）
probs = torch.softmax(logits, dim=-1)
print(f"softmax 之后    : {tuple(probs.shape)}   ← 每行和为 1，就是概率分布")

print("\n===== 3. 三条规律总结 =====")
print("规律 1：Transformer 里'序列长度'在第 0 维、'批量'在第 1 维（batch_first=False）")
print("规律 2：Encoder/Decoder 只做内容变换，形状 (长, 批, 维) 进 → (长, 批, 维) 出")
print("规律 3：最后一步 Linear 把 维度 压到 词表大小 → softmax → '下一个词的概率分布'")
