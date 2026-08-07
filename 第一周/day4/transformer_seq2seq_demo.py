# -*- coding: utf-8 -*-
"""
Day 4 配套脚本 4：官方风格 Transformer seq2seq 小 demo —— 数字倒序翻译
=====================================================================
运行方法（在 llm 环境中）：
    conda activate llm
    cd "D:\\Lan\\研究生\\技术学习\\大模型算法\\第一周\\day4"
    python transformer_seq2seq_demo.py

任务：把一串数字倒过来（输入 1234 → 输出 4321）。
选这个任务的用心：
    * 它完全靠"位置信息"才能做对 —— 不记住每个数字在第几位，模型学不会
    * 它就是"翻译"的最简版：源串进编码器，目标串进解码器，逐词生成
    * 和 PyTorch 官方教程的 seq2seq demo 是同一个结构（Encoder+Decoder+生成循环）

【这段代码是机器学习/深度学习吗？—— 是的，而且是标准的深度学习流程】
    机器学习 = 让计算机从"数据"里自动学规律，而不是程序员手写规则。
    深度学习 = 机器学习的一个分支，用"神经网络"（很多层可学习的函数叠起来）来学。
    本脚本五大件一个不缺，全是机器学习/深度学习的标准配置：
        1. 模型（Seq2Seq）   ：神经网络，含词嵌入 Embedding + Transformer + 输出层 Linear
        2. 数据              ：每步随机生成一批"数字串 → 倒序"的样本对
        3. 损失函数          ：CrossEntropyLoss，衡量"模型预测"和"正确答案"差多远
        4. 优化器            ：Adam，负责按"梯度"把参数调得更准
        5. 训练 + 测试        ：先拿海量样本把模型参数调好，再用"没见过的例子"检验
    你现在跑的大模型（Qwen、GPT 等）就是把这个流程放大几百万倍。

【它如何训练？】每跑一批样本做 4 件事（对应 train_one_batch 函数）：
    1. 造一批数据（make_batch，一批 128 个样本）
    2. 前向传播：把样本喂给模型，得到"预测的分布"；训练时用"教师强制"——
       解码器喂的是真实目标串（只让模型练"预测下一个数字"这一件事）
    3. 算损失 loss：CrossEntropyLoss 对比预测分布和正确答案
    4. 反向传播 + 更新：loss.backward() 算出每个参数该往哪调（梯度），
       optimizer.step() 按梯度把参数调准一点点
    重复"1~4"就叫训练；80 个 epoch × 每 epoch 40 批 = 3200 次调整，
    loss 从 1.856 一路降到 0.000，就是参数越调越准的证据。

【它如何测试？】训练完，模型参数不再更新，用"没见过的新样本"检验（generate 函数）：
    1. model.eval() + @torch.no_grad()：关掉 dropout、关掉梯度，让输出稳定
    2. 只把输入喂给模型，用"贪心解码"逐字生成：
       每步从"下一个字符的概率分布"里挑概率最大的字符，拼上去，再喂回下一步
    3. 和正确答案比对，数正确率（10 个新例子里对了几个）
    "测试用新样本"很关键：如果只测背过的题，分数高也没意义——
    测试测的是模型"学会了规律"而不是"死记硬背"。

流程（对应教程第 5 步）：
    1. 打印模型结构（对照第 4 步学的形状）
    2. 训练：教师强制（喂真实目标）+ 交叉熵损失
    3. 测试：贪心解码（每步取概率最大的字符，喂回下一步）
    4. 灵魂实验：把位置编码关掉再训一遍，你会看到 loss 几乎不降

超参数在下面常量区，觉得慢就调小（EPOCHS / ITERS_PER_EPOCH / BATCH_SIZE）。
"""

import math
import random
import time

import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# 超参数（觉得训练慢，就把 EPOCHS / ITERS_PER_EPOCH / BATCH_SIZE 调小）
# ---------------------------------------------------------------------
EPOCHS = 80
ITERS_PER_EPOCH = 40
BATCH_SIZE = 128      # 批量大一点，小模型的 Python 开销占比高，大批量跑得更快
D_MODEL = 64        # 每个词的向量维度
NHEAD = 2           # 多头数（必须整除 D_MODEL）
NUM_LAYERS = 2      # 编码器 / 解码器层数
MAX_LEN = 8         # 源串最长 8 位数字

# ---------------------------------------------------------------------
# 词表：PAD(补齐) + BOS(起始) + EOS(结束) + 10 个数字 = 13 个符号
# ---------------------------------------------------------------------
SYMBOLS = ["PAD", "BOS", "EOS"] + [str(i) for i in range(10)]
stoi = {s: i for i, s in enumerate(SYMBOLS)}   # 字符 → 编号
itos = {i: s for s, i in stoi.items()}         # 编号 → 字符
PAD, BOS, EOS = stoi["PAD"], stoi["BOS"], stoi["EOS"]
TGT_MAX = MAX_LEN + 2                          # 目标串最多 BOS + 8 位 + EOS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("设备:", device)


# ---------------------------------------------------------------------
# 位置编码模块（今天第 1 步用 numpy 画过，这里用 torch 实现同一件事）
# ---------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维用 sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维用 cos
        self.register_buffer("pe", pe)   # 存进模型，但不需要梯度

    def forward(self, x):
        # x 形状 (seq_len, batch, d_model)，把对应位置的编码加进去
        # pe[:seq_len] 是 (seq_len, d_model)，unsqueeze(1) 补成 (seq_len, 1, d_model) 才能和批量维广播
        return x + self.pe[:x.size(0)].unsqueeze(1)


# ---------------------------------------------------------------------
# 模型：词嵌入 + 位置编码 + 官方 nn.Transformer + 输出层
# ---------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(SYMBOLS), D_MODEL, padding_idx=PAD)
        self.pos = PositionalEncoding(D_MODEL)
        self.transformer = nn.Transformer(
            d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS, num_decoder_layers=NUM_LAYERS,
            dim_feedforward=D_MODEL * 2, dropout=0.1, batch_first=False,
        )
        self.fc = nn.Linear(D_MODEL, len(SYMBOLS))   # d_model 维 → 13 个字符的分数

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask, tgt_mask):
        # 位置编码加在嵌入之上（Day 4 第 1 步的"座位号"）
        src_e = self.pos(self.embed(src))            # (src_len, B, D_MODEL)
        tgt_e = self.pos(self.embed(tgt))            # (tgt_len, B, D_MODEL)
        out = self.transformer(
            src_e, tgt_e,
            src_key_padding_mask=src_pad_mask,       # (B, src_len)，PAD 处为 True
            tgt_key_padding_mask=tgt_pad_mask,       # (B, tgt_len)
            tgt_mask=tgt_mask,                       # (tgt_len, tgt_len) 因果掩码
        )
        return self.fc(out)                          # (tgt_len, B, 13)


# ---------------------------------------------------------------------
# 数据：随机生成数字串，目标 = 倒序
# ---------------------------------------------------------------------
def make_sample():
    n = random.randint(2, MAX_LEN)
    src = [random.randint(0, 9) for _ in range(n)]
    tgt = src[::-1]                     # 倒过来
    return src, tgt


def encode(digits, add_special):
    """把数字列表变成"编号 + 补齐到定长"的一维数组。
    注意：词表里 0=PAD, 1=BOS, 2=EOS，数字 0~9 的编号是 3~12，
    所以每个数字要经 stoi 转换成词表编号（不能直接用数字本身当编号）。"""
    tokens = [stoi[str(d)] for d in digits]   # 数字 → 词表编号（0→3, 1→4, ..., 9→12）
    if add_special:                     # 目标串：BOS + 数字 + EOS
        ids = [BOS] + tokens + [EOS]
        max_len = TGT_MAX
    else:                               # 源串：不加特殊符，直接补齐
        ids = tokens
        max_len = MAX_LEN
    ids = ids[:max_len]
    ids = ids + [PAD] * (max_len - len(ids))   # 后面补 PAD 到定长
    return ids


def make_batch(batch_size):
    """造一个批次：返回 src、tgt，形状都是 (序列长度, 批量)。"""
    src_list, tgt_list = [], []
    for _ in range(batch_size):
        s, t = make_sample()
        src_list.append(encode(s, add_special=False))
        tgt_list.append(encode(t, add_special=True))
    src = torch.tensor(src_list, dtype=torch.long).T   # (MAX_LEN, B)
    tgt = torch.tensor(tgt_list, dtype=torch.long).T   # (TGT_MAX, B)
    return src, tgt


def pad_mask(ids):
    """把 (seq, batch) 的编号张量变成 (batch, seq) 的布尔掩码，True = 该位置是 PAD。"""
    return (ids == PAD).T


def causal_mask(sz):
    """上三角因果掩码：让解码器每一步只能看到"当前位置及以前"（不能偷看未来）。
    返回布尔矩阵（True = 该位置被遮住），和 key_padding_mask 类型一致，避免告警。"""
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)


# ---------------------------------------------------------------------
# 贪心解码：每步从"下一个字符的概率分布"里挑最大的，喂回下一步
# ---------------------------------------------------------------------
@torch.no_grad()
def generate(model, src_digits):
    model.eval()                        # 关掉 dropout（训练随机失活），保证结果稳定
    src = torch.tensor([encode(src_digits, add_special=False)],
                       dtype=torch.long).T.to(device)     # (MAX_LEN, 1)
    src_e = model.pos(model.embed(src))
    memory = model.transformer.encoder(src_e,
                                       src_key_padding_mask=pad_mask(src))   # 编码器读一遍
    dec_ids = [BOS]                     # 从"起始符"开始
    for _ in range(TGT_MAX):
        tgt = torch.tensor([dec_ids], dtype=torch.long).T.to(device)
        tgt_e = model.pos(model.embed(tgt))
        dec_out = model.transformer.decoder(tgt_e, memory,
                                            tgt_mask=causal_mask(tgt.size(0)).to(device))
        logits = model.fc(dec_out[-1])              # 只看最后一步 → (1, 13)
        next_id = logits.argmax(-1).item()          # 概率最大的那个字符
        if next_id == EOS:                          # 遇到结束符就停
            break
        dec_ids.append(next_id)
        if len(dec_ids) >= TGT_MAX - 1:             # 防止死循环
            break
    return "".join(itos[i] for i in dec_ids[1:])    # 去掉 BOS，拼成字符串


# ---------------------------------------------------------------------
# 主流程：打印结构 → 训练 → 测试
# ---------------------------------------------------------------------
print("\n===== 1. 任务 =====")
print("把一串数字倒过来（1234 → 4321）。这个任务必须靠位置信息才能做对。")

print("\n===== 2. 模型结构 =====")
model = Seq2Seq().to(device)
print(model)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)   # 只惩罚"非 PAD"位置

def train_one_batch():
    src, tgt = make_batch(BATCH_SIZE)
    src, tgt = src.to(device), tgt.to(device)
    tgt_in = tgt[:-1]              # 解码器输入 = 去掉最后一个位置（教师强制）
    tgt_out = tgt[1:]              # 目标 = 往右移一位
    tmask_c = causal_mask(tgt_in.size(0)).to(device)

    optimizer.zero_grad()
    logits = model(src, tgt_in, pad_mask(src), pad_mask(tgt_in), tmask_c)
    loss = criterion(logits.reshape(-1, len(SYMBOLS)), tgt_out.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

print(f"\n===== 3. 开始训练（{EPOCHS} 个 epoch × {ITERS_PER_EPOCH} 步）=====")
start = time.time()
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for _ in range(ITERS_PER_EPOCH):
        total += train_one_batch()
    avg = total / ITERS_PER_EPOCH
    if epoch % 10 == 0 or epoch == 1:
        s, t = make_sample()
        pred = generate(model, s)
        s_str = "".join(map(str, s))
        t_str = "".join(map(str, t))
        mark = "[OK]" if pred == t_str else "[X]"
        print(f"epoch {epoch:3d} | loss {avg:.3f} | 输入 {s_str} → 预测 {pred} (期望 {t_str}) {mark}")

print(f"训练完成，用时 {time.time() - start:.1f} 秒")

# 测试 10 个随机例子
print("\n===== 4. 测试 10 个随机例子 =====")
correct = 0
for _ in range(10):
    s, _ = make_sample()
    s_str = "".join(map(str, s))
    pred = generate(model, s)
    ok = pred == s_str[::-1]
    correct += int(ok)
    print(f"输入 {s_str:8s} → 预测 {pred:8s}  (正确 {s_str[::-1]}) {'[OK]' if ok else '[X]'}")
print(f"\n正确率: {correct}/10")

print("\n===== 5. 数学视角 =====")
print("没有位置编码，这个任务永远学不会 —— 位置编码 = 模型的'顺序感'。")
print("可以试试把上面模型里 self.pos(...) 删掉重新训练，观察 loss 几乎不降。")

