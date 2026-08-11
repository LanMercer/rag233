# 第一周工作汇报（8/1 ~ 8/5 · 前 5 个工作日）

> 汇报人：数学（概率统计）硕士 ｜ 目标：2026 秋招（江浙沪央国企数字化/AI 岗 + 互联网备选）
> 依据：`00-基础内容总纲.md`（总背景）、`01-第一周详细计划.md`（周计划）、每日教程与工作汇报（day1~day5）
> 时间说明：**一周只工作五天**，本报告只汇报前 5 个工作日（Day1~Day5，8/1~8/5）的成果，已全部完成并 git 存档（commit `bad0656`→`3b49046`）。原计划中 8/6~8/7 的 **Qwen 本地部署**与**第一周复盘**两项工作已**移至第 2 周**完成（详见第八部分）。

---

## 一、总背景

| 项目 | 情况 |
|---|---|
| 个人画像 | 985/211 数学（概率统计）硕士；Python 基础语法；ML/DL 零基础；大模型此前只调过 API |
| 硬件 | NVIDIA RTX 3060 Laptop（6G 显存），CUDA 12.3，GPU 版 PyTorch 2.5.1+cu121 |
| 最大短板 | 简历上无成型项目 → 8 月必须产出一个能讲清楚的项目 |
| 8 月主线 | 补齐"部署 → RAG → 微调"大模型全链路，冲刺 8 月下旬提前批 |
| 双线并行 | 主线：大模型应用开发；副线：统计建模基础（覆盖银行风控岗） |
| 时间安排 | 集中 3~4h/天（主线攻坚）+ 零散 0.5~1h/天（力扣刷题）+ 睡前 15 分钟（git 存档） |

**差异化策略（本周核心心法）**：用概率语言讲大模型——softmax 输出 = 概率分布、注意力权重矩阵 = 转移概率矩阵（每行和为 1）、多头 = 多张转移概率矩阵并行、位置编码 = 频率指纹、LoRA = 低秩近似（SVD）。数学（概率统计）背景从第一天就开始变现，作为面试差异化亮点。

---

## 二、本周计划与完成总览

**周主题**：环境 + Transformer 认知 + Python 补缺
**周目标**：确认 Qwen 部署方案（本地部署执行移至第 2 周）+ 画出 Transformer 结构图 + 同步开启刷题节奏

| 日期 | 计划内容 | 主线完成状态 | 核心产出 |
|---|---|---|---|
| 8/1 周六 | 环境搭建日 | ✅ 完成 | 环境截图 + requirements.txt + 第一个 commit `bad0656` |
| 8/2 周日 | Transformer 图解入门 | ✅ 完成 | `transformer_structure.png` 结构图草稿 + 一页概率语言概念笔记 |
| 8/3 周一 | 3Blue1Brown 注意力可视化 | ✅ 完成 | softmax 手算 notebook（3×3 转移概率矩阵）+ 概念笔记补全 |
| 8/4 周二 | 多头注意力与位置编码 | ✅ 完成 | 手写多头注意力（vs 官方误差 5e-07）+ seq2seq demo（loss→0，测试 10/10） |
| 8/5 周三 | 结构图定稿 + 微调前置认知 | ✅ 完成 | 定稿结构图（三件套）+ LoRA 数学 demo + Qwen 硬件方案（3B 4bit） |
| 8/6~8/7 周四五 | ~~Qwen 本地部署 + 部署收尾 + 第一周复盘~~（原计划） | ⏭ 移至第 2 周 | 本周只工作 5 天，8/6~8/7 休息/复盘；原 day6/day7 工作归入第 2 周完成 |

> **刷题情况**：本周专题"数组 + 哈希表"，每日 1~2 题（均为前 5 个工作日内安排的零散任务）。已完成 8/10 题（1、217、26、27、88、121、242、383），2 题待收尾（283、169，教程均已备好思路提示）。

---

## 三、每日工作成果汇总

### Day 1（8/1 周六）· 环境搭建日 —— 全部完成 ✅

- Miniconda + `llm` 虚拟环境（Python 3.11.15，`D:\miniconda1\envs\llm`），与系统环境隔离。
- GPU 版 PyTorch 2.5.1+cu121；**`torch.cuda.is_available()` 返回 True**（周里程碑第 1 项达成）；迷你模型生成测试通过，全链路（环境→GPU→模型）跑通。
- 核心库就位：transformers / accelerate / datasets / langchain / chromadb / bitsandbytes / jupyter。
- `test_env.py` 验收脚本跑通；`pip freeze > requirements.txt` 生成依赖清单；git 首次 commit `bad0656`。
- 力扣 1. 两数之和、217. 存在重复元素（哈希入门）✅，能讲清思路。
- **关键踩坑**：conda 服务条款报错 → `conda tos accept`；新版 transformers 的 torch 安全限制（CVE-2025-32434）→ 换用 safetensors 格式迷你模型绕开；VS Code 终端 activate 不生效 → `conda init powershell` + 重启。

### Day 2（8/2 周日）· Transformer 图解入门 —— 全部完成 ✅

- 精读 Jay Alammar《The Illustrated Transformer》中文版（只看"发生了什么"，跳过矩阵推导与训练章节）。
- 用概率语言重写笔记：softmax 输出 = 概率分布；注意力 = 对 Value 的加权平均，权重由 Q、K 相似度经 softmax 决定；位置编码 = 给词编座位号。
- 用 `draw_transformer.py` 生成整体结构图 `transformer_structure.png`（标注 Encoder / Decoder / 多头注意力 / 位置编码）。
- 产出 `day2-Transformer图解入门详细教程.md` + 一页概念笔记参考范文；下载原论文中英文版 PDF。
- 力扣 26、27（双指针入门）✅；Python 列表推导式、字符串切片 ✅。

### Day 3（8/3 周一）· 3Blue1Brown 注意力机制可视化 —— 全部完成 ✅

- 观看 3Blue1Brown《But What Is a Transformer?》第 1、2 部，理解注意力动机、Q/K/V 三向量、softmax 权重。
- **亲手手算 3×3 注意力权重矩阵**（`softmax_attention.py` / `.ipynb`），确认每行和为 1 = 转移概率矩阵（周里程碑：softmax 手算 notebook ✅）。
- 数学加成实锤：用**方差可加性**解释"为什么除以 √d"（d 个独立分量求和方差 = d，除以 √d 让方差回 1，防止 softmax 退化）。
- 改词向量实验：给"猫""狗"加共享"动物"维度后，二者互相关注权重 0.212→0.468，"追"掉到 0.063 → 词向量承载语义，注意力自动聚焦同类词。
- 力扣 88、121（双指针进阶）✅；Python 类与对象初步 ✅。
- **踩坑**：第一次用 Jupyter（输出在格子正下方而非终端）；PowerShell 中文乱码（控制台编码问题）；脚本升级为逐行教学版。

### Day 4（8/4 周二）· 多头注意力与位置编码 —— 主线完成 ✅

- 跑通 `positional_encoding_demo.py`：生成位置编码热力图 `positional_encoding.png`；**关键实验**：不加位置编码时"我爱你/你爱我"注意力输出完全一样，加了之后不一样。
- 手写多头注意力 `multihead_attention_demo.py`：头 1（看身份）≈ 对角矩阵、头 2（看动物类别）让"猫狗互相关注"；**手写 vs 官方 `nn.MultiheadAttention` 最大误差 5e-07**——证明做的是同一件事。
- 摸清 `torch.nn.Transformer` 源码与张量形状：`(序列长度, 批量, 维度)`，Encoder/Decoder 只改内容不改形状。
- 跑通 seq2seq 倒序小 demo `transformer_seq2seq_demo.py`：**loss 从 1.856 → 0.000，测试 10/10 全对**，GPU 约 2 分钟（第一次亲手训练出模型）。
- 数学视角：多头 = 多张转移概率矩阵并行；位置编码 = 频率指纹（与信号处理/傅里叶分解同源）。
- **踩坑**：奇数维位置编码 sin/cos 列数对不上（裁剪修复）；手写 vs 官方误差 1.87（PyTorch 线性层是 `y = x·Wᵀ`，转置修复 → 5e-07）；seq2seq 生成全错（数字没转成词表编号 3~12，`stoi[str(d)]` 修复 → 10/10）。
- **待补（后续已补齐）**：3Blue1Brown 第 3、4 部观看 ✅、力扣 242/383 ✅、Python 文件读写练习 ✅（见 `day4/day4工作汇报.md`）。

### Day 5（8/5 周三）· 结构图定稿 + 微调前置认知 —— 主线完成 ✅

- 跑通 `draw_transformer_full.py`，生成**定稿结构图** `transformer_structure_full.png`（上半整体结构 + 下半 Encoder 单层解剖，红色 ⊕ 残差跳线、LayerNorm、FFN 全标注）——周里程碑第 2 项达成。
- 三件套公式与作用：残差 `x + 子层(x)`、LayerNorm `(x-均值)/标准差×γ+β`、FFN `W2·ReLU(W1·x+b1)+b2`；标准"子层 → 残差加法 → LayerNorm"包装模式。
- 跑通 `lora_math_demo.py`（纯 numpy 数学模拟）：① 预训练底座 W0 通用任务误差≈0、特定任务误差 4.05；② **完整微调把旧任务冲掉（任务 A 误差 0→3.93）= 灾难性遗忘**；③ ΔW 奇异值只有 3 个 → 低秩；④ LoRA 用 B·A（秩 r）近似，r 越小越温和；⑤ 512×512 模拟 r=16 误差 1.63e-15 几乎无损。
- **LoRA 参数账**：全量 d²=12,845,056 vs LoRA 2dr=57,344（d=3584, r=8）→ **节省 99.55%**；Qwen2.5-7B LoRA 只训 642 万参数 ≈ 全模型的 **0.084%**。
- 跑通 `qwen_memory_calculator.py` 确认硬件方案：**6G 显存首选 Qwen2.5-3B-Instruct + 4bit（NF4）≈ 3.0GB**；7B 4bit ≈ 5.6GB 需 8G 显卡；7B fp16（15.2GB）不推荐。
- 零散时间：`numpy_basics.py` 跑通（7 招 + numpy 手写 softmax）✅；力扣 283、169（🔄 周报时点未打完）。
- **踩坑**：matplotlib 下标字形缺失（x₁ 改成 x1）；正规方程加 λI=1e-6 防奇异；显存测算为经验估算（第 2 周部署时实测为准）。

---

## 四、本周里程碑验收清单

| # | 里程碑 | 状态 |
|---|---|---|
| 1 | `torch.cuda.is_available()` 返回 True，环境依赖齐全 | ✅ 8/1 达成 |
| 2 | 画出 Transformer 整体结构图并讲清每层作用 | ✅ 8/2 草稿、8/5 定稿（含三件套） |
| 3 | 用概率语言解释 softmax / 注意力 / 位置编码 / 温度 | ✅ softmax、注意力、位置编码已讲透；温度将在第 2 周部署 Qwen 时实测落地 |
| 4 | softmax 手算 notebook（3×3 转移概率矩阵） | ✅ 8/3 达成 |
| 5 | 完整结构图（残差 / LayerNorm / 前馈网络） | ✅ 8/5 达成 |
| 6 | 微调一页认知笔记（预训练 / 微调 / LoRA） | ✅ 8/5 参考范文与数学 demo 达成，手写笔记待补 |
| 7 | 本地加载 Qwen2.5 量化版成功生成文本 | ⏭ 移至第 2 周：硬件方案已定（3B 4bit ≈ 3.0GB），第 2 周开头执行 |
| 8 | 力扣完成 10 题（数组 + 哈希，简单为主） | 🔄 已完成 8/10，2 题待收尾（283、169，属本周前 5 个工作日任务） |
| 9 | 完成第一周复盘笔记 | ⏭ 移至第 2 周：8/6~8/7 为休息/复盘日，复盘笔记与 Qwen 部署一同在第 2 周完成 |

---

## 五、踩坑与解决汇总（本周共 15+ 个，精选高频）

| 类别 | 问题 | 解决办法 |
|---|---|---|
| 环境 | conda 服务条款未接受，创建环境被拒 | `conda tos accept` 后重试 |
| 环境 | 新版 transformers 要求 torch≥2.6（安全限制），显卡驱动最高只支持 cu121（torch 2.5.1） | 换用 safetensors 格式迷你模型绕开 |
| 环境 | VS Code PowerShell 里 `conda activate` 不生效 | `conda init powershell` + 重启 VS Code |
| 运行 | PowerShell 运行含中文脚本乱码 / `✓✗` 报 GBK 编码错误 | 属控制台编码问题：VS Code 终端正常；输出重定向 UTF-8 文件核对；脚本改用 `[OK]/[X]` 标记 |
| 数学 | 手写多头注意力 vs 官方误差 1.87 | PyTorch 线性层约定 `y = x·Wᵀ`，把权重转置后喂给官方模块 → 误差降到 5e-07 |
| 数学 | seq2seq 训练完 loss 很低但生成全错（0/10） | 数字 0~9 的词表编号是 3~12，不能直接当 token → `stoi[str(d)]` 转换 → 10/10 全对 |
| 数学 | 位置编码奇数维 sin/cos 列数对不上 | 对奇数维裁剪：cos 只用前"偶数维个数"个频率 |
| 数学 | 位置编码加批量维度时维度不匹配 | `pe[:seq_len]` 补 `unsqueeze(1)` 成 `(seq, 1, d_model)` 再广播 |
| 可视化 | matplotlib 缺下标数字字形（x₁ 显示方块） | 脚本里下标字符全部改为普通写法（x1/x2/W1/W2） |
| 数值 | LoRA 最小二乘矩阵接近奇异 | 正规方程加 λI（λ=1e-6）防奇异，结果与理论完全一致 |
| 估算 | 显存测算的"运行开销"是经验估算，不同来源有出入 | 注明"以第 2 周部署时 `torch.cuda.memory_allocated()` 实测为准"，不影响方案结论 |
| 学习 | 找不到 `torch.nn.Transformer` 核心结构 | 教程修正：核心结构在 `__init__` 中间；`inspect.getsource` 一次只打印一个类，需跑第二条命令 |
| 学习 | 看不懂 `(5, 3, 6)` 三个数字 | 教程新增"序列长度 / 批量 / 维度"大白话小节（点名册类比） |
| 学习 | 不确定 demo 是不是机器学习/深度学习 | 脚本文件头 + 教程新增"五件套"（模型/数据/损失/优化器/训练+测试）说明 |

---

## 六、本周产出文件清单

| 天数 | 文件 |
|---|---|
| Day1 | `day1-环境搭建详细教程.md`、`requirements.txt`、`test_env.py`、`大模型算法Day1-工作汇报.md` |
| Day2 | `day2-Transformer图解入门详细教程.md`、`draw_transformer.py`、`transformer_structure.png`、`概念笔记-参考范文.md`、原论文中英文 PDF |
| Day3 | `day3-3Blue1Brown注意力机制可视化详细教程.md`、`softmax_attention.py`、`softmax_attention.ipynb`、`概念笔记-补全参考范文.md`、`day3工作汇报.md` |
| Day4 | `day4-多头注意力与位置编码详细教程.md`、`positional_encoding_demo.py`、`positional_encoding.png`、`multihead_attention_demo.py`、`transformer_shape_demo.py`、`transformer_seq2seq_demo.py`、`概念笔记-参考范文.md`、`day4工作汇报.md` |
| Day5 | `day5-结构图定稿与微调前置认知详细教程.md`、`draw_transformer_full.py`、`transformer_structure_full.png`、`lora_math_demo.py`、`qwen_memory_calculator.py`、`numpy_basics.py`、`概念笔记-参考范文.md`、`day5工作汇报.md` |
| 本周汇总 | `week1工作汇报.md`（本文件）、`week1概念笔记汇总.md`、`week1力扣题目与答案汇总.md` |

**Git 存档**：commit `bad0656`（8/2 Day1）→ `701a9c4`（8/4 Day2）→ `63415d2`（8/5 Day3）→ `7aa4cee`（8/7 Day4）→ `3b49046`（8/11 Day5）。

---

## 七、本周数据与亮点（数字会说话）

- **环境**：PyTorch 2.5.1+cu121，GPU 可用，6 个核心库 + bitsandbytes 全部就位。
- **手算能力**：3×3 注意力权重矩阵每行和为 1 = 转移概率矩阵；词向量改造实验权重 0.212→0.468。
- **代码可信度**：手写多头 vs 官方误差 5e-07；seq2seq demo loss 1.856→0.000、测试 10/10。
- **微调数学**：灾难性遗忘任务 A 误差 0→3.93；LoRA 参数节省 99.55%；7B 只训 0.084%；512×512 用 r=16 近似误差 1.63e-15。
- **部署算账**：6G 显存首选 Qwen2.5-3B-Instruct 4bit ≈ 3.0GB（7B 4bit ≈ 5.6GB 需 8G，7B fp16 15.2GB 不推荐）。
- **力扣**：已完成 8/10（1、217、26、27、88、121、242、383），全部能讲清思路。

**最大跨越**：从"只调过 API"到——能画出 Transformer 完整结构图、能亲手算出注意力权重矩阵、能训练出第一个 seq2seq 模型、能用 SVD/低秩近似讲清 LoRA、能自己算显存账定部署方案。数学背景从"简历上的专业"变成了"每天都能用的武器"。

---

## 八、小结与下周安排

### 小结
第一周目标（前 5 个工作日）基本达成：**环境就绪 + 结构图定稿 + 微调前置认知 + 刷题节奏开启**。主线学习（Day1~5）已全部完成并 git 存档（commit `bad0656`→`3b49046`，Day5 于 8/11 存档，时间比计划稍滞后）。**按"一周只工作五天"的新安排，原 day6/day7 的 Qwen 本地部署与第一周复盘两项工作已移至第 2 周**。本周验证了"用概率语言讲大模型"的差异化路线完全走得通，且每个结论都有亲手跑的代码/数字支撑。

### 本周遗留（前 5 个工作日任务，建议尽快补上）
1. **力扣收尾**：283、169 两题做完并入 `leetcode` 仓库（按数组/哈希分类，题解注释写"数学视角"）。
2. 概念笔记手写补全（范文已全部备好；Day 4 部分已补全，Day 5 微调认知部分待补）。

### 已移至第 2 周的工作（原 day6/day7，不占本周）
1. **Qwen 本地部署**——按已定方案（Qwen2.5-3B-Instruct 4bit ≈ 3.0GB）下载模型 → `transformers` + `bitsandbytes` 4bit 加载 → 推理脚本生成文本 → `torch.cuda.memory_allocated()` 实测显存 → 调 temperature/top_p 对比输出。
2. **第一周复盘笔记**——完成情况 / 卡点 / 下周调整三部分，并与 Qwen 部署产出一同整理进 git。

### 下周（第 2 周 8/8~8/12）预告：Qwen 部署收尾 + Prompt 工程 + RAG 全链路
- **第 2 周开头先承接本周移过来的两项**：Qwen 本地部署（3B 4bit，产出本地推理截图）+ 第一周复盘笔记。
- 主线产出：**上传 PDF 能问答的 demo**（RAG，检索增强生成）。
- 本周储备直接可用：RAG 的交叉注意力 = 用"查到的文档"当 K、V，"问题"当 Q（`torch.nn.MultiheadAttention` 直接可用）；注意力三步走就是检索时打分、选 Top-K、融合的过程。
- 第 2 周详细计划（含承接项）见 `第二周\01-第二周详细计划.md`（第 2 周开始时创建）。

---

> 汇报依据：《00-基础内容总纲》《01-第一周详细计划》及各日教程/工作汇报、脚本实测运行结果、git 提交记录整理。未完成项如实标注，绝不虚报。
