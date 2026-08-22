# Day 12（第三周第 2 个工作日 · LoRA/QLoRA 微调实战（二）：跑通微调训练）· 超详细新手教程

> 时间安排：主线 4~~5 小时 + 零散时间 0.5~~1 小时 + 睡前 15 分钟
> 日期说明：第三周计划按"第 N 个工作日"锚定（第二周 day10 实际 8/20 完成，本周工作日从 8/21 起），**内容顺序不变**——day12 就是原计划 8/16 的内容：LoRA/QLoRA 微调实战（二）。
> 适用对象：已完成 Day11——手里有 `sft_data.json`（300 条三字段机器人行业数据）、`check_env.py` 五件套全绿（transformers 5.14.1 / peft 0.20.0 / bitsandbytes 0.50.0 / accelerate 1.14.0 / datasets 5.0.1）、本地 `download\Qwen2.5-3B-Instruct\`（Day6 下载，4bit 部署过）、Day9 `local_api.py`（标准生成层）。今天**第一次真正训练模型**。
> 本教程的目标：① 讲清训练"五件套"每个概念（新名词必带简介）；② 读懂****** `train_lora.py`**（文件头五件套 + LoraConfig / BitsAndBytesConfig / get_peft_model 三件套装配）；③ 跑通 QLoRA 微调训练，用** `torch.cuda.memory_allocated()` **盯显存（6G 预算内）、观察 loss 下降；④ 只存 adapter（几 MB~几十 MB，不存全量模型）；⑤ 微调前后效果对比 →** `微调前后效果对比表.md`**。
> 本教程的铁律：今天不下载新模型、不改数据；训练产物** `lora_adapter/`**（adapter 权重）不进 git（会在 .gitignore 排除，另行说明）；跑训练前**关掉浏览器/游戏等占显存程序**。教程里的预期输出是"形态示例"，**以你实际跑出的结果为准**（下面写的是形态说明，不是这台机器的实测）。

---



## 📌 今天你要做什么（大白话版）

Day11 我们造好了"食材"（300 条 SFT 数据）和"灶台"（五件套环境全绿），还复习了 LoRA 的数学（ΔW 低秩、r 是旋钮、省 99.55% 参数）。今天**开火**：让 Qwen2.5-3B 在这 300 条机器人行业数据上真正学一遍——这就是微调。

1. **先补训练概念**：LM loss（交叉熵）、AdamW、epoch/验证集、梯度累积、adapter——新名词必带简介；
2. **读懂** `train_lora.py`：文件头把五件套写清（模型=Qwen2.5-3B + QLoRA r=8 alpha=16、数据=300 条 SFT、损失=交叉熵、优化器=AdamW lr 2e-4、训练=3 epoch + 验证集），正文是 LoraConfig + BitsAndBytesConfig + get_peft_model 三件套装配；
3. **跑通训练**：盯显存（6G 预算内）、看 loss 一路下降；
4. **只存 adapter**：`lora_adapter/` 只有几 MB（`adapter_config.json` + `adapter_model.safetensors`），不是几 GB 的全量模型；
5. **微调前后对比**：同一组 5 个问题，原模型 vs 原模型+adapter，肉眼找差异 → 对比表。

**今天结束时的"验收标准"一句话：**

> 你手里有一个 `lora_adapter/`（只含 adapter）+ 一张 `微调前后效果对比表.md`（至少 3 个问题有肉眼可见差异），且能讲清训练五件套、"LoRA 只存 adapter、推理时合入权重 W0+B·A"和本周参数账/显存账。

**和你的数学背景接上的点（今天会反复出现）：**

- **LM loss = 交叉熵 = 负对数似然的平均**：模型对词表输出一个概率分布，loss 就是"标准答案 token 处概率的负对数"。最小化它 = 最大化 `P(答案 | 问题)`——正是 SFT 在调整的那个条件概率；
- **梯度累积 = 显存不够时的"数学凑法"**：`等效 batch = per_device_batch × gradient_accumulation_steps`，用"攒几步梯度再更新一次"换显存；
- **验证集 = 留出法**：Day11 统计八股讲的交叉验证思想今天直接落地——90% 训练、10% 只看不练（算 eval_loss），防过拟合。

---



## 🗺️ 今天的学习路线图（先看这里，心里有个数）


| 步骤       | 内容                                     | 预计时间     |
| -------- | -------------------------------------- | -------- |
| 第 0 步    | 准备：复制数据、重跑环境检查、关占显存程序                  | 10 分钟    |
| 第 1 步    | 概念补课：训练五件套详解（新名词必带简介）                  | 30~40 分钟 |
| 第 2 步    | 读懂 `train_lora.py`（五件套 + 三件套配置）⭐       | 40~60 分钟 |
| 第 3 步    | 跑通训练：盯显存 + 观察 loss + 只存 adapter ⭐ 今日核心 | 60~90 分钟 |
| 第 4 步    | 微调前后效果对比：`compare_lora.py` + 对比表 ⭐     | 40~60 分钟 |
| 第 5 步    | 验收：五件套 + 参数账/显存账 + 面试话术                | 20~30 分钟 |
| 零散时间     | 力扣 56、75（排序应用）＋ 统计八股：交叉验证              | 0.5~1 小时 |
| 睡前 15 分钟 | 收尾 + git commit（adapter 不进 git）        | 15 分钟    |


> 主线合计约 4~~5 小时。**第 3 步（跑通训练）是今天最大的坎**——训练本身可能要 20~~60 分钟（以你机器实际为准），这段时间正好拿去刷力扣/看统计八股，别干瞪眼。

---



## 🔧 第 0 步：准备（10 分钟）



### 0.1 认识今天的学习素材


| 素材           | 位置/链接                                               | 用途                 |
| ------------ | --------------------------------------------------- | ------------------ |
| 本文件夹 2 个脚本   | `train_lora.py` / `compare_lora.py`                 | 第 2、3、4 步主线        |
| Day11 SFT 数据 | `第三周\day11\sft_data.json`（300 条）                    | 今天要**复制一份**到 day12 |
| Day11 环境检查   | `第三周\day11\check_env.py`                            | 训练前再跑一遍            |
| 本地 Qwen 模型   | `download\Qwen2.5-3B-Instruct\`（Day6 下载，6GB 不进 git） | 微调底座（今天只读不下载）      |
| Day9 接口      | `第二周\day9\local_api.py`                             | Day13 把微调模型接入接口的参考 |
| 第三周周计划       | `第三周\01-第三周详细计划.md`（8/16 章节）                        | 全周对照               |




### 0.2 复制数据 + 打开文件夹

数据放哪？两种方案：① 复制到 day12（本教程默认，`train_lora.py` 直接读 `sft_data.json`）；② 不改动数据，脚本里写相对路径 `..\day11\sft_data.json`。本教程用方案①，路径最简单：

```
copy "D:\Lan\研究生\技术学习\大模型算法\第三周\day11\sft_data.json" "D:\Lan\研究生\技术学习\大模型算法\第三周\day12\sft_data.json"
cd "D:\Lan\研究生\技术学习\大模型算法\第三周\day12"
conda activate llm
```

看到行首 `(llm) PS ...\day12>` 即可。✅

### 0.3 重跑环境检查 + 清出显存

```
python "D:\Lan\研究生\技术学习\大模型算法\第三周\day11\check_env.py"
```

重点盯两项：

1. `CUDA 可用 : True`——硬门槛，False 回 Day6 查 CUDA 安装；
2. `当前可用显存 : ≥4GB`——今天训练峰值比推理高，**<4GB 就先关浏览器/游戏/其他模型进程**（避坑表第 1 条）。



### ✅ 第 0 步验收标准

终端停在 day12、行首有 `(llm)`、`sft_data.json` 已在本地、check_env 全绿且可用显存 ≥4GB = 成功。

---



## 🧠 第 1 步：概念补课——训练"五件套"详解（30~40 分钟）

> 对应周计划 8/16 第 1 条："写微调脚本 `train_lora.py`，**文件头写清五件套**：模型=Qwen2.5-3B-Instruct + QLoRA（r=8, alpha=16）、数据=自制 SFT 数据、损失=交叉熵（LM loss）、优化器=AdamW + 学习率 1e-4~2e-4、训练=若干 epoch + 验证集观察 loss。"
>
> "五件套"从第一周就跟我们，今天终于到"训练"这一环——每一件套都需要真正的数学直觉。



### 1.1 第一件：模型——Qwen2.5-3B-Instruct + QLoRA

- 底座 = Day6 下载的 `Qwen2.5-3B-Instruct`，今天**冻结它**（4bit 量化，参数不更新）；
- 可训练部分 = 挂在它上面的 LoRA 低秩增量 `B·A`（r=8, alpha=16）——这就是 **QLoRA = 4bit 底座 + LoRA 适配器**（Day11 讲过）；
- 为什么用 Instruct 版？它的 ChatML 对话格式和我们数据格式一致，Day9 已经用它做过接口，今天训练、明天（Day13）接入接口都顺。



### 1.2 第二件：数据——300 条 SFT 数据怎么喂进去

- 每条 `(instruction, input, output)` 拼成一段 **ChatML**：`<|im_start|>user\n指令+材料<|im_end|><|im_start|>assistant\n标准答案<|im_end|>`（Day9 手拼过，今天用 `apply_chat_template` 自动拼）；
- **90/10 切验证集**：270 条训练、30 条验证——验证集只看 loss（eval_loss），不参与梯度更新，防过拟合（第 1.5 节展开）。



### 1.3 第三件：损失——LM loss（交叉熵，数学主场）⭐

**先给一句总纲：loss 是一个数字，专门衡量"模型这次答得有多差"——答得越离谱数字越大，答得越准数字越小，训练就是让这个数字一路变小。** 交叉熵（cross-entropy）就是这个"打分器"的名字，也是语言模型唯一常用的损失。

**第一步，看模型在哪个位置打分：**

语言模型是**逐 token 生成**的。训练时模型读一句话，在每个位置都要预测"下一个 token 是什么"。以标准答案 `谐波减速器是……` 为例，模型读到"谐"字时，要对词表里所有候选（几十万个字/词）各给一个概率——比如给"波"0.7、"合"0.2、"和"0.1……真实答案是"波"。

**第二步，交叉熵就是"标准答案位置的负对数概率"：**


| 模型给真实答案"波"的概率 p | 损失 = -log p | 感觉         |
| --------------- | ----------- | ---------- |
| 0.99            | 0.01        | 胸有成竹，几乎不罚  |
| 0.70            | 0.36        | 猜得不错       |
| 0.50            | 0.69        | 和抛硬币一样，没把握 |
| 0.10            | 2.30        | 猜得很差       |
| 0.01            | 4.61        | 压根不觉得这是答案  |


交叉熵这一个位置的损失 = **-log(模型给标准答案的概率)**，就这么简单。

**第三步，为什么用 log？三个理由：**

1. **单调**：p 越大 -log p 越小——想让 loss 变小，就等于想让标准答案概率变大，方向完全一致；
2. **连乘变连加**：一整句话的生成概率是每个位置概率的**连乘** p₁×p₂×…×p_T（比如 0.9 的 50 次方，会小到数值下溢）。取 log 后连乘变连加：log(p₁p₂…) = Σlog(pᵢ)，再加个负号就是 Σ[-log(pᵢ)]——既稳定又方便。这就是**负对数似然（NLL）**这个名字的由来；
3. **凸函数、惩罚不均匀**：从 0.9 提到 0.99，loss 只降一点；但从 0.1 提到 0.9，loss 狂降——把一个"几乎不认的字"教对，收获远大于"本来就差不多的字"，符合直觉。

**第四步，为什么名字叫"交叉"熵？**

信息论里，两个分布 q（真实）和 p（预测）的交叉熵是 `H(q,p) = -Σᵢ qᵢ·log pᵢ`。训练时真实答案 q 是**独热分布**（答案"波"的概率为 1，其余全 0），求和时其余项全是 0，只剩一项 → 恰好就是 **-log p(真实答案)**。所以"交叉熵"在我们这儿 = "标准答案位置的负对数概率"，名称的来龙去脉到此闭环。

**第五步，整条数据的 loss 怎么算？**

一条数据（题干 + 标准答案）有几百个 token，对**每个答案位置**都算一个 -log p(该位置的真实 token)，然后**求平均**：

```
loss = (1/T) · Σ_{t} -log p_t(该位置真实 token)
```

教程里说的"损失从 1.4 降到 0.8"翻译成人话：平均每个位置模型给标准答案 token 的概率，从 e^(-1.4) ≈ 25% 涨到 e^(-0.8) ≈ 45%——**模型对这份数据的平均把握提升了**。

**第六步，为什么只算答案部分（output）？**

题干"请解释什么是谐波减速器？"里的字也是已知的，如果也算 loss，模型会把"记住题目怎么拼"也当成任务——但我们真正想让它学的是"看到问题 → 给出答案"。所以实现上把题干位置的标签设为 -100，PyTorch 的交叉熵自动忽略这些位置，**只对答案部分打分**（第 2 步会看到代码）。

**数学视角（面试加分）**：交叉熵 = 负对数似然（NLL）的期望，而期望形式下交叉熵 = KL 散度 + 常数项，最小化交叉熵 ⇔ 最小化 KL(q‖p) ⇔ 最大化模型给标准答案的概率——**这就是最大似然估计（MLE）**。所以"SFT 在 300 条数据上训练"本质是在做 MLE：找到让 P(答案|问题) 最大的 LoRA 参数，把分布从"大而平"收成"小而尖"。

### 1.4 第四件：优化器——AdamW + 学习率

**先给一句总纲：loss 告诉模型"答得有多差"，优化器负责回答"接下来往哪个方向、迈多大步子，才能让 loss 变小"。** 交叉熵算出一个数字后，必须把它变成"对参数的修改"——这个"翻译"就是优化器干的活。

**第一步，回到最朴素的想法：梯度下降（下山）。**

把 loss 想象成一座山的海拔，模型参数 θ 是你在山上的位置。你要下山（让 loss 变小）：算出当前位置的梯度 g = ∂loss/∂θ（指向"最陡的上坡方向"），然后朝反方向迈一步：

\theta \leftarrow \theta - \text{lr} \cdot g

lr（learning rate，学习率）就是"一步迈多大"。

具体数字走一遍：假设某参数 θ=5.0，当前梯度 g=2.0（往正方向走 loss 会变大），lr=0.01，那么新 θ = 5.0 − 0.01×2.0 = **4.98**。梯度在"推着" θ 往 loss 变小的方向挪。

**第二步，为什么朴素梯度下降不够好？**


| 痛点     | 现象                               | 解决                 |
| ------ | -------------------------------- | ------------------ |
| ① 步长难定 | lr 大：在山谷里来回震荡、不收敛；lr 小：龟速，几万步走不动 | 给每个参数"自适应步长"（Adam） |
| ② 峡谷震荡 | 地形是窄长的峡谷，梯度方向剧烈抖动，直走会被两壁弹来弹去     | 引入动量（Momentum）     |
| ③ 悬崖   | 个别参数梯度极大，一步把参数甩飞（loss 变 NaN）     | 梯度裁剪（今天不展开）        |


**第三步，动量 Momentum——统计里的"指数滑动平均（EMA）"。**

下山不只看"当前这一脚"，还要记住"之前一直在往哪个方向滚"——像推下坡的球：连续朝一个方向走，速度会累积；方向突然变了，也不会立刻掉头，而是被"惯性"拉着平滑转弯。

实现上，维护一个速度变量 v，用 EMA 混合当前梯度：

v \leftarrow \beta v + (1-\beta)g \qquad (\beta=0.9)

β=0.9 时，v ≈ 最近 10 步梯度的加权平均。**EMA 就是统计课上学过的指数加权移动平均**——近期权重指数衰减、远处几乎忽略。这一下就把"峡谷来回震荡"磨平了（正负梯度互相抵消），沿主方向越走越快。

**第四步，Adam——给每个参数独立的"步子大小"（自适应学习率）。**

Adam 记录两个量（都是 EMA）：

- 一阶矩 m = EMA 梯度（≈ 梯度均值，方向）；
- 二阶矩 v = EMA 梯度平方（≈ 梯度平方的均值，反映"这一步的梯度通常多大"）。

更新公式（加了偏差修正，写作 m̂, v̂）：

\theta \leftarrow \theta - \text{lr} \cdot \frac{\hat m}{\sqrt{\hat v} + \epsilon}

**数学视角（你的主场）**：`m̂ / √v̂` 本质上是对梯度做了一次"标准化"——**除以它自己的均方根（RMS），就像标准化里的除标准差**。结果是每个参数的步长"单位一致、尺度无关"：

- 梯度一直很大的参数（频繁被更新）→ v 大 → 除下来步子变小，不会乱冲；
- 梯度一直很小的参数（像"谐波"这种冷门词关联的权重，稀疏更新）→ v 小 → 步子变大，不会被饿死。

一个公式同时解决痛点①②：既有动量方向，又自适应步长。

**第五步，W = Weight Decay——AdamW 比 Adam 多出来的 W。**

**Weight decay（权重衰减）**：每次更新时，除了按梯度走一步，还把参数朝 0 拉一小把：

\theta \leftarrow \theta - \text{lr}\cdot g - \text{lr}\cdot\lambda\cdot\theta

为什么？参数太大会让模型"太自信"、对训练集死记硬背——惩罚大参数 = 防过拟合。数学上等价于给 loss 加一项 L2 正则 `λ‖θ‖²`（梯度里多出 λθ 这一项，就是"往回拉"的力）。

**AdamW 的 W**：在 Adam 里，λθ 会混进梯度、被 `m̂/√v̂` 一起缩放——衰减幅度就被自适应步长搅乱了；AdamW（W = Weight Decay）把"权重衰减"从梯度里**解耦**出来，直接在参数上按原始比例衰减。所以 AdamW 的权重衰减是"干净、可预期"的。

> **和你的统计背景接上（面试加分）**：L2 正则 = 给参数加**高斯先验**，最小化 `loss + λ‖θ‖²` 等价于**最大后验估计（MAP）**，λ 是先验的强度——权重衰减就是"我事先相信参数不该太大"的贝叶斯话术。

**第六步，学习率为什么取 1e-4~2e-4？**

lr 是"每一步的基准步长"。微调的任务不是"从头学"，而是"在已有底座上小幅调整"——步子太大，会把底座学好的东西一脚踢飞（**灾难性遗忘**，Day5 的 0→3.93 实验）；太小又学不动。经验值：


| 场景           | 学习率                   |
| ------------ | --------------------- |
| 全量微调         | 1e-5 ~ 5e-5           |
| LoRA / QLoRA | 1e-4 ~ 2e-4（今天取 2e-4） |
| 学习率太高        | loss 震荡 / NaN / 灾难性遗忘 |


另外配合两个调度：

- **warmup（前 3% 步线性升温）**：训练刚开始梯度估计还不可靠（尤其 Adam 的 EMA 还没积累起来），先小步走稳住，再逐步放大；
- **cosine 余弦退火**：后期让学习率平滑降下来，收敛更细（像下山快到谷底时放慢脚步，不会在谷底来回弹）。

**一句话总结**：Adam = **动量（方向惯性）+ 每个参数自适应步长（梯度标准化）**，W = **权重衰减（往 0 拉一点防过拟合）**；今天用它微调 LoRA，学习率 2e-4 + warmup + 余弦退火，在"改得动"和"改不坏"之间取平衡。

---

**🎈 还是没懂？看这版：一个"摸黑下山"的故事**

把上面所有名词忘掉，想象这个场景：

> 你半夜被困在一座山上，手电没电，什么都看不见。你手里只有一个**海拔表**（loss 数字），它告诉你"你现在多高"，但**不告诉你怎么走**。目标：摸黑走到山脚（loss 最小）。

现在，整个优化器就是你的"**脚步控制器**"。每一步要做三件事：

**① 先探路，决定往哪走 → 梯度 g**

你伸手往左探探、往右探探，感觉哪边"海拔下降最快"，就往那边走。数学上这个"探路"动作叫**算梯度**：`g = 山坡最陡的方向`。有方向了，走起。

**② 决定一步迈多大 → 学习率 lr**

- 迈**太大**：一脚踩空，滚下去又弹上来，在山谷里来回晃（loss 忽高忽低，永远到不了底）；
- 迈**太小**：走一整晚还在原地磨蹭。

所以 lr 就一句话：**一步迈多大**。今天取 `2e-4` = 一个很小的数字 = **小碎步**。为什么这么小？因为微调只是给一个已经很好的模型"稍微调整姿势"，不是从头爬山——步子一大，容易把底座本来学会的东西踢坏（灾难性遗忘）。全量微调是 1e-5（更小），LoRA 只改一点参数，2e-4 刚好。

**③ 三个"外挂"，让脚步更聪明 → Adam + W**

现在你已经会"探方向 + 迈步子"了，但纯靠这个走山还是笨，于是加三样东西：


| 外挂     | 山里的故事                                                                                       | 术语                     | 管什么  |
| ------ | ------------------------------------------------------------------------------------------- | ---------------------- | ---- |
| 记住惯性   | 你走了几步发现一直在往"东北"方向下坡，中间只是偶尔踩到小坑。如果每步只看脚下，会被小坑带得左右乱晃；**记住最近 10 步大致的方向**，小坑的抖动互相抵消，大方向越走越快     | **动量 Momentum**（EMA）   | 方向   |
| 按坡度调步子 | 缓坡上小步挪太慢——反正坡度小、不会摔，**大胆迈大步**；陡坡上必须**收小步子**——不然冲下去。每个方向独立调：坡度大自动小步、坡度小自动大步                  | **Adam 自适应步长**（m̂/√v̂） | 步幅   |
| 兜里的磁铁  | 每走一步，兜里都装着一块**往下吸的磁铁**，把参数朝 0 轻轻拉一点。为什么？怕你走得太飘、太自信——参数太大 = 死记硬背训练数据 = 过拟合。每步往回拽一点，模型就"谦逊"一点 | **权重衰减 Weight Decay**  | 防过拟合 |


> 名字来历："Adaptive Moment Estimation" = 自适应矩估计（m 是"动量"、v 是"方差/矩"），**Adam**；多了那兜磁铁（权重衰减）就是 **AdamW** 里的 **W**。

**把整个故事压成一句话：**

> 优化器 = 你的脚步控制器：**探方向（梯度）→ 定步幅（学习率）→ 记惯性（动量）→ 按坡度自动调步（Adam）→ 每步被磁铁往 0 拉（W）**，合起来就是 **AdamW**。

**再对着** `train_lora.py` **第 1 区看一遍，全对上了：**


| 代码                           | 下山故事里对应                           |
| ---------------------------- | --------------------------------- |
| `learning_rate=2e-4`         | 基准步幅：小碎步微调，怕踢坏底座                  |
| `warmup_ratio=0.03`          | 起步先原地热身（开头路况不明，前 3% 步先用小步走稳）      |
| `lr_scheduler_type="cosine"` | 快到山脚自动放慢脚步，稳稳停在谷底（余弦退火）           |
| `GRAD_ACCUM=8`               | 攒 8 步的"探路结果"再迈一次大脚（梯度累积，第 1.5 节讲） |


**面试背诵版一句话：**

> "优化器把 loss 变成对参数的更新。AdamW 就是：用动量记住更新方向，按每个参数的梯度大小自动缩放步长，再加上权重衰减防过拟合。学习率取 2e-4，是因为 LoRA 只在底座上小幅调整，步子太大就会灾难性遗忘。"



### 1.5 第五件：训练——epoch / 验证集 / 梯度累积

**先给一句总纲：训练 = 让模型在这 300 道题上"刷题"，但这三件事管的是"刷几遍、怎么知道自己学会没、显存不够时怎么刷"——全都可以用一个"考试"的故事装下。**

**通俗版：一场只考 300 道题的考试**

> 训练模型 = 准备一场"只考 300 道题"的考试。训练数据 = 300 道题（带标准答案）。三个概念全来自这个场景：

**① epoch——这 300 道题反复背几遍。**

**epoch = 把全部训练数据完整过一遍**。今天 `EPOCHS = 3` = 300 道题刷 3 遍。

- 刷 **1 遍**：每道题只见过一次，还没学透就到考试了（loss 没降到位）；
- 刷 **3 遍**：从"眼熟"到"会做"，loss 明显下降——微调恰到好处；
- 刷 **100 遍**：300 道题背得滚瓜烂熟，但**只会这 300 道**，题目一变就懵——这就是**过拟合**（背题 ≠ 学会）。

**② 验证集——不能拿"背过的题"给自己打分。**

怎么知道模型学得好不好？拿训练数据考它当然全对——**它刚背过**。就像考完试问自己"我背过没"，答案永远是好，没参考价值。

所以留出 30 道题（10%）**从头到尾不让它见**：270 道背着练，30 道考完用来验货。这 30 道"新题"的分数 = **eval_loss**。数学上就是 Day11 讲的**留出法（hold-out）**——交叉验证思想的最小实现。

两个 loss 怎么读（训练日志里交替出现）：


| 现象                | 含义        | 怎么办                   |
| ----------------- | --------- | --------------------- |
| train↓ 且 eval↓    | 真学会了      | 继续训                   |
| train↓ 但 eval 回升  | 开始背题（过拟合） | 提前停 / 减 epoch / 调低 lr |
| train↓ 但 eval 一直高 | 数据本身太偏    | 回 Day11 补数据质量         |


**③ 梯度累积——显存小，攒几步"探路结果"再迈一步。**

一步更新 = 先喂一小撮数据算出梯度。显存只够一次喂 **2 条**（`BATCH_SIZE=2`），但只看 2 条就改参数太"以偏概全"——方向会抖。于是攒够 8 小步再正式迈一步：

```
第 1 小步：喂 2 条 → 算出梯度 g₁     ← 只存梯度，不更新参数
第 2 小步：喂 2 条 → 算出梯度 g₂
   ...（共 8 小步）
第 8 小步：喂 2 条 → 算出梯度 g₈
─────────────────────────────
攒齐 → 8 个梯度相加，正式更新一次参数
```

**等效 batch = 2 × 8 = 16**：每次更新相当于"看了 16 条数据"，方向更稳。

**为什么这么干能省显存？** 不攒 = 一次喂 16 条 = 显存同时装 16 条的激活值 → 6G 爆；攒梯度 = 每次只喂 2 条 = 显存永远只装 2 条的量，算完梯度就释放。**代价只是慢一点，换来等效 batch=16 的稳定性**。

> 常考的一个坑：**梯度累积 ≠ 增大 batch 的显存**。真 batch=16 是"同时"算 16 条的前向（峰值高）；梯度累积是"先后"算 2 条×8 次（峰值永远是 2 条的量）。区别就在同时 vs 先后。

---

**技术版（背给面试官听）：**

- **epoch = 把全部训练数据完整过一遍**。今天 3 epoch = 300 条数据被"读 3 遍"；
- **验证集 = 留出法**（Day11 统计八股的落地）：每 epoch 末算一次 eval_loss。eval_loss 比训练 loss 高一点是正常的（模型没见过验证题）；若 eval 回升而 train 猛降 = 过拟合信号（减 epoch / 调小 lr）；
- **梯度累积 gradient_accumulation_steps**：显存只够 batch=2，但想让每步"看"更多数据 → 攒 8 步梯度再更新一次。`等效 batch = 2 × 8 = 16`。**这是显存不够时的"数学凑法"，不额外占显存**。



### 1.6 只存 adapter，推理时合入权重

- 训练产物是 **adapter**（`adapter_config.json` + `adapter_model.safetensors`，几 MB~几十 MB），不是全量模型；
- **推理时合入**：`W_最终 = W0 + B·A`（PEFT 的 forward 自动把低秩增量加到原权重上）。所以 Day13 把 adapter 挂到 Day9 接口上，模型就"微调过了"——换的只是权重路径。



### ✅ 第 1 步验收标准

能讲清五件套每一件"是什么 + 为什么"；重点能讲"交叉熵 = -log P(答案 token)、为什么只算 output 部分"。

---



## 📜 第 2 步：读懂 `train_lora.py`（40~60 分钟）⭐

> 对应周计划 8/16 第 1、2 条。代码我已经写好，**先读再跑**。整体分 6 块：



### 2.1 整体结构（读的时候对照看）


| 区块            | 干什么                                                            | 对应五件套     |
| ------------- | -------------------------------------------------------------- | --------- |
| 文件头 docstring | 五件套清单 + 运行前提醒                                                  | 全         |
| 第 1 区 配置区     | 所有常量（模型路径 / R / ALPHA / 数据 / 学习率 / epoch…）                     | 全         |
| 第 2 区 加载底座    | `AutoModelForCausalLM` + `BitsAndBytesConfig(4bit nf4)`        | 模型        |
| 第 3 区 挂 LoRA  | `LoraConfig` + `get_peft_model` + `print_trainable_parameters` | 模型        |
| 第 4 区 数据处理    | `load_dataset` → 90/10 切分 → ChatML 拼接 → **标签掩码**               | 数据        |
| 第 5 区 Trainer | `TrainingArguments` + `Trainer`（自动前向/反向/更新/日志/验证）              | 损失+优化器+训练 |
| 第 6 区 收尾      | `train()` → 显存报告 → `save_pretrained`（**只存 adapter**）           | 训练        |




### 2.2 关键 API（新名词必带简介）


| API                            | 一句话简介                                                          |
| ------------------------------ | -------------------------------------------------------------- |
| `BitsAndBytesConfig`           | 4bit 量化配置（Day6/Day9 同款：nf4 + double quant），让 3B 底座只占 ~1.9GB 显存 |
| `LoraConfig`                   | LoRA 的"参数配置单"：r（秩）/ alpha（缩放）/ dropout / target_modules（挂哪几层）  |
| `get_peft_model`               | 把底座包成"只训练 adapter"的外壳——底座冻结，B·A 可训练                            |
| `print_trainable_parameters()` | 打印"可训练参数 / 总参数 / 占比"——参数账的官方出处                                 |
| `Trainer`                      | HuggingFace 的训练管家：自动做前向、反向、梯度更新、loss 打印、epoch 末验证              |
| `TrainingArguments`            | 训练参数（batch / 学习率 / epoch / 验证策略 / 混合精度…）                       |
| `DataCollatorForSeq2Seq`       | 自动 padding（对齐 batch 内长度）+ 生成 attention_mask                    |
| `save_pretrained`              | 保存 adapter（在 PeftModel 上调用 = 只存增量，不存全量）                        |




### 2.3 数据怎么变 token（第 4 区精读）

```
(instruction, input, output)
   │  ① apply_chat_template 拼 ChatML
   ▼
<|im_start|>user\n<指令>（材料：<input>）<|im_end|>
<|im_start|>assistant\n<output><|im_end|>
   │  ② 分词 + 标签掩码
   ▼
input_ids = [题干 tokens] + [答案 tokens]
labels    = [-100, ..., -100] + [答案 tokens]   ← 题干部分 -100 = 不参与 loss
```

**自测题**（答案在代码注释里，先自己想再看）：

1. `TARGET_MODULES` 是干嘛的？为什么要把注意力四件套和 MLP 都挂上？（→ LoRA 只改指定线性层，挂得越多可学参数越多、效果越好；本脚本 7 个全挂）
2. 为什么 `labels` 里题干位置填 -100？（→ 让损失只对答案打分；-100 是 PyTorch 交叉熵的"忽略标记"）
3. `GRAD_ACCUM = 8`，等效 batch 是多少？（→ 2 × 8 = 16）
4. `eval_strategy="epoch"` 什么意思？（→ 每个 epoch 末在验证集上算一次 eval_loss 并打印）
5. 为什么 `model.save_pretrained` 存出来只有几 MB？（→ PeftModel 只存 B·A 增量 + 配置，不存 4bit 底座）



### 2.4 参数账（第 3 区会打印，先算一遍）

Qwen2.5-3B 有 36 层、隐藏维 2048。挂 r=8 的 LoRA 后，每个线性层增加 `2 × 2048 × 8 ≈ 3.3 万` 参数。**全部 7 类模块挂上 ≈ 1180 万参数 ≈ 总参数 3.09B 的 0.4% 左右**——以 `print_trainable_parameters()` 实际打印为准。

> 衔接 Day11 的账：Day11 里说的 0.084% 是"只挂注意力层（q/k/v/o 的简化口径）"下的数字；本脚本按官方推荐把 MLP 层也挂上（效果更好），占比约 0.4%。**两种口径都远小于 1%，"LoRA 只训不到 1% 参数"的结论不变**——这就是 6G 显存能训练的根本原因。



### 2.5 逐区块精读（Python 入门版）⭐ 本日代码阅读重点

> 本节按"从上到下的执行顺序"把脚本每个区块拆开讲。**阅读策略：先读** `main()`**（主流程），它是顺序执行的流水线，遇到不认识的函数再跳进去看。** 每个 Python 语法点都标注出来了。



#### 区块 0：文件头 docstring（第 1~20 行）

```python
# -*- coding: utf-8 -*-
"""
train_lora.py —— QLoRA 微调训练脚本...
【五件套】...
"""
```

- 第 1 行 `# -*- coding: utf-8 -*-`：告诉 Python"这个文件用 UTF-8 编码保存"（文件里有中文）。
- 第 2~20 行 `"""..."""` 是 **docstring（文档字符串）**：不是代码，是"说明书"，运行时不执行。开头写清五件套是总纲规定的规矩。
- Python 语法：`#` 开头 = 注释（整行被忽略）；`"""..."""` = 多行字符串（写在文件开头就自动成为文档）。



#### 区块 1：导入区（第 21~38 行）

```python
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model
```

- 第 24~26 行：**防中文乱码**。`hasattr(对象, "属性名")` = "判断对象有没有这个功能"，有才调用 `reconfigure` 把控制台输出改成 UTF-8。先判断再调用，旧版 Python 没这功能也不会报错（加固写法）。
- `import 包`：引入整个包，用 `包名.功能()`；`from 包 import 名字`：只引入一个名字，直接用 `名字()`。
- 第 30~37 行括号折行：括号内换行合法，是长 import 的标准写法。



#### 区块 2：配置区（第 40~72 行）

```python
MODEL_PATH = r"D:\...\Qwen2.5-3B-Instruct"
R = 8            # LoRA 秩
ALPHA = 16       # 缩放系数
DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
...
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
```

- **全大写命名 = 常量**（约定：值以后不改）。整个脚本的"旋钮"都在这，调参只改这个区——读代码先读这区，就知道脚本要干什么。
- `r"..."`：`r` = raw（原始字符串），反斜杠 `\` 原样保留。Windows 路径必须这么写。
- `TARGET_MODULES` 是**列表**（`[ ]`），装着要给哪些层挂 LoRA。



#### 函数 `mem_report`（第 75~79 行）—— 小工具：盯显存

```python
def mem_report(tag: str):
    """打印当前显存占用（字节 -> GB），全程盯显存用"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"[显存] {tag}：allocated={alloc:.2f} GB")
```

- `def 函数名(参数):`：**定义函数** = 打包一段可反复调用的代码。
- `tag: str`：类型标注（提示参数该是字符串，不强制）。
- `torch.cuda.memory_allocated()`：当前**已占用显存（字节数）**；`/ 1024 ** 3` 把字节除以 1024³ 换算成 GB。
- `f"..."`：**f-string**，`{}` 里的表达式被替换成真实值；`:.2f` = 保留两位小数。



#### 函数 `main`（第 82~205 行）—— 主流程，四步流水线

**第 1/4 步：加载 4bit 底座（第 84~105 行）**

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,               # 4bit 加载总开关（省显存核心）
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=quant_config, device_map="auto", torch_dtype=torch.float16,
)
model.config.use_cache = False      # 训练关 KV cache 省显存
mem_report("加载 4bit 底座后")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # 借用结束符当填充符
```

- `BitsAndBytesConfig`：4bit 量化配置，和 Day6/Day9 部署**完全同款**——保证基座加载方式一致（Day11 避坑表第 6 条：基座与训练时保持同款量化配置）。
- `from_pretrained(路径)`：从本地磁盘把模型读进显存；`device_map="auto"` 自动放 GPU/CPU。
- **tokenizer（分词器）**：负责"文字 ↔ 数字编号"互转。`pad_token is None` 表示没有"填充符"，就借用 eos（结束符）——否则 batch 里长度不齐的数据没法自动 padding，训练会报错。这是 transformers 的常规补丁写法。

**第 2/4 步：挂 LoRA 补丁（第 107~121 行）**

```python
lora_config = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT,
                         target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
```

- `LoraConfig`：LoRA 的"配置单"（只描述，不行动）。
- `get_peft_model(模型, 配置单)`：**真正动手**——冻结底座，给指定层挂上可训练小矩阵 B·A，返回 PEFT 模型。
- `print_trainable_parameters()`：打印参数账（`trainable%: 0.38` 从这里来）。
- `enable_input_require_grads` + `gradient_checkpointing_enable`：两个省显存技巧，梯度检查点 = 前向不存中间结果、反向时重算——**用"多算一遍"换"少占显存"**（6G 关键）。

**第 3/4 步：数据加载 + 转 token（第 123~158 行）⭐ 最需要仔细看**

```python
raw = load_dataset("json", data_files=DATA_PATH, split="train")
split = raw.train_test_split(test_size=VAL_RATIO, seed=SEED)
print(f"训练集 {len(split['train'])} 条，验证集 {len(split['test'])} 条")
```

- `load_dataset("json", data_files=...)`：读 JSON 数据（Day11 干跑同款）。
- `.train_test_split(test_size=0.1, seed=42)`：**切成 90% 训练 + 10% 验证**；`seed=42` 保证每次切分结果一致。

嵌套函数 `tokenize_batch`（第 130~155 行，Python 允许函数里定义函数，第 157 行才被调用）：

```python
def tokenize_batch(batch):
    prompts, full_texts = [], []
    for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
        user_content = f"{ins}\n\n材料：{inp}" if inp else ins
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)
        full_texts.append(prompt + out + tokenizer.eos_token)
    enc_p = tokenizer(prompts, truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False)
    enc_f = tokenizer(full_texts, truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False)
    labels = []
    for ids, pids in zip(enc_f["input_ids"], enc_p["input_ids"]):
        lab = ids.copy()
        mask = min(len(pids), len(ids))
        lab[:mask] = [-100] * mask
        labels.append(lab)
    return {"input_ids": enc_f["input_ids"], "attention_mask": enc_f["attention_mask"], "labels": labels}
```


| 语法点                                        | 在做什么                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------- |
| `[], []`                                   | 创建两个空列表                                                                               |
| `for a, b, c in zip(x, y, z):`             | `zip` 把三列数据**按位置配对**：第一条的 (instruction, input, output) 依次给 ins/inp/out                |
| `A if 条件 else B`                           | 三元表达式：input 非空就拼"指令+材料"，否则只有指令                                                        |
| `apply_chat_template(...)`                 | 自动拼 ChatML（`<                                                                         |
| `.append(x)`                               | 往列表末尾添加                                                                               |
| `prompt + out + tokenizer.eos_token`       | 字符串拼接：题干 + 标准答案 + 结束符                                                                 |
| `tokenizer(列表, ...)`                       | 文字列表 → 数字编号（token ID）；`truncation=True` 超 512 截断                                      |
| `ids.copy()`                               | 复制列表，改副本不动原数据                                                                         |
| `lab[:mask] = [-100] * mask`               | **标签掩码**：把前 mask 个（题干）位置全改成 -100；`[:mask]` 切片取前 mask 个，`[-100] * mask` 造一个全是 -100 的列表 |
| `return {"input_ids": ..., "labels": ...}` | 返回字典：输入编号 / 注意力掩码 / 标签（答案编号 + 题干 -100）                                                |


```python
train_ds = split["train"].map(tokenize_batch, batched=True, remove_columns=raw.column_names)
eval_ds = split["test"].map(tokenize_batch, batched=True, remove_columns=raw.column_names)
```

- `.map(函数)`：把函数套到每一条数据上；`batched=True` 一次处理一批（快）；`remove_columns=...` = 转完删掉原三列，只留新字段。

**第 4/4 步：训练（第 160~192 行）**

```python
args = TrainingArguments(
    output_dir=LOG_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE, warmup_ratio=WARMUP_RATIO, lr_scheduler_type=SCHEDULER,
    fp16=True, logging_steps=5, eval_strategy="epoch", save_strategy="no",
    seed=SEED, report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  tokenizer=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))
mem_report("训练开始前")
trainer.train()
mem_report("训练结束后")
metrics = trainer.evaluate()
print(f"[验证] 最终 eval_loss = {metrics.get('eval_loss'):.4f}")
```

- `TrainingArguments`：训练参数配置单（和配置区常量一一对应）。
- `Trainer`：**训练管家**——前向、反向、梯度更新、loss 打印、epoch 末验证全部自动做，不用手写训练循环。
- `DataCollatorForSeq2Seq(..., padding=True)`：把 batch 内长度不齐的序列自动补齐成矩阵。
- `trainer.train()`：开始训练；`trainer.evaluate()`：验证集算分，返回字典，`.get('eval_loss')` 取 eval_loss 数值。

**保存 adapter（第 194~205 行）**

```python
os.makedirs(OUTPUT_ADAPTER, exist_ok=True)
model.save_pretrained(OUTPUT_ADAPTER)
tokenizer.save_pretrained(OUTPUT_ADAPTER)
for fname in sorted(os.listdir(OUTPUT_ADAPTER)):
    size_mb = os.path.getsize(os.path.join(OUTPUT_ADAPTER, fname)) / 1024 / 1024
    print(f"  {fname:<40s} {size_mb:.2f} MB")
```

- `os.makedirs(路径, exist_ok=True)`：创建目录，已存在不报错。
- `model.save_pretrained(目录)`：**只存 LoRA 补丁**（PEFT 模型 save 的就是 adapter，不是全量模型）——参数高效的核心。
- 遍历打印文件大小：`os.listdir(目录)` 列出文件名；`os.path.join(路径, 名)` 拼完整路径；`os.path.getsize(路径)` 拿字节数 ÷1024÷1024 转 MB；`f"{fname:<40s}"` 左对齐占 40 字符（对齐好看）。



#### 入口（第 208~209 行）

```python
if __name__ == "__main__":
    main()
```

- 标准入口写法。`python train_lora.py` 直接运行时 `__name__` 是 `"__main__"` → 执行 `main()`；被别的脚本 `import` 时不自动跑。既能直接运行、又能被安全引用。



#### 本节 Python 语法速查（本脚本用到的全部语法）


| 语法                           | 例子                            | 意思                  |
| ---------------------------- | ----------------------------- | ------------------- |
| `#` 注释                       | `# 省显存`                       | 整行被忽略               |
| `"""..."""`                  | 文件头                           | 文档字符串（说明书）          |
| `import x`                   | `import os`                   | 引入整个包，用 `x.功能()`    |
| `from x import y`            | `from peft import LoraConfig` | 只引入一个名字             |
| 常量                           | `R = 8`                       | 全大写，约定不改            |
| `def f(a, b):`               | `def mem_report(tag: str):`   | 定义函数                |
| 类型标注                         | `tag: str`                    | 提示参数类型（不强制）         |
| `f"{x:.2f}"`                 | 打印显存                          | 格式化字符串，2 位小数        |
| `if ... :`                   | `if inp:`                     | 条件分支                |
| `for x in y:`                | `for fname in ...:`           | 循环遍历                |
| `zip(a, b, c)`               | 配三列数据                         | 按位置打包成对             |
| `A if 条件 else B`             | 拼材料                           | 三元表达式               |
| `.append(x)`                 | 收集题干                          | 往列表末尾添加             |
| `[:mask]` 切片                 | `lab[:mask]`                  | 取前 mask 个元素         |
| `[-100] * mask`              | 造掩码                           | 生成全是 -100 的列表       |
| `x.copy()`                   | 复制编号                          | 复制列表，改副本不动原数据       |
| 嵌套函数                         | `tokenize_batch`              | 函数里再定义函数            |
| 字典                           | `{"a": 1}`                    | 键值对集合，用 `d["a"]` 取值 |
| `.map(函数)`                   | 数据转换                          | 把函数套到每条数据上          |
| `if __name__ == "__main__":` | 入口                            | 直接运行时才执行            |




### ✅ 第 2 步验收标准

能指着脚本说出每个区块在干嘛、五件套对应哪几个常量、能答出上面 5 个自测题。

---



## 🏋️ 第 3 步：跑通训练（60~90 分钟）⭐ 今日核心

> 对应周计划 8/16 第 3 条："训练并观察：`torch.cuda.memory_allocated()` 盯显存（6G 预算内），观察 loss 下降；训完**只保存 adapter**（`save_pretrained` 到 `lora_adapter/`，不是全量模型）。"



### 3.1 开始训练

```
python train_lora.py
```

**你会看到**（形态示例，数字以你实际跑出为准——固定了 seed，同一台机器两次跑结果一致）：

```
============================================================
第 1/4 步：加载 4bit 底座模型（Qwen2.5-3B-Instruct）
============================================================
[显存] 加载 4bit 底座后：allocated=1.93 GB
============================================================
第 2/4 步：挂 LoRA 适配器（LoraConfig + get_peft_model）
============================================================
trainable params: 11,796,480 || all params: 3,096,xxx,xxx || trainable%: 0.38
============================================================
第 3/4 步：加载数据 + 切验证集 + 转 token（ChatML + 标签掩码）
============================================================
训练集 270 条，验证集 30 条
============================================================
第 4/4 步：开始训练（Trainer 托管）
============================================================
[显存] 训练开始前：allocated=2.10 GB
{'loss': 1.42, 'learning_rate': 0.00015, 'epoch': 0.11}
{'loss': 1.05, 'learning_rate': 0.00019, 'epoch': 0.22}
...
{'eval_loss': 0.96, 'epoch': 1.0}
{'loss': 0.81, 'epoch': 0.44}
...
{'eval_loss': 0.78, 'epoch': 2.0}
...
{'eval_loss': 0.72, 'epoch': 3.0}
[训练完成]
[显存] 训练结束后：allocated=2.15 GB
[验证] 最终 eval_loss = 0.72
保存 LoRA 适配器（只存增量，不存全量模型）
  adapter_config.json                0.00 MB
  adapter_model.safetensors          23.42 MB
  tokenizer.json                     6.52 MB
  tokenizer_config.json              0.00 MB
  ...
[完成] adapter 已保存到 lora_adapter/，接下来运行 compare_lora.py 做效果对比。
```



### 3.2 盯两样东西（训练时别发呆）

1. **loss 下降**：训练 loss 应逐行下降（1.4 → 1.0 → 0.8 → …）；每 epoch 末的 `eval_loss` 也应下降（1.0 → 0.8 → 0.7）。**eval_loss 比 train loss 略高是正常的**（验证题没见过）；若 eval 回升而 train 猛降 → 过拟合，下一步减小 epoch 或学习率；
2. **显存**：`[显存]` 三行对比——底座约 1.9GB，训练峰值不应超 6GB。若 OOM，看避坑表第 1 条（关程序 / 减 BATCH_SIZE / 减 MAX_SEQ_LEN）。



### 3.3 训练时长预期

300 条、3 epoch、等效 batch 16 → 约 60 个优化步。RTX 3060 6G 大概 **20~60 分钟**（以实际为准）。**等的时候去做零散任务**（力扣 56/75、交叉验证），别干瞪眼。

### 3.4 检查产物：只存了 adapter

```
dir lora_adapter
```

应看到 `adapter_config.json`（LoRA 配置：r / alpha / target_modules）+ `adapter_model.safetensors`（训练好的 B·A 增量，约 20~30MB）+ tokenizer 文件。**就这么点东西 = 今天全部成果**——没有几 GB 的全量模型，这就是"参数高效微调"。

### ✅ 第 3 步验收标准

训练正常结束、train/eval loss 都下降、`lora_adapter/` 只有 adapter（小而轻）。

---



## ⚖️ 第 4 步：微调前后效果对比（40~60 分钟）⭐

> 对应周计划 8/16 第 4 条："加载'原模型' vs '原模型 + adapter'，同一组 3~5 个测试问题对比输出 → 产出**微调前后效果对比表**。"



### 4.1 为什么"依次加载"而不是同时加载

6G 显存放不下两个 3B 模型。所以 `compare_lora.py` 先加载原模型跑完 5 个问题、释放显存，再加载"原模型+adapter"跑同一组问题——**保证两次提问完全一样，差异只可能来自微调**。

### 4.2 跑对比脚本

```
python compare_lora.py
```

脚本会打印每个问题两段输出，并自动生成 `微调前后效果对比表.md`。

**5 个测试问题**（覆盖 Day11 SFT 数据里的 5 类任务）：


| #   | 类别     | 测试问题                                        |
| --- | ------ | ------------------------------------------- |
| 1   | 公告信息抽取 | 请从以下埃斯顿的年度报告中，提取营业收入的数据。                    |
| 2   | 数据计算   | 汇川技术去年净利润 28.26 亿元，今年净利润 25.77 亿元，请计算同比增长率。 |
| 3   | 概念解释   | 请用通俗的语言解释什么是谐波减速器？                          |
| 4   | 公告要点总结 | 请总结以下绿的谐波公告的核心要点。                           |
| 5   | 合规判断   | 优必选拟向特定地区出口工业机器人，请问是否合规？                    |




### 4.3 手动补"差异分析"列

脚本生成的表格里"差异分析"列留空，**你手动补**。对比时看 4 个维度：

1. **格式**：微调后是否学会 ①②③ 分点；
2. **行话**：是否用上谐波减速器 / 减速比 / PID / 产能利用率等术语；
3. **计算步骤**：计算题是否学会"先列式 → 再代入 → 给结果"（Day11 特意教的）；
4. **口吻**：是否更"研报风"（评级、风险提示）。

**示例行**（真实内容以你跑出的为准）：


| #   | 类别   | 原模型回答                 | 微调后回答                                                                  | 差异分析                |
| --- | ---- | --------------------- | ---------------------------------------------------------------------- | ------------------- |
| 2   | 数据计算 | 同比增长率约为 -8.8%。（只给了结果） | 同比增长率 = (今年-去年)/去年 × 100% = (25.77-28.26)/28.26 × 100% ≈ -8.81%。（列式清晰） | ✅ 学会"先列式再代入"，计算过程完整 |




### 4.4 验收：至少 3 个问题有肉眼可见差异

- 差异不明显？按避坑表排查：数据质量 / epoch 太少 / 学习率太低 / 测试问题与训练分布差太远；
- **能解释"为什么有差异"**：微调把 `P(答案|问题)` 的分布往这 300 条数据上收了——模型在机器人行话、计算步骤、输出格式上"更像"这份数据了。



### ✅ 第 4 步验收标准

对比表完成、至少 3 个问题差异肉眼可见、能讲清差异的来源。

---



## ✅ 第 5 步：验收——五件套 + 账本 + 面试话术（20~30 分钟）



### 5.1 训练五件套背诵


| 件   | 今天的内容                                                         |
| --- | ------------------------------------------------------------- |
| 模型  | Qwen2.5-3B-Instruct，4bit 底座（冻结）+ LoRA r=8 alpha=16（7 类线性层）    |
| 数据  | 300 条机器人行业 SFT 数据，90/10 切验证集，ChatML 拼接，题干标签 -100              |
| 损失  | 交叉熵（LM loss）：最小化 -E[log P(output                              |
| 优化器 | AdamW，lr 2e-4，warmup 3% + cosine 退火                           |
| 训练  | 3 epoch，等效 batch 16（2×8 梯度累积），eval_strategy=epoch 看 eval_loss |




### 5.2 面试话术（背下来）

> "我用 QLoRA 微调了 Qwen2.5-3B-Instruct：4bit 量化底座冻结，只训练 LoRA 低秩增量（r=8，alpha=16，挂在注意力和 MLP 共 7 类线性层上），训练参数只占全模型的约 0.4%。数据是 Day11 自制的 300 条机器人行业 SFT 数据（三字段 instruction/input/output，6 类任务，90/10 切验证集）。损失是标准的 LM 交叉熵——只对标准答案部分计算，相当于最大化 P(答案|问题)。优化器用 AdamW，学习率 2e-4，3 个 epoch，等效 batch 16。训练时我用 torch.cuda.memory_allocated() 盯显存，6G 预算内跑完。训完只保存了 adapter（adapter_model.safetensors，约 20MB），推理时 PEFT 自动把 B·A 合回原权重。微调前后我拿同一组 5 个问题对比，至少在格式、术语、计算过程上有肉眼可见差异。"



### 5.3 账本更新（Day11 三张账 + 今天新增训练账）


| 账   | 数字                                              | 一句话                  |
| --- | ----------------------------------------------- | -------------------- |
| 数据账 | 300 条 / 三字段 / 6 类 / 0 重复                        | 微调的"食材"              |
| 参数账 | 全量 d² → LoRA 2dr，省 99.55%；训练参数 ≈ 0.4%（约 1180 万） | 只训 adapter，所以省显存     |
| 显存账 | 4bit 底座约 1.9GB，训练峰值 <6GB（实测）                    | 6G 显存跑 QLoRA 微调 OK   |
| 训练账 | 3 epoch / 等效 batch 16 / ~60 步 / eval_loss 0.7x  | 训练结束，adapter 只约 20MB |




### ✅ 第 5 步验收标准

五件套能背、面试话术能讲、四张账能算 = 今天主线全部完成 🎉

---



## 🏃 零散时间任务（0.5~1 小时，穿插在休息/训练等待时做）

> 按第三周计划 8/16：力扣 **56、75**（排序应用）+ 统计八股：**交叉验证**（K 折 / 留一法，衔接 RAG 评测）。



### A. 力扣 56. 合并区间（中等 · 排序 + 贪心）⭐ 必做

**题目**：以数组 `intervals` 表示若干个区间的集合，合并所有重叠区间。

**思路**：先按左端点排序，再贪心扫描——若当前区间与"已合并的最后一个"重叠（`a <= res[-1][1]`）就并进去（右端点取大），否则开新区间。

**数学视角**：**排序让"重叠区间必相邻"**——排完序后贪心只需和上一个比较。复杂度 O(n log n)（排序主导）。"排序 + 贪心"是区间类问题的黄金组合，面试高频。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])          # 1. 按左端点排序
        res = []
        for a, b in intervals:                      # 2. 贪心扫描
            if not res or a > res[-1][1]:           # 与上一个不重叠
                res.append([a, b])
            else:                                   # 重叠 -> 右端点取大
                res[-1][1] = max(res[-1][1], b)
        return res
```

**复杂度**：O(n log n) 时间、O(n) 空间（存答案）。

### B. 力扣 75. 颜色分类（中等 · 三指针/荷兰国旗）⭐

**题目**：只含 0、1、2 的数组原地排序（不许用内置排序，O(1) 额外空间、一趟扫完）。

**思路**：荷兰国旗三指针。维护三个区间的不变量：`[0, l)` 全是 0、`[l, i)` 全是 1、`(r, n]` 全是 2，`i` 一路扫过去把元素归位。

**数学视角**：**不变量驱动算法**——每一步都保持三段区间性质不变。注意碰到 2 时 `i` 不前进（交换来的值还要再判一次），这是最容易写错的地方。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        i, l, r = 0, 0, len(nums) - 1   # 三指针：l 是 0 区右界，r 是 2 区左界
        while i <= r:
            if nums[i] == 0:            # 归到左边 0 区
                nums[l], nums[i] = nums[i], nums[l]
                l += 1; i += 1
            elif nums[i] == 2:          # 归到右边 2 区（i 不动，再判一次）
                nums[r], nums[i] = nums[i], nums[r]
                r -= 1
            else:                       # 1 留在中间
                i += 1
```

**复杂度**：一趟 O(n)、O(1) 额外空间——"一趟扫描 + 常数空间"的经典。

**做完两题**：提交通过后存入 GitHub `leetcode` 仓库（`排序` 分类），题解注释写"数学视角"（排序让重叠相邻 / 不变量三段分区）。

### C. 统计八股：交叉验证（10~15 分钟）

> 对应第三周计划 8/16："交叉验证：K 折 / 留一法，为什么需要验证集（偏差-方差权衡的实操落地）。"

- **为什么需要验证集**：模型超参数（学习率、epoch、LoRA 的 r）不能拿训练集自己评——训练集上的分数永远虚高（模型"背过"了）。需要一块"没见过的"数据评真实水平。今天训练脚本的 **90/10 验证集就是最简单的"留出法"（hold-out）**；
- **K 折交叉验证**：把训练集均分 K 份，轮流取 1 份当验证、其余 K-1 份训练，得到 K 个分数取平均。常用 K=5 或 10；
- **留一法（Leave-One-Out, LOO）**：K = N（样本数）的极端——每份只有 1 个样本。小数据上最充分，但计算贵 N 倍；
- **数学视角（偏差-方差权衡）**：K 大 → 每次训练数据多（低偏差），但 K 次训练数据高度重叠（相关性高，方差大）且贵；K 小 → 便宜但偏差大。**5~10 折是经验折中点**。

> 面试加分句式："交叉验证的本质是用'留出法'的重复版来同时评估性能和稳定性——我微调时先用 90/10 留出法看 eval_loss，Day14 写 eval.py 时会固定一套评测问题集当测试集，评估口径先定死再谈优化。"

---



## 🌙 睡前 15 分钟：收尾 + 存档（git commit）



### 8.1 汇总今天的产出

检查 day12 文件夹里应该有：


| 文件                             | 说明                                        |
| ------------------------------ | ----------------------------------------- |
| `day12-LoRA微调实战二跑通微调训练详细教程.md` | 本教程                                       |
| `train_lora.py`                | QLoRA 训练脚本（五件套写清文件头 + 三件套配置 + 只存 adapter） |
| `compare_lora.py`              | 微调前后对比脚本（依次加载两个模型 + 自动生成对比表）              |
| `sft_data.json`                | Day11 复制来的训练数据（300 条）                     |
| `lora_adapter/`                | 训练好的 adapter（约 20~30MB，**不进 git**）        |
| `微调前后效果对比表.md`                 | 对比脚本生成 + 你手动补差异分析                         |




### 8.2 把 lora_adapter 排除出 git

`lora_adapter/` 是训练产物、可以随时重训，**不进 git**。在仓库根目录 `.gitignore` 末尾追加两行：

```
# LoRA 微调适配器权重（Day12 训练产物，可重新训练生成，不进 git）
第三周/day12/lora_adapter/
```

> 若你之前已经 add 过，先执行 `git rm -r --cached 第三周/day12/lora_adapter` 再提交。



### 8.3 存档到 git

在仓库根目录执行：

```
cd "D:\Lan\研究生\技术学习\大模型算法"
git add .
git commit -m "Day12: QLoRA 微调训练跑通（Qwen2.5-3B-Instruct + r=8 adapter，300条机器人行业SFT数据，3 epoch，只存adapter）+ 微调前后效果对比表（compare_lora.py 生成）"
```

**你会看到**：新文件列表 + 存档成功提示 = 完成。`lora_adapter/` 不在列表里 = .gitignore 生效。

---



## 🚧 常见问题速查表（出问题先看这里）


| 现象                                           | 原因                              | 解决办法                                                                                                  |
| -------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 训练 OOM / 显存不够                                | 6G 不够（浏览器占显存 / batch 太大 / 序列太长） | 关浏览器等程序；`BATCH_SIZE=1`；`GRAD_ACCUM` 保持或调大；`MAX_SEQ_LEN` 降到 256                                        |
| `ModuleNotFoundError: No module named 'xxx'` | 环境没装全                           | 回 Day11 跑 `check_env.py`；缺哪个装哪个（`pip install peft` 等）                                                 |
| 加载模型报 bitsandbytes 相关错误                      | bitsandbytes 版本/兼容问题            | `pip install --upgrade bitsandbytes`；确认 0.43+                                                         |
| 训练 loss 不降或 NaN                              | 学习率太高 / 数据格式问题                  | lr 降到 1e-4；确认 `sft_data.json` 三字段齐全（跑 Day11 `check_sft_data.py`）                                      |
| eval_loss 一直 > train_loss 且不降                | 过拟合 / 数据量少                      | 减 epoch 到 2、lr 降到 1e-4；或回 Day11 扩语料池补数据                                                               |
| 误把 adapter 存成全量模型                            | 保存方式不对                          | 确认用 `model.save_pretrained`（PeftModel）；保存后文件只有几 MB，否则检查是否误存了 base_model                               |
| 加载 adapter 报错（shape 不匹配等）                    | 基座配置与训练时不一致                     | 加载时用与训练**同款 4bit 配置**；用 `PeftModel.from_pretrained(base_model, adapter)` 或 `AutoPeftModelForCausalLM` |
| 对比结果差异不明显                                    | 微调力度不够 / 问题太偏                   | 检查数据质量、epoch（可到 4~5）、学习率；选与训练数据分布更近的测试题                                                               |
| 终端中文乱码                                       | Windows 控制台编码                   | 运行前 `chcp 65001` 或 `python -X utf8 脚本.py`（脚本已内置 `sys.stdout.reconfigure` 加固）                          |
| 训练很久没动静                                      | 模型加载/训练本身慢                      | 正常，3B 训练 20~60 分钟；看 `[显存]` 和 loss 日志确认在跑                                                              |
| 今天时间不够                                       | 正常                              | 优先级：第 3 步训练 + 第 4 步对比必做；第 1 步概念可在零散时间补；第 5 步话术睡前背                                                     |


---



## ✅ 今日验收清单（完成一项打一个勾）

**主线：**

- [x] 能讲清训练五件套：模型（4bit 底座 + QLoRA r=8 alpha=16）/ 数据（300 条 + 90/10 验证集）/ 损失（交叉熵，只算答案）/ 优化器（AdamW lr 2e-4）/ 训练（3 epoch + eval_loss）
- [x] 能答出 2.3 的 5 个自测题（target_modules / -100 / 等效 batch / eval_strategy / adapter 为什么小）
- [x] `python train_lora.py` 跑通：显存 <6GB、train/eval loss 均下降
- [x] `lora_adapter/` 只有 adapter（adapter_config.json + adapter_model.safetensors，几 MB~几十 MB），**没有**几 GB 的全量模型
- [x] 能讲清"LoRA 只存 adapter、推理时合入权重 W0+B·A"与本周参数账/显存账
- [ ] `python compare_lora.py` 跑出对比表，**至少 3 个问题有肉眼可见差异**，能解释差异来源

**零散时间：**

- [ ] 力扣 56（合并区间）完成，能讲"排序让重叠相邻 + 贪心"
- [ ] 力扣 75（颜色分类）完成，能讲"三指针 + 不变量三段分区"
- [x] 统计八股：能讲 K 折 / 留一法 / 为什么需要验证集 + 偏差-方差权衡

**睡前：**

- [ ] `.gitignore` 已加 `第三周/day12/lora_adapter/`（adapter 不进 git）
- [ ] git commit 存档成功

**全部打勾 = Day 12 圆满结束。** 今天你**第一次真正训练了模型**——adapter 就是项目微调层的第一版产物。**Day13 将把它接入 Day9 接口，开始项目定型。** 🎉

---



## 📎 附录：脚本定位与 Day13 衔接


| 脚本                | 一句话定位                        | Day13 怎么用                        |
| ----------------- | ---------------------------- | -------------------------------- |
| `train_lora.py`   | QLoRA 训练脚本（五件套 + 只存 adapter） | 微调结果 `lora_adapter/` 是项目"风格/能力层" |
| `compare_lora.py` | 微调前后效果对比（自动生成对比表）            | 对比表是效果评估报告的素材                    |


**Day13 的剧本（预告）**：项目定性（机器人领域垂直 RAG 问答系统）→ 架构图（Qwen 生成层 + 微调 adapter 风格层 + `rag_demo/` 检索层 + 模板库 v2 答案组织层）→ **把微调模型接入 Day9** `local_api.py`（换模型路径 + `load_adapter`，确认 `/v1/chat/completions` 依旧兼容，curl 冒烟测试）→ 准备 20 条评测问题集。**今天训好的 adapter = 项目微调层的正式交付物。**