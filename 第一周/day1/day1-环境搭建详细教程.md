# Day 1（8/1 周六）· 环境搭建日 —— 超详细新手教程

> 时间安排：主线 3~~4 小时 + 零散时间 0.5~~1 小时 + 睡前 15 分钟
> 适用对象：电脑小白，我假设你只会用鼠标点开软件，不会用命令行
> 本教程的目标：把"能跑大模型的电脑环境"搭好，并验证你的 GPU 能干活

---

## 📌 今天你要做什么（大白话版）

打个比方：跑大模型就像**开餐厅**。

- **Python 环境** = 厨房（做饭的地方）
- **PyTorch 等库** = 锅碗瓢盆、燃气灶（工具）
- **GPU（显卡）** = 大厨的"好厨艺"——有了它上菜才快
- **虚拟环境** = 给厨房装个独立房间，不会把家里其他地方弄乱

今天你的任务就是：**把厨房搭好、把工具备齐、确认大厨（GPU）真的能上班**。

今天结束时的"验收标准"只有一句话：

> 在终端里运行一句 `python 命令`，电脑显示 `True`（意思是：GPU 可以用了），并且能成功让一个小模型"开口说话"。

---

## 🖥️ 第 0 步：学会打开"终端"（5 分钟）—— 这是你今天的必修技能

**什么是终端（也叫"命令行 / 命令提示符"）？**
就是你用一个黑色（或蓝色）小窗口，**用打字的方式指挥电脑干活**。以后 90% 的操作都要在这里完成，所以先学会怎么打开它。

**打开方法（二选一）：**

- 方法 A：按键盘上的 `Win` 键（就是画着窗户那个键），直接输入 **powershell**，然后按回车，会弹出一个蓝色窗口。
- 方法 B：按 `Win` 键 → 输入 **终端**（Windows 11 自带），回车。

> 💡 蓝色窗口 = Windows PowerShell，就是我们用的"终端"。
> 以后教程里写"在终端输入 XXX"，就是在这个窗口里打字，然后按回车。

**先学 3 个最常用的终端命令**（今天必须会，以后天天用）：


| 命令         | 作用           | 类比         |
| ---------- | ------------ | ---------- |
| `cd 文件夹路径` | 让终端"走进"某个文件夹 | 双击进入文件夹    |
| `dir`      | 看看当前文件夹里有什么  | 看一眼文件夹里的东西 |
| `cls`      | 把终端屏幕清空      | 擦黑板        |


> 💡 特别提醒：在终端里**按鼠标右键 = 粘贴**（Ctrl+V 在终端里不总好使，右键粘贴最稳）。

**练习一下**：输入下面这行，按回车，看看能不能进入我们今天的文件夹：

```
cd "D:\Lan\研究生\技术学习\大模型算法\第一周\day1"
```

- 如果能看到 `PS ...\day1>` 这样的提示，说明你成功了！✅
- 如果提示"找不到路径"，多半是路径打错了，或者路径里有引号问题——直接复制粘贴上面的命令，一字不差就行。

---



## 🔍 第 1 步：检查电脑上已有的软件（15 分钟）

你已经装了不少软件，我们一条条查，**看到什么说明有，看不到什么说明没有**。每查一条我都解释一下它在干什么。

打开终端，依次输入下面每一条命令并回车，对照"你会看到什么"：

### 1.1 检查 Python

**命令：**

```
python --version
```

- **这是在干什么**：问电脑"你装没装 Python？装的是哪个版本？"
- **你会看到**：`Python 3.14.4`
- **结论**：✅ 你装了 Python，而且很新。不过它**太新了**，很多 AI 库还没跟上，所以今天我们会再装一个"独立的小房间"（虚拟环境）来用，系统里的这个先不管它。



### 1.2 检查 pip（Python 的"安装管家"）

**命令：**

```
pip --version
```

- **这是在干什么**：pip 负责下载和安装 Python 的各种库，相当于手机上的"应用商店"。
- **你会看到**：`pip 26.0.1 from C:\Users\...`
- **结论**：✅ 有。



### 1.3 检查 Anaconda（今天的关键）

**命令：**

```
conda --version
```

- **这是在干什么**：conda 是管理 Python 环境的大管家（也就是前面说的"独立房间"）。计划里要求装 Anaconda，它自带的这个管家叫 conda。
- **你会看到**：`'conda' 不是内部或外部命令...`（报错）
- **结论**：❌ **没装**。这是今天**唯一需要新装的大软件**，第 2 步就装它。



### 1.4 检查 Git（版本管理工具，用来保存你的学习进度）

**命令：**

```
git --version
```

- **这是在干什么**：Git 像"游戏存档"，每天学完可以把当天成果存档，以后随时可以回退到任意一天的进度。
- **你会看到**：`git version 2.54.0.windows.1`
- **结论**：✅ 有。



### 1.5 检查 VS Code（写代码的编辑器）和 Jupyter（交互式笔记本）

**命令：**

```
code --version
```

- **你会看到**：一堆版本号数字。
- **结论**：✅ VS Code 已装。

**命令：**

```
jupyter --version
```

- **你会看到**：`jupyterlab 4.5.7` 等一堆版本号。
- **结论**：✅ Jupyter 已装。（Jupyter 以后会常用来写大模型的练习笔记）



### 1.6 检查你的显卡（GPU）—— 这是你最值钱的硬件

**命令：**

```
nvidia-smi
```

- **这是在干什么**：问电脑"你的 NVIDIA 显卡在不在？驱动装好没有？"
- **你会看到**：
  - 第 2 行写：`NVIDIA GeForce RTX 3060 Laptop GPU`（你的显卡型号）
  - 第 2 行末尾写：`CUDA Version: 12.3`（驱动支持的最高 CUDA 版本，CUDA 是让显卡干活的语言）
  - 还有一行：`1659MiB / 6144MiB`（显卡内存用量 / 总量，6GB 显存）
- **结论**：✅ 显卡完全正常，驱动已装好。

> ⚠️ **重要发现**：你的显存是 **6GB**，而周计划里写的是"8G+ 显存"。6GB 也能跑，但第 2 周部署 Qwen2.5-7B 时会比较紧张，到时候我们要么用更小的模型（比如 Qwen2.5-3B），要么加几个参数硬跑。**今天不用管，先记着这个事，第 5 天我们再商量。**



### 1.7 检查 AI 核心库装没装（PyTorch、transformers）

**命令：**

```
pip show torch
```

- **你会看到**：如果没装，会提示 `WARNING: Package(s) not found` 或者一堆报错。
- **结论**：❌ 没装，今天装。

**命令：**

```
pip show transformers
```

- **结论**：❌ 也没装，今天装。



### 📋 第 1 步总结表（你可以截图保存）


| 检查项          | 命令                      | 你的结果                      | 结论               |
| ------------ | ----------------------- | ------------------------- | ---------------- |
| Python       | `python --version`      | Python 3.14.4             | ✅ 有（太新，今天不用它）    |
| pip          | `pip --version`         | pip 26.0.1                | ✅ 有              |
| conda        | `conda --version`       | 报错"不是内部或外部命令"             | ❌ 要装 ⬅️今天唯一的软件安装 |
| Git          | `git --version`         | git 2.54.0                | ✅ 有              |
| VS Code      | `code --version`        | 版本号                       | ✅ 有              |
| Jupyter      | `jupyter --version`     | jupyterlab 4.5.7          | ✅ 有              |
| GPU 驱动       | `nvidia-smi`            | RTX 3060 Laptop，CUDA 12.3 | ✅ 正常             |
| PyTorch      | `pip show torch`        | 未找到                       | ❌ 要装             |
| transformers | `pip show transformers` | 未找到                       | ❌ 要装             |


---



## 📦 第 2 步：安装 Miniconda（30~40 分钟，唯一要新装的大软件）



### 2.1 为什么要装它

你的电脑已经有 Python 了，但那个 Python 是"3.14 新版"，太新容易跟 AI 库打架，而且**直接在系统里乱装库，装坏了很难收拾**（就像在客厅乱堆杂物）。

Miniconda 能帮你：

1. **建独立房间**（虚拟环境）——每个项目一个房间，互不打扰，装坏了直接拆掉重建，不伤系统；
2. 方便地给每个房间指定 Python 版本（我们要的是 3.11，稳妥好使）。

> 📌 计划里写的是"安装 Anaconda"。Anaconda 和 Miniconda 是**一家公司的两个产品**：Anaconda 自带几百个库（又大又重，约 1GB），Miniconda 只带最核心的管家（约 80MB，轻快）。**我们只需要管家 + 自己装库，所以选 Miniconda，省时间省硬盘**。核心命令完全一样，学会 Miniconda = 学会 Anaconda。



### 2.2 下载与安装

1. 打开浏览器，访问官网下载页：
  **[https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)** （这是 Miniconda 下载页，往下滚找 "Miniconda installers"，下载 **Windows** 的 **64-bit** 那个 `.exe` 文件）
  > 也可以直接用这个镜像链接（国内更快）：`https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe`
2. 双击下载好的 `.exe` 文件开始安装。
3. 一路点 **Next**（下一步），但**注意三个关键地方**：
  - 安装路径随意（默认即可，但**记一下你装到哪了**。你的机器上装在了 `D:\miniconda1`，下文提到 Miniconda 路径的地方我都按这个写，如果你装到别处就换成你的路径）；
  - **勾选** `Add Miniconda3 to my PATH environment variable`（把管家加入系统路径，这一步很重要，不然终端找不到 conda）；
  - 其余选项默认即可。
4. 安装完成后，**必须把终端窗口全部关掉，重新打开一个新终端**（不重开的话，电脑还没"记住"刚装的 conda）。
5. 在新终端里验证：

```
conda --version
```

- **你会看到**：`conda 25.x.x` 之类。
- 如果还是提示"不是内部或外部命令"，说明第 3 步那个勾没勾上，最省事的办法：**重装一遍 Miniconda，这次记得勾上**。



### ✅ 第 2 步验收标准

新终端里 `conda --version` 能显示版本号 = 成功。

---



## 🏠 第 3 步：创建虚拟环境 `llm`（10 分钟）

> ✅ **温馨提示：这一步你已经完成了**（2026-08-01 我已帮你把 llm 环境建好，Python 3.11.15，在 `D:\miniconda1\envs\llm`）。
> **不用重新创建环境！** 你只需要从下面 3.2 的"进入环境"开始操作即可。

现在管家有了，我们让它建一个叫 `llm` 的"独立房间"，并指定用 Python 3.11（这个版本跟所有 AI 库都兼容，最稳）。

### 3.1 创建环境

在终端输入（一条命令，一次回车）：

```
conda create -n llm python=3.11 -y
```

**每个词是什么意思**（讲给你听，不用背）：

- `conda` = 呼叫管家
- `create` = 创建
- `-n llm` = 给房间起名叫 llm
- `python=3.11` = 这个房间里装 Python 3.11
- `-y` = 中途问"确定吗？"时自动答"是"，省得你手动按

**你会看到**：一长串下载过程，最后显示 `done` 或 "To activate this environment, use..." 字样。

> ⚠️ 如果你自己从头建，可能会遇到 `CondaToSNonInteractiveError`（服务条款报错），先执行 `conda tos accept` 再重跑这条命令即可（详见文末"实际遇到的报错"）。



### 3.2 进入这个房间（激活环境）

```
conda activate llm
```

- 命令输入后，你会看到**行首变成了** `(llm)`，像这样：
`(llm) PS C:\...>`
- `(llm)` 出现 = 你现在在 llm 房间里，后面装的所有东西都只装在这个房间，系统其他地方干干净净。

> ⚠️ 以后每次开新终端、想用 AI 环境，都要先敲 `conda activate llm`。忘了就会装错地方。



### 3.3 验证房间里的 Python 版本

```
python --version
```

- **你会看到**：`Python 3.11.x` ✅（注意：不是之前的 3.14，因为现在在 llm 房间里了）



### ✅ 第 3 步验收标准

行首出现 `(llm)` 且 `python --version` 显示 3.11 = 成功。

---



## 🔥 第 4 步：安装 GPU 版 PyTorch（30~60 分钟，下载最大的一步）



### 4.1 为什么要装 GPU 版

你的显卡（GPU）是大厨，CPU（电脑主脑）是普通员工。跑大模型时，**GPU 版 PyTorch 让显卡亲自下厨，速度能快几十倍**。如果装成 CPU 版，小模型还能勉强跑，7B 大模型基本跑不动。所以一定要装 GPU 版。

> 安装前保证终端行首有 `(llm)`（还不行就敲 `conda activate llm`）。



### 4.2 安装命令（联网 5~20 分钟，请耐心等）

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**这条命令在干什么**：

- `pip install` = 用"应用商店"安装
- `torch==2.5.1` 等三个包 = 要装的三个包，`==版本号` 表示固定版本（torch 是核心，torchvision / torchaudio 是配套工具，必须跟 torch 配套）
- `--index-url ...` = 去 PyTorch 官方仓库下载，`cu121` 表示 **CUDA 12.1 的 GPU 版**
- **为什么不用更新的 cu124**：你的显卡驱动版本是 546.30，最高支持 CUDA 12.3。cu124 需要驱动 551.61 以上，装了你电脑反而用不了 GPU。cu121 只需要驱动 ≥ 531.14，**这是你这台电脑最稳妥的组合**。

> 💡 下载包大约 2.6GB，如果很慢或断掉，**重新运行一次同样的命令即可**，pip 支持断点续传（接着下）。



### 4.3 立刻验证 GPU 能不能用（兴奋时刻！）

```
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('GPU 可用:', torch.cuda.is_available()); print('显卡:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

- **这是在干什么**：让 Python 问 torch"显卡能用吗？"
- **你会看到**：

```
PyTorch 版本: 2.5.1+cu121
GPU 可用: True
显卡: NVIDIA GeForce RTX 3060 Laptop GPU
```

- 看到 `GPU 可用: True` 就是大成功！🎉 记住这个 `True`，这就是周计划验收清单里的第一项。

> ⚠️ 如果显示 `False`：先别慌，最常见的三种原因——
>
> 1. 装成了 CPU 版（看版本号是不是带 `+cpu`）。重新装一次 4.2 的官方命令。
> 2. 装成了 CUDA 版本太新的包（版本号带 `+cu124` 及以上）。这台电脑的驱动最高只支持 CUDA 12.3，必须用 cu121 版本，重跑 4.2 的命令。
> 3. 显卡驱动太旧。到时候把 `nvidia-smi` 的截图发给我，我们远程解决。



### ✅ 第 4 步验收标准

`torch.cuda.is_available()` 返回 `True` = 成功。

---



## 📚 第 5 步：安装 AI 核心库（15~30 分钟）



### 5.1 这些库都是干嘛的（一句话认识它们）


| 库              | 作用（大白话）                   | 哪周用         |
| -------------- | ------------------------- | ----------- |
| `transformers` | 大模型的"通用遥控器"，加载、使用各种大模型全靠它 | 第 2/4/5/6 天 |
| `accelerate`   | 让大模型跑得又快又不爆显存的加速器         | 第 2 周       |
| `datasets`     | 下载和管理训练数据的"资料库"           | 后面微调用       |
| `langchain`    | 组装 AI 应用的"乐高积木"，连接模型和工具   | 第 2 周 RAG   |
| `chromadb`     | 向量数据库，AI 的"记忆仓库"          | 第 2 周 RAG   |
| `bitsandbytes` | 4bit 量化工具，把大模型"压缩"到小显存能跑  | 第 2 周       |
| `jupyter`      | 交互式笔记本，写练习笔记用             | 天天用         |




### 5.2 安装方式（推荐用我准备好的清单文件）

我已经帮你在 `day1` 文件夹里放好了一个 `requirements.txt` 文件（它就是个"购物清单"，列出所有要装的库）。

**先确保在 llm 环境里**（行首有 `(llm)`），然后输入：

```
cd "D:\Lan\研究生\技术学习\大模型算法\第一周\day1"
pip install -r requirements.txt
```

**这条命令在干什么**：

- `cd ...` = 走进 day1 文件夹（找清单文件）
- `pip install -r requirements.txt` = 照着清单把库全装上

> ⏳ 这一步要下载几个大包（尤其 bitsandbytes），5~20 分钟，耐心等。中途失败就重跑一遍这条命令。



### 5.3 全部装完，验证一下清单上的库是否都在

```
pip list
```

- 你会看到一长串库名和版本号。往里面找找 `torch`、`transformers`、`langchain`、`chromadb`、`accelerate`、`datasets`、`bitsandbytes` 这几个名字，**都在 = 成功**。



### ✅ 第 5 步验收标准

`pip list` 里能看到上面 7 个关键库 = 成功。

---



## 🧪 第 6 步：大验收——让小模型"开口说话"（15~20 分钟）

这是今天最高光的时刻！我们要运行一个验收脚本：

1. 再确认一次 GPU 可用；
2. 在 GPU 上做一次真实的小计算；
3. **加载一个迷你大模型，让它真的生成一句话**。

> 这个迷你模型（tiny-random-gpt2）约 34MB，是"发动机点火测试"用的，不追求说得漂亮（它是随机初始化的参数，生成的文字可能是乱码，**这完全正常**），**能跑通就说明整个链路 OK**，等第 2 周我们再换真正的中文大模型 Qwen。



### 6.1 运行我准备好的验收脚本

确保 `(llm)` 环境 + 在 day1 文件夹里，输入：

```
python test_env.py
```

**第一次运行**，程序会自动下载 tiny-random-gpt2 模型（约 34MB，很快），然后开始生成。

**你会看到**（大概长这样）：

```
PyTorch 版本: 2.x.x+cu124
CUDA 是否可用: True
GPU 名称: NVIDIA GeForce RTX 3060 Laptop GPU
GPU 计算测试: [2.0, 4.0, 6.0]

生成中... 请稍候
迷你模型生成结果: Hello, I am a robot that ...
环境验收通过！🎉
```

看到 `环境验收通过！🎉` = **今天的核心任务完成**！

### 6.2 如果遇到下载慢/失败

国内下载 HuggingFace 模型偶尔很慢，先设一下国内镜像（一行命令，设完再跑一次 `python test_env.py`）：

```
set HF_ENDPOINT=https://hf-mirror.com
```

> 注意：这个设置只对当前终端窗口有效，下次新开终端要重新设。第 2 周部署 Qwen 时我们再系统处理镜像问题。



### ✅ 第 6 步验收标准

看到 `环境验收通过！🎉` = 成功。

---



## 💻 第 7 步：配置 VS Code 认准 llm 环境（20 分钟）

以后写代码基本都在 VS Code 里。我们要让 VS Code 知道你装好的 `llm` 环境（不然它默认用系统那个 3.14 的 Python，白搭）。

1. 打开 VS Code（开始菜单搜 `Visual Studio Code`）。
2. 点击菜单栏 **文件 → 打开文件夹**，选择 `D:\Lan\研究生\技术学习\大模型算法\第一周`。
3. 首次打开会提示"安装推荐的扩展"，点 **安装**（或者点左侧竖排第 5 个图标"扩展"，搜索 **Python**，点安装 Microsoft 出品的那个）。
4. 打开任意一个 `.py` 文件，看右下角显示的解释器版本。如果是 3.11 就不用管；如果不是：
  - 按快捷键 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，回车；
  - 在列表里选**带** `llm` **字样的那个**（你的机器上路径是 `D:\miniconda1\envs\llm\python.exe`）。
5. 再打开 VS Code 里自带的终端（菜单栏 **终端 → 新建终端**），输入：

```
conda activate llm
```

- 确认行首出现 `(llm)`，以后在 VS Code 里写代码、跑程序都在这个环境里。

> 💡 **以后每天开 VS Code，第一步就是：新建终端 → 敲** `conda activate llm`**。** 记不住就把这句话贴在手边。

---



## 🏃 零散时间任务（0.5~1 小时，穿插在休息时做）

今天大块时间给环境，零散时间做两件事：

### A. 力扣刷题 2 道（哈希入门）

去 [https://leetcode.cn](https://leetcode.cn) 注册/登录，做今天的两道题：

1. **1. 两数之和**（几乎每次笔试都出现，必做）
2. **217. 存在重复元素**（简单题）

**做题方法（重要）**：

- 先**自己想 10 分钟**，哪怕想不出来也要想（这 10 分钟才是涨功力的部分）；
- 再看官方题解，看它用什么思路（今天都是"哈希表"思路）；
- 看懂后，**用自己的话把思路讲一遍**（讲给空气听也算）——面试考的就是这个；
- 提交通过后，把代码和思路注释存起来，为后面建 leetcode 仓库做准备（今天先不用建）。

**简单解释一下"哈希表"是什么**：可以理解成一个"翻牌台"。你想知道"某个数字以前见没见过"，不用挨个翻旧牌，直接把牌放到对应的格子，一问就知道在不在——**查找速度极快**。两数之和就是"我缺谁，去牌台上翻谁"。

### B. Python 语法速查（今天补 3 个点）

今天的三个概念：**函数定义与调用、列表、字典**。

- 函数：`def 名字(参数):` 然后写功能，像给电脑写"指令卡片"，用的时候叫名字就执行。
- 列表：`a = [1, 2, 3]`，一排放东西的格子，用 `a[0]` 取第 1 个。
- 字典：`d = {"name": "张三"}`，带标签的格子，用 `d["name"]` 取值——**字典就是 Python 里的"哈希表"，刷题里天天见**。

> 推荐快速资料：廖雪峰 Python 教程（[https://liaoxuefeng.com/books/python/），只读"函数""list""dict"这三节，10](https://liaoxuefeng.com/books/python/），只读"函数""list""dict"这三节，10) 分钟搞定，不用细读。

---



## 🌙 睡前 15 分钟：收尾 + 存档（git commit）

每天结尾固定动作，把今天的成果存进"游戏存档"。

### 8.1 生成依赖清单（今天的产出）

在 `(llm)` 环境的终端里，输入：

```
cd "D:\Lan\研究生\技术学习\大模型算法\第一周\day1"
pip freeze > requirements.txt
```

- `pip freeze` = 把当前环境所有已装的库和版本列出来
- `> requirements.txt` = 把列出来的内容存进这个文件
- 这样以后任何人（或未来的你）拿到这个文件，一条命令就能复刻你的环境。

> 我之前帮你放的 requirements.txt 是"购物清单"（装之前用），这个新生成的是"购买记录"（装完之后），两个不冲突，用新的覆盖即可。



### 8.2 存档到 git（第一次 commit）

在终端输入：

```
cd "D:\Lan\研究生\技术学习\大模型算法"
git status
git add .
git commit -m "Day1: 完成环境搭建，llm环境+GPU版PyTorch+核心库，验证通过"
```

**每条命令在干什么**：

- `git status` = 看看有哪些文件变了（新文件会显示为红色/未跟踪）
- `git add .` = 把今天的成果放进"待存档区"
- `git commit -m "说明"` = 正式存档，`-m` 后面是这次存档的备注（以后翻存档就能看懂那天干了啥）

**你会看到**：一堆新文件的名字 + `1 file changed...` 之类的提示 = 存档成功。

> 如果第一次提示要设置用户名邮箱（`git config --global user.name "你的名字"`），照着提示设一次即可，永久生效。
> 如果 `git add .` 之前仓库里还没有东西，可能需要先 `git init` 一次，报错的话把终端信息截图发我。



### 8.3 写明日计划（3 分钟）

用记事本新建一个 `day2-计划.md`（放第一周文件夹），写下：

- 明日主题：Transformer 图解入门
- 预计完成的 3 件事（看 `01-第一周详细计划.md` 里 8/2 的内容）
- 明天零散时间刷的题：26、27

> 不用写长，几行就行，重点是"养成习惯"。

---



## 🚧 常见问题速查表（出问题先看这里）


| 现象                                    | 原因                         | 解决办法                                                       |
| ------------------------------------- | -------------------------- | ---------------------------------------------------------- |
| `conda` 不是内部或外部命令                     | Miniconda 没加进 PATH         | 重装 Miniconda，勾选 "Add to PATH"                              |
| `CondaToSNonInteractiveError`（服务条款报错） | 新版 Miniconda 要求先同意官方源的服务条款 | 终端输入 `conda tos accept`，然后重新创建环境                           |
| `python --version` 显示 3.14.x          | 你看的是系统自带 Python，版本会被系统自动更新 | 没关系，先 `conda activate llm`，再 `python --version` 就该是 3.11 了 |
| 行首没有 `(llm)`                          | 环境没激活                      | 敲 `conda activate llm`                                     |
| `torch.cuda.is_available()` 返回 False  | 装了 CPU 版 torch             | 重跑第 4.2 步官方命令                                              |
| pip 安装很慢/超时                           | 网络问题                       | 重跑同一条命令（支持续传），或稍后再试                                        |
| 下载模型失败                                | HuggingFace 国内访问慢          | `set HF_ENDPOINT=https://hf-mirror.com` 后再跑                |
| `ModuleNotFoundError`                 | 环境不对或库没装                   | 确认行首有 `(llm)`，然后重跑 `pip install -r requirements.txt`       |
| VS Code 里 import torch 报错             | 解释器没选 llm 环境               | 按第 7 步重新选解释器                                               |


---



## ✅ 今日验收清单（完成一项打一个勾）

- [ ] 会用终端，能 `cd` 到指定文件夹
- [ ] `conda --version` 能显示版本号（Miniconda 装好）
- [ ] `conda activate llm` 后行首出现 `(llm)`，`python --version` 是 3.11
- [ ] `torch.cuda.is_available()` 返回 **True**（GPU 可用）✅ 周里程碑第 1 项
- [ ] `pip list` 里能看到 transformers / langchain / chromadb / accelerate / datasets / bitsandbytes
- [ ] `python test_env.py` 跑出"环境验收通过！🎉"并**截图**
- [ ] VS Code 能认出 llm 环境
- [ ] 力扣完成 1、217 两道题，能讲出思路
- [ ] `pip freeze > requirements.txt` 生成了依赖清单
- [ ] git commit 存档成功
- [ ] 写了明日计划（day2）

**全部打勾 = Day 1 圆满结束，你已经比 90% 的人更接近"能跑大模型的电脑"了！** 🎉

---



## 📎 附录：我（AI）帮你预检测的结果（2026-07-31）

以下是我替你先跑了一遍检查命令后，你电脑的真实情况（写教程用的，供对照）：


| 检查项          | 命令                      | 我检测到的结果                                            | 结论             |
| ------------ | ----------------------- | -------------------------------------------------- | -------------- |
| Python       | `python --version`      | Python 3.14.4                                      | ✅ 有（太新，弃用）     |
| pip          | `pip --version`         | pip 26.0.1                                         | ✅ 有            |
| conda        | `conda --version`       | 命令不存在                                              | ❌ 需装 Miniconda |
| Git          | `git --version`         | git 2.54.0.windows.1                               | ✅ 有            |
| VS Code      | `code --version`        | 已安装                                                | ✅ 有            |
| Jupyter      | `jupyter --version`     | jupyterlab 4.5.7                                   | ✅ 有            |
| 显卡           | `nvidia-smi`            | RTX 3060 Laptop GPU（6GB 显存），驱动 546.30，支持 CUDA 12.3 | ✅ 正常           |
| torch        | `pip show torch`        | 未安装                                                | ❌ 需装           |
| transformers | `pip show transformers` | 未安装                                                | ❌ 需装           |


**额外提醒**：

1. 你系统里已经装了很多数据分析库（numpy / pandas / scikit-learn / matplotlib / jupyterlab），说明你之前用过 Python，基础不错，今天的环境搭建会很快上手。
2. 系统那个 Python 3.14 就别用了，**今天之后所有操作都在 llm 环境里做**。
3. 你的显卡是 3060 Laptop 6GB 版，第 2 周部署 Qwen 时我会帮你选合适方案（大概率用 Qwen2.5-3B 或 7B 的 4bit 量化+调参），现在不用操心。



### 📌 你现在的进度位置（先看这里）


| 教程步骤                   | 状态                                                          |
| ---------------------- | ----------------------------------------------------------- |
| 第 0 步 学用终端             | ✅ 你已经在用了（能自己 cd 进文件夹）                                       |
| 第 1 步 检查已有软件           | ✅ 完成                                                        |
| 第 2 步 安装 Miniconda     | ✅ 早就装好了（在 `D:\miniconda1`）                                  |
| 第 3 步 创建 llm 环境        | ✅ 已创建成功（Python 3.11.15）                                     |
| 第 4 步 安装 GPU 版 PyTorch | ✅ **已完成**（你装的是 2.5.1+cu121，GPU 可用=True）                     |
| 第 5 步 安装 AI 核心库        | ✅ 至少 transformers 已装好（你才能跑 test_env.py）；其余库请用 `pip list` 自查 |
| 第 6 步 验收脚本             | 🔄 **正在收尾**（GPU 全通过，模型报错已修好，等你重跑）                           |
| 第 7 步 配置 VS Code       | 🔄 终端激活问题已解决（方案 A）；还差"选解释器"这一步                              |
| 零散时间 + 睡前存档            | ⬜ 还没开始                                                      |


---



### 🐛 你实际遇到的报错 + 解决办法（如实记录，2026-08-01）

**报错 1：**`CondaToSNonInteractiveError: Terms of Service have not been accepted...`

- 现象：你执行 `conda create -n llm python=3.11 -y` 时，conda 直接报错拒绝，列出三个官方源（pkgs/main、pkgs/r、pkgs/msys2）需要先同意服务条款。
- 原因：新版 Miniconda（2025 年后）加了新规则——第一次使用官方下载源之前，必须同意该源的服务条款。因为创建命令带了 `-y`（自动确认），conda 没法弹窗问你同不同意，所以直接报错停止。**不是你操作错了**。
- 解决办法：在终端执行 `conda tos accept`，然后**重新执行一次**创建命令即可。
- ✅ 这条已经帮你处理好，见下方"我帮你干的活"。

**报错 2：**`EnvironmentNameNotFound: Could not find conda environment: llm`

- 现象：你执行 `conda activate llm` 时提示找不到 llm 环境。
- 原因：这是**报错 1 的连锁反应**——环境根本没建成（被服务条款拦住了），自然也就激活不了。
- 解决办法：不需要单独处理，报错 1 解决、环境创建成功后，`conda activate llm` 就能正常工作了。
- ✅ 已随之解决。

**报错 3：**`python --version` **显示** `Python 3.14.6`**，和教程写的 3.14.4 不一样**

- 原因：Windows 会把系统自带的 Python 静默自动更新（3.14.4 → 3.14.6），属正常现象，不用管。
- 解释：这个版本是我们**故意不用的**系统 Python。只要行首出现 `(llm)`，`python --version` 显示的就是 llm 环境里的 3.11，与系统无关。
- ✅ 不用处理。

**报错 4：验收脚本报** `ValueError: ... require users to upgrade torch to at least v2.6`**（CVE-2025-32434 安全限制）**

- 现象：你运行 `python test_env.py`，前面 GPU 部分全部通过（版本 2.5.1+cu121、CUDA True、计算测试通过），但加载模型时抛错。
- 原因：2025 年发现 `torch.load`（PyTorch 读取模型文件的函数）有严重安全漏洞（编号 CVE-2025-32434）。新版 transformers 为了安全，强制要求 torch ≥ 2.6 才能加载 `.bin` 格式的模型文件。而你装的 torch 是 2.5.1——**这不是你操作错了**，是因为你的显卡驱动 546.30 最高支持 CUDA 12.3，而 cu121 仓库最高只提供 torch 2.5.1，torch 2.6+ 只有 cu124 及更新版本，你的驱动带不动。**升级 torch 这条路在你的电脑上走不通**。
- 解决办法：报错信息自己给了出路——"用 safetensors 格式的模型文件就不受此限制"。已把验收脚本里的模型从 `sshleifer/tiny-gpt2`（只有 .bin 格式）换成 `hf-internal-testing/tiny-random-gpt2`（带 safetensors 格式），绕开安全限制，torch 保持 2.5.1 不动。
- ✅ 已修好，重新运行 `python test_env.py` 即可。

> 补充解释：安全漏洞的**前提是"加载别人给的、可能被篡改的模型文件"**。我们测试用的都是 HuggingFace 官方审核过的迷你模型，风险极低；第 2 周部署 Qwen 时我们会用官方 safetensors 格式，同样不受影响。学安全规范、保持好习惯即可，不必恐慌。

**报错 5：VS Code 终端里** `conda activate llm` **不生效（行首没有** `(llm)`**）**

- 现象：在 VS Code 自带的终端里敲 `conda activate llm`，没有报错，但行首始终不出现 `(llm)`。中途还出现过一次 `^C 终止批处理操作吗(Y/N)? y` 的提示。
- 原因：VS Code 默认终端是 **PowerShell**，而 conda 安装时只把"激活钩子"装进了开始菜单那个 Anaconda Prompt（cmd），**没有装进 PowerShell 的配置文件**。在 PowerShell 里 `conda activate` 会退化为调用一个批处理文件的方式，它只在一个临时子进程里执行，改不到当前 PowerShell 的环境变量，所以激活不生效。那个 `^C` 提示正是"批处理方式运行"的痕迹。
- 解决办法（方案 A，已使用）：在 VS Code 的 PowerShell 终端里执行一次 `conda init powershell`（把 conda 钩子正式装进 PowerShell 配置），然后**完全关闭 VS Code 再重新打开**（配置文件是启动时加载的，只关终端不够）。重开后新建终端再 `conda activate llm` 即可。
- 另一个替代法（方案 B，未使用）：在 VS Code 终端面板右上角 `+` 旁边的下拉箭头里选"命令提示符"，用 cmd 终端操作，效果和 Anaconda Prompt 一样。
- ✅ 你已用方案 A 解决。验证方法：`python --version` 显示 3.11.15、`$env:CONDA_DEFAULT_ENV` 显示 llm、`where.exe python` 第一行是 `D:\miniconda1\envs\llm\python.exe`，三条全对 = 激活成功。

---



### 🤝 我（AI）帮你干的活（如实记录，2026-08-01）

1. **找到 Miniconda 的真实位置**：你安装时选了自定义路径，装在 `D:\miniconda1`（不是默认的 C 盘位置），已确认 conda 26.5.3 可用。
2. **接受服务条款**：执行 `conda tos accept`，成功接受了 pkgs/main、pkgs/r、pkgs/msys2 三个源的服务条款。
3. **创建 llm 环境**：执行 `conda create -n llm python=3.11 -y`，成功创建。环境里是 **Python 3.11.15**，位置在 `D:\miniconda1\envs\llm`。
4. **修正了教程里两处与你的电脑不符的地方**：
  - PyTorch 安装命令从 `cu124` 改为 `cu121`（你的显卡驱动 546.30 最高只支持 CUDA 12.3，cu124 需要驱动 551.61 以上，装了用不了 GPU）；
  - 所有 Miniconda 路径统一改为 `D:\miniconda1`。
5. **修改了验收脚本** `test_env.py`：把加载的迷你模型从 `sshleifer/tiny-gpt2`（.bin 格式，撞上新版 transformers 的 torch≥2.6 安全限制）换成 `hf-internal-testing/tiny-random-gpt2`（safetensors 格式，不受限制）。你的 torch 2.5.1 保持不变。

> ⚠️ 这些处理完，**下一步仍是你的**：重新运行 `python test_env.py` 跑通验收。之后第 7 步 VS Code、刷题、存档仍由你按教程自己做。

---



### ⏭️ 你接下来该做什么

1. **回到第 7 步收尾**：你的终端激活问题已解决，重开 VS Code 后还差最后一步——按第 7 步第 4 点，用 `Ctrl + Shift + P` 调出命令面板，输入 `Python: Select Interpreter`，在列表里选带 `llm` 的那个（`D:\miniconda1\envs\llm\python.exe`）。
2. **重新运行一次验收**（如果你还没看到 `环境验收通过！🎉`）：
  ```
   python test_env.py
  ```
   看到 `环境验收通过！🎉` = Day 1 主线任务全部完成，**请截图**。
3. 然后用 `pip list` 自查第 5 步的核心库（transformers / langchain / chromadb / accelerate / datasets / bitsandbytes）是否都在。
4. 接着做零散时间刷题（力扣 1、217）→ 睡前 `pip freeze > requirements.txt` + git 存档。
5. 每做完一步就打勾 ✅ 今日验收清单；卡住了就把终端报错截图发给我，我帮你修报错，但**活还是你自己干**。

