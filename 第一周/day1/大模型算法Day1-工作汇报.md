# Day 1 工作汇报 · 环境搭建日

> 汇报人：本人 ｜ 汇报周期：冲刺月第一天（8/1 环境搭建日，部分收尾跨至 8/2）
> 依据：`00-基础内容总纲.md`（总背景）、`01-第一周详细计划.md`（周计划）、`day1-环境搭建详细教程.md`（当日成果）

---

## 一、总背景

**2026 秋招冲刺**：985/211 数学（概率统计）硕士，目标是江浙沪央国企数字化/AI 岗 + 互联网备选。

- **冲刺节奏**：8 月为冲刺月（每天集中 4~5h + 零散 0.5~1h），9 月秋招黄金月边投边学。
- **主攻方向**：大模型应用开发（部署 → RAG → 微调），目标是产出一个**能讲清楚的项目**，补上简历空白。
- **个人优势**：数学/概率统计背景，适合用概率语言理解并讲解大模型原理，作为面试差异化亮点。
- **硬件基础**：NVIDIA RTX 3060 Laptop（6GB 显存），具备本地部署大模型的硬件条件。

---

## 二、本周计划（8/1~8/7）

**主题：环境搭建 + Transformer 认知 + Python 补缺**

- **本周目标**：本地跑通 Qwen + 画出 Transformer 结构图 + 开启刷题节奏。
- **主线节奏**（每天 3~4h）：环境搭建 → Transformer 理解 → Qwen 本地部署。
- **零散时间**（每天 0.5~1h）：力扣刷题（本周专题：数组 + 哈希表）+ Python 语法速查。
- **睡前 15 分钟**：Git 提交当天产出 + 写明日计划。
- **本周验收**：`torch.cuda.is_available()` 返回 True、能画 Transformer 结构图、本地加载 Qwen 量化版成功生成文本、力扣完成 14 题。

**Day 1 当日计划**：安装 Anaconda + 创建虚拟环境 `llm`（Python 3.11）→ 配置编辑器 + Jupyter → 验证 GPU + 安装 CUDA 版 PyTorch → 安装核心库（transformers / langchain / chromadb / accelerate / datasets）→ 验收跑通 + 力扣 1、217 题。

---

## 三、第一天工作成果

### 3.1 软件与环境检测（第一步）

| 项目 | 检测结果 | 处理 |
|---|---|---|
| 系统 Python | 3.14.4（后自动更新为 3.14.6） | 版本过新，弃用 |
| Miniconda | 未安装 | 已安装（`D:\miniconda1`，conda 26.5.3） |
| Git / VS Code / Jupyter | 均已安装 | 直接使用 |
| NVIDIA 显卡 | RTX 3060 Laptop 6GB，驱动 546.30，支持 CUDA 12.3 | 正常，作为 GPU 计算主力 |

### 3.2 环境搭建完成

- **虚拟环境 `llm` 创建成功**：Python 3.11.15，位置 `D:\miniconda1\envs\llm`，与系统环境完全隔离。
- **GPU 版 PyTorch 安装成功**：`torch 2.5.1+cu121`（匹配显卡驱动上限 CUDA 12.3）。
- **核心库安装**：transformers / accelerate / datasets / langchain / chromadb / bitsandbytes / jupyter 等。
- **编辑器配置**：VS Code 已接入 `llm` 解释器，终端 `conda activate llm` 正常生效。

### 3.3 关键验收结果

- ✅ **`torch.cuda.is_available()` 返回 `True`** —— GPU 可用，达成本周验收清单第 1 项。
- ✅ **GPU 计算测试通过**：显卡实际完成张量计算（`[2.0, 4.0, 6.0]`）。
- ✅ **迷你模型生成测试通过**：成功加载 tiny 模型完成一次文本生成，全链路（环境→GPU→模型）跑通。
- ✅ **Git 首次提交成功**：commit `bad0656`，环境搭建成果已存档。

### 3.4 踩坑与解决（过程中发现并修复的 5 个问题）

| # | 问题 | 解决办法 |
|---|---|---|
| 1 | conda 服务条款未接受，创建环境被拒 | 执行 `conda tos accept` 后重试 |
| 2 | 环境未建成导致 `activate llm` 报找不到环境 | 报错 1 解决后自动恢复 |
| 3 | `python --version` 显示系统 3.14.6（与教程不符） | 属系统自动更新，改用 llm 环境内 3.11 |
| 4 | 验收脚本加载模型触发 torch 安全限制（CVE-2025-32434） | 换用 safetensors 格式迷你模型，绕过限制 |
| 5 | VS Code 终端 `conda activate llm` 不生效 | 执行 `conda init powershell` 并重启 VS Code |

### 3.5 当日产出文件（已全部提交 git）

| 文件 | 说明 |
|---|---|
| `day1-环境搭建详细教程.md` | 手把手新手教程（含踩坑记录、验收清单） |
| `requirements.txt` | 核心库依赖清单 |
| `test_env.py` | 环境验收脚本 |

---

## 四、小结与明日安排

- **小结**：Day 1 目标全部达成——本地"能跑大模型的电脑"已就绪，GPU 可用性已验证，为本周后续的 Transformer 学习和 Qwen 部署打下基础。
- **明日（Day 2）**：Transformer 图解入门——精读《图解 Transformer》中文版，画 Transformer 整体结构图；零散时间刷力扣 26、27（双指针入门），补 Python 列表推导式与字符串切片。
