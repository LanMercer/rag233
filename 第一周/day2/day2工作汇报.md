# day2工作汇报 · Day1+Day2 中期进展汇报

> 汇报日期：2026-08-04 ｜ 汇报人：数学（概率统计）硕士 ｜ 目标：2026 秋招（江浙沪央国企数字化/AI 岗 + 互联网备选）
> 注：本文件为 Day1+Day2 的中期进展汇报，完整周报告见 `第一周\week1工作汇报.md`。

---

## 一、总背景

| 项目 | 情况 |
|---|---|
| 个人画像 | 985/211 数学（概率统计）硕士；Python 基础语法；ML/DL 零基础；大模型此前只调过 API |
| 硬件 | NVIDIA GPU（RTX 3060 Laptop，6G 显存） |
| 最大短板 | 简历上无成型项目 → 8 月必须产出一个能讲清楚的项目 |
| 8 月主线 | 补齐"部署 → RAG → 微调"大模型全链路，冲刺 8 月下旬提前批 |
| 双线并行 | 主线：大模型应用开发；副线：统计建模基础（覆盖银行风控岗） |
| 时间安排 | 集中 3~4h/天（主线攻坚）+ 零散 0.5~1h/天（力扣刷题）+ 睡前 15 分钟（git 存档） |

**差异化策略**：用概率语言讲大模型（softmax=概率分布、注意力权重矩阵=转移概率矩阵、温度=分布熵），发挥数学背景优势。

## 二、第一周计划（8/1 ~ 8/7）

**周主题**：环境 + Transformer 认知 + Python 补缺
**周目标**：本地跑通 Qwen + 画出 Transformer 结构图 + 同步开启刷题节奏

| 日期 | 内容 | 主线产出 |
|---|---|---|
| 8/1 周六 | 环境搭建日 | 环境截图 + requirements.txt + 第一个 commit |
| 8/2 周日 | Transformer 图解入门 | 结构图草稿 + 一页概念笔记 |
| 8/3 周一 | 3Blue1Brown 注意力可视化 | softmax 手算 notebook |
| 8/4 周二 | 多头注意力与位置编码 | 概念笔记 |
| 8/5 周三 | 结构图定稿 + 微调前置认知 | 完整结构图 + 认知笔记 |
| 8/6 周四 | Qwen 本地部署（第一天） | 本地 Qwen 推理截图 + 温度对比 |
| 8/7 周五 | 部署收尾 + 第一周复盘 | 部署 README + 复盘笔记 |

> 注：按"一周只工作五天"的新安排，上表 **8/6~8/7 的 Qwen 本地部署与部署收尾/复盘已移至第 2 周完成**（详见 `week1工作汇报.md`）。上表其余为当时的周计划快照。

**刷题安排**：本周专题"数组 + 哈希表"，每日 1~2 题（8/1 两数之和/存在重复元素；8/2 删除重复项/移除元素……8/7 丢失的数字/搜索插入位置）。

## 三、第一天工作成果（8/1 周六 · 环境搭建日）

**目标达成情况：全部完成 ✅**

| 完成项 | 结果 |
|---|---|
| Miniconda + llm 虚拟环境 | 已装好，Python 3.11.15（路径 `D:\miniconda1\envs\llm`） |
| GPU 版 PyTorch | 2.5.1+cu121，`torch.cuda.is_available()` 返回 **True** ✅ 周里程碑第 1 项 |
| 核心库安装 | transformers / langchain / chromadb / accelerate / datasets / bitsandbytes / jupyter 全部就位 |
| 验收脚本 | `test_env.py` 跑通，迷你模型成功生成一句话 |
| VS Code 配置 | 已认准 llm 环境（conda init powershell 已处理） |
| 力扣刷题 | 完成 1. 两数之和、217. 存在重复元素（哈希入门），能讲清思路 |
| 依赖清单 + 存档 | `pip freeze > requirements.txt` 生成；git 首次 commit 完成 |

**过程中解决的关键问题**（如实记录）：
1. Miniconda 服务条款报错 → `conda tos accept` 解决；
2. 新版 transformers 要求 torch≥2.6（安全限制），但显卡驱动最高只支持 cu121（torch 2.5.1）→ 换用 safetensors 格式的迷你模型绕开；
3. VS Code 的 PowerShell 里 `conda activate` 不生效 → `conda init powershell` + 重开 VS Code 解决。

**遗留提醒**：显存 6G，第 2 周部署 Qwen 时需用 4bit 量化（或改用 Qwen2.5-3B），已计划在第 5 天确认硬件方案。

## 四、第二天进展（8/2 周日 · Transformer 图解入门，进行中）

已在 day2 文件夹产出：
- `day2-Transformer图解入门详细教程.md`（超详细精读教程）
- `transformer_structure.png` + `draw_transformer.py`（整体结构图，脚本可一键生成）
- `概念笔记-参考范文.md`（一页概率语言笔记范文）
- 已下载原论文中英文版：`attention is all you need_en.pdf` / `_cn.pdf`

## 五、下一步

- 继续完成第 2~7 天主线的同时，保持每日力扣 1~2 题；
- 第 5 天确认 Qwen 部署硬件方案（6G 显存下 4bit 量化的可行性）；
- 第 2 周本地部署 Qwen 并产出推理截图。

---

> 汇报依据：《00-基础内容总纲》《01-第一周详细计划》《day1 环境搭建教程》及当日实际完成情况整理。
