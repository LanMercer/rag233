---
title: 机器人领域 RAG 文档问答系统 Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 机器人领域 RAG 文档问答系统 · 在线 Demo

> **不用上传也能直接问**：内置示例语料（GMR 论文 105 段）→ 提问 → 返回**带出处**的答案 + 召回片段；
> 也可以上传自己的领域 PDF 覆盖内置语料。

在线部署在**魔搭创空间（ModelScope Studio）**；本目录同时兼容 HuggingFace Spaces
（上表 front-matter 即为 HF 所需，魔搭忽略它、只读 `license`）。

## 内置示例语料（默认知识库）

- 文件：`chunks.json` —— 来源为项目 `第三周/day13/chunks.json`，是 **GMR 论文**
  （*Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking*，arXiv:2510.02252）的切块文本，共 **105 段**。
- 与项目**离线评测用的是同一批切块**，因此页面上的 `[资料§N]` 出处编号可以直接对照离线评测表
  （实测：20 道评测题 × Top-8 的 `chunk_id` 序列与离线落盘的 `top_ids` **20/20 完全一致**）。
- 上传 PDF 会**覆盖**内置语料（`建立知识库` 按钮 → 新库替换旧库）。

## 说明（诚实标注 · 必读）

- **检索层真实运行**：`bge-small-zh-v1.5` 向量化 + Chroma 余弦相似度 Top-K + 模板 09（带 `[资料§N]` 出处 + 防幻觉约束）。
- **生成层使用 DeepSeek API**（`deepseek-chat`）：免费 CPU 环境跑不动 3B 模型，所以线上用 DeepSeek 代生成以省算力。
  **⚠️ 它 ≠ 本项目微调模型**：本地版本是 **QLoRA 微调过的 Qwen2.5-3B-Instruct 4bit**（只训 0.48% 参数）；
  DeepSeek 更强、且**没有加载本项目的 LoRA adapter**。本页面只演示"检索链路 + 交互形态"。
- **评测数字归属**：下列数字（若有）**只代表"本地 Qwen2.5-3B-Instruct-LoRA"版本，不是本页 DeepSeek 的**。
  微调效果与优化前后对比见项目博客/离线评测与演示 GIF。

## 部署要点（踩过的坑，改用别的平台时先读这段）

| 项 | 要求 | 不照做会怎样 |
|---|---|---|
| 监听地址 | `0.0.0.0:7860`（已写在 `app.py` 的 `launch()` 里） | gradio 默认绑 `127.0.0.1`，平台反代连不进来 → 页面「Could not load this space」+ 网关 502 |
| 平台 Gradio 版本 | 必须 = `requirements.txt` 里的 `gradio==4.44.0` | 两端版本不一致 → pip 解析冲突 → 容器起不来（**但创空间状态仍显示 Running**） |
| 镜像 | 选**预装 torch** 的镜像（如 `ubuntu22.04-py311-torch2.9.1-modelscope1.35.0`） | 否则要从零拖 Linux CUDA 版 torch（800MB~2.5GB），免费 CPU 档易构建超时/爆盘 |
| `HF_ENDPOINT` | 设 `https://hf-mirror.com` | 容器内从 HF 拉 `bge-small-zh-v1.5`（约 95MB）会超时，提问时报错 |
| 依赖 pin | `pydantic==2.10.6` + `starlette==0.46.2` 不可删 | 会让首屏与每个请求都 500（详见 `requirements.txt` 里的坑 1/2） |

### 改代码前必读：四个"不报错也查不出来"的坑（2026-09-22 全部实测踩到）

| # | 现象 | 根因 | 修法（已写进 `app.py`） |
|---|---|---|---|
| A | 答案里的出处**重复**：`[资料§103]、[资料§103]、[资料§4]、[资料§4]`（有效命中从 4 段掉到 2 段） | `Chroma.from_documents(...)` 不传 `collection_name` 时用默认名 `"langchain"`，而 chromadb 在同一进程**共享同一个 in-memory system** → 第二次建库是**往同一个 collection 追加**（实测 105 → 210 条） | 每次建库都给唯一 `collection_name`（`_new_collection_name()`）；内置库另加进程内缓存，只建一次 |
| B | 同一个问题，线上出处的编号与离线评测表**对不上**（20 题里 13 题不一致，且每次建库结果还不一样） | chromadb 的 `hnsw_params.py` 里 `search_ef` 默认只有 **10**：候选池比库还小 → 漏掉真近邻；本语料同篇论文段落高度相似，分数挤在窄带里，边界处会抖动 | `collection_metadata` 里显式给 `hnsw:search_ef: 200`（> 段数 105 → 退化为穷尽搜索）→ 实测与离线落盘 **20/20 一致** |
| C | 第一次提问就报 `IndexError: list index out of range`，页面 500 | `if progress:` 会调用 `gr.Progress.__len__()`，其实现是 `self.iterables[-1].length` —— 还没报过进度时 `iterables` 为空 → 直接抛错（gradio 4.44 `helpers.py:672`） | 判空写 `progress is not None`；并统一走 `_tick()` 兜住异常（进度条不该有能力把建库搞崩） |
| D | 预热/建库后过一会儿提问报 `sqlite3.OperationalError: no such table: collections`（**间歇性**） | 内存库用 `file::memory:?cache=shared` + `LockPool`，而 LockPool 的连接放在 `threading.local()` 里、`_connections` 只存弱引用 → **建库线程一退出，内存库可能整体销毁**，Gradio worker 线程新建连接只看到空库 | 不用内存库：`persist_directory` 指向系统临时目录（`PerThreadPool` + 磁盘文件，与线程生死无关） |

> 另外：内置库**启动时后台线程预热**（约 25~40s，期间端口照常绑定），首问从 31s 降到 2~3s；
> 同时保留懒加载兜底（预热失败或还没好时，首问自己建库，不会白屏）。

## 环境变量（在平台的「设置 → 环境变量 / Secrets」里配置）

| 变量名 | 说明 | 本项目的值 |
|---|---|---|
| `API_BASE_URL` | OpenAI 兼容端点 | `https://api.deepseek.com/v1/chat/completions` |
| `API_KEY` | DeepSeek 调用密钥（**只放 Secrets，绝不写进代码**） | `sk-****` |
| `MODEL_NAME` | 生成用模型名 | `deepseek-chat` |
| `HF_ENDPOINT` | 国内拉 HF 权重走镜像 | `https://hf-mirror.com` |
| `EMBED_MODEL` | （可选）向量模型 | `BAAI/bge-small-zh-v1.5` |
| `GRADIO_ANALYTICS_ENABLED` | （可选）关掉 gradio 版本自检外呼 | `False` |

## 本地运行

```bash
pip install -r requirements.txt
# 配置环境变量（Windows PowerShell 示例）
$env:API_BASE_URL="..."; $env:API_KEY="..."; $env:MODEL_NAME="..."
python app.py
# 打开 http://127.0.0.1:7860
```

## 目录

- `app.py`：Gradio 界面（内置语料/上传 PDF → 建库 → 提问 → 答案 + 出处 + 召回片段）
- `chunks.json`：内置示例语料（GMR 论文 105 段，与项目离线评测同一批切块）
- `requirements.txt`：依赖（含"新容器必崩"的坑位说明）

> 向量库落在**系统临时目录**（不写进本目录）；模型权重、LoRA adapter 等大文件不在本仓库内。
