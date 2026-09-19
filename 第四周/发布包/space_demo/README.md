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

> 上传领域 PDF → 自动建库 → 提问 → 返回**带出处**的答案 + 召回片段。

## 说明（诚实标注 · 必读）

- **检索层在本 Space 真实运行**：`bge-small-zh-v1.5` 向量化 + Chroma 余弦相似度 Top-K + 模板 09（带 `[资料§N]` 出处 + 防幻觉约束）。
- **生成层使用 DeepSeek API**（`deepseek-chat`）：免费 CPU 环境跑不动 3B 模型，所以线上用 DeepSeek 代生成以省算力。
  **⚠️ 它 ≠ 本项目微调模型**：本地版本是 **QLoRA 微调过的 Qwen2.5-3B-Instruct 4bit**（只训 0.48% 参数）；
  DeepSeek 更强、且**没有加载本项目的 LoRA adapter**。本页面只演示"检索链路 + 交互形态"。
- **评测数字归属**：下列数字（若有）**只代表"本地 Qwen2.5-3B-Instruct-LoRA"版本，不是本页 DeepSeek 的**。
  微调效果与优化前后对比见项目博客/离线评测与演示 GIF。

## 环境变量（在 Space → Settings → Variables and secrets 配置）

| 变量名 | 说明 | 本项目的值 |
|---|---|---|
| `API_BASE_URL` | OpenAI 兼容端点 | `https://api.deepseek.com/v1/chat/completions` |
| `API_KEY` | DeepSeek 调用密钥（**只放 Secrets，绝不写进代码**） | `sk-****` |
| `MODEL_NAME` | 生成用模型名 | `deepseek-chat` |

## 本地运行

```bash
pip install -r requirements.txt
# 配置环境变量（Windows PowerShell 示例）
$env:API_BASE_URL="..."; $env:API_KEY="..."; $env:MODEL_NAME="..."
python app.py
```

## 目录

- `app.py`：Gradio 界面（上传 PDF → 建库 → 提问 → 答案 + 出处 + 召回片段）
- `requirements.txt`：依赖
- `index/`：（可选）预建向量库；不存在时首次运行自动建库

> 模型权重、LoRA adapter 等大文件不在本仓库内。
