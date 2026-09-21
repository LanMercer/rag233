# -*- coding: utf-8 -*-
r"""
RAG 评测脚本 v2.1 · 参数化版（第四周 Day17 · O0 尺子校准收尾 + O1-R1 检索实验）⭐ 本周全部优化实验的"尺子"

Day17 相对 Day16 版的四处改动（其它逻辑一字未动，保证"尺子本身没被换掉"）：
    ① 【新增】--query-instruction on/off：检索 query 侧加 bge 官方 instruction 前缀（O1-R1）。
       实现方式：薄封装 BgeQueryInstructionEmbeddings（只重写 embed_query，段落侧 embed_documents 不动）。
       原因：本机 langchain_huggingface.HuggingFaceEmbeddings 不支持 query_instruction（Day17 现场实测）。
    ② 【新增】--keyword-patch on/off：判据校准补丁（默认 on）。
       修 Day16 发现的"关键词过严误杀"：Q5 期望关键词 LAFAN1 -> LAFAN1/LAFAN。
       做法：不直接改第三周的 eval_questions.json，而在本脚本用显式补丁表覆盖 + 打印留痕，
             保证"判据改动一次性、可追溯、前后口径一致"。
    ③ 【修复】--append-log 的插入位置：Day16 版是追加到文件末尾（会落进「三、失败尝试记录」）；
       Day17 版定位到「## 一、实验记录表」，插到最后一行数据行之后。
    ④ 输出目录命名新增 _qi 后缀（--query-instruction on 时），避免与"无前缀"的实验互相覆盖。

Day19 相对 Day17 版的改动（O1-R4 查询改写；本轮只动三处，保证"尺子本身没被换掉"）：
    ① 【新增】--rewrite-cache / --rewrite-mode：读 rewrite_queries.py 生成的改写缓存，
       把"检索用的查询"从原中文问句换成改写结果（plain/term/hyde/both 四档）。
       ⭐ 纪律：改写【只给检索层用】——生成层仍然喂原中文问句（generate_answer 的实参没变），
         这样（a）只改了一个变量，（b）判分口径（expected_keywords 是按原问题写的）依然成立。
    ② 每题结果里新增 rec["retrieval_query"] / rec["rewrite_used"]，并落进 eval_results.json
       → 事后能回答"它到底问了什么"（面试追问"你怎么证明改写生效了"的材料）。
    ③ run_log / 日志行 / --append-log 的配置列都补上 rw=<档位> 或 rw=off
       → 让 R4 行与 R3 行在「实验记录表」里一眼可分（否则两行配置长得一样）。
    另：汇总里新增「⑦ 改写生效 N/M 题」，且缓存里取不到该档时打印题号（避免"接线没生效"
       被误读成"改写不涨分"）。

Day19 第 3 步追加（O2-G2 模板消融 + O2-G3 引文合法性；同样只做加法）：
    ① 【新增】模板 v2a / v2b（v2 补上"只输出拒答句"一条）→ 模板层的**单句消融**：
       v2a 只做①②④（治"该答却拒答"FP），v2b 只做③④（治"编造引用/数字"FN）。
    ② 【新增】legal_citation_filter()：把答案里**不在本轮 Top-K** 的引文编号摘出来，
       落进 rec["citation_bad"] / rec["citation_n"] / rec["answer_citation_cleaned"]。
       ⚠ **只留痕、不参与判定**（否则清理后的答案会改判分，等于又混进一个变量）。
       ⚠ 同时认 `[资料§N]` 与裸 `[N]`：因为模板写的是前者、而 build_context() 的标签是后者，
         模型两种都写过 → 只认一种会漏检过半（详见 CITE_RE 处的注释）。
    ③ 汇总新增「⑧ 引文合法性」；每题打印「引文 N 处（非法 M 处）」。

Day19 第 3 步追加之二（9/21 复核三条 out 题答案后收紧；仅动第 3 条规则，v2a 一字未动）：
    ① 【收紧】v2 / v2b 的第 3 条：原措辞"只能引用资料中出现过的段落编号"**太宽** ——
       实测 Q18 把**参考文献的方括号编号 `[41][42][43]`** 和**页码 `pp. 425–432 / 679–686 / 248:1–248:16`**
       当成了"资料段落号"与"被引次数"（逐句回 chunks.json 核过，见 day19 教程 §3.2）。
       现改为显式限定："N 只取【资料】每一段开头标出的编号；参考资料里的 [数字]、页码、卷号
       一律不得当作段落编号或被引次数"。
    ② 保持**单变量消融**不变：v2a 仍只做①②④、v2b 仍只做③④，两者只差第 3 条 → 收紧后仍可比。
    ③ ⚠ **已知未覆盖的一条**：Q11（"缺点有哪些？"）与 Q14（"奖励函数细节？"）的错法是
       **答非所问**（拿相邻话题的段落顶上），属第 2 条规则的范围，本轮**故意不动** ——
       留给 R5 结果来判定"答非所问"是否是独立瓶颈（预判写在 day19 教程 §3.2 末尾）。

Day19 第 3 步后追加（9/21：O2-G3 从"加分项"升级为"必备兜底"）：
    ① **R5 的结论逼出来的**：三档 v2 全线回退，且 Q20/Q15/Q13 的错法都是"内容取自检索、
       只是用错了"（把时长列当仿真步数 / 抄 chunk 31 却标参考文献号 [37]）→
       **提示词治不了这类幻觉** → 只能靠生成之后的工程兜底。
    ② 【新增】--citation-policy {record,strip,retry}（**默认 record**，与旧轮完全同行为）：
       strip = 非法编号就地剪除（纯文本后处理，不调模型）；
       retry = 带"你引错了这些编号，白名单是这些"的修正指令重答一次，仍非法再剪。
    ③ 【新增】⑧ 改报**两套口径**：`citation_*` = 原始生成（与 R3/R4/R5 同口径）、
       `*_delivered` = 兜底后的交付答案；并单独报"**判定随兜底翻转**"的题 ——
       那是后处理造成的，**绝不记进"生成层变好"**。
    ④ 【设计约束】剪除非法编号后**不把答案改写成拒答句** → 不会把 FN 机械地变成 TP
       （否则就是用后处理"造分"）。"剪完没内容"只记进 `citation_only_bad` 当诊断。
    ⑤ 【新增】legal_citation_filter 在**没有非法引用时原样返回**（连空白都不动）——
       避免"清理"本身成为一个隐性变量。

它和第三周 day14 的 eval.py 是什么关系？
    eval.py   = 一次性尺子：模型/Top-K/温度全写死在代码里，只出"首版基线"那一组数字；
    eval_v2.py = 可配置尺子：同一份代码 + 不同命令行参数 = 一轮受控实验，
                 结果按参数分目录落盘，互不覆盖，且可随时复现（这就是 HR 要的"可归因优化"）。

五件套说明（这个脚本在机器学习的哪一环）：
- 模型   ：生成层 = 由 --profile 决定——
             lora     → Qwen2.5-3B-Instruct-LoRA（第三周 day12 微调，day13 local_api_lora.py 起服务，默认）
             original → Qwen2.5-3B-Instruct（原模型，第二周 day9 local_api.py 起服务）
           检索层 = bge-small-zh-v1.5（把问题向量化，去 Chroma 里找相似段落）
- 数据   ：第三周 day13 定好的 20 条评测题（eval_questions.json：in_material 10 + out_of_material 10）
           + 第三周 day13 建好的 GMR 论文向量库（chroma_db）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试/评估环节）：对每条题
           ① 检索：问题向量化 → Chroma 余弦相似度 Top-K（记下召回了哪些 chunk）
           ② 组织：按 --template 选模板（v1 = 第三周模板 09；v2/v2a/v2b = 强化约束与单句消融）拼上下文
           ③ 生成：POST 本地 OpenAI 兼容接口拿答案（同一题重复 --runs 次）
           ④ 判分：规则判分（in 看期望关键词 / out 看是否如实拒答）
                   + 每题多跑时"多数投票" + 记录一致率（看生成稳不稳）
                   + 检索命中判据"双轨"：strict（Top-K ∩ source_chunk）+ loose（关键词是否落在 Top-K 上下文里）
           ⑤ 汇总：检索命中 / 生成正确 / 防幻觉 / 防幻觉 F1 / 总正确率 → 写评测表 + 明细 json
    ↑ 属于机器学习的【评估（Evaluation）】环节：不训练、不动权重，只"打分"。

配置从哪来（不重复造轮子）：
    复用 第三周\day15\config.json 的 profile（api_url / model_name / top_k / temperature）、
    embed_model、persist_dir（其相对路径以 config.json 所在目录为基准解析）。
    想换模型名/端口/embedding，只改那一份 config.json，本脚本不用动。

运行前置（顺序别乱）：
    1. 向量库在：第三周\day13\chroma_db（没有就先跑 day13 的 build_index.py）；
    2. 与 --profile 对应的服务已在 8000 端口运行（6G 显存一次只能跑一个 3B）：
         lora     → cd 第三周\day13 ; python -m uvicorn local_api_lora:app --host 127.0.0.1 --port 8000
         original → cd 第二周\day9  ; python -m uvicorn local_api:app --host 127.0.0.1 --port 8000
       等日志出现"[OK] 模型已就绪：…"再跑本脚本。

运行方法（llm 环境；先 cd 到本文件所在目录）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day17"

    # 看全部参数
    python eval_v2.py --help

    # 冒烟：只跑 2 条题，确认"接线"没坏（数字无意义）
    python eval_v2.py --top-k 4 --runs 1 --limit 2 --out result_smoke

    # D2 步骤①：O0 收尾——用 3 次生成取多数投票，复跑"校准后"的稳定基线（B0r3）
    python eval_v2.py --top-k 4 --runs 3 --note "B0r3 基线：判据校准 + 3 次投票" --exp-id B0r3 --date 9/21

    # D2 步骤②：O1-R1——只打开 query instruction 前缀（k 仍为 4，一次只改一个变量）
    python eval_v2.py --top-k 4 --runs 3 --query-instruction on --note "R1: bge query instruction" --exp-id R1 --date 9/21

    # D2 步骤③：O1-R1+R2——前缀 + TOP_K 8（三组对比收尾）
    python eval_v2.py --top-k 8 --runs 3 --query-instruction on --note "R1+R2: instruction + TOP_K 8" --exp-id R1R2 --date 9/21

    # 省时备选：先用 --runs 1 快速看方向，再对要引用的两组用 --runs 3 复跑定稿
    python eval_v2.py --top-k 4 --runs 1 --query-instruction on --note "R1 快速轮"

参数一览：
    --profile            lora | original            默认取 config.json 的 default_profile
    --top-k              检索返回条数                默认 4
    --temperature        生成温度                    默认 0.2
    --runs               每题重复生成次数（≥3 取多数投票）默认 1
    --template           v1 | v2 | v2a | v2b         默认 v1（v2=约束全上；v2a=只弱化拒答；v2b=只禁编造 → 单句消融）
    --rewrite-cache      改写缓存 json（O1-R4）        默认空=不改写（Day19；只换检索用的查询）
    --rewrite-mode       plain | term | hyde | both  默认 term（用缓存里的哪一档当检索查询）
    --query-instruction  on | off                   默认 off（on = query 侧加 bge 官方 instruction 前缀，O1-R1）
    --keyword-patch      on | off                   默认 on（判据校准补丁：Q5 关键词放宽）
    --out                输出目录                    默认 result_<profile>_k<K>[_qi][_temp<t>][_<template>][_runs<n>]
    --limit              只跑前 N 条（冒烟用，0=全部）  默认 0
    --max-tokens         单条答案上限                 默认 512
    --manual-fix         on | off                   默认 on（沿用 day14 的人工改判表，复现基线必须 on）
    --citation-policy    record | strip | retry     默认 record（O2-G3 引文兜底；strip/retry = 强制白名单）
    --append-log         跑完把一行结果插入 第四周\实验日志.md 的「一、实验记录表」
    --note               本轮改了什么变量（写进日志行）

运行后输出（都在 --out 目录里）：
    评测表.md        —— 指标汇总 + 双轨检索命中 + 逐题结果（贴进报告/日志用）
    eval_results.json —— 全部明细（每题检索、答案、逐次判定、一致率）
    run_log.txt      —— 本轮完整运行日志（脚本自己用 UTF-8 写，避免 shell 重定向存成乱码）
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import requests

# 镜像兜底（与本项目其它脚本保持一致）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固 + 导入 langchain 组件
# ---------------------------------------------------------------------------
# Windows 控制台默认 GBK，打不出 emoji / 部分符号会崩，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain_chroma import Chroma                          # 连向量库
from langchain_huggingface import HuggingFaceEmbeddings      # 问题向量化

# ---------------------------------------------------------------------------
# 第 2 区：路径常量（相对本脚本定位，换机器也能跑）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       # 第四周\day16
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))   # 仓库根（大模型算法）

# 复用第三周 day15 的 config.json（profile / embed_model / persist_dir 都读它）
CONFIG_PATH = os.path.join(REPO_DIR, "第三周", "day15", "config.json")
# 默认评测题（第三周 day13 定）
DEFAULT_EVAL_JSON = os.path.join(REPO_DIR, "第三周", "day13", "eval_questions.json")
# 实验日志（第四周\实验日志.md）
DEFAULT_LOG_MD = os.path.join(REPO_DIR, "第四周", "实验日志.md")

# 拒答词表：命中任一即认为模型"如实拒答"（防幻觉成功的信号），与 day14 一字不差
REFUSAL_MARKERS = [
    "资料中没有提到", "资料中未提到", "资料中没有提及", "资料中未提及",
    "资料里没有", "资料中并未", "资料中并没有", "材料中没有", "文中没有",
    "没有提到", "没有提及", "未提到", "未提及", "未涉及",
    "没有相关内容", "无法从资料", "资料未提供", "没有找到相关内容",
]

# ---------------------------------------------------------------------------
# Day17 新增 ①：判据校准补丁（KEYWORD_PATCH）
# ---------------------------------------------------------------------------
# 背景（Day16 实跑发现，属"尺子自身的刻度问题"）：
#   B0 轮 Q5 答 "a diverse LAFAN dataset"（语义正确），但 eval_questions.json 里关键词写死
#   "LAFAN1" → 被规则判 ❌。这是"关键词过严误杀正确答案"，会让生成正确率被低估。
#
# 纪律：判据改动必须 ①一次性做完 ②可留痕 ③前后一致，否则数字不可比。
#   做法：不直接改第三周的 eval_questions.json（那是历史文件），而在本脚本用显式补丁表覆盖，
#         并在评测表 + 终端 + 运行日志里打印"本轮判据补丁"，做到"改了什么都看得见"。
KEYWORD_PATCH = {
    5: {"expected_keywords": ["LAFAN1", "LAFAN"]},   # Q5：LAFAN1 -> LAFAN1/LAFAN（LAFAN 是 LAFAN1 的前缀）
}

# ---------------------------------------------------------------------------
# Day17 新增 ②：bge query instruction 前缀（O1-R1）
# ---------------------------------------------------------------------------
# bge 系列（BAAI General Embedding）官方用法：检索时"查询侧"要加一句 instruction，
# "段落侧"不加。原因是 bge 训练时 query 与 passage 的编码方式就是不对称的：
#     相似度 = cos( E_query(instruction + 问题), E_passage(段落) )
# 当前脚本用的 HuggingFaceEmbeddings 默认两边都不加前缀 → query 侧的编码约定与训练时不一致，
# 属系统性欠对齐（对应第三周诊断的"待验证假设①"）。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 模板库（答案组织层）：
#   v1 = 第三周模板 09（基线口径，和 day14 一字不差，复现基线必须用它）
#   v2 = 强化约束草稿（D1 先占位，day19 的 O2-G2 再做 v1/v2 并排消融）
#        改动点：① 弱化"无内容就直说"的触发条件 → 治"过拒答 FP"
#                ② 加"只能引用出现过的编号、禁止编造数字与引用" → 治"编造引用 FN"
PROMPT_TEMPLATES = {
    "v1": """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

只输出答案；如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}""",
    "v2": """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

【规则】
1. 只输出答案本身，不要复述题目、不要输出无关解释；
2. 只有当上面资料确实找不到相关信息时，才回答「资料中没有提到」；资料里有依据就照实回答；
3. 引用只能写成 [资料§N]，N 只取【资料】每一段开头标出的编号；参考资料（参考文献）里的 [数字]、页码、卷号一律不得当作段落编号或被引次数；禁止编造编号，禁止编造数字或统计量；
4. 如果资料不足以回答，**只输出拒答句**，不要先给一段猜测再拒答。
5. 答案不超过 3 句。

【问题】
{question}""",

    # v2a = 只做"①②④"（治"过拒答"，即 FP）
    "v2a": """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

【规则】
1. 只输出答案本身，不要复述题目、不要输出无关解释；
2. 只有当上面资料确实找不到相关信息时，才回答「资料中没有提到」；资料里有依据就照实回答；
3. 每条要点末尾用 [资料§N] 标出依据的段落编号；答案不超过 3 句。

【问题】
{question}""",

    # v2b = 只做"③④"（治"编造引用/数字"，即 FN）
    "v2b": """你是一名严谨的资料问答助手。
请只依据下面给定的资料回答用户问题。

【资料】
{context}

【规则】
1. 只输出答案本身，不要复述题目；
2. 如果资料里没有相关内容，请直接回答「资料中没有提到」，不要编造；
3. 引用只能写成 [资料§N]，N 只取【资料】每一段开头标出的编号；参考资料（参考文献）里的 [数字]、页码、卷号一律不得当作段落编号或被引次数；禁止编造编号，禁止编造数字或统计量；
4. 如果资料不足以回答，**只输出拒答句**，不要先给一段猜测再拒答；
5. 答案不超过 3 句。

【问题】
{question}""",
}

# ---------------------------------------------------------------------------
# Day19 O2-G3 引文合法性（后处理）
#   9/21 升级：从"只留痕"（record）扩到"**强制白名单**"（strip / retry）。
#   动机（有实测依据）：R5/R5a/R5b 三档证明**提示词约束治不了"误读型幻觉"**——
#     Q20 把 Tab. I 的"运动时长列"当"仿真步数"、Q15 抄 chunk 31 的公式却标 [37]（参考文献编号）、
#     Q13 抄 chunk 91 的参考文献条目当"发表的期刊"。内容**取自检索、只是用错了**，
#     所以模型不认为自己在编 → 只能靠**生成之后的工程兜底**。
# ---------------------------------------------------------------------------
# 为什么必须接受两种写法（day19 实测 + 代码核对）：
#   模板写的是 [资料§N]，但 build_context() 给段落的标签其实是 [N]（`f"[{chunk_id}] {text}"`）
#   → 模型两种都写过：day17 的答案里 `[35]`、`[97]` 与 `[资料§68]` 混着出现。
#   只认 [资料§N] 会**漏检一半以上的非法引用**，比不检查更危险（会让你误以为引文很干净）。
#   ⚠ 这里**不改模板/上下文**——那会动到 R3 的可复现性（v1 模板必须一字不变）。
CITE_RE = re.compile(r"\[(?:资料§|资料|ref|§)?\s*(\d+)\]")

# 9/21 新增：**非数字编号的"伪引文"**。实测 R6b 的 Q3 输出里成片出现
#   `[资料©] [资料®] [资料°] [资料ª] [資料½] [resourceì] [resourceshadow] [resourceright]`
#   —— 模型在用"引文形状"假装自己在标注来源（内容多是编的学校名单）。
#   `CITE_RE` 只认数字 → **这类一个都抓不到**，所以"交付口径非法 0 处"≠"答案干净"。
#   这里只做**计数**（不改答案）：让口径边界在报告里是**可量化**的，而不是一句"可能有漏检"。
PSEUDO_CITE_RE = re.compile(
    r"\[(?:资料|資料|資源|资源|資訊|资讯|情報|信息|信頑|ref|resource)[^\]\d]{0,16}\]")

# ⚠ 必须过滤两类误报（离线复算时踩到）：
#   ① `[资料中没有提及]` —— 这是**拒答残句**被方括号包起来，不是引文（R6b 的 Q1 就是这样）；
#   ② 方括号里有数字的 —— 那是 CITE_RE 的活，别重复计。
REFUSAL_IN_BRACKET = ("没有提到", "没有提及", "未提到", "未提及", "没有", "未涉及", "无法")


def count_pseudo_citations(text):
    """数"非数字伪引文"（`[资料©]` / `[resourceì]` 之类）。只计数，不改答案。"""
    n = 0
    for m in PSEUDO_CITE_RE.finditer(text or ""):
        s = m.group(0)
        if any(w in s for w in REFUSAL_IN_BRACKET) or re.search(r"\d", s):
            continue
        n += 1
    return n


def legal_citation_filter(answer, context_ids):
    """把**不在本轮上下文里**的引文编号摘掉，返回 (清理后的答案, 非法编号列表)。

    合法性口径：编号必须出现在**本轮 Top-K 的 chunk_id**（= 模型实际看到的段落集合）里。
    ⚠ **没有非法引用时原样返回**（连空白都不动）——避免"清理"本身变成一个隐性变量。
    """
    ctx = set(int(i) for i in context_ids if i is not None)

    def repl(m):
        return m.group(0) if int(m.group(1)) in ctx else ""

    bad = [int(m.group(1)) for m in CITE_RE.finditer(answer) if int(m.group(1)) not in ctx]
    if not bad:
        return answer, []
    cleaned = _tidy_after_strip(CITE_RE.sub(repl, answer))
    return cleaned, bad


def _tidy_after_strip(text):
    """剪掉编号后收拾残局：连续标点、标点前的空格、首尾空标点。

    只在**确实剪过东西**的答案上调用（`legal_citation_filter` 提前 return 保证了这点），
    所以正常答案一个字都不会被动。
    例：`① 踢步内收伪影 [2]；② 身高伪影 [52]；资料中没有提及③ 拖动伪影。`
        → `① 踢步内收伪影；② 身高伪影；资料中没有提及③ 拖动伪影。`
    """
    t = re.sub(r"\s{2,}", " ", text)
    t = re.sub(r"([，,、；;：:])\1+", r"\1", t)          # 连续同种标点压成一个
    t = re.sub(r"[，,、；;：:]{2,}", lambda m: m.group(0)[0], t)  # 不同种标点连排只留第一个
    t = re.sub(r"\s+([，,。；;：:、）)】])", r"\1", t)     # 标点前的空格
    t = re.sub(r"^[\s，,、；;：:。]+", "", t)              # 开头的空标点
    t = re.sub(r"[\s，,、；;：:]+$", "", t)               # 末尾的空标点（句号保留）
    return t.strip()


# 重答指令（--citation-policy retry 用）：把"你引了不存在的编号"直接点名告诉模型。
# 设计要点：**给白名单**（列出本轮真实存在的编号）+ **给出口**（资料不足就只输出拒答句），
#   否则模型会为了"避免引用错误"而胡乱改写内容（那是更大的问题）。
RETRY_INSTRUCTION = """

【必须修正的问题】
你上一次的回答引用了这些段落编号：{bad}；但本次【资料】里**只有**这些编号：{ctx}。
请重新回答上面那个问题，并且：
1. 只允许引用上面列出的编号，格式 [资料§N]；
2. 如果现有资料不足以回答，**只输出**「资料中没有提到」，不要给出任何猜测、也不要标注编号；
3. 不要输出"你上次引错了"这类解释，直接给新答案。"""


def apply_citation_policy(policy, profile, template, question, context, answers,
                          ids, temperature, max_tokens, dry_run=False):
    """O2-G3 强制白名单。返回 (交付答案列表, 元信息 dict)。

    policy：
      record = 原样交付（**默认**，与 R3/R4/R5 各行完全同行为，保证数字可复现）
      strip  = 逐条把非法编号就地剪掉（纯文本后处理，**不再调模型**）
      retry  = 先带"你引错了这些编号"的修正要求重答一次；仍非法再剪（最多重答 1 次）
    ⚠ 剪完**不会**把答案改写成拒答句 —— 那会把 FN 机械地变成 TP，等于用后处理"造分"。
       "剪完没内容了"只记进 `only_bad_after` 当诊断，不改判定取向。
    """
    meta = {"removed": 0, "retried": 0, "only_bad_after": 0}
    if policy == "record" or dry_run:
        return answers, meta

    ctx = [int(i) for i in ids if i is not None]
    delivered = []
    for a in answers:
        cur = a
        _, bad = legal_citation_filter(cur, ctx)
        if bad and policy == "retry":
            meta["retried"] += 1
            prompt = template.format(context=context, question=question) + \
                RETRY_INSTRUCTION.format(bad=bad, ctx=ctx)
            cur = chat_completion(profile, prompt, temperature, max_tokens)
        cleaned, bad2 = legal_citation_filter(cur, ctx)
        meta["removed"] += len(bad2)
        if bad2:
            cur = cleaned
        if not re.sub(r"[\s，,。；;：:、（）()\[\]【】\-—]+", "", cur):
            meta["only_bad_after"] += 1
        delivered.append(cur)
    return delivered, meta

# 人工复核修正表（沿用第三周 day14，--manual-fix off 可关闭）：
#   自动规则会被"表面命中"骗过，跑完人工看一遍明细再在这里覆盖。
MANUAL_FIXES = {
    9: {"correct": False, "detail": "答的是 ProtoMotions 的仓库（NVLabs/ProtoMotions），不是 GMR 论文代码（YanjieZe/GMR），github 关键词碰巧命中，人工改判错"},
    18: {"correct": False, "detail": "先编造一串引用统计数字（425/679/248 等），末尾才补「资料中没有提到」——拒答词命中但内容已编造，属规则漏网，人工改判错"},
}

# ---------------------------------------------------------------------------
# 第 3 区：配置读取（复用 day15 config.json）
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p, base_dir):
    """相对路径按 base_dir 解析成绝对路径；已是绝对路径则原样返回。"""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def load_config():
    """读 day15 的 config.json，返回 (配置字典, 配置所在目录)。"""
    cfg = load_json(CONFIG_PATH)
    return cfg, os.path.dirname(CONFIG_PATH)


# ---------------------------------------------------------------------------
# 第 4 区：服务检查（活着 + 是不是我们想要的那个模型）
# ---------------------------------------------------------------------------

def api_base(api_url):
    """从 .../v1/chat/completions 截出服务根地址（用于拼 /health、/v1/models）。"""
    return api_url.rsplit("/v1/chat/completions", 1)[0]


def check_service(profile):
    """
    两查（沿用 day15 ask.py 的经验，防"服务在跑 ≠ 服务正确"）：
      ① GET /health      → 服务活着、模型加载完（model_ready=True）
      ② GET /v1/models   → 服务端真实模型名与 profile 一致
    返回 (是否通过, 提示文本)。
    """
    base = api_base(profile["api_url"])
    try:
        r = requests.get(base + "/health", timeout=10)
        r.raise_for_status()
        if r.json().get("model_ready") is not True:
            return False, "服务已启动但模型还没加载完（model_ready=False），等日志出现「模型已就绪」再跑。"
    except Exception as e:
        return False, f"连不上服务（{e}）。请先启动与 --profile 对应的服务再跑。"

    try:
        served = requests.get(base + "/v1/models", timeout=10).json()["data"][0]["id"]
    except Exception as e:
        return False, f"服务活着但读 /v1/models 失败（{e}）。"

    if served != profile["model_name"]:
        return False, (f"⚠ 服务端模型 = {served}，但 profile 要求 = {profile['model_name']}。"
                       "请切换服务（6G 显存一次只能跑一个 3B）后重跑。")
    return True, f"服务检查通过：{served} 已就绪，且与 profile 一致 ✅"


# ---------------------------------------------------------------------------
# 第 5 区：检索 / 生成 / 判分
# ---------------------------------------------------------------------------

def apply_keyword_patch(questions, enabled=True):
    """
    Day17：把 KEYWORD_PATCH 里的判据校准应用到题目上（默认开启）。
    返回 (应用了几条, 人类可读的说明)，供终端/评测表/日志留痕。
    """
    if not enabled:
        return 0, "未应用（--keyword-patch off，沿用 day13 原始判据）"
    applied = []
    for q in questions:
        patch = KEYWORD_PATCH.get(q["id"])
        if patch:
            for k, v in patch.items():
                q[k] = list(v)
            applied.append(f"Q{q['id']}")
    if not applied:
        return 0, "无（题目里没有需要校准的编号）"
    return len(applied), "已应用判据校准补丁：" + "、".join(applied) + "（Q5 关键词 LAFAN1 -> LAFAN1/LAFAN）"


class BgeQueryInstructionEmbeddings(HuggingFaceEmbeddings):
    """
    Day17 新增：带 query instruction 前缀的 bge embedding 薄封装。

    为什么需要自己写？
        本机 langchain_huggingface.HuggingFaceEmbeddings 是 pydantic 模型，字段只有
        [model_name, cache_folder, model_kwargs, encode_kwargs, query_encode_kwargs,
         multi_process, show_progress] —— **没有 query_instruction**（Day17 实测，
        等价于 inspect.signature 看到的是 __init__(**kwargs) 而 model_fields 里没有它）。
        langchain_community 里有个 HuggingFaceBgeEmbeddings 支持 query_instruction，
        但它已标注 sunset（弃用），本项目主线用 langchain_huggingface，故自己封装最稳。

    关键点：只重写 embed_query（查询侧加前缀）；embed_documents（段落侧）完全不改。
        理由：bge 的训练约定就是"query 加、passage 不加"；段落侧加了反而破坏对齐。
        这也意味着：**不需要重建向量库**（库里存的段落向量本来就是不带前缀的）。
    """
    query_instruction: str = ""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self.query_instruction + text)


def load_store(persist_dir, embed_model, query_instruction=""):
    """
    连接 Chroma 向量库；embedding 必须与建库时是同一个模型。
    query_instruction 非空时，只在"查询侧"加 bge 官方前缀（O1-R1）。
    """
    if query_instruction:
        embedder = BgeQueryInstructionEmbeddings(model_name=embed_model,
                                                 query_instruction=query_instruction)
    else:
        embedder = HuggingFaceEmbeddings(model_name=embed_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)


def retrieve(store, question, top_k):
    """
    检索层：question 向量化 → 余弦相似度 Top-K。
    返回 (docs, ids)：docs = [(Document, 相似度), ...]，ids = [chunk_id, ...]。
    """
    docs = store.similarity_search_with_relevance_scores(question, k=top_k)
    ids = []
    for d, _ in docs:
        try:
            ids.append(int(d.metadata.get("chunk_id")))
        except (TypeError, ValueError):
            ids.append(None)
    return docs, ids


def build_context(docs):
    """把检索到的片段拼成上下文文本（模板里 {context} 的内容）。"""
    return "\n".join(f"[{d.metadata.get('chunk_id')}] {d.page_content}" for d, _ in docs)


def chat_completion(profile, prompt, temperature, max_tokens):
    """调本地 OpenAI 兼容接口拿一次生成（generate_answer 与 O2-G3 retry 共用）。"""
    payload = {
        "model": profile["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(profile["api_url"], json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def generate_answer(profile, template, question, context, temperature, max_tokens, dry_run=False):
    """生成层：套模板 → POST 本地接口 → 返回答案文本。dry_run=True 时用桩答案（不连模型服务）。"""
    if dry_run:
        return "【dry-run 桩答案】资料中没有提到。"   # 只验证流程，数字无意义
    prompt = template.format(context=context, question=question)
    return chat_completion(profile, prompt, temperature, max_tokens)


def is_refusal(text):
    """判分规则①：模型是否"如实拒答"（说资料里没有）。"""
    return any(m in text for m in REFUSAL_MARKERS)


def hit_keywords(answer, keywords):
    """判分规则②：期望关键词是否命中答案。返回 (bool|None, 命中列表)。"""
    if not keywords:
        return None, []
    low = answer.lower()
    hit = [kw for kw in keywords if kw.lower() in low]
    return bool(hit), hit


def judge_once(q, answer):
    """对"单次生成"判分，返回 (correct|None, detail)。口径与 day14 一致。"""
    if q["type"] == "in_material":
        hit, matched = hit_keywords(answer, q.get("expected_keywords", []))
        if hit is True:
            return True, "命中关键词：" + "、".join(matched)
        if hit is None:
            return None, "未配关键词，需人工判分"
        if is_refusal(answer):
            return False, "该答未答（资料里有答案却拒答）"
        return False, "未命中任何期望关键词"
    # out_of_material / 反事实：如实拒答才算防幻觉成功
    ok = is_refusal(answer)
    return ok, ("如实拒答「资料中没有提到」" if ok else "疑似幻觉：未拒答、自行编造或臆测")


def judge_question(q, top_ids, context, answers):
    """
    对一条题的全部重复生成做判定 + 多数投票。
    answers = [第1次答案, 第2次答案, ...]（--runs 次）
    返回一条结果记录（含 strict/loose 双轨检索命中、投票结果、一致率）。
    """
    per_run = []
    for a in answers:
        ok, detail = judge_once(q, a)
        per_run.append({"correct": ok, "detail": detail, "answer": a})

    # --- 多数投票：取判定众数；平票时取第一次的判定（保守，不做额外猜测）---
    valid = [p["correct"] for p in per_run if p["correct"] is not None]
    if valid:
        counter = Counter(valid)
        top_count = max(counter.values())
        winners = [k for k, v in counter.items() if v == top_count]
        final_correct = per_run[0]["correct"] if len(winners) > 1 else winners[0]
        consistency = top_count / len(valid)
    else:
        final_correct, consistency = None, 0.0

    # 代表的答案：取与最终判定相同判定的第一次答案（便于贴表）
    repr_answer = next((p["answer"] for p in per_run if p["correct"] == final_correct), per_run[0]["answer"])
    repr_detail = next((p["detail"] for p in per_run if p["correct"] == final_correct), per_run[0]["detail"])

    rec = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "note": q.get("note", ""),
        "top_ids": top_ids,
        "source_chunk": q.get("source_chunk", []),
        "answer": repr_answer,
        "refused": is_refusal(repr_answer),
        "correct": final_correct,
        "detail": repr_detail,
        "runs": len(answers),
        "consistency": round(consistency, 3),
        "per_run": per_run,   # 每次生成的答案 + 规则原判（--manual-fix off 时用它还原规则判定）
    }

    # --- 检索命中：双轨判据（口径透明，两项并列报）---
    src = set(rec["source_chunk"])
    if q["type"] == "in_material":
        # strict：Top-K 里是否真出现 source_chunk 中任一编号（day14 旧口径，偏严）
        rec["retrieval_hit_strict"] = bool(src) and bool(src & set(top_ids))
        # loose：期望关键词是否落在 Top-K 的上下文里（治 Q4 那种"没召回该段但答对了"的反例）
        kws = q.get("expected_keywords", [])
        if kws:
            low_ctx = context.lower()
            rec["retrieval_hit_loose"] = any(kw.lower() in low_ctx for kw in kws)
        else:
            rec["retrieval_hit_loose"] = None
        # 兼容旧字段名（方便和 day14 结果对照）
        rec["retrieval_hit"] = rec["retrieval_hit_strict"]
    else:
        rec["retrieval_hit_strict"] = None
        rec["retrieval_hit_loose"] = None
        rec["retrieval_hit"] = None

    # --- 人工复核修正（--manual-fix on 时生效，复现基线必须 on）---
    fix = MANUAL_FIXES.get(q["id"])
    if fix is not None:
        if "correct" in fix:
            rec["correct"] = fix["correct"]
        if "detail" in fix:
            rec["detail"] = fix["detail"]
        rec["manual_fixed"] = True
    else:
        rec["manual_fixed"] = False
    return rec


# ---------------------------------------------------------------------------
# 第 6 区：指标汇总
# ---------------------------------------------------------------------------

def summarize(results):
    """把结果汇总成指标字典（三组正确率 + 双轨检索命中 + 防幻觉 F1 + 总正确率 + 投票一致率）。"""
    in_mat = [r for r in results if r["type"] == "in_material"]
    out_mat = [r for r in results if r["type"] == "out_of_material"]

    # ① 检索层命中率（双轨，只看带 source_chunk 的 in_material 题）
    hit_den = [r for r in in_mat if r.get("source_chunk")]
    strict_num = sum(1 for r in hit_den if r.get("retrieval_hit_strict"))
    loose_den = [r for r in hit_den if r.get("retrieval_hit_loose") is not None]
    loose_num = sum(1 for r in loose_den if r.get("retrieval_hit_loose"))

    # ② 生成层正确率（in_material）
    gen_den = [r for r in in_mat if r["correct"] is not None]
    gen_num = sum(1 for r in gen_den if r["correct"])

    # ③ 防幻觉正确率（out_of_material）
    anti_den = [r for r in out_mat if r["correct"] is not None]
    anti_num = sum(1 for r in anti_den if r["correct"])

    # ④ 防幻觉 F1（二分类口径，与 day14 完全一致）
    tp = anti_num
    fp = sum(1 for r in in_mat if r["refused"])   # 资料里有答案却拒答
    fn = len(out_mat) - tp                         # 该拒没拒（幻觉漏网）
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0

    # ⑤ 总正确率（宏观平均）
    judged = [r for r in results if r["correct"] is not None]
    overall = sum(1 for r in judged if r["correct"]) / len(judged) if judged else 0.0

    # ⑥ 投票一致率（--runs ≥ 2 才有意义）：看生成"稳不稳"
    cons = [r["consistency"] for r in results if r.get("runs", 1) > 1]

    # ⑦ 引文合法性（O2-G3）：**两套口径**
    #   原始口径 = 模型自己生成的引文（与 R3/R4/R5 各行同口径，模型侧没变）
    #   交付口径 = 经过 strip/retry 兜底之后真正给出去的引文
    cite_total = sum(r.get("citation_n", 0) for r in results)
    cite_bad_total = sum(len(r.get("citation_bad") or []) for r in results)
    cite_bad_qs = [r["id"] for r in results if r.get("citation_bad")]
    cite_used_qs = [r["id"] for r in results if r.get("citation_n")]
    cite_total_d = sum(r.get("citation_n_delivered", 0) for r in results)
    cite_bad_total_d = sum(len(r.get("citation_bad_delivered") or []) for r in results)
    cite_bad_qs_d = [r["id"] for r in results if r.get("citation_bad_delivered")]
    cite_removed = sum(r.get("citation_removed", 0) for r in results)
    cite_retried_qs = [r["id"] for r in results if r.get("citation_retried")]
    cite_only_bad_qs = [r["id"] for r in results if r.get("citation_only_bad")]
    # 兜底改变了判定的题（**必须单独报**：这是后处理造成的，不能记成"生成变好了"）
    guard_changed = [{"id": r["id"], "raw": r.get("correct_raw"), "delivered": r["correct"]}
                     for r in results
                     if r.get("correct_raw") is not None and r.get("correct_raw") != r["correct"]]
    # ⚠ 9/21 补：F1 的 FP 是由 **refused**（不是 correct）算的 → 只比 correct 会漏报。
    #   实测 R6b：Q1 的 correct 两次都是 ❌，但 refused False→True，FP 从 0 变 1 —— 不在上面那个名单里。
    guard_refused_changed = [{"id": r["id"], "raw": r.get("refused_raw"), "delivered": r["refused"]}
                             for r in results
                             if r.get("refused_raw") is not None
                             and r.get("refused_raw") != r["refused"]]
    cite_pseudo = sum(r.get("citation_pseudo_n", 0) for r in results)

    return {
        "total": len(results),
        "in_material": len(in_mat),
        "out_of_material": len(out_mat),
        "retrieval_hit_strict": (strict_num, len(hit_den)),
        "retrieval_hit_loose": (loose_num, len(loose_den)),
        "gen_correct": (gen_num, len(gen_den)),
        "anti_correct": (anti_num, len(anti_den)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": len(in_mat) - fp},
        "precision": p, "recall": rec, "f1": f1,
        "overall": overall,
        "avg_consistency": (sum(cons) / len(cons)) if cons else None,
        "citation": {"total": cite_total, "bad": cite_bad_total,
                     "bad_qs": cite_bad_qs, "used_qs": cite_used_qs,
                     # 9/21 新增：交付口径 + 兜底动作留痕
                     "total_delivered": cite_total_d,
                     "bad_delivered": cite_bad_total_d,
                     "bad_qs_delivered": cite_bad_qs_d,
                     "removed": cite_removed,
                     "retried_qs": cite_retried_qs,
                     "only_bad_qs": cite_only_bad_qs,
                     "guard_changed": guard_changed,
                     # 9/21 补：F1 看得见的那半边（refused）+ 抓不到的伪引文
                     "guard_refused_changed": guard_refused_changed,
                     "pseudo_n": cite_pseudo},
    }


# ---------------------------------------------------------------------------
# 第 7 区：写评测表 + 追加实验日志
# ---------------------------------------------------------------------------

def preview(text, n=80):
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[:n] + "…"


class Tee:
    """
    把输出同时写给"屏幕 + 文件"的小工具。
    为什么要它：PowerShell 的 `>` / `Tee-Object` 重定向会按控制台编码解码子进程输出，
    中文路径/中文输出下容易存成乱码（实测踩过）。让 Python 自己用 UTF-8 写日志，最稳。
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def pct(n, d):
    return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else "—"


def build_report(meta, results, s, args, profile):
    """把结果拼成《评测表.md》文本。"""
    lines = []
    lines.append(f"# RAG 评测表 v2.1（Day17 · profile={args.profile} · k={args.top_k} · "
                 f"temp={args.temperature} · runs={args.runs} · template={args.template}）\n")
    lines.append(f"- 项目：{meta['project']}")
    lines.append(f"- 知识库：{meta['knowledge_base']}")
    lines.append(f"- 评测题：{meta['total']} 条（in_material {meta['in_material']} + out_of_material {meta['out_of_material']}）")
    if args.limit:
        lines.append(f"- ⚠ 本次为**冒烟子集**：只跑了前 {args.limit} 条题（--limit {args.limit}），指标不代表完整 20 题")
    if args.dry_run:
        lines.append("- ⚠ 本次为 **dry-run**（生成用桩答案、未连模型服务）：只验证脚本流程，**指标数字无意义**")
    lines.append(f"- 评测模型：{profile['model_name']}｜TOP_K={args.top_k}｜temperature={args.temperature}｜"
                 f"模板={args.template}｜--runs={args.runs}｜人工改判={'on' if args.manual_fix else 'off'}")
    if args.citation_policy != "record":
        lines.append(f"- **引文兜底（O2-G3）**：`--citation-policy={args.citation_policy}` —— "
                     + ("非法编号就地剪除（纯后处理，不调模型）" if args.citation_policy == "strip"
                        else "先带白名单重答一次，仍非法再剪除")
                     + "；⚠ 剪除后**不重写成拒答句**（避免把 FN 机械地变成 TP）"
                     + "；本轮「一致率」也是按**交付答案**统计（与旧轮的原始答案口径略有差别）")
    if args.query_instruction == "on":
        lines.append(f"- 检索 query 前缀：**已启用**（bge 官方 instruction：{BGE_QUERY_INSTRUCTION}；段落侧不加）")
    else:
        lines.append("- 检索 query 前缀：未启用（query 与段落两侧都不加前缀）")
    if args.keyword_patch == "on":
        lines.append("- 判据校准：**已应用**（Q5 关键词 LAFAN1 -> LAFAN1/LAFAN，修 day16 发现的「误杀正确答案」）")
    else:
        lines.append("- 判据校准：未应用（沿用 day13 原始判据）")
    if args.note:
        lines.append(f"- 本轮改动：{args.note}")
    if args.rewrite_cache:
        lines.append(f"- 查询改写（O1-R4）：**已启用**（{args.rewrite_cache} 的 `{args.rewrite_mode}` 档；"
                     f"⚠ 只换**检索**用的查询，生成仍用原中文问题）")
    lines.append("")

    lines.append("## 一、汇总指标\n")
    sn, sd = s["retrieval_hit_strict"]
    ln, ld = s["retrieval_hit_loose"]
    gn, gd = s["gen_correct"]
    an, ad = s["anti_correct"]
    c = s["confusion"]
    lines.append("| 指标 | 数值 | 含义 |")
    lines.append("|------|------|------|")
    lines.append(f"| 检索层命中率 · strict | {pct(sn, sd)} | Top-K 与 source_chunk 有交集（day14 旧口径，偏严） |")
    lines.append(f"| 检索层命中率 · loose | {pct(ln, ld)} | 期望关键词是否落在 Top-K 上下文里（新口径，治 Q4 反例） |")
    lines.append(f"| 生成层正确率（in_material） | {pct(gn, gd)} | 资料内有答案的题，期望关键词是否命中答案 |")
    lines.append(f"| 防幻觉正确率（out_of_material） | {pct(an, ad)} | 资料外的题，是否如实答「资料中没有提到」 |")
    lines.append(f"| 总正确率（全部题） | {pct(sum(1 for r in results if r['correct'] is True), len([r for r in results if r['correct'] is not None]))} | 宏观平均 |")
    if s["avg_consistency"] is not None:
        lines.append(f"| 生成一致率（同题多跑多数占比） | {s['avg_consistency']:.3f} | 越大说明生成越稳；小说明单次数字有随机性 |")
    cc = s["citation"]
    lines.append(f"| 引文合法性（O2-G3） | 原始生成 引文 {cc['total']} 处 ｜ 非法 {cc['bad']} 处 | "
                 f"答案里的引文编号是否真在本轮 Top-K 里（口径与 R3/R4/R5 各行一致） |")
    if cc["bad_qs"]:
        lines.append(f"| ↑ 含非法引用的题 | {cc['bad_qs']} | 这些题的答案引了**模型没看过**（或把参考文献编号当段落号）的编号 |")
    if args.citation_policy != "record":
        lines.append(f"| **引文兜底（policy={args.citation_policy}）** | "
                     f"交付 引文 {cc['total_delivered']} 处 ｜ 数字编号非法 {cc['bad_delivered']} 处 | "
                     f"剪除 {cc['removed']} 处"
                     + (f"（= {args.runs} 次生成**累计**；上面『非法』只算代表答案那一次）"
                        if args.runs > 1 else "")
                     + (f"；重答 {len(cc['retried_qs'])} 题" if cc["retried_qs"] else "")
                     + (f"；⚠ 剪完无内容 {cc['only_bad_qs']}" if cc["only_bad_qs"] else "") + " |")
        if cc["pseudo_n"]:
            lines.append(f"| ⚠ 口径边界：抓不到的『伪引文』 | {cc['pseudo_n']} 处 | "
                         "形如 `[资料©]`/`[resourceì]` 的**非数字**引文形状（模型用'引文的样子'伪装内容）；"
                         "`CITE_RE` 只认数字 → **『数字编号非法 0 处』≠『答案干净』** |")
        if cc["guard_changed"]:
            lines.append(f"| ⚠ 判定随兜底翻转（correct） | "
                         + "，".join(f"Q{g['id']} `{g['raw']}` → `{g['delivered']}`" for g in cc["guard_changed"])
                         + " | **后处理造成的翻转，不得计入「生成层变好」** |")
        if cc["guard_refused_changed"]:
            lines.append(f"| ⚠ **拒答状态随兜底翻转（refused）** | "
                         + "，".join(f"Q{g['id']} `{g['raw']}` → `{g['delivered']}`"
                                     for g in cc["guard_refused_changed"])
                         + " | **F1 的 FP 按 refused 算 → 这一项直接改动 F1**；"
                         "只比 correct 会漏报（R6b 实测：Q1 的 correct 两次都 ❌，却让 FP +1） |")
        else:
            lines.append("| ↑ 兜底是否影响 F1 | **未改变任何一题的拒答状态** | "
                         "设计约束：剪除非法编号后**不重写成拒答句** → 不会把 FN 机械地变成 TP |")
    lines.append("")
    lines.append("### 防幻觉 F1（二分类口径：判'拒答'这事做得好不好）\n")
    lines.append("| 混淆矩阵 | 真实：资料外（out，应拒答） | 真实：资料内（in，应作答） |")
    lines.append("|----------|------------------------------|------------------------------|")
    lines.append(f"| 模型拒答 | TP = {c['tp']}（真防住幻觉） | FP = {c['fp']}（该答没答） |")
    lines.append(f"| 模型没拒答 | FN = {c['fn']}（幻觉漏网） | TN = {c['tn']}（正常作答） |")
    lines.append("")
    lines.append(f"- 精确率 = TP/(TP+FP) = {s['precision']:.3f}（拒答的时候，拒得对不对）")
    lines.append(f"- 召回率 = TP/(TP+FN) = {s['recall']:.3f}（该拒的题里，拒住了多少）")
    lines.append(f"- **F1 = 2·P·R/(P+R) = {s['f1']:.3f}**")
    lines.append("")

    lines.append("## 二、逐题结果\n")
    lines.append("| id | 类型 | 检索 strict | 检索 loose | 生成判定 | 一致率 | 判定说明 | 答案摘要 |")
    lines.append("|----|------|-------------|------------|----------|--------|----------|----------|")
    for r in results:
        if r["type"] == "in_material":
            st = f"✅ {r['source_chunk']}∩{r['top_ids']}" if r.get("retrieval_hit_strict") else f"❌ top={r['top_ids']}"
            lo = "✅" if r.get("retrieval_hit_loose") else "❌"
        else:
            st, lo = "—", "—"
        mark = "✅" if r["correct"] else ("❌" if r["correct"] is False else "❓")
        cons = f"{r['consistency']:.2f}" if r.get("runs", 1) > 1 else "—"
        lines.append(f"| {r['id']} | {r['type']} | {st} | {lo} | {mark} | {cons} | {r['detail']} | {preview(r['answer'], 60)} |")

    lines.append("\n## 三、附录：每题完整答案（人工复核用）\n")
    for r in results:
        lines.append(f"### Q{r['id']}（{r['type']}）{r['question']}")
        lines.append(f"- 判定：{'✅' if r['correct'] else ('❌' if r['correct'] is False else '❓')} ｜ {r['detail']}"
                     + ("（人工改判）" if r.get("manual_fixed") else ""))
        if r["type"] == "in_material":
            lines.append(f"- 检索 Top-K：{r['top_ids']}（期望 source_chunk：{r['source_chunk']}）")
        if r.get("retrieval_query"):
            lines.append(f"- 检索用的查询：{r['retrieval_query']}")
        if r.get("citation_n"):
            bad = r.get("citation_bad") or []
            lines.append(f"- 引文：原始生成 {r['citation_n']} 处 ｜ 非法 {('无' if not bad else bad)}"
                         + (f" ｜ 剪除非法引用后：{r['answer_citation_cleaned']}" if bad else ""))
        if args.citation_policy != "record":
            bad_d = r.get("citation_bad_delivered") or []
            lines.append(f"- 兜底（policy={args.citation_policy}）：剪除 {r.get('citation_removed', 0)} 处"
                         + (f"；重答 {r['citation_retried']} 次" if r.get("citation_retried") else "")
                         + f"；交付口径数字编号非法 {('无' if not bad_d else bad_d)}"
                         + (f"；⚠ 另有 {r['citation_pseudo_n']} 处非数字『伪引文』（校验器抓不到）"
                            if r.get("citation_pseudo_n") else "")
                         + ("；⚠ 剪完已无内容" if r.get("citation_only_bad") else ""))
            if r.get("correct_raw") != r["correct"]:
                lines.append(f"- ⚠ **判定随兜底翻转**：原始 `{r.get('correct_raw')}` → 交付 `{r['correct']}`"
                             f"（后处理造成，非生成层变化）")
            if r.get("refused_raw") is not None and r.get("refused_raw") != r["refused"]:
                lines.append(f"- ⚠ **拒答状态随兜底翻转**：原始 `{r['refused_raw']}` → 交付 `{r['refused']}`"
                             f"（**F1 的 FP 按 refused 算 → 这会改动 F1**，非生成层变化）")
        lines.append(f"- 备注（出题依据）：{r['note']}")
        lines.append(f"- 答案：{r['answer']}"
                     + (f"\n- 原始生成（未兜底）：{r['answer_raw']}" if r.get("answer_raw") else "") + "\n")
    return "\n".join(lines)


def append_log(args, s, profile):
    """
    把本轮结果作为一行插入 第四周\\实验日志.md 的「一、实验记录表」末尾（--append-log 时）。

    Day16 版是 `open(..., "a")` 直接追加到文件末尾 → 行会落进「三、失败尝试记录」表
    （Day16 实跑踩到，手工移回）。
    Day17 修正：定位到「## 一、实验记录表」这一节，找"分隔行之后的最后一行数据行"，
               插在它后面；没找到表结构就退回追加，至少不丢数据。
    """
    sn, sd = s["retrieval_hit_strict"]
    ln, ld = s["retrieval_hit_loose"]
    gn, gd = s["gen_correct"]
    an, ad = s["anti_correct"]
    overall_num = sum(1 for r in _last_results if r["correct"] is True)
    overall_den = len([r for r in _last_results if r["correct"] is not None])
    row = (f"| {args.date} | {args.exp_id} | {args.note or '—'} | "
           f"{profile['model_name']} / k={args.top_k} / T={args.temperature} / runs={args.runs} / "
           f"{args.template} / qi={args.query_instruction} / patch={args.keyword_patch} / "
           f"rw={args.rewrite_mode if args.rewrite_cache else 'off'} / cite={args.citation_policy} | "
           f"{pct(sn, sd)} | {pct(ln, ld)} | {pct(gn, gd)} | {pct(an, ad)} | {s['f1']:.3f} | "
           f"{pct(overall_num, overall_den)} | "
           f"{args.conclusion or '（待填）'} | {os.path.basename(args.out_dir)} |")

    if not os.path.exists(DEFAULT_LOG_MD):
        with open(DEFAULT_LOG_MD, "w", encoding="utf-8") as f:
            f.write("# 第四周实验日志（优化工程的\"账本\"）\n\n## 一、实验记录表\n\n")
            f.write("| 日期 | 编号 | 改了哪个变量 | 配置 | 检索 strict | 检索 loose | 生成正确率 | "
                    "防幻觉正确率 | 防幻觉 F1 | 总正确率 | 结论 / 下一步 | 结果目录 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            f.write(row + "\n")
        return DEFAULT_LOG_MD

    with open(DEFAULT_LOG_MD, encoding="utf-8") as f:
        lines = f.readlines()

    sec = next((i for i, l in enumerate(lines) if l.strip().startswith("## 一、实验记录表")), None)
    if sec is None:
        with open(DEFAULT_LOG_MD, "a", encoding="utf-8") as f:
            f.write(row + "\n")
        return DEFAULT_LOG_MD

    sep_idx, last_data = None, None
    for i in range(sec + 1, len(lines)):
        if lines[i].startswith("## "):
            break
        stripped = lines[i].strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        # 分隔行 |---|---|：每个单元都是 "-" / ":" / 空格组成且非空
        if cells and all(c and set(c) <= set("-: ") for c in cells):
            sep_idx = i
            continue
        if sep_idx is not None and any(cells):
            last_data = i          # 分隔行之后的非空数据行（最后一行即最新一轮）
    insert_at = last_data if last_data is not None else sep_idx
    if insert_at is None:
        insert_at = sec

    lines.insert(insert_at + 1, row + "\n")
    with open(DEFAULT_LOG_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return DEFAULT_LOG_MD


# ---------------------------------------------------------------------------
# 第 8 区：主流程
# ---------------------------------------------------------------------------
_last_results = []   # 给 append_log 用的最近一次结果（避免多传一层参数）


def main():
    parser = argparse.ArgumentParser(
        description="RAG 评测脚本 v2.1（参数化 + 多数投票 + 双轨判据 + query instruction / 判据补丁）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--profile", default=None, help="lora | original（默认取 config.json 的 default_profile）")
    parser.add_argument("--top-k", type=int, default=4, help="检索返回条数（默认 4）")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度（默认 0.2）")
    parser.add_argument("--runs", type=int, default=1, help="每题重复生成次数，≥3 时取多数投票（默认 1）")
    parser.add_argument("--template", default="v1", choices=list(PROMPT_TEMPLATES), help="提示词模板版本（默认 v1）")
    parser.add_argument("--query-instruction", default="off", choices=["on", "off"],
                        help="检索 query 侧是否加 bge 官方 instruction 前缀（O1-R1，默认 off）")
    parser.add_argument("--keyword-patch", default="on", choices=["on", "off"],
                        help="是否应用判据校准补丁（Q5 关键词放宽，默认 on）")
    parser.add_argument("--out", default=None, help="输出目录（默认 result_<profile>_k<K>[_qi]...）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用；0=全部）")
    parser.add_argument("--max-tokens", type=int, default=512, help="单条答案上限（默认 512）")
    parser.add_argument("--rewrite-cache", default="", help="改写缓存 json（Day19 O1-R4；留空=不改写）")
    parser.add_argument("--rewrite-mode", default="term", choices=["plain", "term", "hyde", "both"],
                        help="用缓存里的哪一档当检索查询（默认 term）")
    parser.add_argument("--manual-fix", default="on", choices=["on", "off"], help="是否应用人工改判表（默认 on）")
    parser.add_argument("--citation-policy", default="record", choices=["record", "strip", "retry"],
                        help="O2-G3 引文兜底：record=只留痕（默认，与旧轮同行为）；"
                             "strip=非法编号就地剪除；retry=先重答一次仍非法再剪")
    parser.add_argument("--append-log", action="store_true",
                        help="把本轮结果作为一行插入 第四周\\实验日志.md 的「一、实验记录表」")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连模型服务，生成用桩答案（只验证脚本流程；数字无意义）")
    parser.add_argument("--note", default="", help="本轮改了哪个变量（写进评测表与日志）")
    parser.add_argument("--exp-id", default="", help="实验编号（如 R1），写日志用")
    parser.add_argument("--date", default="", help="实验日期，写日志用（如 9/21）")
    parser.add_argument("--conclusion", default="", help="本轮结论（写日志用）")
    args = parser.parse_args()

    cfg, cfg_dir = load_config()
    profile_name = args.profile or cfg.get("default_profile", "lora")
    if profile_name not in cfg["profiles"]:
        print(f"❌ 没有这个 profile：{profile_name}。可选：{list(cfg['profiles'])}")
        sys.exit(1)
    args.profile = profile_name
    profile = cfg["profiles"][profile_name]

    # 输出目录命名：默认 result_<profile>_k<K>；非默认旋钮加后缀，保证每轮互不覆盖
    if args.out:
        out_dir = args.out if os.path.isabs(args.out) else os.path.join(SCRIPT_DIR, args.out)
    else:
        name = f"result_{profile_name}_k{args.top_k}"
        if args.query_instruction == "on":
            name += "_qi"
        if args.temperature != 0.2:
            name += f"_temp{args.temperature}"
        if args.template != "v1":
            name += f"_{args.template}"
        if args.runs > 1:
            name += f"_runs{args.runs}"
        if args.citation_policy != "record":
            name += f"_cite{args.citation_policy}"
        if args.dry_run:
            name += "_dryrun"
        out_dir = os.path.join(SCRIPT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir

    # 本轮运行日志（屏幕 + 文件双写；Python 自己用 UTF-8 写，避免 shell 重定向乱码）
    log_path = os.path.join(out_dir, "run_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = sys.stdout

    eval_json = DEFAULT_EVAL_JSON
    if not os.path.exists(eval_json):
        print(f"❌ 找不到评测题文件：{eval_json}")
        log_file.close()
        sys.exit(1)
    data = load_json(eval_json)
    meta, questions = data["meta"], data["questions"]
    if args.limit > 0:
        questions = questions[:args.limit]

    # ---- day19 新增：读改写缓存（O1-R4）----
    # 纪律 #1（教程 1.2 节）：改写【只给检索层用】，生成层仍是原中文问句 →
    #   ① 保证"只改一个变量"；② 保证判分仍按原问题的 expected_keywords（答案风格不变）。
    rewrite_cache = {}
    rw_label = "off（检索用原始问题）"
    if args.rewrite_cache:
        if not os.path.exists(args.rewrite_cache):
            print(f"❌ 找不到改写缓存：{args.rewrite_cache}（先跑 rewrite_queries.py 生成）")
            log_file.close()
            sys.exit(1)
        with open(args.rewrite_cache, encoding="utf-8") as f:
            rewrite_cache = json.load(f)
        rw_meta = rewrite_cache.get("_meta", {}) or {}
        print(f"[改写] 已加载 {args.rewrite_cache}（档位={args.rewrite_mode}）"
              f"｜模型={rw_meta.get('model', '?')}")
        print(f"[改写] 缓存元信息：time={rw_meta.get('time', '?')}｜"
              f"prompt_version={rw_meta.get('prompt_version', '?')}｜modes={rw_meta.get('modes', '?')}")
        rw_label = f"on（检索用 {args.rewrite_mode} 档改写；生成仍用原中文问题）"

    rw_missing = []          # 缓存里没有该档（或档位字段为空）的题号 → 静默回退到原问题

    def pick_query(q):
        """返回 (用于检索的查询, 是否真的用了改写)。取不到就静默回退原问题，并记下题号。"""
        if not rewrite_cache:
            return q["question"], False
        rec = rewrite_cache.get(str(q["id"])) or {}
        rq = (rec.get(args.rewrite_mode) or "").strip()
        if rq:
            return rq, True
        rw_missing.append(q["id"])
        return q["question"], False

    template = PROMPT_TEMPLATES[args.template]
    persist_dir = resolve_path(cfg["persist_dir"], cfg_dir)
    embed_model = cfg["embed_model"]

    # Day17：先把"判据校准补丁"应用到题目上（默认 on），并在终端留痕
    n_patch, patch_msg = apply_keyword_patch(questions, enabled=(args.keyword_patch == "on"))
    qi_text = BGE_QUERY_INSTRUCTION if args.query_instruction == "on" else ""
    qi_label = "on（查询侧加 bge instruction）" if qi_text else "off（查询侧不加前缀）"

    print("=" * 72)
    print(f"RAG 评测 v2.1 开始｜profile={profile_name}｜TOP_K={args.top_k}｜temp={args.temperature}｜"
          f"runs={args.runs}｜template={args.template}")
    print(f"query instruction：{qi_label}｜判据补丁：{patch_msg}")
    print(f"查询改写：{rw_label}")
    print(f"引文兜底（O2-G3）：{args.citation_policy}"
          + ("（只留痕，不改答案、不改判定）" if args.citation_policy == "record"
             else "（**强制白名单**：非法编号会被剪除" + ("，先重答一次" if args.citation_policy == "retry" else "") + "）"))
    print(f"评测题 {len(questions)} 条（本次运行）｜输出目录：{os.path.relpath(out_dir, SCRIPT_DIR)}")
    if args.note:
        print(f"本轮改动：{args.note}")
    print("=" * 72)

    # 1) 服务两查（dry-run 跳过）
    if args.dry_run:
        print("[dry-run] 跳过服务检查与真实生成：答案用桩文本，只验证脚本流程，指标数字无意义。")
    else:
        ok, msg = check_service(profile)
        print(msg)
        if not ok:
            print("→ 先启动与 profile 匹配的服务，再重跑本脚本。")
            sys.exit(1)

    # 2) 加载向量库
    print(f"[OK] 加载向量库：{persist_dir}")
    if qi_text:
        print(f"[OK] 检索 query 侧已加 bge instruction 前缀：{BGE_QUERY_INSTRUCTION}")
    store = load_store(persist_dir, embed_model, qi_text)

    # 3) 逐题评测（每题检索一次、生成 runs 次）
    results = []
    out_json = os.path.join(out_dir, "eval_results.json")
    try:
        for i, q in enumerate(questions, start=1):
            print("-" * 72)
            print(f"[{i:>2}/{len(questions)}] id={q['id']:>2} {q['type']}｜{q['question']}")
            rq, used_rewrite = pick_query(q)
            if used_rewrite and rq != q["question"]:
                print(f"   检索查询: {rq}")
            docs, ids = retrieve(store, rq, args.top_k)          # ← 检索用改写后的查询
            context = build_context(docs)

            answers = []
            for r in range(args.runs):
                answers.append(generate_answer(profile, template, q["question"], context,
                                               args.temperature, args.max_tokens, dry_run=args.dry_run))
                #                                                    ↑ 生成仍用【原中文问题】（纪律 #1）

            # day19 O2-G3 升级：强制白名单（默认 record = 原样交付 → R3/R4/R5 数字可复现）
            delivered, cite_meta = apply_citation_policy(
                args.citation_policy, profile, template, q["question"], context, answers, ids,
                args.temperature, args.max_tokens, dry_run=args.dry_run)

            # 原始生成先判一次（**只用来做统计可比**：⑧ 的两列要跟 R3/R4/R5 同口径）。
            # judge_question 不调模型（纯规则+投票），多算一次的开销可忽略。
            rec_raw = judge_question(q, ids, context, answers)
            rec = rec_raw if args.citation_policy == "record" \
                else judge_question(q, ids, context, delivered)

            rec["retrieval_query"] = rq                          # ← 落进结果，便于复盘"它到底问了什么"
            rec["rewrite_used"] = used_rewrite
            # ⑧ 引文合法性：**两套口径并列**（这是 9/21 升级的关键）
            #   citation_* = 对【原始生成】的统计 → 与 R3/R4/R5 各行完全同口径（模型侧没变）
            #   *_delivered = 对【交付答案】的统计 → 兜底（strip/retry）之后的效果
            rec["answer_raw"] = rec_raw["answer"]
            rec["correct_raw"] = rec_raw["correct"]
            # ⚠ 9/21 补：`refused` 才是 F1 里 FP 的驱动量（fp = in_mat 里 refused 的题数）。
            #   实测 R6b 的 Q1 就是"correct 两次都 ❌、只有 refused 变了"→ 它让 FP +1
            #   却没出现在"判定随兜底改变"名单里。必须单独记 refused 的原始值。
            rec["refused_raw"] = is_refusal(rec_raw["answer"])
            rec["answer_citation_cleaned"], rec["citation_bad"] = \
                legal_citation_filter(rec_raw["answer"], ids)
            rec["citation_n"] = len(CITE_RE.findall(rec_raw["answer"]))
            rec["answer_delivered_cleaned"], rec["citation_bad_delivered"] = \
                legal_citation_filter(rec["answer"], ids)
            rec["citation_n_delivered"] = len(CITE_RE.findall(rec["answer"]))
            # 9/21 新增：**非数字编号**的"伪引文"计数（实测 Q3 出现 [资料©]/[resourceì]/[resourceright]）
            #   CITE_RE 只认数字 → 这类全部漏检。不计入"非法"，只做**口径边界**的量化。
            rec["citation_pseudo_n"] = count_pseudo_citations(rec["answer"])
            # 兜底动作留痕（"改了什么"必须能逐题回放）
            # ⚠ 单位注意：removed/retried/only_bad 是 **runs 次生成累计**，
            #   而 citation_bad / citation_n 只统计**代表答案**那一次 → 前者通常是后者的 1~3 倍。
            rec["citation_removed"] = cite_meta["removed"]
            rec["citation_retried"] = cite_meta["retried"]
            rec["citation_only_bad"] = cite_meta["only_bad_after"]
            if not args.manual_fix:
                # 关掉人工改判时，把判定还原成"规则原判"（per_run[0] 的规则结果）
                rec["correct"] = rec["per_run"][0]["correct"]
                rec["detail"] = "（规则原判）" + rec["per_run"][0]["detail"]
                rec["manual_fixed"] = False
            results.append(rec)

            mark = "✅" if rec["correct"] else ("❌" if rec["correct"] is False else "❓")
            print(f"   Top-K: {ids}")
            print(f"   检索  : strict={'✅' if rec['retrieval_hit_strict'] else ('❌' if rec['retrieval_hit_strict'] is False else '—')}"
                  f"  loose={'✅' if rec['retrieval_hit_loose'] else ('❌' if rec['retrieval_hit_loose'] is False else '—')}")
            if args.runs > 1:
                print(f"   一致率: {rec['consistency']:.2f}（{args.runs} 次生成）")
            if rec["citation_n"]:
                line = f"   引文  : {rec['citation_n']} 处"
                line += (f"｜⚠ 非法 {rec['citation_bad']}（不在本轮 Top-K 内）"
                         if rec["citation_bad"] else "（全部合法）")
                print(line)
            # O2-G3 强制白名单：把"兜底做了什么"打出来（默认 record 时不打印，保持旧观感）
            if args.citation_policy != "record":
                print(f"   兜底  : policy={args.citation_policy}"
                      f"｜剪除 {rec['citation_removed']} 处"
                      + (f"｜重答 {rec['citation_retried']} 次" if rec["citation_retried"] else "")
                      + (f"｜交付口径非法 {rec['citation_bad_delivered']}"
                         if rec["citation_bad_delivered"] else "｜交付口径非法 0 处 ✅")
                      + (f"｜⚠ 剪完已无内容 {rec['citation_only_bad']} 题" if rec["citation_only_bad"] else ""))
                if rec["correct_raw"] != rec["correct"]:
                    print(f"   ⚠ 判定随兜底改变：raw={'✅' if rec['correct_raw'] else '❌'} → "
                          f"交付={'✅' if rec['correct'] else '❌'}（**这是后处理造成的，须与生成层区分**）")
                if rec["answer_raw"] != rec["answer"]:
                    print(f"   原始答案: {preview(rec['answer_raw'], 140)}")
            print(f"   判定  : {mark} ｜ {rec['detail']}")
            print(f"   A: {preview(rec['answer'], 160)}")
            # 逐题落盘：中断也不丢已跑的题
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    except KeyboardInterrupt:
        print("\n[中断] 已保存已完成的题目到 eval_results.json。")
    except Exception as e:
        print(f"\n[出错] {e}（已跑的题已保存）")

    if not results:
        print("没有任何结果，退出。")
        log_file.close()
        return

    # 4) 汇总 + 写表
    global _last_results
    _last_results = results
    s = summarize(results)
    report_path = os.path.join(out_dir, "评测表.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report(meta, results, s, args, profile))

    print("\n" + "=" * 72)
    print("汇总指标")
    print("=" * 72)
    print(f"① 检索命中 strict : {pct(*s['retrieval_hit_strict'])}")
    print(f"① 检索命中 loose  : {pct(*s['retrieval_hit_loose'])}")
    print(f"② 生成正确率(in)  : {pct(*s['gen_correct'])}")
    print(f"③ 防幻觉正确率(out): {pct(*s['anti_correct'])}")
    print(f"④ 总正确率        : {100.0 * s['overall']:.1f}%"
          f"（{sum(1 for r in results if r['correct'] is True)}/{len([r for r in results if r['correct'] is not None])}）")
    c = s["confusion"]
    print(f"⑤ 防幻觉F1        : P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
          f"（TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}）")
    if s["avg_consistency"] is not None:
        print(f"⑥ 生成一致率      : {s['avg_consistency']:.3f}")
    # day19：改写接线自检——"有几题其实没用上改写"必须看得见，
    # 否则 `retrieval_query` 全是原文时，你会误以为是"改写不涨分"而不是"接线没生效"。
    if args.rewrite_cache:
        n_used = sum(1 for r in results if r.get("rewrite_used"))
        print(f"⑦ 改写生效        : {n_used}/{len(results)} 题用了「{args.rewrite_mode}」档改写")
        if rw_missing:
            print(f"   ⚠ 未取到改写的题号：{rw_missing}"
                  f"（缓存里该档缺失或为空 → 已回退原问题）")
        if n_used == 0:
            print("   ⚠⚠ 一题都没用上改写：确认 --rewrite-mode 与缓存字段名一致（plain/term/hyde/both）")
    # day19 O2-G3：引文合法性（原始口径 vs 交付口径）——"模型引的资料编号是不是它真看过的"
    cc = s["citation"]
    print(f"⑧ 引文合法性      : 引文 {cc['total']} 处（{len(cc['used_qs'])} 题带引文），"
          f"非法 {cc['bad']} 处"
          + (f"｜题号 {cc['bad_qs']}" if cc["bad_qs"] else "（全部落在本轮 Top-K 内 ✅）"))
    if args.citation_policy != "record":
        print(f"   交付口径        : policy={args.citation_policy}"
              f"｜引文 {cc['total_delivered']} 处，非法 {cc['bad_delivered']} 处"
              + (f"｜题号 {cc['bad_qs_delivered']}" if cc["bad_qs_delivered"] else " ✅")
              + ("｜⚠ `非法 0 处` 只保证**数字编号**；另有 "
                 f"{cc['pseudo_n']} 处非数字『伪引文』（[资料©] 之类）抓不到" if cc["pseudo_n"] else ""))
        print(f"   兜底动作        : 剪除 {cc['removed']} 处（= {args.runs} 次生成累计；"
              f"上面『非法』只算代表答案那一次）"
              + (f"｜重答 {len(cc['retried_qs'])} 题 {cc['retried_qs']}" if cc["retried_qs"] else "")
              + (f"｜⚠ 剪完已无内容 {cc['only_bad_qs']}" if cc["only_bad_qs"] else ""))
        if cc["guard_changed"]:
            print(f"   ⚠ 判定随兜底改变 {len(cc['guard_changed'])} 题："
                  + "，".join(f"Q{g['id']} {g['raw']}→{g['delivered']}" for g in cc["guard_changed"]))
            print("      ↑ 这是**后处理造成的判定翻转**，绝不能算进'生成层变好'（报告须单独说明）")
        else:
            print("   ✅ 判定（correct）未因兜底改变任何一题")
        # ⚠ F1 的 FP 看的是 refused，不是 correct → 必须单独报，否则会漏报（R6b 的 Q1 就是这种）
        if cc["guard_refused_changed"]:
            print(f"   ⚠ 拒答状态随兜底改变 {len(cc['guard_refused_changed'])} 题："
                  + "，".join(f"Q{g['id']} {g['raw']}→{g['delivered']}" for g in cc["guard_refused_changed"]))
            print("      ↑ **这一项会直接改动防幻觉 F1 的 FP**（F1 按 refused 算），必须与生成层分开报")
        else:
            print("   ✅ 拒答状态未因兜底改变任何一题（F1 的 FP 不受兜底影响）")
    print("=" * 72)
    print(f"[完成] 已生成：\n  {report_path}\n  {out_json}")

    # 5) 实验日志（可选）
    if args.append_log:
        log_path = append_log(args, s, profile)
        print(f"[日志] 已追加一行到：{log_path}")
    else:
        print("\n[日志行] 想记进实验日志，可复制下面这行（或加 --append-log 自动追加）：")
        sn, sd = s["retrieval_hit_strict"]
        ln, ld = s["retrieval_hit_loose"]
        gn, gd = s["gen_correct"]
        an, ad = s["anti_correct"]
        print(f"| {args.date or '日期'} | {args.exp_id or '编号'} | {args.note or '改动'} | k={args.top_k} T={args.temperature} "
              f"runs={args.runs} {args.template} qi={args.query_instruction} patch={args.keyword_patch} "
              f"rw={args.rewrite_mode if args.rewrite_cache else 'off'} cite={args.citation_policy} | "
              f"{pct(sn, sd)} | {pct(ln, ld)} | {pct(gn, gd)} | {pct(an, ad)} | "
              f"F1={s['f1']:.3f} | | {os.path.basename(out_dir)} |")

    print(f"[日志文件] 本轮完整运行日志：{log_path}")
    log_file.flush()
    log_file.close()


if __name__ == "__main__":
    main()
