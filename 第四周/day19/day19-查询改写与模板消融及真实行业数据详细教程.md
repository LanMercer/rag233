# Day 19（第四周第 4 个工作日 · 项目整改 D4：O1-R4 查询改写落地 + O2-G2 模板 09 v2 消融 + F2 真实行业数据 + F3 真实/合成对照微调 + O3 成果报告初稿）· 超详细新手教程

> 时间安排：主线 4~~5 小时 + 零散时间 0.5~~1 小时 + 睡前 15 分钟
> 日期说明：计划日 **2026/9/23（周三）**，第四周第 4 个工作日（第四周 5 个工作日 = 9/18、9/21、9/22、**9/23**、9/24，day 编号连续 = `day16`~`day20`）。日期可顺延，内容顺序不变；**每天只新建当天那个 day 文件夹**。 **日期口径（沿用 Day18 统一结论）**：本日起，「日期 = **实际执行日**」，不是计划日。日志/报告里写日期前先看一眼` run_log.txt`的时间戳；同一天跑多轮靠**编号**区分（R3/R4/R5…）。 适用对象：**day18 已完成**——你手里有`第四周\day18（`data_selfcheck.py`、`retrieval_lab.py`、`retrieval_lab.json`、`build_paper_qa.py`、`sft_data_paper.json` 90 条、`app.py`）、`第四周\day17\`（`eval_v2.py` v2.1、`start.py`、`start.bat`、`result_lora_k8_runs3\` = **R3**、`****result_lora_k8_qi_runs3\`****、`**result_lora_k4_runs3\`**、`result_lora_k4_qi_runs3\`）、`第四周\实验日志.md`（含 R3 行 + 决策门 + 4 条失败尝试）、`第三周\day13\`（`chroma_db\` **105 chunk、`**chunks.json`、`local_api_lora.py`）、`第三周\day15\config.json`、`第三周\day12\train_lora.py` **+** `lora_adapter\`、`第三周\day11\sft_data.json`（300 条合成数据）、****`第四周\发布包\`（`space_demo\requirements.txt` **已 pin** `pydantic==2.10.6`）。
> 本教程的目标：把 day18 的"诊断结论"变成"计入成绩的优化结果"，并把 HR 意见③（真实数据）彻底闭环。今天有三件"出成果"的事——① 查询改写真实现（day18 的 100% 是人工开卷的"上界"，今天用本地模型自动改写，跑出**能写进报告的成绩**）；② **模板 09 v2 + 逐句消融**（打生成层的稳定靶子：Q11/Q14/Q18 的编造 FN）；③ **真实数据两条腿都落地**（F2 真实行业 QA ≥100 条 + F3 真实 vs 合成对照微调）。再加 O3 成果报告初稿。
> 依据：`00-基础内容总纲.md`、`第四周\01-第四周详细计划.md`（**D4 章节**）、`第四周\01-第四周项目总结与优化路线图.md`（O 线 O1-R4/R5/R6 + O2-G2/G3 + F 线 F2/F3 + O3）、`第四周\day18\day18工作汇报.md`（第六节「明日安排」7 条）、`第四周\实验日志.md`、`第四周\day17\eval_v2.py`（day19 版的改造对象）。

---

## ⚠️ 计划衔接提示（day18 已经把"瓶颈"钉死了，今天怎么接）

> **情况说明**：day18 用**便宜实验**（不调模型、35 秒一轮）把问题定位得非常干净——**卷子是干净的**（数据自检 9/10 自洽、0 错位、0 缺失）→ **库里有答案**（关键词都在）→ **换英文术语问就全对**（strict 40% → 100%，三通道一致）→ **瓶颈在"查询侧的中英语言错位"**。
> 但 day18 那 100% 是**人工写的理想查询**，等于**开卷考试**，只能用来"定位瓶颈"，**不能当成绩**。**今天第一件事就是把这件事做成真实现，并让数字进「一、实验记录表」。**

**Day18 已完成 → 今天的处理：**


| Day18 产出 / 结论                                                                                                       | 今天（Day19）怎么接                                                                 |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 数据自检：**9/10 自洽、0 错位、0 缺失、1 需人工**                                                                                    | 已排除"标注错"这条岔路 → **不再重做**；今天的失败只往"查询侧 / 生成层"归因                                 |
| `retrieval_lab.py` plain：**vector 4/10、bm25 4/10、hybrid 3/10、并集 7/10**                                              | 作为**诊断基线**；今天用**改写后的查询**重跑同一套筛查（看并集是否抬升 → 决定融合值不值得做）                         |
| HyDE 冒烟：三通道 **10/10（上界）**                                                                                           | **今天第 1 步把"人工理想查询"换成"模型自动改写"**，用 `eval_v2.py` 出**计入成绩**的版本 ⭐⭐                |
| RRF 负结果（4/10 → 3/10）+ 三个机制                                                                                          | 反证是"先改写、再融合"→ 今天第 2 步**在改写后复筛**，确认"融合是否有意义"                                  |
| **R3（k=8 / qi=off / patch=on / runs=3）：strict 4/10、loose 5/10、生成 4/10、防幻觉 7/10、F1 0.824、总 11/20、一致率 0.950、唯一 FP=0** | **今天的统一对照基线就是 R3**（已在日志里）→ **今天所有实验只和 R3 比**，别再去和 runs=1 的 day16 R2 相减（口径纪律） |
| 生成层靶子：**R3 的 FN = Q11 / Q14 / Q18**（稳定编造，Q11 一致率 1.00）                                                              | **今天第 3 步的模板 09 v2**：加"不得编造编号/数字"硬约束（O2-G2）                                  |
| F1 论文真实 QA **90 条**（逐字回查 90/90、泄漏 0、五类 18/22/17/25/8）                                                               | **今天的"真实数据"从论文扩到行业**（F2 ≥100 条）；F1 那 90 条直接进 F3 的训练集                         |
| `app.py` 完成版（上传 PDF / 指标对比 / 召回可视化）                                                                                 | 今天**不动界面**；它读 `实验日志.md`，今天新增的 R4/R5 行会**自动出现在页面上**                           |
| `发布包\space_demo\requirements.txt` 补 `pydantic==2.10.6`                                                              | 今天**可选**做一次 DeepSeek 真冒烟（关闭 day18 待补项 4 的后半，见第 7 步）                          |


**Day18 报告里留下的"未关闭项" → 今天的处理：**

1. **HyDE 从"手工冒烟"做成"模型自动改写"** → **今天第 1 步**（`rewrite_queries.py` + `eval_v2.py --rewrite-cache`）⭐ 今天的主线。
2. **改写之后再上融合 / 评估 Rerank** → **今天第 2 步**。
3. **拿 R3 当新对照基线** → **今天全程执行**（同 k=8、同 qi=off、同 runs=3；只改一个变量）。
4. **O2-G2 模板 09 v2 + 消融** → **今天第 3 步**（v2 草稿 day17 已占位，今天复核 + 并排跑 + 逐句消融）。
5. **F2 真实行业数据 / F3 对照微调** → **今天第 4、5 步**（HR 意见③的最后一环）。
6. **⚠ day18 待补项 4 的后半：**`space_demo` **的 DeepSeek 端到端真冒烟**（要 key、花额度）→ **今天第 7 步（可选）**，环境与 UI 冒烟已就绪，只剩"真起页面、真调 API"。
7. **零散时间三项无可核验产物** → 不代填；今天按下面「零散时间」重排（力扣 53 + 6 节串线② + 牛客 0.5 套）。

**今天对照《第四周详细计划》D4 的任务清单：**


| 计划原文（D4 要点）                                                      | 今天怎么落地                                                            |
| ---------------------------------------------------------------- | ----------------------------------------------------------------- |
| F2 真实行业数据采集（公开来源、每条记录 URL/页码，目标 100~200 条）                       | 第 4 步（`build_real_qa.py` + `sft_data_real.json`）                  |
| F3 真实 vs 合成对照微调（复用 `train_lora.py` 配置，训 `lora_adapter_v2/`；显存串行） | 第 5 步（`train_lora_v2.py` + **纯真实 201 条** + 双 adapter 对照）—— 原计划的 `sft_data_v2_mix.json` **未生成**（实际选方案 A，见 §5.2 实测记录） |
| O2-G2 模板 09 v2 + 消融（治过拒答 FP / 编造引用 FN）                           | 第 3 步（`--template v2` / `v2a` / `v2b` 并排）                         |
| O3《优化成果报告》初稿                                                     | 第 6 步（骨架 + 总表 + 贡献表 + 失败记录，定稿留 D5）                                |
| （day18 明日安排）把 HyDE 做成模型自动改写、计入成绩                                 | 第 1 步 ⭐⭐                                                          |
| （day18 明日安排）改写后再融合 / 评估 Rerank                                   | 第 2 步                                                             |
| 零散：力扣 53 ｜ 6 节串线② ｜ 牛客 0.5 套                                     | 见「零散时间任务」                                                         |
| 验收：真实数据 ≥100 条可追溯；v2 adapter 训练完成；模板 v2 出对比；报告初稿成型               | 见「今日验收清单」                                                         |


---

## 📌 今天你要做什么（大白话版）

day18 你干了一件很漂亮的事：**用 35 秒的便宜实验，把"检索为什么难"这个问题钉死在一个点上**——不是库不行、不是题错了，而是**你问的是中文、要搜的段落是英文**。但你也留了一个"不能兑现的支票"：

> **"换成英文理想查询就能 100%" —— 可那个英文查询是你自己手写的。**

面试官一定会追问一句：**"那你的系统自己能改写好吗？改写之后到底几分？"** —— 今天就是来回答这一句的。

所以今天做六件事，顺序**有讲究**：

1. **把改写做成真实现**（第 1 步，O1-R4 落地）：写 `rewrite_queries.py`，让**本地 Qwen 自己**把中文问句改写成"英文术语 + 假设答案片段"，缓存落盘（**可人工复核**），再让 `eval_v2.py` 读缓存跑全量 20 题。
  → 一句话：**上界是"如果问对了能有多好"，成绩是"我的系统自己问得对不对"——今天要的是后者。**
2. **改写后再融合**（第 2 步）：day18 的 RRF 是负结果，原因是"有一路在瞎猜（BM25 读不懂中文）"；现在两路都有信号了，**重新判一次融合值不值得**。
  → 一句话：**负结果不是终点，"什么条件下这个负结果会翻正"才是。**
3. **打生成层的靶子**（第 3 步，O2-G2）：Q11/Q14/Q18 在 R3 里**稳定编造**（Q11 一致率 1.00，不是抖动）→ 模板 09 v2 加"**只能引用出现过的编号、禁止编造数字**"硬约束，并做 **v1/v2 并排 + 逐句消融**。
  → 一句话：**把"检索层问题"和"生成层问题"分开报，这本身就是加分项。**
4. **真实行业数据**（第 4 步，F2）：采**公开可引用**资料（机器人上市公司年报/公告/官网/白皮书/公开研报），构造 **≥100 条**三字段 QA，每条带 `source`（机构 + 标题 + URL + 日期 + 原文摘录）。
  → 一句话：**HR 说"随机生成的没意义"——那就每条都指得出来源。**
5. **真实 vs 合成对照微调**（第 5 步，F3）：同一套训练配置，分别训 v1（合成 300 条，**已有**）和 v2（真实数据），在**同一评测集**上对比，如实记录（**涨了是结论，持平也是结论**）。
  → 一句话：**这唯一能回答"真实数据到底值不值"。**
6. **成果报告初稿**（第 6 步，O3）：把 O1/O2 的 before/after 数字整理成表（定稿留 D5）。
  → 一句话：**今天不追求定稿，追求"表格的形状先立起来"。**

**今天结束时的"验收标准"一句话：**

> 能拿着一张表说清**"我做了哪一步 → 哪个数字从 X 变成 Y"**（第 1、3 步给检索层和生成层的**可引用成绩**）；`sft_data_real.json` **≥100 条且每条带 source**（F2 交付）；`lora_adapter_v2/` 训完且**显存没爆**、有 v1/v2 对照表（F3 交付）；`优化成果报告.md` 骨架成型且 `__` **处一个都没预填**。

**和你的数学背景接上的点（今天会反复出现）：**

- **上界 vs 估计量**：day18 的 100% 是"**用真实（理想）查询算出的可达上界**"，今天要的是"**用模型预测的查询得到的估计量**"。改写模型会犯错 → 估计量必然低于上界，**差距本身就是"改写质量"的度量**。这就是统计里"**oracle 上界 vs 可行估计量**"的差别。
- **误差分解**：改写引入的误差 = **幻觉术语**（编出语料里没有的词）+ **信息丢失**（把关键实体改没了）。两种误差方向不同 → 所以**必须先人工抽查改写质量**，而不是只看最终分数。
- **模板消融 = 对照实验的最小单位**：v2 改了两句话，就要 **v2a（只改第一句）/ v2b（只改第二句）** 分开跑 → 否则你只能说"改了模板后变了"，不能说"**是哪一句让它变的**"。
- **数据泄漏 = 训练集与测试集独立**：F2 的真实数据也必须过 `build_real_qa.py` 的泄漏检查（和评测题太像就作废）；F3 训练前也要确认"真实数据没把评测题抄进去"。
- **n=20 的置信区间**：所有"提升 X 个百分点"都要记住 ±0.22 的量级——所以报告里要说"**提升 N 题 / N 个百分点**"，并**避免在小数点上做文章**。

---

## 🗺️ 今天的学习路线图（先看这里，心里有个数）


| 步骤        | 内容                                                                                      | 预计时间            |
| --------- | --------------------------------------------------------------------------------------- | --------------- |
| 第 0 步     | 准备：认文件、激活环境、四查、账本现状、三条纪律                                                                | 10~15 分钟        |
| 第 1 步     | **O1-R4 落地：查询改写真实现 + 计入成绩**（`rewrite_queries.py` + `eval_v2.py --rewrite-cache`）⭐⭐ 今天最重 | 60~80 分钟        |
| 第 2 步     | **改写后再融合 / Rerank 评估**（`retrieval_lab.py` 复筛 + 加权/空路跳过 + Rerank 判读）                     | 30~40 分钟        |
| 第 3 步     | **O2-G2 模板 09 v2 + 逐句消融**（v1 / v2 / v2a / v2b 并排）+（加分）引文合法性后处理                          | 50~70 分钟        |
| 第 4 步     | **F2 真实行业数据 ≥100 条**（`build_real_qa.py` 校验 + `sft_data_real.json`）⭐⭐ HR 硬要求             | 60~90 分钟        |
| 第 5 步     | **F3 真实 vs 合成对照微调**（`train_lora_v2.py` → `lora_adapter_v2/` → 双 adapter 同评测集对比）⭐        | 60~90 分钟（含训练等待） |
| 第 6 步     | **O3《优化成果报告》初稿**（骨架 + before/after 总表 + 贡献表 + 失败记录）                                     | 40~50 分钟        |
| （可选）第 7 步 | `space_demo` 的 DeepSeek 端到端真冒烟（关闭 day18 待补项 4 后半）                                       | 15~25 分钟        |
| 零散时间      | 力扣 **53. 最大子数组和** ＋ 统计八股 6 节串线② ＋ 牛客真题 0.5 套                                            | 0.5~1 小时        |
| 睡前 15 分钟  | 收尾 + git check-ignore + commit + 写明日计划（Day20 = D5）                                      | 15 分钟           |


> 主线合计约 4~5 小时。**第 5 步的训练是"等待型任务"**：启动后可以去写第 4 步的数据或第 6 步的报告初稿（**但训练期间别开模型服务**，见 5.1 的显存纪律）。
> **时间不够时的取舍（照本周计划的原则）**：**先保第 1、4、5 步**（= 优化成绩 + 真实数据 + 对照实验，正好对应 HR 意见①③），第 3 步可只做 v1/v2 两档（先不做逐句消融），第 2、6、7 步可顺延到 D5。**宁可少做几个实验，也要保证"有提升数字 + 有真实数据 + 有可用界面 + 有公网链接"四件套**（界面和链接在 D5 收口）。

---

## 🔧 第 0 步：准备（10~15 分钟）

### 0.1 认识 day19 文件夹里的文件


| 文件                                         | 类型                                               | 用途                                                                                                                                                     |
| ------------------------------------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `rewrite_queries.py`                       | **已建（可运行）** ⭐⭐                                   | **O1-R4 查询改写**：读 20 条评测题 → 让本地 Qwen 把中文问句改写成"英文术语 / 假设答案片段" → 落盘 `queries_rewritten.json`。**缓存式设计**：只跑一次、可人工复核、后续所有实验复用（省时间、可复现）                       |
| `queries_rewritten.json`                   | 数据（今天生成）                                         | 改写缓存，形如 `{"1": {"plain": "...", "term": "...", "hyde": "...", "both": "..."}, ...}` + `_meta`（模型名 / 时间 / 提示词版本）→ **这是"改写质量"的复核材料，也是成绩可追溯的凭据**          |
| `eval_v2.py`                               | **从 day17 复制改造**（day19 版） ⭐                      | 在 day17 v2.1 上**只做加法**：`--rewrite-cache` / `--rewrite-mode`（见 1.6）+ 模板 v2a/v2b（见 3.3）+ 引文合法性后处理（见 3.5）。**不覆盖 day17 的脚本**（历史轮次的可复现性靠它）                  |
| `build_real_qa.py`                         | **已建（可运行）** ⭐                                    | **F2 校验器**：从 `build_paper_qa.py` 改造——把"chunk 逐字回查"换成"**来源五要素齐全 + quote 非空 + 跨条重复 + 撞评测题（泄漏）**"四查，另加"**类目/机构分布**"统计（**已实测：pass 路径 exit 0、条数不足 exit 1**） |
| `sft_data_real.seed.json`                  | 数据（**已建**）                                       | F2 的**种子文件**：3 条**格式示例**（含五要素 source，`url` 是 `example.com` 占位、内容标了"（格式示例·请替换）"）→ **照它的结构写真实数据，别把示例当数据用**                                               |
| `sft_data_real.json`                       | 数据（今天生成）                                         | **F2 交付物**：≥100 条真实行业 QA，每条带 `source`（机构 / 标题 / URL / 日期 / 原文摘录）                                                                                       |
| `sft_data_v2_mix.json`                     | 数据（**今天未生成** → 留 D5）                                        | **F3 的"配比版"训练集**（真实 + 合成 300，方案 B 才需要）。**本轮实际选了 A（纯真实）**，所以没用上它、也没生成；见 §5.2 实测记录                                                                                                  |
| `train_lora_v2.py`                         | **已建** ⭐（= `第三周\day12\train_lora.py` 的逐字复制件）     | **只有 3 个常量不同**（`DATA_PATH` / `OUTPUT_ADAPTER` / `LOG_DIR`），**其余一字不动** → 保证"对照实验只改数据这一个变量"                                                              |
| `local_api_lora_v2.py`                     | **已建** ⭐（= `第三周\day13\local_api_lora.py` 的逐字复制件） | **只有 2 个常量不同**（`ADAPTER_PATH` → `day19\lora_adapter_v2`、`MODEL_NAME` 加 `-v2`，防"服务在跑 ≠ 服务正确"）；另在 `第三周\day15\config.json` 里**新增**了 `lora_v2` profile     |
| `lora_adapter_v2/`                         | 训练产物（**不进 git**）                                 | F3 的 adapter v2（20~30 MB）                                                                                                                              |
| `优化成果报告.md`                                | 今天生成（初稿）                                         | O3：before/after 总表 + 逐步骤贡献表 + 失败尝试 + 结论与下一步（**定稿留 D5**）                                                                                                |
| `result_rewrite_*/`、`result_template_*/` 等 | 结果目录（今天生成）                                       | 每轮实验的 `eval_results.json` / `评测表.md` / `run_log.txt`（**每轮分目录，互不覆盖**）                                                                                   |


> **为什么要"复制改造"而不是"直接改 day17 的文件"？** 两条纪律：① **历史轮次必须可复现**——day17 的 R3 是用 `day17\eval_v2.py` 跑出来的，如果今天原地改它，日后回看就不是同一把尺子了；② **"一天一个 day 文件夹"** 是本项目惯例，产物归档不串台。**唯一例外是** `实验日志.md`（它是跨天的"账本"，只在 `第四周\` 下维护）。
>
> **唯一的"就地扩展"是 `day18\retrieval_lab.py`**（第 2 步）：它是 day18 教程里就预告过的"**可复用的筛查台**"，今天的 `file` 档是**纯加法**（老路径行为一字未变，day18 的 4-4-3 仍可复现），所以没有再复制一份到 day19。除了它，今天所有脚本/配置都是"复制件 + 只改常量"（`train_lora_v2.py`、`local_api_lora_v2.py`、`config.json` 的 `lora_v2` profile）。

### 0.2 激活环境 + 编码（昨天的坑，别再踩）

```powershell
conda activate llm
cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day19"
```

看到行首 `(llm) PS ...\day19>` 即可。✅

**如果报** `UnicodeEncodeError: 'gbk' codec can't encode ...`（不是脚本 bug，是"输出端 UTF-8 vs 显示端 GBK"不匹配）：

```powershell
chcp 65001
$env:PYTHONIOENCODING='utf-8'
```

> 记忆法：**报错里出现** `gbk` **/ 乱码，先想"编码不匹配"**。
> 今天三个新脚本都输出中文（改写提示词、来源标题），**这个坑出现概率高**。
> 不想用 `conda activate` 也行，直接调解释器（**注意：**`eval_v2.py` **需要 langchain，必须 llm 环境**）：
> `& "D:\miniconda1\envs\llm\python.exe" rewrite_queries.py --help`

### 0.3 五查（3 分钟，缺啥补啥）

**① 查"库"和"卷子"（今天的检索对象）：**

```powershell
dir ..\..\第三周\day13\chroma_db
dir ..\..\第三周\day13\eval_questions.json
dir ..\..\第三周\day13\chunks.json
```

**② 查 day17 / day18 的产物还在**（今天要接着用）：

```powershell
dir ..\day17\eval_v2.py
dir ..\day17\result_lora_k8_runs3          # R3 = 今天的对照基线
dir ..\day18\retrieval_lab.py
dir ..\day18\sft_data_paper.json           # F1 的 90 条，F3 要用
dir ..\实验日志.md
```

**③ 查"合成数据 + 训练脚本 + adapter"三件套**（F3 要用）：

```powershell
dir ..\..\第三周\day12\train_lora.py
dir ..\..\第三周\day12\lora_adapter
dir ..\..\第三周\day11\sft_data.json
dir ..\..\第三周\day13\local_api_lora.py
```

**④ 查新脚本能不能跑（几秒钟）：**

```powershell
python rewrite_queries.py --help
python build_real_qa.py --help
python ..\day17\eval_v2.py --help
```

**⑤ 查依赖（只有** `eval_v2.py` **需要第三方包）：**

```powershell
python -c "import langchain_chroma, langchain_huggingface, requests; print('deps OK')"
```

> ⚠ **今天特别容易踩的坑（和昨天同款）**：`rewrite_queries.py` 与 `build_real_qa.py` **只用标准库 + requests**，系统 python 也能跑；但 `eval_v2.py` 会 `import langchain_chroma`，**必须 llm 环境**。混着用就会遇到"一个能跑一个不能跑"的假故障。

### 0.4 先看一眼"账本现状"（避免重复记账）

打开 `第四周\实验日志.md`，确认「一、实验记录表」现在的最后一行是：


| 日期   | 编号     | 改了哪个变量              | 检索 strict      | 生成         | 防幻觉        | F1        | 总正确率            |
| ---- | ------ | ------------------- | -------------- | ---------- | ---------- | --------- | --------------- |
| 9/20 | **R3** | 关前缀、保留 k=8（**确认项**） | **4/10 = 40%** | 4/10 = 40% | 7/10 = 70% | **0.824** | **11/20 = 55%** |


**今天所有实验的统一对照基线 = R3**（k=8 / qi=off / patch=on / runs=3 / 模板 v1）。三条口径记牢：

1. **只与 R3 比**（同 k、同 qi、同 runs）——**别再和 runs=1 的 day16 R2 相减**，投票本身会改数字（这是 day17 就立下的纪律）；
2. **每轮只改一个变量**（今天要改的变量依次是：`rewrite` / `template` / **训练数据**）；
3. **对外引用的数字必须** `--runs 3`，且 **F1 要回** `run_log.txt` **对一次 TP/FP/FN/TN**（day18 那条笔误的教训）。

### 0.5 今天要守住的三条纪律


| #   | 纪律                           | 为什么                                                                        |
| --- | ---------------------------- | -------------------------------------------------------------------------- |
| 1   | **改写只用于"检索"，答案提示词仍用原问题**     | 否则你同时改了"检索输入"和"生成输入"两个变量 → 说不清是谁起的作用；而且答案要按中文关键词判分，问句不能变成英文                |
| 2   | **诊断数字 ≠ 成绩**                | `retrieval_lab.py` 只测检索 → 结论进报告的"分析"章节；**能进「一、实验记录表」的成绩必须来自** `eval_v2.py` |
| 3   | **训练与推理串行**（6G 显存一次只能跑一个 3B） | 训练前**必须先停模型服务**（`Ctrl+C` 或按端口 kill），训练时盯显存峰值 <6G；否则 `os error 1455` / 直接崩  |


### ✅ 第 0 步验收标准

- [x] `(llm)` 环境 + 编码设置就绪，`eval_v2.py --help` 能打印参数表
- [x] 五查全绿：库/卷子在、R3 结果目录在、`sft_data_paper.json` 在、训练三件套在
- [x] 能一句话说出"今天的对照基线是 R3（strict 4/10 / F1 0.824 / 总 55%）"和"今天只改哪三个变量"

---

## 🔎 第 1 步：O1-R4 落地——把 day18 的"上界"做成"成绩"（60~80 分钟）⭐⭐

### 1.1 为什么今天必须先做这件事

回顾 day18 的推导链：

```
数据自检 9/10 自洽（卷子没问题）
        ↓
中文查询 strict 40%（现状）
        ↓  只换查询、库不变
英文理想查询 strict 100%（三通道一致）  ← 上界
        ↓
所以：信息在库里，是"查询端对不上"
```

**这条链的最后一环是空的**：那个"英文理想查询"是**人手写的**。真实的系统不能用人的脑子当组件。所以今天要补上：

> **让本地 Qwen2.5-3B 自己把中文问题改写成"贴英文语料"的查询，然后用同一把尺子跑全量 20 题。**

改写会犯错（编术语、漏实体、句式跑偏），所以**必然低于 100%**——**这个"低于上界的成绩"才是能写进报告的数字**。

### 1.2 关键设计：改写只给检索用，不给生成用（⭐ 别改错地方）

```
                ┌───────────────── 原问题（中文，判分口径不变）
                │                        │
                ▼                        ▼
        ┌───────────────┐        ┌──────────────────┐
问题 ──▶│ 模型改写（一次）│──▶检索──▶│ 模板 09 拼上下文  │──▶ 答案（中文）
        └───────────────┘   Top-8  └──────────────────┘
              （缓存到 json）                ▲
                                          仍用原中文问题当 prompt 的【问题】
```

**三个理由**（面试可以直接讲）：

1. **单变量纪律**：改的只有"检索输入"，生成侧的输入、温度、模板、判分口径全不动 → 数字变了只能是改写的功劳。
2. **判分口径**：`eval_questions.json` 的 `expected_keywords` 是**按原问题的答案**写的；若把 prompt 里的问题换成英文，答案风格会变、判分失真。
3. **用户体验**：真实系统里用户问的就是中文，**界面上的问题不能被你偷偷替换掉**；改写只是"内部检索技巧"。

### 1.3 三种改写强度（今天都生成，分档跑）


| 档位                | 提示词要求模型输出                                                        | 优点                     | 风险                   |
| ----------------- | ---------------------------------------------------------------- | ---------------------- | -------------------- |
| `term`（术语抽取）      | 只输出**英文关键词/术语**，逗号分隔，如 `Stanford University, author affiliation` | 稳、少幻觉、BM25 直接受益        | 丢句子结构，向量语义可能变弱       |
| `hyde`（假设答案片段）    | 输出**一段英文的"假设答案"**（像论文里会怎么写）                                      | 语义最贴段落（day18 上界就是这么来的） | 模型会**编**（写出语料里没有的事实） |
| `both`（术语 + 假设答案） | 先一行术语，再一段假设答案                                                    | 两路兼顾                   | 提示词更长、更易跑偏           |


> **今天的推荐路线**：**先跑** `term`**（最稳），再跑** `hyde`**，**`both` **作为备选**。三档都用**同一份缓存文件**里的不同字段，**不需要重跑模型** → 一轮 `eval_v2.py` 十几分钟，但**换档位零成本**。

**改写提示词（照抄，注意"不许解释、只输出查询"这条，否则模型会跟你聊天）**：

```text
【term 档·system】
你是检索查询改写助手。把用户的中文问题改写成适合检索英文技术论文的英文检索词。
要求：
1. 只输出英文关键词与术语，用英文逗号分隔，最多 8 个；
2. 保留问题里的实体（人名、机构、数据集、方法名、仓库名的原文拼写）；
3. 不要输出解释、不要输出句子、不要加引号或前缀。

【term 档·user】
问题：{question}

【hyde 档·system】
你是检索查询改写助手。请先假设一段英文论文里"回答了该问题"的文字，然后把它作为检索查询输出。
要求：
1. 只输出这段英文文字（1~2 句），不要解释、不要复述问题；
2. 尽量使用论文语体（如 "We evaluate on the LAFAN1 dataset…"）；
3. 如果不确定事实，只写"可能出现的表述"，不要编造具体数字。

【hyde 档·user】
问题：{question}
```

> ⚠ **一个必须提前说清的边界**：`hyde` 档**允许模型编"假设表述"**（这是 HyDE 的机制），但**编出来的术语会成为噪声**。所以第 1.5 步的**人工抽查**是硬性动作，不是可选项。

### 1.4 `rewrite_queries.py`（**已建**：~140 行，标准库 + requests）

> **文件已在** `第四周\day19\rewrite_queries.py`（随教程一起生成，`--help` 与 `--dry-run` 已实测通过）。**核心代码与逐行说明如下**——直接用或对照读都行。

```python
# -*- coding: utf-8 -*-
"""
rewrite_queries.py —— O1-R4 查询改写（Day19 核心产出）

作用：把 20 条评测题的中文问句，改写成"更贴英文论文语料"的检索查询，落盘成缓存文件。
      缓存下来有两个好处：① 只跑一次，省时间；② 可人工复核（改写质量本身要检查）。

设计纪律（Day19 第 1.2 节）：
    改写【只给检索层用】。生成层的问题仍是原中文问句 → 保证"只改一个变量"，
    也保证答案按原关键词判分。

用法：
    python rewrite_queries.py                    # 三档全跑（term / hyde / both）
    python rewrite_queries.py --modes term       # 只跑术语档
    python rewrite_queries.py --dry-run          # 不调模型，只写 plain（先验证流程）
    python rewrite_queries.py --show 3           # 打印第 3 题的改写结果，人工抽查
"""
import argparse
import json
import os
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
QUESTIONS = os.path.join(REPO_ROOT, "第三周", "day13", "eval_questions.json")
DEFAULT_OUT = os.path.join(BASE_DIR, "queries_rewritten.json")
API_URL = "http://127.0.0.1:8000/v1/chat/completions"

SYSTEM_TERM = """你是检索查询改写助手。把用户的中文问题改写成适合检索英文技术论文的英文检索词。
要求：
1. 只输出英文关键词与术语，用英文逗号分隔，最多 8 个；
2. 保留问题里的实体（人名、机构、数据集、方法名、仓库名的原文拼写）；
3. 不要输出解释、不要输出句子、不要加引号或前缀。"""

SYSTEM_HYDE = """你是检索查询改写助手。请先假设一段英文论文里"回答了该问题"的文字，然后把它作为检索查询输出。
要求：
1. 只输出这段英文文字（1~2 句），不要解释、不要复述问题；
2. 尽量使用论文语体（如 "We evaluate on the LAFAN1 dataset..."）；
3. 如果不确定事实，只写"可能出现的表述"，不要编造具体数字。"""


def load_questions(path):
    """评测题文件是 {"meta":..., "questions":[...]}；也兼容直接给一个 list。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("questions", [])
    return data


def call_model(system, user, temperature=0.01, max_tokens=160, timeout=120):
    """调本地 OpenAI 兼容服务（就是评测用的那个 8000 端口，不额外占显存）。"""
    payload = {
        "model": "Qwen2.5-3B-Instruct-LoRA",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(API_URL, json=payload, timeout=timeout)
    if r.status_code != 200:
        # FastAPI 把真实异常放在响应体的 detail 里（服务端是 `raise HTTPException(500, f"生成失败：{e}")`）；
        # 只用 raise_for_status() 的话，终端只剩一句 "500 Server Error"，真正的原因会被吞掉。
        raise RuntimeError(f"服务端返回 {r.status_code}：{r.text[:500]}")
    # 常见 500 原因：temperature 必须 > 0（transformers 的 TemperatureLogitsWarper 校验，
    # 因服务端写死 do_sample=True，所以 temperature=0 会直接抛 ValueError）。
    return r.json()["choices"][0]["message"]["content"]


def clean(text):
    """去掉模型爱加的前后缀（'改写：' / 引号 / markdown 代码块 / 多余换行）。"""
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()      # 去代码块围栏
    t = re.sub(r"^(改写|查询|检索查询|查询改写)\s*[:：]\s*", "", t).strip()
    t = t.strip("\"'“”‘’ ")
    t = re.sub(r"\s*\n+\s*", " ", t).strip()                    # 合成单行
    return t


def rewrite_one(question, modes, dry_run=False):
    """返回 {"plain":..., "term":..., "hyde":..., "both":...}（只填要求跑且成功的档）。"""
    rec = {"plain": question}
    if dry_run:
        return rec
    if "term" in modes:
        rec["term"] = clean(call_model(SYSTEM_TERM, f"问题：{question}", max_tokens=96))
    if "hyde" in modes:
        rec["hyde"] = clean(call_model(SYSTEM_HYDE, f"问题：{question}", max_tokens=200))
    if "both" in modes:
        term = rec.get("term") or clean(call_model(SYSTEM_TERM, f"问题：{question}", max_tokens=96))
        hyde = rec.get("hyde") or clean(call_model(SYSTEM_HYDE, f"问题：{question}", max_tokens=200))
        rec["both"] = f"{term} | {hyde}"
    return rec


def main():
    ap = argparse.ArgumentParser(description="O1-R4 查询改写（Day19）")
    ap.add_argument("--questions", default=QUESTIONS, help="评测题 json")
    ap.add_argument("--out", default=DEFAULT_OUT, help="改写缓存输出路径")
    ap.add_argument("--modes", default="term,hyde,both", help="要跑的档位（逗号分隔）")
    ap.add_argument("--dry-run", action="store_true", help="不调模型，只写 plain")
    ap.add_argument("--show", type=int, default=None, help="打印某题的改写结果（人工抽查）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    if args.dry_run:
        print("⚠ dry-run：不调模型，只写 plain（用来先验证文件读写与后续 eval 的接线）")

    result = {}
    t0 = time.time()
    for i, q in enumerate(questions, start=1):
        print(f"[{i:>2}/{len(questions)}] id={q['id']:>2} {q['question']}")
        result[str(q["id"])] = rewrite_one(q["question"], modes, dry_run=args.dry_run)
    result["_meta"] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen2.5-3B-Instruct-LoRA",
        "modes": modes,
        "api_url": API_URL,
        "prompt_version": "day19-v1",
        "elapsed_sec": round(time.time() - t0, 1),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[落盘] {args.out}（共用时 {result['_meta']['elapsed_sec']} 秒）")

    if args.show:
        rec = result.get(str(args.show), {})
        print(f"\n=== 抽查 id={args.show} ===")
        for k in ("plain", "term", "hyde", "both"):
            if rec.get(k):
                print(f"[{k:>5}] {rec[k]}")
    print("\n下一步：人工抽查几条（重点看有没有编造术语），再跑 eval_v2.py --rewrite-cache")


if __name__ == "__main__":
    main()
```

**先起服务**（改写要用本地模型；**注意：一个终端只能跑一个 3B，这条服务也是后面评测要用的那一条**）：

```powershell
# 终端 A（另开一个）：起微调模型服务
conda activate llm
cd "D:\Lan\研究生\技术学习\大模型算法\第三周\day13"
python -m uvicorn local_api_lora:app --host 127.0.0.1 --port 8000
# 等日志出现「模型已就绪」和「LoRA adapter 已加载」再进行下一步
```

### 1.5 先跑改写 + 人工抽查（⭐ 这一步不能省）

```powershell
# 终端 B：先 dry-run 验证流程（几秒）
python rewrite_queries.py --dry-run --limit 2

# 正式跑三档（20 题 × 约 3 次调用 = 60 次；本机实测 3~4 分钟）
python rewrite_queries.py

# 抽查 3 条最典型的失败题（day18 的"专名题"）：Q1 英文全称 / Q3 作者单位 / Q5 数据集
python rewrite_queries.py --show 1
python rewrite_queries.py --show 3
python rewrite_queries.py --show 5
```

> **⚠** `--show` **不是"只读缓存"**（`day19-v1` 实测发现）：它和正式跑走的是**同一条路径**——照样把 20 题全部重跑一遍，并**覆盖** `queries_rewritten.json`（实测三次分别耗时 198.4 / 213.1 / 249.1 秒）。
> **后果**：① 抽查很贵（看 3 条 = 跑 3 轮全量）；② **你抽查看到的那一批，可能已经不是最终落盘的那一批**（后一次把前一次覆盖了）。
> **正确姿势**：**先只跑一次全量，再连着** `--show` **看**；想根治就在 `main()` 里把 `--show` 分支**提到重跑循环之前**（命中缓存就直接打印并 `return`），10 行以内。
> **本次"覆盖"没造成串台**（顺带得到一个证据）：第 1 轮打印的 Q3、第 2 轮打印的 Q5，与第 3 轮落盘的 Q3/Q5 **逐字相同** → **同一提示词下改写是可复现的**（`T=0.01` 近似贪心），这也是"改写数字可以对外引用"的前提。

**怎么算"抽查合格"**（四看，写进日志）：


| 看什么                 | 合格的样子                                                                                                                                     | 不合格的样子                                             | 处置                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------- |
| **① 实体保留**          | Q1 出现 `General Motion Retargeting`、Q3 出现 `Stanford`、Q5 出现 `LAFAN1`（**这三个词就是** `eval_questions.json` **里的** `expected_keywords`，别自己另立一套标准） | 实体被意译没了（只剩 `dataset` / `university`），或被**改错的实体**顶替 | 该档降级；提示词里补"**原样保留实体拼写**"                                             |
| **② 术语幻觉**          | 改写里冒出的每个新词，都能在 `chunks.json` 里搜到                                                                                                          | 冒出论文里没有的方法名 / 数据集名 / 数字                            | 回库**计数**核对（0 次 = 幻觉）→ **记进失败尝试**；`hyde` 档收紧提示词                       |
| **③ BM25 可读性**      | `term` 档**几乎全是 ASCII 关键词**                                                                                                                | `term` 档混中文、或输出整句问句                                | 检查 `clean()` 与提示词                                                    |
| **④ 指令泄漏**（9/20 新增） | 输出里**只有查询本身**                                                                                                                             | 把提示词里的话抄进输出（实测多见 `可能出现的表述：`）                       | **改提示词**：把"只写…"这种正向指示改成"**不要**输出解释/前缀"的否定式；否则 8 个汉字的小句混进英文查询，会把向量往下拉 |


> **⚠ 别跳过抽查直接看分数**：改写引入的是**系统性误差**（错误的术语会稳定地把排名带偏），不是随机噪声。**"分数涨了"也可能是"改写把某几题蒙对了"**——只有看过改写内容，你才知道该不该信这个数字。这是 day18 那条"文档数字要回脚本输出对账"的同一条思路。

#### 1.5.1 实测复核记录（9/20，`prompt_version = day19-v1`，20 题全量，**复核结论：不合格**）

数据来源：`queries_rewritten.json` 的 `_meta` = `{"time": "2026-09-20 19:24:44", "model": "Qwen2.5-3B-Instruct-LoRA", "modes": ["term","hyde","both"], "prompt_version": "day19-v1"}`。抽查命令 `--show 1 / 3 / 5`。

**① 三题抽查明细**（判据列直接抄 `第三周\day13\eval_questions.json` 的 `expected_keywords`）：


| 抽查题         | 期望实体（判据）                     | `term` 档实测                                                      | `hyde` 档实测                                                                     | 判定                                                |
| ----------- | ---------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------- |
| **Q1** 英文全称 | `General Motion Retargeting` | `GMR: Generalized Mean Regression, generalized mean regression` | `GMR stands for Ground Motion Reconstructions Project.可能出现的表述：…`               | ❌ **两档都编错实体**                                     |
| **Q3** 作者单位 | `Stanford`                   | `GMR, authors, university`                                      | `The authors are from University of Science and Technology of China.可能出现的表述：…` | ❌ 实体丢失（`term` 只剩 `university`）+ `hyde` 编成 USTC    |
| **Q5** 数据集  | `LAFAN1`                     | `data set, experiment, dataset`                                 | `We evaluate on LAFAN-1 dataset.`                                              | ⚠ **半合格**：实体出现了，但写成 `LAFAN-1`；`term` 档全是泛化词，实体没保住 |


**② 回** `chunks.json` **核对**（把"感觉像是幻觉"变成"计数为 0 的幻觉"——**这一步就是"看②"的落地动作**）：


| 改写里冒出来的词                                        | 出自哪题             | 库里命中次数 | 结论                                                                                                                                                                                                                                                      |
| ----------------------------------------------- | ---------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Generalized Mean Regression`                   | Q1 `term`        | **0**  | ❌ 编造。**注意：它是统计学里真实存在的另一个术语**，属"看起来很像"的伪实体                                                                                                                                                                                                               |
| `Ground Motion Reconstructions Project`         | Q1 `hyde`        | **0**  | ❌ 编造（库里的真词是 `General Motion Retargeting`，出现 **7** 次；模型历次**答案**里也爱写 `Ground Motion Retargeting`，同样是库外词）                                                                                                                                                  |
| `University of Science and Technology of China` | Q3 `hyde`        | **0**  | ❌ 编造（库里 `Stanford` 出现 **7** 次）                                                                                                                                                                                                                          |
| `DGM`、`MCMC`                                    | Q4 `hyde`        | **0**  | ❌ 编造（库里真正的对比方法是 `PHC` 32 次 / `ProtoMotions` 25 次 / `Unitree` 17 次）                                                                                                                                                                                      |
| `FERET`、`CASIA`                                 | Q7 `hyde`        | **0**  | ❌ 编造，且这俩**是人脸识别数据集**（整题跑到了别的领域）                                                                                                                                                                                                                         |
| `F-measure`、`4%`                                | Q8 `hyde`        | **0**  | ❌ 编造指标名与数字                                                                                                                                                                                                                                              |
| `IEEE Transactions on Image Processing`         | Q13 `hyde`       | **0**  | ❌ 编造期刊                                                                                                                                                                                                                                                  |
| `50,000 steps`                                  | Q20 `hyde`       | **0**  | ❌ 编造数字                                                                                                                                                                                                                                                  |
| `4,096 parameters`                              | Q15 `hyde`       | **2**  | ⚠ **最隐蔽的一类**：库里 `4096` 是"**仿真试验跑了 4096 次**"，不是"模型参数量 4096" → 检索能"命中"含 `4096` 的段落，但那是**别的意思**                                                                                                                                                            |
| `GMRES`（Q16 把 `GMR` 写成了它）                       | Q16 `term`       | **0**  | ❌ 编造。`GMRES` 是数值线性代数里**真实存在**的算法名（与本文毫无关系）→ 另一例"真实但错位"的伪实体                                                                                                                                                                                              |
| `本文研究解决了目标检测中尺度不变性的问题。`（**整句中文 + 整题跨领域**）       | Q2 `hyde`        | **0**  | ❌ 编造，且**违反了** `hyde` **提示词"输出英文"的第 1 条**：论文讲的是**人形机器人运动重定向**，跟"目标检测"毫无关系。**危害最大的一类**——整句中文在 BM25 里 `TOKEN_RE` 一个 token 都切不出来（等于该通道全空），向量侧则被拉到一个完全无关的语义区（实测 Q2 的 Top-8 从 `[14,22,1,9,3,25,13,55]` 变成 `[74,55,63,61,11,58,9,14]`，**期望段 chunk 3 直接掉出前 8**） |
| `LAFAN-1`                                       | Q5/Q8/Q19 `hyde` | **0**  | ❌ 多了一个连字符。库里是 `LAFAN1`（**10** 次）；而 `retrieval_lab.py` 的 `TOKEN_RE = [a-zA-Z0-9]+` 会把 `LAFAN-1` 切成 `lafan` + `1`，库里 token 是 `lafan1` → **稀疏通道直接打不中**（这一条解释了"为什么真实现会低于 day18 的上界"）                                                                        |


> 核查命令（可复现，`cd 第三周\day13`）：
>
> ```powershell
> foreach ($t in @('Generalized Mean','Ground Motion','FERET','CASIA','DGM','MCMC','F-measure','Image Processing','50,000','4096','GMRES','LAFAN-1','LAFAN1','General Motion Retargeting','Stanford')) {
>     $c = (rg -o --no-filename -i ([regex]::Escape($t)) chunks.json | Measure-Object).Count
>     "{0,-26} {1}" -f $t, $c
> }
> ```
>
> **读数注意**：`DGM` 会报 **2** 次，但那是 `retargeting` / `ACKNOWLEDGMENT` 里的**子串误报**（`rg -o -i ".{40}DGM.{40}" chunks.json` 一看便知），**实际 0**；`4096` 的 2 次是"仿真试验 4096 次"，**不是参数量**。→ **计数不是终点，"这个词在这一段里是不是这个意思"才是**。

**③ 四看结论**


| 看什么        | 判定    | 证据                                                                                                         |
| ---------- | ----- | ---------------------------------------------------------------------------------------------------------- |
| ① 实体保留     | ❌ 不合格 | 抽查 3 题里 **2 题（Q1/Q3）核心实体被改写器"主动改错"**，1 题（Q5）写成 `LAFAN-1`                                                   |
| ② 术语幻觉     | ❌ 不合格 | 上表 12 组新词，**10 组库里 0 命中**；2 组（`4096`、`Generalized Mean Regression`）是"真实存在但与本文语义错位"                         |
| ③ BM25 可读性 | ❌ 不合格 | `term` 档混中文：Q6 `paper: 方法, 策略训练, 方法研究`、Q7 `…, 引入, 分析, 研究, 特点, 影响, 检测`、Q8 `GMR 结论 实验 数据:实验结论`、Q15 `…,数量,大小` |
| ④ 指令泄漏     | ❌ 不合格 | `可能出现的表述：` 出现在 Q1/Q3/Q9/Q11/Q15/Q17/Q18 共 **7 题**的 `hyde` 档（提示词 SYSTEM_HYDE 第 3 条被模型当成"要输出的模板文字"抄了出来）      |


**结论一句话**：**不是"改写没生效"，而是"改写太自信"** —— 面对库里没有的事实（机构 / 期刊 / GPU 数 / 参数量 / 被引次数），模型选择**编一个具体答案**，而不是**保留原词**。day18 的 10/10 上界是用**人类写的理想查询**得到的，真实现的主要漏损就在这一步（**系统性误差，不是多跑几次能消掉的抖动**）。

**④ 处置（今天就落地，并进「三、失败尝试记录」）**

1. `hyde` **提示词收紧（v2）**：把第 3 条"只写'可能出现的表述'"改成**否定式**——"**不要**在输出里解释、加前缀或加引号"；并加一条"**除问题中已出现的事实外，不许给出具体的机构名 / 期刊名 / 数字**"（允许模糊表述，禁止细节编造）。
2. `term` **提示词收紧（v2）**：强制"只输出 ASCII 英文关键词"+"**原样保留实体拼写**（`LAFAN1` 不加连字符、`Stanford` 不意译成 `university`）"。
3. **工程兜底（推荐做，10 分钟）**：把本节扫出来的词写进 `hallucination_blacklist.txt`，改写后**后处理过滤**（命中即丢弃该词）——改写器能力不够时，靠过滤保底，且这个文件本身就是报告里"我知道我的改写器会瞎编"的证据。
4. **R4 两档都跑**：`term` 只当兜底/对照，`hyde` 作为主攻（并在 1.9 的溯源三件套里保留 `retrieval_query`，让面试官看到"它到底问了什么"）。

### 1.6 给 `eval_v2.py` 加两个旋钮（day19 版只动两处）

复制 `..\day17\eval_v2.py` → `.\eval_v2.py`，然后**只做两处改动**（模板 v2a/v2b 见第 3 步，一起加更省事）：

**改动 ①：在** `main()` **的** `parser` **区加两个参数**（约在第 714 行 `--max-tokens` 附近）：

```python
    parser.add_argument("--rewrite-cache", default="", help="改写缓存 json（Day19 O1-R4；留空=不改写）")
    parser.add_argument("--rewrite-mode", default="term", choices=["plain", "term", "hyde", "both"],
                        help="用缓存里的哪一档当检索查询（默认 term）")
```

**改动 ②：在答题主循环里，把"检索用的查询"和"生成用的查询"分开**（约在第 810~816 行）：

```python
    # ---- day19 新增：读改写缓存（只影响检索，不影响生成）----
    rewrite_cache = {}
    if args.rewrite_cache:
        with open(args.rewrite_cache, encoding="utf-8") as f:
            rewrite_cache = json.load(f)
        print(f"[改写] 已加载 {args.rewrite_cache}（档位={args.rewrite_mode}）"
              f"｜模型={rewrite_cache.get('_meta', {}).get('model', '?')}")

    def pick_query(q):
        """返回 (用于检索的查询, 是否真的用了改写)。"""
        if not rewrite_cache:
            return q["question"], False
        rec = rewrite_cache.get(str(q["id"])) or {}
        rq = (rec.get(args.rewrite_mode) or "").strip()
        return (rq, True) if rq else (q["question"], False)

    # ...（循环内）...
            rq, used_rewrite = pick_query(q)
            docs, ids = retrieve(store, rq, args.top_k)          # ← 检索用改写后的查询
            context = build_context(docs)

            answers = []
            for r in range(args.runs):
                answers.append(generate_answer(profile, template, q["question"], context,
                                               args.temperature, args.max_tokens, dry_run=args.dry_run))
                #                                                    ↑ 生成仍用【原中文问题】（纪律 #1）
            rec = judge_question(q, ids, context, answers)
            rec["retrieval_query"] = rq                          # ← 落进结果，便于复盘"它到底问了什么"
            rec["rewrite_used"] = used_rewrite
```

同时在 `run_log.txt` 的输出里补一行，**让"本轮用了哪档改写"可追溯**（找打印配置的那处，加 `rw={args.rewrite_mode}`）：

```python
    print(f"... runs={args.runs} {args.template} qi={args.query_instruction} "
          f"patch={args.keyword_patch} rw={args.rewrite_mode if args.rewrite_cache else 'off'} | ...")
```

> `day19\eval_v2.py` **里实际补了 3 处**（本教程的代码块只给了最典型的一处，别以为漏改了）：
> ① 开头的配置块多打一行 `查询改写：on/off（…档…）`（**看** `run_log.txt` **第一屏就知道这轮有没有改写**）；
> ② `[日志行]` 那行补 `rw=…`；
> ③ `--append-log` 写进「实验记录表」的**配置列**也补 `rw=term`——**必须补**，否则 **R4 行与 R3 行在表里长得一模一样**（两行都是 `… / v1 / qi=off / patch=on`），"每轮只改一个变量"的可追溯性就断了。

> **⚠ 9/21 补充：还有一个第 4 处改动/参数**——`--citation-policy`（O2-G3 引文兜底）。它的代码在 **§3.5.4**，**默认 `record` 与 day17 行为完全一致**，所以第 1 步的 R4 跑法**不受影响**；但如果你是从 `..\day17\eval_v2.py` 重新复制一份，记得三样一起加（`--rewrite-cache` / v2 模板 / `--citation-policy`），别漏。

**另外加了一个"接线自检"**（不在原计划里，但强烈建议加）：汇总区多打一行

```python
    if args.rewrite_cache:
        n_used = sum(1 for r in results if r.get("rewrite_used"))
        print(f"⑦ 改写生效        : {n_used}/{len(results)} 题用了「{args.rewrite_mode}」档改写")
        if rw_missing:
            print(f"   ⚠ 未取到改写的题号：{rw_missing}（缓存里该档缺失或为空 → 已回退原问题）")
```

**为什么值这几行**：`pick_query()` 在"缓存里没这道题 / 该档字段为空"时是**静默回退原问题**的。没有这一行，你会看到一堆 `retrieval_query` 等于原文，然后得出"**改写不涨分**"的结论——而真实原因是"**缓存里那一档根本没生成**"（比如只跑了 `--modes plain`，或 `--dry-run` 生成的缓存）。这两个结论**方向完全相反**，必须靠数字分开。

> **第 3 步会再补一行「⑧ 引文合法性」**（同上：只统计不改判定）——所以终端汇总区最终是 ①~⑦ + ⑧ 共 8 行，见 3.5。

**改完立刻验证"接线通了"**（这是 day17 学到的"先验证工具生效，再跑实验"）：

```powershell
# 用 dry-run 缓存跑 2 条，看终端有没有打印 [改写] 已加载
python rewrite_queries.py --dry-run            # 生成一份只含 plain 的最小缓存
python eval_v2.py --top-k 8 --limit 2 --dry-run --rewrite-cache queries_rewritten.json --rewrite-mode plain
# 期望：打印 [改写] 已加载 + 查询改写：on + rw=plain，且结尾 ⑦ 改写生效 = 2/2 题
# 答案因为 --dry-run 是占位（数字无意义），重点看"有没有报错、有没有取到改写查询"

# 反向验证（同样重要）：故意传一个缓存里没有的档位 → 应看到 ⚠ 未取到改写的题号
python eval_v2.py --top-k 8 --limit 2 --dry-run --rewrite-cache queries_rewritten.json --rewrite-mode hyde
```

> **⚠ 注意** `--dry-run` **生成的缓存会覆盖你正式用的那份**：`python rewrite_queries.py --dry-run` 会把 `queries_rewritten.json` 写成**只有** `plain` **档**的版本（正式的三档全没了）。
> **正确做法二选一**：① 用 `--out` 另存一份：`python rewrite_queries.py --dry-run --out queries_dryrun.json`（推荐）；② 验证完**重跑一遍正式三档**（约 4 分钟）。
> 这也是上面"反向验证"那条命令的意义：它不依赖缓存内容，纯粹检查**接线逻辑**通不通。

**✅ 接线已验证（9/20 实测，用现有** `queries_rewritten.json` **跑了两轮** `--dry-run`**，产物已删）**：


| 轮次               | 命令要点                                                             | 终端证据                                                                                                                                                          |
| ---------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 第 1 轮            | `--rewrite-mode plain --limit 2`                                 | `[改写] 已加载 queries_rewritten.json（档位=plain）｜模型=Qwen2.5-3B-Instruct-LoRA` / `查询改写：on（检索用 plain 档改写；生成仍用原中文问题）` / `⑦ 改写生效 : 2/2 题用了「plain」档改写` / 日志行含 `rw=plain` |
| 第 2 轮（含**反向验证**） | 先手工把缓存里 **Q1 的** `hyde` **删掉**，再 `--rewrite-mode hyde --limit 2` | `⑦ 改写生效 : 1/2 题用了「hyde」档改写` + `⚠ 未取到改写的题号：[1]（缓存里该档缺失或为空 → 已回退原问题）`                                                                                           |


**关键的一条硬证据：改写查询确实进了检索层**（不是只打了个标签）。第 2 轮里 Q2 的 `Top-K` 与第 1 轮**完全不同**：


| 题   | `plain` 档 Top-8                 | `hyde` 档 Top-8                    |
| --- | ------------------------------- | --------------------------------- |
| Q2  | `[14, 22, 1, 9, 3, 25, 13, 55]` | `[74, 55, 63, 61, 11, 58, 9, 14]` |


而 **Q1（缓存里没有** `hyde`**）两轮的 Top-8 逐位相同**（`[36, 76, 69, 35, 75, 70, 72, 100]`）→ 证明"回退原问题"是**真正的 no-op**，没有偷偷用一个空字符串去检索。这两条一起看，才叫"接线通了"。

### 1.7 正式跑 R4（今天最重的一轮）

**先冒烟 2 条**（别一上来全量；`--limit 2` 会只跑前两条，约 30 秒）：

```powershell
python eval_v2.py --top-k 8 --runs 1 --limit 2 --rewrite-cache queries_rewritten.json --rewrite-mode term --out result_rewrite_smoke --note "改写档=term 冒烟"
```

**再跑全量（runs=3，与 R3 同口径）**：

```powershell
# R4：只改"检索查询"这一个变量（term 档）
python eval_v2.py --top-k 8 --runs 3 --rewrite-cache queries_rewritten.json --rewrite-mode term `
  --out result_rewrite_term_runs3 --exp-id R4 --date 填你的执行日 `
  --note "O1-R4 模型自动改写（term 档），其余同 R3" `
  --conclusion "" --append-log
```

> ⚠ `--date` **一定写你的实际执行日**（日期口径）——**本教程示例里的 `9/23` 是计划日，照抄就会污染日志的日期列**：9/20 实跑时 R4 行填对了（9/20），**R4b / R4c 两行却照抄成了 `9/23`**，事后已按目录时间戳 + `run_log.txt` 改回 **9/20**（详见实验日志的「日期口径第二次生效」注）。**记住：`--date` 是"跑的时候填"，不是"抄模板"。**
> 📌 **本教程已把示例改成 `--date 填你的执行日`**（累计被抄四次：R4b/R4c、R5/R5a/R5b、F3v2）——它**故意长得不像一个日期**，抄下去会一眼看出问题。**别写成 `<你的执行日>`**：PowerShell 里 `<` 是保留运算符，python 根本不会启动（day18 踩过）。

跑完 3 档（每档约十几分钟，可串行跑；服务不用重启）：

```powershell
# R4-hyde
python eval_v2.py --top-k 8 --runs 3 --rewrite-cache queries_rewritten.json --rewrite-mode hyde `
  --out result_rewrite_hyde_runs3 --exp-id R4b --date 填你的执行日 `
  --note "O1-R4 改写 hyde 档（假设答案片段）" --append-log

# R4-both（备选）
python eval_v2.py --top-k 8 --runs 3 --rewrite-cache queries_rewritten.json --rewrite-mode both `
  --out result_rewrite_both_runs3 --exp-id R4c --date 填你的执行日 `
  --note "O1-R4 改写 both 档（术语+假设答案）" --append-log
```

**跑的时候顺手做一件"对账"动作**（day18 的教训沉淀）：打开结果目录的 `run_log.txt`，数一下 `④ 生成正确率` / `⑤ 防幻觉F1` 那两行的 TP/FP/FN/TN，**写日志时直接抄这四个数**，别抄别的行。

### 1.8 怎么读这三轮（三种结果，都要如实写）

与 **R3** 对照，只看**两个数**：`检索 strict`（改写直接作用的地方）和 `总正确率 / F1`（改写是否真的传导到了最终结果）。


| 若出现                                        | 说明                                                  | 怎么写 / 下一步                                                                                                               |
| ------------------------------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **strict 明显上升**（如 4/10 → 7/10 以上），总分/F1 也升 | 假设②在"真实现"下依然成立 → **报告的主结论成立**                       | 写进报告"优化成果"：**检索命中率 20%（原始基线）→ 40%（k=8）→ X%（+ 查询改写）**；并用 `retrieval_query` 字段举 1~2 个具体例子（改写前搜不到、改写后搜到了）                  |
| **strict 上升但总分不升**（甚至降）                    | **检索层变好 ≠ 答案变好**：上下文变了，生成层可能"看到更多反而更乱"，或新召回的段落把答案带偏 | **这是很有价值的一轮**：把它写成"**分层归因**"的证据（检索层与生成层的收益不同步），并指向第 3 步的模板约束                                                            |
| **strict 几乎没动**                            | 说明**本地 3B 的改写质量不够**（幻觉/丢实体），比不上人写的理想查询              | 如实写清"上界 100% 与真实现 X% 的差距 = **改写器的能力瓶颈**"；试 `hyde`/`both` 档，或退回"轻量改写"（只抽术语、不生成句子）。**别把责任推给检索层**——day18 已经证明检索层在"问对"的时候很强 |


> **⚠ 一条必须写进报告的边界**：**R4 是在同一套 20 题上迭代出来的第二个旋钮**（R3 是 k=8）。本轮数字**可以对外引用**（runs=3、口径一致、只改一个变量），但**不能把 R3 和 R4 的差说成"改写的全部价值"**——因为 k 的贡献（+2 题）已经含在 R3 里了。正确说法：**"在 k=8 的基础上，查询改写让检索命中率从 40% 提升到 X 个百分点"**。

**✅ 实测（9/20，三档全跑完）——落在"strict 上升但总分不升"这一支，上面那句模板句里的 X 要按下面填：**


| 行          | ① 检索 strict            | ④ 生成正确率          | ⑤ 防幻觉       | ⑩ 总正确率   | 读法                                  |
| ---------- | ---------------------- | ---------------- | ----------- | -------- | ----------------------------------- |
| R4 `term`  | 4/10（**没动**，loose 5→6） | 3/10 ↓           | 6/10 ↓      | 45% ↓    | 改写把命中题集合**重排**（`v∪b` 7→6）→ 成绩反退     |
| R4b `hyde` | **5/10 ↑**             | **6/10 ↑（+2 题）** | **4/10 ↓↓** | 50% ↓    | 检索与生成同涨、防幻觉崩 → **"看到更多反而更乱"的教科书案例** |
| R4c `both` | **6/10 ↑（最好）**         | **6/10 ↑**       | 5/10 ↓      | 55% = 打平 | 检索最好，总分只是"不亏"                       |


**→ R4 这段成绩，照下面这么写（别照上面表格的模板句写，否则等于只报好消息）：**

1. **能写的**："在 **k=8** 的基础上，查询改写（Qwen2.5-3B 自动改写）让**检索 strict 从 40% 提升到 50%~~60%（+1~~2 题）**、**生成正确率从 40% 提升到 60%**。"
2. **必须一起写的**："**但防幻觉从 70% 掉到 40%~50%，总正确率没有提升（最好也只是打平 55%）**。"（`hyde` 档 F1 0.571 是四轮最低）
3. **分层归因（这一轮最值钱的一句）**：**改写的作用点在"检索 + 生成"，副作用在"该不该拒答"** —— HyDE 拿"假设答案片段"当查询，检索回来的段落**更像答案本身**，但这条机制**不区分"资料里到底有没有答案"**，于是 out 题（Q13 期刊 / Q17 / Q19 / Q20）从"如实拒答"变成编造。
4. **绝不能写**：① "做了查询改写，检索命中率 100%"（那是**人工手写**的上界，真实现是 50%~60%）；② 只报 strict 不报 F1；③ 把 R4 的 −1 题说成"3B 太弱所以改写没用"（**逐题追查已证明原因是提示词替换专名**，见日志 R4 行的"逐题追查"与「Day19 诊断结论」）。

### 1.9 记账时"数字溯源"三件套

每轮跑完，日志行里除数字外，**必须能追溯到**：

1. `queries_rewritten.json` 的 `_meta`（模型名 + 时间 + 提示词版本 `day19-v1`）；
2. 结果目录 `result_rewrite_*/`（`run_log.txt` 的 TP/FP/FN/TN）；
3. `eval_results.json` 里每题的 `retrieval_query`（**"它到底问了什么"** —— 这是面试被追问"你怎么证明改写生效了"时最有说服力的材料）。

### ✅ 第 1 步验收标准

- [x] `rewrite_queries.py` 跑通，`queries_rewritten.json` 落盘且含 `_meta`
- [x] **人工抽查 ≥3 条**（Q1/Q3/Q5），逐条按 1.5 的**四看**给出合格/不合格判定（9/20 实测结论：**不合格**），并能说出"实体保住了没有、有没有编造术语、编造的词在 `chunks.json` 里是几次"
- [x] `eval_v2.py --rewrite-cache` 接线验证通过（终端打印 `[改写] 已加载` + `rw=<档位>`，**且** `⑦ 改写生效 = N/N` **不为 0**；`--rewrite-mode` 换成缓存里没有的档位时能打出 ⚠ 题号）
- [x] R4（至少 `term` 档）跑完并 `--append-log` 入表；**结果目录含** `retrieval_query` **字段**
- [x] 能对着 R3 说出"改了哪一步 → 哪个数字从 X 变成 Y"，以及"上界与真实现差多少、为什么"

---

## 🔬 第 2 步：改写之后再融合 / 评估 Rerank（30~40 分钟）

### 2.1 为什么现在才轮到"融合"

day18 的 RRF 是负结果（vector 4/10 → hybrid 3/10），但**归因很清楚**：BM25 只吃 ASCII token，**中文查询对它近乎零信号**（Q2/Q5/Q6/Q9 直接 `top=[]`），而 RRF 仍给它**等权**，等于一半噪声。

**反证也已经在 day18 拿到了**：把查询换成英文术语（hyde 档）后，**两路都有信号，三通道一致 10/10**。

> 所以今天的判据不是"融合好不好"，而是：**"在真实改写（不是理想查询）之后，融合值不值得做？"** —— 这正是 day18 留下的那个"什么条件下负结果会翻正"的问题。

### 2.2 用 `retrieval_lab.py` 复筛（零成本，约 35 秒）

`day18\retrieval_lab.py` 的 `--query-mode` 原来只有 `plain` / `hyde`。今天给它加一档 `file`：从 `queries_rewritten.json` 读**真实改写结果**当查询（这样筛查和正式评测**用的是同一批查询**，数字才对得上）。

> **✅ 已改好**：改动已落进 `第四周\day18\retrieval_lab.py`（**纯加法**：`plain` / `hyde` 两条老路径的行为与数字一字未变，day18 记录的 4-4-3 仍可复现）。下面是**与实际代码一致**的清单——你要读代码就照这几处对。

**改动 ①：在通道循环之前加载缓存**（`run()` 里 `store = None` 那段之后）：

```python
    # ---- Day19 新增：--query-mode file 时，从改写缓存读"真实改写结果" ----
    rewrite_cache = {}
    rw_missing = []
    if args.query_mode == "file":
        if not os.path.exists(args.rewrite_cache):
            print(f"❌ 找不到改写缓存：{args.rewrite_cache}")
            print("   → 先在 day19 目录跑 rewrite_queries.py 生成（或改 --rewrite-cache 指向正确路径）")
            sys.exit(1)
        with open(args.rewrite_cache, encoding="utf-8") as f:
            rewrite_cache = json.load(f)
        rw_meta = rewrite_cache.get("_meta", {}) or {}
        print(f"[改写] 已加载 {args.rewrite_cache}（档位={args.rewrite_mode}）"
              f"｜模型={rw_meta.get('model', '?')}")
        print(f"[改写] 缓存元信息：time={rw_meta.get('time', '?')} ｜ "
              f"prompt_version={rw_meta.get('prompt_version', '?')}")
```

**改动 ②：在取查询的地方加** `file` **分支**（紧挨着原来的 `hyde` 分支）：

```python
            query = q["question"]
            if args.query_mode == "file":
                rec = rewrite_cache.get(str(q["id"])) or {}
                rq = (rec.get(args.rewrite_mode) or "").strip()
                if rq:
                    query = rq
                else:
                    rw_missing.append(q["id"])      # 该档缺失 → 回落原问题（必须记账，否则会误读成"改写没用"）
            elif args.query_mode == "hyde":
                query = HYDE_QUERIES.get(q["id"], query)
```

**改动 ③：把** `--query-mode` **的 choices 扩一档，并加两个参数**：

```python
    parser.add_argument("--query-mode", default="plain", choices=["plain", "hyde", "file"],
                        help="plain=原题；hyde=手工理想查询（上界）；file=读 day19 的真实改写缓存")
    parser.add_argument("--rewrite-cache", default=DEFAULT_REWRITE_CACHE,
                        help="改写缓存路径（query-mode=file 时用；默认 day19\\queries_rewritten.json）")
    parser.add_argument("--rewrite-mode", default="term", choices=["plain", "term", "hyde", "both"],
                        help="用缓存里的哪一档当查询（默认 term；plain 档=原题，可用于接线自检）")
```

> `--rewrite-mode` 比教程原稿多了一个 `**plain` 档**：缓存里的 `plain` 就是原题，所以
> `--query-mode file --rewrite-mode plain` 的四个数字**必须与 `--query-mode plain` 完全一样**——
> 这是这一步最省事的**接线自检**（不一样就说明缓存读串了）。

**改动 ④（教程原稿没有，实测发现必须补）：`--json` 的落盘名要带模式。**
原来的代码无论哪种 `query-mode` 都写 `retrieval_lab.json` → **跑一次 `hyde` 或 `file` 就会把 day18 的 plain 证据覆盖掉**（而它是报告里"4-4-3"那组数字的原始明细）。现在：

```python
        # Day19 修复：落盘名带模式，避免"跑一次 hyde/file 就把 day18 的 plain 证据覆盖掉"
        out_name = "retrieval_lab.json" if args.query_mode == "plain" else f"retrieval_lab_{args.query_mode}.json"
```

→ 于是今天是 `retrieval_lab_file.json`（file 模式）和 `retrieval_lab_hyde.json`（hyde 模式），**day18 那个 `retrieval_lab.json` 不会被碰**。

**跑复筛**（顺手做三档融合调参，都在这一步零成本完成）：

```powershell
cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day18"

# ⓪ 接线自检（先跑这条，几秒钟）：file+plain 档 ⇔ plain 模式，四个数字必须一模一样
python retrieval_lab.py --top-k 8 --query-mode file --rewrite-mode plain

# ① 真实改写后的三通道（关键：看并集是否比 7/10 更高）
python retrieval_lab.py --top-k 8 --query-mode file --rewrite-mode term --json

# ② 调小 rrf_k（day18 的三个机制之一：1/(60+rank) 把名次差压平）
python retrieval_lab.py --top-k 8 --query-mode file --rewrite-mode term --retrievers hybrid --rrf-k 10

# ③ 换个说法：只在 vector 上前几名很确信时……（做不到"加权"就先用 rrf_k 与 池大小 两个旋钮）
python retrieval_lab.py --top-k 8 --query-mode file --rewrite-mode term --retrievers hybrid --rrf-k 20 --pool 10
```

> ① 的 `--json` 会**新增**一个 `day18\retrieval_lab_file.json`（**不会覆盖** day18 原来的 `retrieval_lab.json`，见改动 ④）。⚠ 但**同一个 `--query-mode file` 重跑会覆盖同名文件**（落盘名只带 query-mode、不带 rewrite-mode）→ 想同时留 `term` 与 `plain` 两份 json，得先备份或给脚本加 `--out-name`。
> 若 ① 打印出 `⚠ 未取到改写的题号：...`，说明这几题在这一档里是空的 → 它们的 strict 变化**不能归因给改写**。

**读表（记录到日志的「Day19 诊断结论」小节）**：


| 观察                         | 结论                                                                  |
| -------------------------- | ------------------------------------------------------------------- |
| `vector ∪ bm25` 从 7/10 再上升 | 改写把 BM25 从"瞎的"变成"有用的" → **融合终于有了意义**                                |
| `hybrid` 从 3/10 上升到接近并集    | `rrf_k` **调小/加权有效** → 可以把"混合检索"写进简历（并附上"必须先解决查询侧语言错位"这个前提）          |
| `hybrid` 仍然 < vector       | 保持 day18 的结论：**本项目当前阶段"单路向量 + 改写"最优**，融合列为"学到能讲"（面试可讲三个机制，这本身就是加分项） |


> **⚠ 这一步的数字仍然是"诊断"**（`retrieval_lab.py` 只测检索、不测生成）→ **别把上表的诊断数字（本轮实测：hybrid 6/10、`v∪b` 6/10、三通道并集 7/10）写进「一、实验记录表」**。它进报告的"分析/失败尝试"章节。要在成绩表里体现融合，得用 `eval_v2.py` 再跑一轮——**但 `eval_v2.py` 目前只有单路向量**，所以得先给它接 BM25/RRF（时间不够就 D5 做）。

**✅ 实测（9/20 已跑完，结论已记进日志的「Day19 诊断结论」小节）** —— 上面三行只有**中间那行的方向**对，数字与分支都要按实测改：


| 观察行                      | 预期  | 实测（9/20）                             | 判定                                                                          |
| ------------------------ | --- | ------------------------------------ | --------------------------------------------------------------------------- |
| `v∪b` 从 7/10 **再上升**     | 上升  | **7/10 → 6/10（−1）**                  | ❌ 不成立：BM25 确实被唤醒（4→5，空列表 4 题→0 题），但 Q1/Q4/Q7 这三道"靠**专名**命中"的题被改写成泛化词后**失守** |
| `hybrid` 上升到**接近**并集     | 接近  | **3/10 → 6/10 = 并集规模**（三通道并集仍是 7/10） | ✅ 成立，且是**本轮唯一成立的一条**：hybrid 第一次**超过**最好的单通道（vector 4/10）                    |
| `hybrid` **仍然 < vector** | ——  | **6/10 > 4/10**                      | ❌ 不成立 → day18 的"本项目当前阶段单路向量最优"在"查询侧先对齐"之后**翻正**                             |


**三条实测结论（比数字更重要）**：

1. **融合翻正，但机制不是"并集变大"**：plain 时 hybrid 连 vector 已有的命中都丢（hy={5,9,10} ⊊ v={3,5,9,10}）；term 时 hybrid **完整保住 vector 的 4 题、还多捞 2 题**（Q10 来自 BM25 的强信号；**Q8 在两路各自的 Top-8 里都不存在，是 RRF 从候选池里捞回来的**）。
2. **改写做的是"命中集合重排"（+4/−4 换手），不是净增益**：救回来的全是"中文→英文"的通路（Q2/Q3/Q5/Q9），弄丢的全是**它自作主张替换掉的专名**（Q1/Q4/Q7）与**被泛化词稀释的原句**（Q10 的 `embodiment gap` 本来已对齐）。→ 与 §1.5 的"改写器会编造实体"**互相印证**。
3. **旋钮**：`rrf_k` 60 → 10 **不改变题目级结果**（day18 "名次被压平"的归因在本轮**证伪一半**：两路都有信号时它不构成瓶颈）；`pool` 20 → 10 **丢 Q8**（chunk 3 只排在第 9~20 名 → 进不了融合候选）→ **别缩池子**。⚠ 命令 ③ 同时动了 `rrf_k` 与 `pool`，掉题应归因到 `pool`；**要严格单变量，`--rrf-k 20` 需配 `--pool 20` 再跑一次（35 秒）**。

**真正该背下来的一句**：融合的价值**不在做大并集，而在把并集里已有的题兑现进 Top-8** —— 兑现率 **3/7 = 43% → 6/7 = 86%**。写报告的"优化步骤"用这句，比"并集涨了 X"准确得多。

### 2.3 Rerank（O1-R6）怎么评估（时间不够就"学到能讲"）

Rerank 的定位是**两段式召回**：先粗召回 `k=20~50`（宁滥勿缺）→ 用 **cross-encoder（如** `bge-reranker-base`**）** 对"问题-段落"逐对打分重排 → 取 Top-4/8。

**今天不一定要跑，但要能讲清四件事**（面试常问）：


| 问题               | 答题要点                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| 为什么不直接用向量 Top-K？ | 双塔（bi-encoder）把"问题"和"段落"**各自编码**再算余弦 → 快，但**没有交互**；cross-encoder 让两者**拼在一起过一遍模型** → 准，但**慢**（每对都要前向一次） |
| 为什么是"两段式"？       | 用便宜的粗召回把候选从 105 缩到 20~50，再用贵的精排 → **精度与延迟的折中**（工程标配）                                                   |
| 成本多大？            | CPU 上 `bge-reranker-base` 对 20 个候选约几百毫秒~2 秒；105 个 chunk 的小库完全跑得动                                       |
| 什么时候不该上？         | 库小（本题 105 段）+ 查询改写已经解决"对齐"问题时，**收益可能被改写吃掉** → 要测才知                                                     |


**如果想真跑一次（约 30 分钟，注意别和训练抢显存）**：`pip install FlagEmbedding` 太重，更轻的做法是用 `sentence_transformers` 的 `CrossEncoder` 加载 `BAAI/bge-reranker-base`，写一个 `rerank_lab.py`：取 `vector` 的 Top-20 → 用 cross-encoder 打分 → 取 Top-8 → 用与 `eval_v2.py` 一致的双轨判据算 strict。**结果同样只作诊断**。

### ✅ 第 2 步验收标准

- [x] `retrieval_lab.py --query-mode file` 跑通，得到**真实改写后的**三通道数字
- [x] 至少试了 `rrf_k=10/20` 两档，能说出"融合在什么条件下翻正/仍不划算"
- [x] 日志里新增「Day19 诊断结论」小节（**标注为诊断口径，不进成绩表**）
- [x] 能对着白板讲清"双塔 vs cross-encoder / 两段式召回"（Rerank 不跑也要能讲）

---

## 🧩 第 3 步：O2-G2 模板 09 v2 + 逐句消融（50~70 分钟）⭐

### 3.1 先把"两层问题"分开（这是今天最容易被混淆的地方）

day17 就已经定调：**Q11/Q14/Q18 是"生成层"的靶子**（它们不是"没检索到"，而是"检索到的资料不足以答，模型硬编"）。day18 的自检又把"检索层"钉死在"查询侧语言错位"上。所以：


| 层   | 症状                                      | 认领它的手段                                     |
| --- | --------------------------------------- | ------------------------------------------ |
| 检索层 | in 题 `top_ids` 里没有 `source_chunk`       | O1 系列（k / 改写 / 融合 / Rerank）→ **今天第 1、2 步** |
| 生成层 | out 题**不拒答、自行编造**（FN）；in 题**该答却拒答**（FP） | **O2 系列（模板 / 约束 / 引文校验）→ 今天第 3 步**         |


> **报告里必须分层报**：`防幻觉 F1` 里 **FN 主因在生成层、FP 主因在检索层**（day18 已确认 R3 的 FP=0，而 FN=Q11/Q14/Q18）。**"把两层分开"本身就是面试加分点**——很多 demo 只会报一个总正确率。

### 3.2 靶子清单（先对账，再改模板）

打开 **R3** 的结果目录 `..\day17\result_lora_k8_runs3\run_log.txt`，找到这两行：

```text
④ 生成正确率（in 题）：4/10
⑤ 防幻觉F1 : P=0.824 R=0.824 F1=0.824（TP=7 FP=0 FN=3 TN=10）
```

→ **R3 的靶子 = FN 3 条 = Q11 / Q14 / Q18**（Q11 在 R3 里三次生成一致率 1.00，**稳定编造，不是抖动** → 最适合当消融的观察点）。

**改模板前，先做一件事：把这 3 条答案原文抄下来**（`评测表.md` 里有），写清"它们编了什么"：

> **✅ 已抄（9/21，源 = `第四周\day17\result_lora_k8_runs3\评测表.md` 附录「每题完整答案」，即 R3 那一轮的原文，逐字未改）。**
> **抄完顺手干了一件更重要的事**：拿每条答案的 `Top-K` 去 `第三周\day13\chunks.json` **逐句回查**是"编的"还是"抄的"。**结论是：这三条里没有一条是凭空编造**——内容**全部来自它检索到的段落**，错的是"**把它们当成了什么**"。所以下面第三列不叫"编造形态"，叫**"错在哪"**（附核过的原文出处）。


| 题   | 类型  | 答案原文（R3 逐字抄）                                                                                                                                                         | 错在哪（按 `chunks.json` 逐句核过；R3 的 Top-K 见括注）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | v2 该拦哪一句                                                                                                                                                      |
| --- | --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q11 | out | `GMR方法的缺点：① 脚滑动;② 地面穿透;③ 自相交。[36];④ 初始参考运动帧影响跟踪政策是否能开始跟踪或立即失败(无论使用哪种重定向方法);⑤ 用户难以区分GMR和Unitree重定向。[35];⑥ 跟踪误差在所有存活帧上计算。[48];⑦ Empjpe高。[46];⑧ 成功回放数/总回放次数低。[47]。`     | **不是编造，是"把资料里的各种句子逐条搬来、统一贴上『缺点』标签"**（Top-K = `[36,18,76,9,35,17,48,46]`）：③ = chunk 9 `ground penetrations, self-intersections, and sudden jumps in joint values`（原文，但那是**伪影**不是缺点）；④ = chunk 17 `the initial frame of the reference motion can greatly impact whether the policy is able to start tracking it or if it fails immediately (regardless of retargeting method used)`（**一字不差照抄**）；⑤ = chunk 35 `users also have a harder time distinguishing the two`（**照抄**）；⑥ = chunk 48 `The tracking errors are computed for all frames that a policy is alive.`（**照抄**）；⑦ `Empjpe` = chunk 46 里的**指标符号** `(Empjpe, 10−3 rad)`（是"平均关节旋转角误差"的缩写，库里真有，模型当成"缺点名"搬出来）；⑧ = chunk 46 `the number of successful rollouts and the total number of rollouts`（**照抄**，那是**成功率**的定义）。**唯一"凭空"的是"缺点"这个框**（论文根本没有 limitations 段）；另 `**[47]` 不在 Top-K → 非法引用 1 处** | **"没有直接回答问题的资料时，不得拿相邻话题的段落顶上"**（第 2 条那句"资料里有依据就照实回答"必须补"**且必须直接针对问题**"）                                                                                       |
| Q14 | out | `GMR 主要从源运动学数据找到目标机器人运动：[76§7];通过两阶段优化寻找机器人运动：① §;主差异是处理源运动缩放：② 。`                                                                                                   | **整段就是 chunk 76 的原文照抄**（Top-K 首条 = 76）：`GMR is described in detail in Sec. IV. The main difference from the prior two methods is how it handles source motion scaling... This is followed by a two-stage optimization to find the robot motion.` —— 模型只把语序调成"① ② "。**零编造**；错在 **① 答非所问**（问"奖励函数细节"，答"方法总览"，而论文恰恰"抑制奖励调参"）+ **② 引用格式崩坏**：`[76§7]`、`① §`、`②` 里的 `§` **后面编号被吃掉**                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 同上一格（**答非所问 → 拒答**）；`§` 后必须带编号这条留给 `v2b` 的引用约束                                                                                                                |
| Q18 | out | `资料§100：IEEE, 21次; 资料§36：A. Tang等, 1次; [资料§41]: K. Zakja, 0次; [資料§42]: F. G.Harvey等, 0篇; [资料 §43]: G. Pavloks等, 2次; [资枓§84]: man Performance Alliance等, 4次。资料中没有提及。` | **全是"看过的内容 + 认错了语义"**（Top-K = `[100,103,84,101,75,91,74,25]`）：**① 把参考文献的页码当成"被引次数"** —— `425` = chunk 100 的 `pp. 425–432`、`679` = chunk 84 的 `pp. 679–686`（day14 记的 `425/679/248` 中，`248` = chunk 101 的 `pp. 248:1–248:16`）；**② 人名是参考文献里的真作者** —— `A. Tang`（chunk 100 `[36]`）、`K. Zakka`（chunk 103 `[41]`）、`F. G. Harvey`（chunk 103 `[42]`）、`G. Pavlakos`（chunk 103 `[43]`）、`man Performance Alliance`（chunk 84 致谢，原文 `Human Performance Alliance`），但**抄错两个字母**：`Zakka→Zakja`、`Pavlakos→Pavloks`；**③ 编号体系张冠李戴** —— 答案里的 `[资料§41]/§42/§43` **不是段落号，是参考文献编号 `[41][42][43]`**，被直接当成了"资料段落号"（按 chunk 口径这 3 处 + `§36` 都不在 Top-K，即 **O2-G3 会记 4 处"非法引用"**，但**根因是"两套编号混了"，不是"引了没看过的段落"**——这条必须写进报告，否则会被读成 4 处凭空引用）；**④ 拒答句在末尾** → 拒答词规则被骗过（这正是 `MANUAL_FIXES` 判它错的原因）                                                                                      | **"引用格式只能是 `[资料§N]`，N 只取【资料】里每段开头标注的编号；正文/参考文献里的 `[数字]`、页码一律不得当作段落号或被引次数"**（现有措辞"只能引用资料中出现过的段落编号"**太宽**——模型把"参考文献编号"也算成"资料中出现过的编号"了）+ **"先拒答，不要先编一段"**（第 4 条） |


> **⚠ 这次核对改变了 v2 的设计预期（重要）**：三条**没有一条是凭空编内容**，全部来自检索到的段落 → **只写"禁止编造数字/引用"治不了它们**。真正要拦三件事：
>
> 1. **该拒答就先拒答**（治 Q11/Q14 的"不拒答"与 Q18 的"先编一段再补拒答"）→ 现有第 2 条 + 第 4 条；
> 2. **不许给"编号/数字"换语义** → 第 3 条要**加一句显式禁令**（见上表 Q18 格）；
> 3. **答非所问也算"资料不足"** → 第 2 条补"且必须直接针对问题"。
>
> **顺带解开一个疑点**：`MANUAL_FIXES` 里 Q18 的注写的是"编造一串引用统计数字（425/679/248 等）"，但 R3 那轮的答案里**根本没有 425/679/248**（是 `21次/1次/0次/2次/4次`）——因为那句注是**照抄 day14 那一轮的观察**。回了库才知道 **425/679/248 是页码**（`pp. 425–432` / `pp. 679–686` / `pp. 248:1–248:16`）→ **Day14 的观察和 R3 的复现，其实是同一个机制**：模型把参考文献里的数字读成了统计量。**这份"跨两轮的同一机制"本身就是报告里能写的一条证据。**
>
> **对 R5 消融的预判（跑完回来验证）**：`v2b`（只加引用/拒答约束）应能同时治 **Q11 与 Q18**；**若 R5b 只治好 Q18、治不了 Q11**，说明瓶颈在"**答非所问**"而非"引用约束"，那就得给 v2 单独加一条**直接相关性**判据（把这句预判原样留着，跑完对一下——这就是"受控实验"该有的样子）。

### 3.3 复核 + 补齐 v2 模板（含逐句消融用的 v2a / v2b）

day17 的 `eval_v2.py` 里 `PROMPT_TEMPLATES` 已经有 `v1` / `v2`（草稿）。今天**补齐 v2a / v2b**，让"改哪一句"可以分开测：

> **✅ 已改好**：`第四周\day19\eval_v2.py` 的 `PROMPT_TEMPLATES` 现在就是下面这四档（`v1` 一字未动；`v2` 在 day17 草稿上补了第 4 条"只输出拒答句"）。`--template` 的 choices 是自动从这张表生成的，**不用另外改参数**。
>
> **⚠ 9/21 复核后收紧了一处（只动第 3 条规则，`v2a` 一字未动）**：原措辞"只能引用资料中出现过的段落编号"**太宽** —— 回库核对发现 Q18 把**参考文献的方括号编号 `[41][42][43]`** 与**页码 `pp. 425–432 / 679–686 / 248:1–248:16`** 当成了"资料段落号"与"被引次数"（详见 §3.2 表）。故 `v2` / `v2b` 的第 3 条改为**显式限定 N 的来源**（见下方代码）。**消融结构不受影响**：`v2a` 与 `v2b` 仍只差第 3 条这一句，两者照样可比。**故意没动第 2 条**（"答非所问"那一类错法留作 R5 的判定对象，预判见 §3.2 末尾）。

```python
PROMPT_TEMPLATES = {
    "v1": """（第三周模板 09 原样，一字不改 —— 复现基线必须用它）""",

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
```

> **⚠ 为什么必须拆 v2a / v2b**：v2 一次改了两件事（"弱化拒答"和"禁止编造"），若只跑 v2，**涨了也说不清是谁的功劳、跌了也不知道该删哪句**。v2a / v2b 就是"**单句消融**"——这是"每轮只改一个变量"在提示词层面的落地。

### 3.4 并排跑（同 k=8 / qi=off / runs=3，与 R3 只差模板）

```powershell
# R5：模板 v2（两处改动一起）
python eval_v2.py --top-k 8 --runs 3 --template v2 --out result_template_v2_runs3 `
  --exp-id R5 --date 填你的执行日 --note "O2-G2 模板 09 v2（弱化拒答+禁止编造）" --append-log

# R5a：只弱化拒答（v2a）
python eval_v2.py --top-k 8 --runs 3 --template v2a --out result_template_v2a_runs3 `
  --exp-id R5a --date 填你的执行日 --note "消融：只弱化拒答条件（治 FP）" --append-log

# R5b：只加引用/数字约束（v2b）
python eval_v2.py --top-k 8 --runs 3 --template v2b --out result_template_v2b_runs3 `
  --exp-id R5b --date 填你的执行日 --note "消融：只加禁止编造编号/数字（治 FN）" --append-log
```

> ⚠ `**--date` 又一次被抄成 `9/23`（第三次了这个坑）**：本轮实际执行日是 **9/21**（三个结果目录时间戳 12:12 / 12:40 / 12:59，`run_log.txt` 同为 9/21），已把日志里被写成 `9/23` 的三行改回 `9/21`。**教程示例里的 `9/23` 只是占位（计划日）**，抄命令时**务必换成你的实际执行日**；`--exp-id` 与 `--note` 可以照抄。

**出对比表**（这是第 6 步报告里"逐步骤贡献表"的一行）：

> **✅ 已跑完（9/21），下表数字逐个抄自终端「汇总指标」，未做任何换算。** 括号里是 `run_log`/终端同源的 **TP/FP/FN/TN**（day18 纪律：F1 必须回原始输出核对四格，不抄 P）。


| 轮次     | 模板  | 检索 strict  | 检索 loose   | 生成（in）   | 防幻觉（out） | 防幻觉 F1                      | 总正确率              | 一致率   |
| ------ | --- | ---------- | ---------- | -------- | -------- | --------------------------- | ----------------- | ----- |
| R3（基线） | v1  | 4/10 40.0% | 5/10 50.0% | 4/10     | **7/10** | **0.824（TP7 FP0 FN3 TN10）** | 11/20 = **55.0%** | 0.950 |
| R5     | v2  | 4/10 40.0% | 5/10 50.0% | **5/10** | **3/10** | 0.462（TP3 FP0 FN7 TN10）     | 8/20 = 40.0%      | 0.950 |
| R5a    | v2a | 4/10 40.0% | 5/10 50.0% | 3/10     | 4/10     | 0.571（TP4 FP0 FN6 TN10）     | 7/20 = **35.0%**  | 0.950 |
| R5b    | v2b | 4/10 40.0% | 5/10 50.0% | 4/10     | 6/10     | 0.706（TP6 **FP1** FN4 TN9）  | 10/20 = 50.0%     | 0.967 |


**✅ 自检信号通过**：**四轮的 `检索 strict/loose` 一字不差（4/10、5/10）** —— 模板只动生成层，检索层没被碰 → 说明这三轮确实是"单变量"。

> **⑧ 引文合法性**（v1 那轮跑在 day17 的脚本上、还没有这一项，故基线缺）：


| 轮次  | 模板  | 引文总数 | 其中带引文的题数 | 非法处数     | 非法题号                    |
| --- | --- | ---- | -------- | -------- | ----------------------- |
| R5  | v2  | 19 处 | 9 题      | **11 处** | [3, 13, 15, 17, 18, 19] |
| R5a | v2a | 22 处 | 15 题     | 4 处      | [5, 7, 18]              |
| R5b | v2b | 14 处 | 6 题      | **11 处** | [3, 7, 13, 14, 18]      |


> ⚠ **读这列必须带一句话**（不然会读错）：**"非法"里相当一部分不是"引了没看过的段落"，而是"编号体系混用"** —— 模型把正文/参考文献里的方括号编号 `[N]`（= **参考文献编号**）当成了"资料段落编号"。本次已逐条回库证实（见下方"逐题机制"的 Q13/Q15/Q18）。**报告里照抄"非法 11 处"会被读成"11 处凭空引用"，务必补上这句。**

> 📌 **另注（抄录忠实性）**：R5a 的终端里 `① 检索命中 strict` 那一行**出现了两次**（值相同、都是 4/10）。脚本本身只打印一次（`eval_v2.py` 第 1026 行），`--append-log` 落进日志的也只有一行 → 这是**复制终端输出时的重复**，不是脚本重复输出的 bug。

**怎么读**（三种情形都要如实写）：


| 情形                           | 说明                          | 怎么写                                                            |
| ---------------------------- | --------------------------- | -------------------------------------------------------------- |
| FN 减少（防幻觉 ↑）且 in 题不降         | **v2 双赢** → 按最好的一版出"定稿数字"   | 报告写"模板 09 v2 让防幻觉从 7/10 提到 X/10，同时生成正确率保持/提升"                  |
| FN 减少但 in 题**出现了新的 FP**（过拒答） | **v2b 的约束太硬**（模型宁可不答）       | 明确写"**这是精度-召回的权衡**"：加约束提精度降召回 → 报告里给"两版都留着，按场景选"               |
| 都没动                          | **模板不是瓶颈** → 编造是这个 3B 的固有倾向 | 如实写进「失败尝试记录」：**"这 3 条的编造不是提示词能救的"** → 指向 O2-G3（引文校验）或更大模型/更好数据 |


**✅ 实测：上面三种情形一种都没中 —— 实际是第四种：`FN 增加`（防幻觉反而 ↓）。**


| 判据                | 实测                                | 结论                                 |
| ----------------- | --------------------------------- | ---------------------------------- |
| 三档 v2 的 F1 vs v1  | 0.824 → **0.462 / 0.571 / 0.706** | **全部低于 v1**，v2 这一轮**是负结果**         |
| 三档 v2 的总正确率 vs v1 | 55% → **40% / 35% / 50%**         | 同上；`v2` 最差（−3 题）、`v2a` 次差（−4 题）    |
| 防幻觉 vs v1         | 7/10 → **3 / 4 / 6**              | **强化约束把"如实拒答"打掉了一半**               |
| 生成正确率 vs v1       | 4/10 → 5 / 3 / 4                  | 唯一亮点是 `v2` 的 5/10，但代价是 out 崩到 3/10 |


**消融分解 —— 哪一句有毒（这是本轮最值钱的结论）**：


| 组合                            | 含"弱化拒答"② | 含"禁编造/只输出拒答句"③④ | 防幻觉（out） |
| ----------------------------- | -------- | --------------- | -------- |
| `v1`（day13 模板 09 原文，**无**②③④） | ✗        | ✗               | **7/10** |
| `v2b` = ③④                    | ✗        | ✓               | 6/10     |
| `v2a` = ①②④                   | ✓        | 部分              | 4/10     |
| `v2` = ①②③④                   | ✓        | ✓               | **3/10** |


→ **② 那句"只有当资料确实找不到相关信息时才拒答；资料里有依据就照实回答"是全组最有害的一句**：带 ② 的两档 out 只有 **3~4/10**，不带 ② 的有 **6~7/10**，**差 2~4 题，超出运行波动**。
**机制**：out 题的 Top-K **永远有 8 条**（检索总会返回东西），② 等于告诉模型"有东西就是有依据" → 它就把**检索到的相邻话题段落**当依据答了出来。**"乐于助人"的提示语在 out 题上是负资产。**
⚠ **但要如实说清另一半**：`**v2b`（6/10）仍然低于 `v1`（7/10）** → **③④ 也没能净赚**，所以正确结论不是"留 ③④ 去掉 ②"，而是**"这一版提示词对生成层没有净收益"**。

**逐题机制（"v2 为什么反而更敢答"的实锤，逐条回 `chunks.json` 核过）**：


| 题                     | v2 轨道上的变化                                                                                                     | 核过的机制                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q20（仿真步数，out）         | v1 拒答 → **v2/v2a 编造**：`Run (slow): 50步；Run: 11步；Run (stop &go): 37步。共约 112 步`                                 | **50 / 11 / 37 是 Tab. I 的"运动时长"列**（chunk **66**，**在 Top-K 内 ✅**）→ **误读表格列**，把"时长"当"仿真步数"；`50+11+37=98`，连"共约 112 步"也是算错的                           |
| Q15（参数量，out）          | v1 拒答 → **v2 编造**：`[37] mesh,v h = f SMPL(β h , θ h ) … 关节回归器 J∈R (n_j×3)×(n_v×3)`                            | 这段公式在 **chunk 31**（**在 Top-K 内 ✅**）→ **抄对了内容、标错了出处**：标 `[37]` 是因为原文写着 `SMPL [37]`，`**[37]` 是参考文献编号**（chunk 100 里 `[37] M. Loper…SMPL`），该标的是 `§31` |
| Q13（期刊，out）           | v1 拒答 → **三档全部编造**（v2b 为 `[18]: Motion adaptation based on Physically Based Retargeting [Lyard Thalmann 20]`） | 这条参考文献在 **chunk 91**（**在 Top-K 内 ✅**）→ 同 Q15 的机制：**把参考文献条目当成"论文发表的期刊"**，编号也照参考文献写 `[18]`                                                          |
| Q17（与 DeepMind 比，out） | v1 拒答 → **v2/v2a 编造**（v2a 直接贴英文原文）                                                                            | 把"源运动缩放的差异/伪影"段当成"对比结论"                                                                                                                           |
| Q18（总被引，out）          | 一直是编造，**v2b 出现新形态**：`[1]+[36]+[37]+[41]+[42]+[43]+[84]=1+1+1×2+1+0+1+2=7。`                                    | **把参考文献编号当成数据做加法** —— 比 v1 的"页码当被引次数"更离谱                                                                                                          |


> **⚠ 一条预注册预判落空了，如实记下**：§3.2 末尾我写过"`v2b` 应能同时治 Q11 + Q18；若只治 Q18 治不了 Q11，说明瓶颈在'答非所问'"。**实测：两个都没治好**（Q11 三档全是编造形态；Q18 三档全错，v2b 还多出一个新形态）。
> **→ 修正后的结论**：**提示词约束对"内容取自检索、但被误读/错用"这一类幻觉基本无效**（模型"看过"这些字，所以它不认为自己在编），**必须靠工程手段**：把 `⑧ 引文合法性` 从"只留痕"升级为**强制白名单**（非 Top-K 编号直接剪掉/重答），或对 out 题加**独立的"是否答非所问"判据**。这条直接改写 O2-G3 的定位：**它不再是"加分项"，而是必需的兜底。**

> **📌 与检索侧合看（别把两件事混成一件事）**：本轮的**检索四列完全没动**，说明这一轮**只测了生成层**；而 §2.2 的结论是"改写 + 融合"在**检索层**有效。→ **"检索层有收益、生成层这版提示词无收益"两句话必须分开写**，各归各的证据。

**→ 工程结论（可直接写进报告）**：

1. **默认模板仍是 `v1`**（R3 口径：F1 0.824 / 总 55%）；`**v2`/`v2a`/`v2b` 三档都不采用**，作为"负结果 + 分层归因"的一轮留档。
2. **报告里的写法**：不写"模板优化提升了 X"，写"**我们按逐句消融验证了模板约束，结果是负的；归因清楚 —— `②` 的'有依据就作答'在 out 题上是负资产；③④ 单独使用也无净增益**"。**一个**能讲清归因的失败**，比一个说不清来源的提升更值钱**。
3. **下一步**：模板这条线**不靠提示词继续磨**，转 ① O2-G3 升级为强制引文白名单；② 或把"拒答判据"做成独立模块；③ 编造的根治指向**更大模型 / 更好的训练数据**（与第 5 步的 F3 真实数据线合并讲）。

### 3.5 O2-G3 引文合法性兜底：从"只留痕"升级为"强制白名单"（20 分钟，区分度强）

> **本节的形态在 9/21 变过一次**：初版（9/20）是"**只留痕**"——只统计、不改答案；9/21 跑完 R5 之后**升级为"强制白名单"**。原因写在 3.5.2，两版代码都在下面（报告里"演进的动机"本身就是可讲的东西）。

#### 3.5.1 初版（9/20）：只留痕

**思路**：把答案里的引文编号 `[资料§N]`（以及裸写的 `[N]`）抽出来，检查编号是否真的出现在**本轮上下文**的段落编号里；把结果落盘留痕。

> **✅ 已改好**：`day19\eval_v2.py` 里的 `CITE_RE` + `legal_citation_filter()`：

```python
import re

# ⚠ 为什么必须接受两种写法（Day19 读代码 + 读 day17 答案时发现的）：
#   模板里写的是 [资料§N]，但 build_context() 给段落的标签其实是 [N]
#   （`f"[{chunk_id}] {text}"`）→ 模型两种都写过：day17 的答案里 `[35]`、`[97]`
#   与 `[资料§68]` 混着出现。**只认 [资料§N] 会漏检一半以上的非法引用**，
#   比不检查更危险（会让你误以为引文很干净）。
CITE_RE = re.compile(r"\[(?:资料§|资料|ref|§)?\s*(\d+)\]")


def legal_citation_filter(answer, context_ids):
    """把不在本轮上下文里的引文编号剪掉，返回 (新答案, 被剪掉的编号列表)。"""
    ctx = set(int(i) for i in context_ids if i is not None)

    def repl(m):
        return m.group(0) if int(m.group(1)) in ctx else ""

    bad = [int(m.group(1)) for m in CITE_RE.finditer(answer) if int(m.group(1)) not in ctx]
    if not bad:
        return answer, []          # ← 9/21 加的这一行，见 3.5.5 约束③
    cleaned = _tidy_after_strip(CITE_RE.sub(repl, answer))
    return cleaned, bad
```

> **两个实现细节，和教程初稿不同（都是实测逼出来的）**：
>
> 1. **正则同时认 `[资料§N]` 与裸 `[N]`**（原因见上面那段注释）：这正是"**文档里说的格式 ≠ 模型实际用的格式**"的一个活例子——校验器要按**实际数据**写，不能按**设计文档**写。
> 2. **剪完要收拾残局**（9/21 补的 `_tidy_after_strip`）：直接 `""` 替换会留下 `伪影 ；② 身高伪影 ；` 这种"悬空标点"，正文会变脏。收拾规则只在**确实剪过东西**时生效（见 3.5.5 约束③）。

#### 3.5.2 9/21 升级：为什么"只留痕"不够

R5 三档模板跑完后，靶子题的错法被逐条回库证实（见 §3.2 与 §3.4"逐题机制"）：

| 题 | 模型输出的错法 | 内容到底哪来的 |
| --- | --- | --- |
| Q20 | 把 Tab. I 的**"运动时长"列**当成"仿真步数" | 数字**在资料里**，列取错了 |
| Q15 | 抄了 chunk 31 的公式，却标成 `[37]` | 内容对，**编号是参考文献号** |
| Q13 | 抄了 chunk 91 的参考文献条目，当成"发表的期刊" | 内容在资料里，**语义域用错了** |

三条的共同点：**内容取自检索、只是用错了** → 模型不认为自己在编 → **在提示词里写"禁止编造"治不好**（R5b 的 F1 只到 0.706，还多出 1 个 FP）。既然生成层这一侧按不住，就只能在**生成之后**加工程兜底。

#### 3.5.3 三个策略：`--citation-policy record | strip | retry`

| 取值 | 行为 | 调模型？ | 用在哪 |
| --- | --- | --- | --- |
| `record`（**默认**） | 只统计、一字不改 | 否 | 复现 R3/R4/R5 等历史行**必须**用这个 |
| `strip` | 非法编号**就地剪除**（纯文本后处理） | 否 | 零成本、零随机性，作为兜底基线 |
| `retry` | 先带"你引了不存在的编号，白名单是这些"的修正指令**重答一次**，仍非法再剪 | 是（最多 +1 次/题） | 想看"给了约束后模型能不能自己改对" |

`--citation-policy` **默认 `record`**，所以**已跑过的轮次数字全部不受影响**——这是特意设计的：新功能不能悄悄改掉旧结论。

#### 3.5.4 升级后的代码（**已改好**）

**① 重答指令**（`retry` 用）——注意它同时给**白名单**和**出口**：

```python
RETRY_INSTRUCTION = """

【必须修正的问题】
你上一次的回答引用了这些段落编号：{bad}；但本次【资料】里**只有**这些编号：{ctx}。
请重新回答上面那个问题，并且：
1. 只允许引用上面列出的编号，格式 [资料§N]；
2. 如果现有资料不足以回答，**只输出**「资料中没有提到」，不要给出任何猜测、也不要标注编号；
3. 不要输出"你上次引错了"这类解释，直接给新答案。"""
```

**② 策略执行**（`eval_v2.py`，`legal_citation_filter` 旁边）：

```python
def apply_citation_policy(policy, profile, template, question, context, answers,
                          ids, temperature, max_tokens, dry_run=False):
    """O2-G3 强制白名单。返回 (交付答案列表, 元信息 dict)。"""
    meta = {"removed": 0, "retried": 0, "only_bad_after": 0}
    if policy == "record" or dry_run:
        return answers, meta                      # record = 原样交付（一字不动）

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
            meta["only_bad_after"] += 1           # 剪完啥都不剩 → 只当诊断记着
        delivered.append(cur)
    return delivered, meta
```

**③ 主循环：判两次**（`record` 时判一次，省得白算）：

```python
            # day19 O2-G3 升级：强制白名单（默认 record = 原样交付 → R3/R4/R5 数字可复现）
            delivered, cite_meta = apply_citation_policy(
                args.citation_policy, profile, template, q["question"], context, answers, ids,
                args.temperature, args.max_tokens, dry_run=args.dry_run)

            # 原始生成先判一次（**只用来做统计可比**：⑧ 的两列要跟 R3/R4/R5 同口径）。
            # judge_question 不调模型（纯规则+投票），多算一次的开销可忽略。
            rec_raw = judge_question(q, ids, context, answers)
            rec = rec_raw if args.citation_policy == "record" \
                else judge_question(q, ids, context, delivered)
```

> **第三个实现细节，和初稿不同（9/21 改的）**：初稿写"位置放在 `judge_question` **之后**"。那是 `record` 的语义。**`strip`/`retry` 必须放在判分之前**——因为我们判的应该是**真正交付出去的那份答案**；同时**额外**用原始答案判一次，专门用来报"⑧ 原始口径"，免得新版数字跟旧轮不可比。

**④ 落盘字段**（每题都记，出问题能逐题回放）：

```python
            rec["answer_raw"] = rec_raw["answer"]            # 未兜底的原文
            rec["correct_raw"] = rec_raw["correct"]          # 未兜底的判定
            rec["citation_n"], rec["citation_bad"] = ...      # 原始口径（与 R3/R4/R5 同口径）
            rec["citation_n_delivered"], rec["citation_bad_delivered"] = ...   # 交付口径
            rec["citation_removed"] = cite_meta["removed"]    # 剪了几处
            rec["citation_retried"] = cite_meta["retried"]    # 重答了几次
            rec["citation_only_bad"] = cite_meta["only_bad_after"]  # 剪完没内容
```

**⑤ 终端 ⑧ 变成四行**（`record` 时仍只打第一行，观感跟以前一样）：

```text
⑧ 引文合法性      : 引文 19 处（9 题带引文），非法 11 处｜题号 [3, 13, 15, 17, 18, 19]
   交付口径        : policy=strip｜引文 8 处，非法 0 处 ✅
   兜底动作        : 剪除 11 处
   ✅ 判定未因兜底改变任何一题（9/21 设计约束：剪除不重写成拒答，故不会机械造分）
```

#### 3.5.5 三条设计约束（**报告里就靠这三条体现"懂实验"**）

| # | 约束 | 为什么 |
| --- | --- | --- |
| ① | 剪除非法引用后**不把答案改写成"资料中没有提到"** | 那会把 out 题的 FN **机械地变成 TP**，F1 立刻好看——但那是**用后处理造分**，不是系统变好。所以"剪完没内容"只记进 `citation_only_bad` 当诊断。 |
| ② | 因兜底**翻转的判定必须单列**（`guard_changed`） | 交付口径的 `correct` 与 `correct_raw` 不同时，终端会打 ⚠ 并列出题号；报告里也单开一行。**这类翻转绝不能计入"生成层变好"**（`record` 轮里这个列表天然为空）。 |
| ③ | **没有非法引用时，函数原样返回**（连空白都不动） | 否则"清理"本身（`\s{2,}` 压缩、`strip()`）会悄悄改动**所有**答案，变成一个隐性变量。 |

> **`retry` 的额外免责声明**（写进报告）：重答指令里告诉了模型"你引的编号不存在"，这**等于暗示它"资料可能不够"**。所以 `retry` 若带来 FN→TP，**不能全记成"模型自己变好了"**——其中一部分是这句提示诱导的。要严格分离这个效应，得再加一档"原样重答、不给提示"做对照（本轮不做，**只写进局限性**）。

#### 3.5.6 怎么跑（两轮，约 5 分钟）

```powershell
# R6：只做剪除（不调模型、零随机性）——与 R3 只差 citation-policy
python eval_v2.py --top-k 8 --runs 3 --template v1 --citation-policy strip `
  --out result_cite_strip_runs3 `
  --exp-id R6 --date 9/21 --note "O2-G3 引文强制白名单：非法编号剪除（只改交付，不改生成）" --append-log

# R6b：带提示重答一次（会多花时间；想省时间就只跑 R6）
python eval_v2.py --top-k 8 --runs 3 --template v1 --citation-policy retry `
  --out result_cite_retry_runs3 `
  --exp-id R6b --date 9/21 --note "O2-G3 retry：点名非法编号后重答一次（含提示诱导效应）" --append-log
```

> `--date` 请填**你的实际执行日**（前几轮已经三次把占位日期 `9/23` 抄进日志了）。

**开跑前 30 秒自检**（零成本，先确认新旋钮真的接上了）：

```powershell
python eval_v2.py --dry-run --limit 2 --citation-policy strip --out _smoke_cite
# 期望看到：① 抬头多一行「引文兜底（O2-G3）：strip（**强制白名单**：非法编号会被剪除）」
#          ② 每题多一行「兜底  : policy=strip｜剪除 0 处｜交付口径非法 0 处 ✅」
#          ③ 汇总区 ⑧ 下面多出「交付口径 / 兜底动作」两行，且有一句
#             「✅ 判定未因兜底改变任何一题」
Remove-Item -Recurse -Force _smoke_cite     # 验完就删，别把冒烟目录当结果交上去
```

**看什么**（三件事，按重要性排）：

1. **交付口径的"非法"是不是 0** —— 这是这个功能**唯一保证**的东西：给出去的答案里不会再有编造的编号。
2. **`判定未因兜底改变任何一题`** —— 应该成立（约束①）。若真翻了一题，必须逐题去看 `eval_results.json` 里的 `answer_raw` vs `answer`，在报告里点名。
3. **`剪除 N 处` 与"总正确率一个点没动"** —— 这个组合恰好是**负结果中最有价值的一句话**：

> **"引文兜底能让交付答案零非法引用，但**救不回** F1。"** 因为 Q18 那句 `[1][36][37][41][42][43][84] = 7。` 剪完还是 `[84] = 7。`——**它本来就不是拒答句**，剪引用改变不了"该拒没拒"这个事实。所以结论指向下一步：**要治 out 题，缺的不是"引文干净"，而是一个能识别"答非所问"的判据**（F3 数据侧 / 更大模型去做）。

> 它为什么仍然"区分度强"：面试官问"你怎么防模型编引用"，你能答出一套**可执行的校验机制**（而不是"我在提示词里写了不要编"），并且能讲清四件事——**① 校验器为什么要兼容两种引用格式；② 为什么它必须放在判分之前；③ 为什么剪完不能顺手改成拒答；④ 为什么它救不回 F1，以及下一步该往哪走**。

#### 3.5.7 实测结果（9/21 已跑完，数字逐个抄自终端）

**主表**（与 §3.4 同格式；括号里是终端同源的 TP/FP/FN/TN）：

| 轮次        | 兜底策略              | 检索 strict  | 检索 loose   | 生成（in）   | 防幻觉（out） | 防幻觉 F1                      | 总正确率               | 一致率   |
| --------- | ----------------- | ---------- | ---------- | -------- | -------- | --------------------------- | ------------------ | ----- |
| R3（基线）    | `record`（无兜底）     | 4/10 40.0% | 5/10 50.0% | 4/10     | **7/10** | **0.824（TP7 FP0 FN3 TN10）** | 11/20 = **55.0%**  | 0.950 |
| R6        | `strip`（剪除）        | 4/10 40.0% | 5/10 50.0% | 5/10     | **7/10** | **0.824（TP7 FP0 FN3 TN10）** | 12/20 = 60.0%      | 0.933 |
| R6b       | `retry`（点名后重答）     | 4/10 40.0% | 5/10 50.0% | **3/10** | **7/10** | **0.737（TP7 FP2 FN3 TN8）**  | 10/20 = **50.0%**  | 0.883 |

> **✅ 检索自检通过**：三轮 `strict/loose` 一字不差（4/10、5/10）→ 兜底只碰交付层，检索层没被碰。

**⑧ 引文合法性明细**：

| 轮次  | 策略       | 原始口径         | 交付口径（数字编号）    | 剪除（runs 累计） | 重答题号                | 抓不到的「伪引文」             | 判定翻转        | 拒答翻转（**改 F1**）        |
| --- | -------- | ------------ | ------------- | ---------- | ------------------- | --------------------- | ----------- | --------------------- |
| R6  | `strip`  | 24 处／非法 5 处  | 19 处／**0 处** ✅ | 13 处       | 0                   | **23 处**（Q3 22、Q8 1）  | 无 ✅        | **无** ✅               |
| R6b | `retry`  | 25 处／非法 7 处  | 14 处／**0 处** ✅ | 8 处        | 7 题 `[1,3,4,5,8,14,18]` | 1 处（Q8）               | Q3、Q5 ✅→❌  | **Q1、Q3（False→True）** |

**三条读数（预登记命中一半、落空一半，都要如实写）：**

1. **命中**：R6 的 F1 与 R3 **逐格相同**（TP7 FP0 FN3 TN10）→ "引文兜底不动防幻觉指标"成立。
2. **落空**：总正确率 55% → 60%（生成 4/10 → 5/10）。**但那不是兜底的功劳**——R6 是**重新生成**的（T=0.2 重掷），Q8 换了答案就命中了"接近"（R3 `…更加忠实参考运动且成功率高：[36][76]。` → R6 `…信度评分接近参考运动; [资料§36]; …`）。**证据链**：轮内 raw vs delivered 逐题一致（判定未翻转、拒答未翻转）说明兜底对答案**零改动**，而 out 侧四格又完全没变 → 这 +1 题只能归给**生成的随机性**。
   > ⭐ **本轮最重要的方法论结论**：**一旦引入会改答案的后处理，跨轮比较在生成层就失效了**（两次运行本来就会掷出不同答案）。判断"兜底有没有副作用"只能靠**轮内 `correct_raw`/`refused_raw` vs 交付**的对照——这正是代码里存"原始口径"的意义；只存交付答案的话，这轮就说不清了。
3. **retry 是负结果，且根因又一次锁在一句提示语上**：F1 0.824 → **0.737**（FP 0→2）、总 55% → **50%**。
   - **Q3**：重答 3 次 → 交付 `资料中没有提及。`（**本来 ✅ 的题被改成拒答**）→ FP
   - **Q1**：重答 2 次 → 交付 `SMPL: A skinnied multi-person linear modell [资料中没有提及]`（半截英文 + 拒答词）→ FP
   - **Q5**：重答 3 次 → 交付**空字符串**（3 次重答全被剪空，`剪完已无内容 2`）→ ✅ 翻 ❌
   - **Q14**：重答后引文合法了，但内容**还是幻觉**（`通过两阶段优化寻找最优解 [76] §`）→ **"引文合法" ≠ "内容正确"**
   - **根因**：`RETRY_INSTRUCTION` 第 2 条给了出口"资料不足就**只输出**「资料中没有提到」"→ 3B **滥用出口**。**这与 R5 的规则②完全同型**：R5 证明"提示词里给出口 → 模型把出口当答案"，R6b 证明"**重答提示词里给出口 → 同样被滥用**"。→ 提示词这条线到此为止，**不采用 `retry`**。

**两条新暴露的读数坑（必须写进报告口径，代码 9/21 已补）：**

- **坑A（F1 的记账口径）**：**F1 的 FP 是按 `refused` 算的，只比 `correct` 会漏报**。R6b 的 Q1 就是"`correct` 两次都 ❌、只有 `refused` 从 False 变 True"→ 它让 FP +1，却**不进**"判定随兜底翻转"名单。9/21 已修：落盘加 `refused_raw`、汇总加 `guard_refused_changed`，⑧ 区把"判定翻转"和"拒答翻转"**分开打**（现在能正确报出 Q1、Q3）。
- **坑B（校验器的边界）**：**"数字编号非法 0 处" ≠ "答案干净"**。R6 的交付答案里仍有 **23 处非数字「伪引文」**——Q3 一口气写了 `[资料©] [资料®] [资料°] [資料½] [resourceì] [resourceright] …`（共 22 处），Q8 写 `[资料第Ⅱ节]`。**模型在用"引文的样子"伪装内容**，而 `CITE_RE` 只认数字 → **一个都抓不到**。9/21 已加 `count_pseudo_citations()` 计数（并**排除** `[资料中没有提及]` 这种"拒答残句被方括号包起来"的误报），报告里单列一行。**这句必须写进报告，否则"0 处"会被读成"答案干净"。**
- 另注（单位不矛盾）：`剪除 13 处` 是 **3 次生成累计**，而 `非法 5 处` 只算**代表答案**那一次 → 前者天然是后者的 1~3 倍。⑧ 输出里已标注这层区别。

**结论怎么落进报告（第 6 步）：**

- **表二加一行**：`O2-G3 引文兜底（strip）| R3 → R6 | 检索不变 | 总 55%→60%（**归因于生成重掷，不是兜底**）| 交付口径非法 5→0 处 | 结论：**交付质量保险，不是效果提升**`
- **表三（失败尝试）加一行**：`O2-G3 retry | F1 0.824→0.737（FP 0→2）、总 55%→50% | 归因：重答提示词的"出口条款"被 3B 滥用（与 R5 规则②同型）`
- **一句话总结**：

> **"引文白名单能让交付答案不出现编造的编号，但它既救不回 F1（strip 逐格不变），也可能反噬（retry 把该答的题改成拒答、甚至交空答案）。它的正确位置是『交付前的保险』，不是『提升指标的手段』。"**

#### 3.5.8 把旧结果补齐：`backfill_citation_fields.py`（已跑完，可复用的"考古"套路）

**问题**：`R6` / `R6b` 跑在 **9/21 早版脚本**上，那时还没有 `refused_raw` / `citation_pseudo_n` 两个字段 → 已落盘的 `eval_results.json` 里缺这两项，**报告就少两行留痕**（"拒答状态随兜底翻转"、"抓不到的伪引文"）。

**为什么不重跑**：重跑会**重掷生成**（T=0.2）→ F1 / 总正确率全变 → **已定稿的结论作废**。这两个字段**只影响留痕行，不影响任何成绩指标**，所以正确做法是**离线补算**。

```powershell
# 先看会补什么（不写盘）
python backfill_citation_fields.py --dry-run
# 真补：补字段 + 重生成 评测表.md（原 json 自动备份）
python backfill_citation_fields.py
```

**三道安全设计**（这是"能放心改动实验产物"的前提，面试可讲）：

1. **断言**：补算前后 `summarize()` 的**全部成绩指标逐格比对**，不一致就**中止且不写盘** —— 保证"补算没动成绩"。实测输出 `F1=0.824（TP7 FP0 FN3 TN10）` / `F1=0.737（TP7 FP2 FN3 TN8）` 与终端完全一致 ✅
2. **备份**：原 `eval_results.json` → `eval_results.pre_backfill.json`
3. **留痕**：重生成的 `评测表.md` **末尾自动附一段"补算说明"**（说明用什么代码、依据什么重算、数字是否变），避免"这张表和首跑对不上"的悬案

**补算后新露出的信息**：R6b 的 `拒答翻转 [Q1, Q3]`（→ 解释了 FP 为什么 0→2）、R6 的 `伪引文 23 处`（→ 解释了"非法 0 处"的真实含义）。

> **套路本身可复用**：以后任何"脚本升级了、但旧结果不想/不能重跑"的场合，都按这三步走 —— **补算 + 断言 + 留痕**。
> ⚠ 但**必须先确认"补的是留痕字段、不是成绩字段"**；如果新字段会进成绩（例如换了判分口径），那**只能重跑**，补算等于篡改历史。

### ✅ 第 3 步验收标准

- [x] **对账过 R3 的** `run_log.txt`，能说出靶子是 Q11/Q14/Q18（且 Q18 是"先编后拒答"的形态）
- [x] `v2 / v2a / v2b` 三档都跑完（至少 v2 一档），对比表填齐、`检索 strict` 四轮一致
- [x] 能把结论写成"**检索层 vs 生成层分开报**"的一段话（含"精度-召回权衡"的判断）
- [x] （加分）引文兜底跑通：`--citation-policy strip` 至少一轮，**交付口径非法 = 0 处**且**判定/拒答都未翻转**，`实验日志.md` 有一行 R6
- [x] 能说清**坑A/坑B**（F1 按 `refused` 记账；"非法 0 处"≠"答案干净"）

---

## 📊 第 4 步：F2 真实行业数据采集（60~90 分钟）⭐⭐ HR 硬要求

### 4.1 合规红线（先读这一段，再动手）


| 红线          | 具体做法                                                 |
| ----------- | ---------------------------------------------------- |
| **只采公开资料**  | 上市公司年报/公告（交易所公开披露）、公司官网产品页/技术博客、公开行业标准与白皮书、公开研报/论文   |
| **注明来源**    | 每条 QA 的 `source` 必须有：**机构 + 标题 + URL + 日期 + 原文摘录**   |
| **不转载大段原文** | `quote` 只留**回答问题的那一两句**（≤120 字），不做整段搬运               |
| **不外传**     | 数据只用于本项目 SFT 对照实验；**发布包不上传原始数据文件**（发布包里只有展示文章与 Demo） |
| **不确定就换来源** | 遇到"需付费/需登录/来源不明"的资料，**直接换一个**，不要"先用着"                |


> 一句心法：**"真实"不是"我从网上抄的"，而是"我能指着公开来源说这句话是它写的"。**

### 4.2 数据规格（照这个写，校验器就认）

```json
{
  "instruction": "2024 年优必选 Walker S 主要应用在哪些场景？",
  "input": "",
  "output": "依据公司公开披露，Walker S 主要面向工业制造场景（如汽车厂的质检、搬运等工位）。",
  "source": {
    "org": "优必选（UBTECH）",
    "title": "2024 年年度报告",
    "url": "https://www.ubtrobot.com/...",
    "date": "2025-04",
    "quote": "（回答问题的那一两句原文，≤120 字）"
  },
  "category": "公司经营"
}
```

**五类分布建议**（HR 说"要真实数据"，本质是"要**有信息量**的数据，不是模板套壳）：


| 类目          | 占比建议 | 例子                                   |
| ----------- | ---- | ------------------------------------ |
| **公司经营**    | ~25% | 营收/研发投入/产能/订单（年报、公告）                 |
| **产品技术**    | ~30% | 自由度/负载/续航/传感器方案（官网、产品页、技术博客）         |
| **行业标准与政策** | ~15% | 国标/团标、行业白皮书、政策文件                     |
| **市场与产业链**  | ~15% | 市场规模、上下游、公开研报                        |
| **论文/研报结论** | ~15% | 公开论文的结论段（与 F1 的论文 QA 区分：这里是**行业**视角） |


### 4.3 来源清单（今天照着找，别现搜）


| 类型     | 可用来源（公开可引用）                               | 适合出的题          |
| ------ | ----------------------------------------- | -------------- |
| 年报/公告  | 上交所/深交所/港交所披露的机器人相关上市公司年报、业绩公告            | 营收、研发费用、产能、订单  |
| 公司官网   | 优必选 / 宇树 / 傅利叶 / 智元 / 云深处 等官网的产品页、新闻、技术博客 | 产品参数、技术路线、发布节奏 |
| 行业标准   | 国标（GB/T）、团标、IEEE/ISO 公开摘要页                | 术语定义、测试方法、安全要求 |
| 白皮书/研报 | 行业协会白皮书、券商公开摘要、咨询机构公开页面                   | 市场规模、渗透率、产业链   |
| 公开论文   | arXiv / 期刊公开页（**方法/结论段**）                 | 技术指标、对比结论      |


> **采集节奏建议**：把 100 条拆成 **4 批 × 25 条**，每批 **15~20 分钟**，做完一批就跑一次校验器（早发现格式错误）。**别一口气写 100 条最后一起校验**——格式错要全改一遍（day18 抄 90 条 quote 踩了三个 PDF 坑，就是这个教训）。

### 4.3.1 本轮采集记录（**111 条，已过 `--min 100`**）

> 已写入 `sft_data_real.json`。**分两批采**：第 1 批 80 条（5 类全覆盖），第 2 批 31 条（补「公司经营」「产品技术」的短板）。
> 最终 `python build_real_qa.py --min 100` → **exit 0**、**0 警告**、**0 重复**、**0 泄漏**。

**类目分布**（校验器实跑输出）——**和"建议占比"对照着看，能看到第 2 批在补哪两块**：

| 类目          | 第 1 批 | 第 2 批 | **合计**  | 建议占比  | 第1批占比 → 最终占比   |
| ----------- | ----- | ----- | ------- | ----- | -------------- |
| 市场与产业链      | 32    | 0     | **32**  | ~15%  | 40.0% → **28.8%** |
| 产品技术        | 18    | +9    | **27**  | ~30%  | 22.5% → **24.3%** |
| 公司经营        | 7     | +17   | **24**  | ~25%  | 8.8% → **21.6%**  |
| 行业标准与政策     | 9     | +5    | **14**  | ~15%  | 11.3% → **12.6%** |
| 论文/研报结论     | 14    | 0     | **14**  | ~15%  | 17.5% → **12.6%** |
| **合计**      | **80** | **+31** | **111** |       |                |

> **读法**：第 2 批**刻意只补「公司经营」和「产品技术」**，因为这两块离建议占比最远（8.8% / 22.5%）。补完后最大类目从 40% 降到 **28.8%**，「公司经营」从 8.8% 抬到 **21.6%**。
> 但**「市场与产业链」仍然超标（28.8% vs 建议 15%）**——它一条没减，只是分母变大了。**如实标注，没有为了好看去删数据**。

**机构 / 域名集中度**：**26 个机构、26 个域名；最大机构占比 10.8%**（宇树科技 12 条 / arXiv 12 条）——**远低于 50% 的告警线**。

**第 1 批实际用到的来源**（都在 `source.url` 里可点开复核）：

| 来源                                      | 类型      | 用在哪          |
| --------------------------------------- | ------- | ------------ |
| 港交所 `hkexnews.hk` 年报 / 业绩公告 PDF          | 一手披露    | 公司经营（营收/研发/毛利/亏损/业务构成） |
| 宇树官网 `unitree.com/cn/g1/` + 文档中心        | 官网页     | 产品技术（自由度/尺寸/扭矩/负载/电池/传感器/售价） |
| `openstd.samr.gov.cn`（国家标准全文公开系统）       | 官方标准页   | GB/T 12643-2025 的五要素 |
| `ndls.org.cn`（国家数字标准馆）                  | 官方标准馆   | 《人形机器人技术要求 第1部分：总则》计划页 |
| 新华社 / 人民日报 / 新华网                        | 央媒      | 市场规模、专利、政策、应用案例 |
| 中国电子学会 `cie.org.cn`                     | 行业协会    | 2024 WRC 现场数据、量产进度 |
| TrendForce `trendforce.cn`              | 咨询机构一手  | 2026 出货量预测、美中日路线差异 |
| 财联社 `cls.cn`（转载上海证券报 / 国泰君安）            | 财经媒体    | GGII 预测、成本占比、产业链企业 |
| 观点网 `guandian.cn`                        | 财经媒体    | GGII 2025 蓝皮书预测 |
| arXiv（`arxiv.org/html/...` 与 DOI）        | 公开论文    | 论文/研报结论（BFM / sim-to-real / 15 分钟训练） |

**第 2 批新增的来源**（这批的重点是"把上游零部件公司的一手财报搬进来"）：

| 来源                                            | 类型       | 用在哪                                    |
| --------------------------------------------- | -------- | -------------------------------------- |
| `hkexnews.hk` 越疆科技年度业绩公告 PDF                   | 港交所公告    | 公司经营（收入/亏损/经调整亏损/毛利率/研发开支）             |
| `notice.10jqka.com.cn` 绿的谐波 2024 年报 PDF（公告镜像）   | A 股年报    | 公司经营（营收净利、人形机器人轻量化、丝杠小批量、国标起草）         |
| `static.cninfo.com.cn`（巨潮资讯）三花智控 2024 年报 PDF   | A 股年报（法定披露） | 公司经营（总营收/营业利润/净利、分业务收入、仿生机器人业务段）       |
| 三花智控 GDR 申请回复 PDF（`file.finance.sina.com.cn`）  | 监管问询回复   | 公司经营（机器人项目**尚无定点客户、无量产订单或收入**）         |
| `szzhaowei.net` 兆威机电 2024 年报 PDF（**公司官网**）      | 公司官网年报   | 公司经营（灵巧手发布、17 主动自由度）                   |
| 证券时报网 `stcn.com`                               | 财经媒体     | 公司经营（兆威营收净利同比、研发投入占比）                  |
| 工信部 `miit.gov.cn`（通知页 + 解读页）                   | **部委官网** | 行业标准与政策（193 号文五要素、2025/2027 目标、5 方面任务） |
| 傅利叶文档中心 `support.fftai.com`                    | 官方文档站    | 产品技术（GR-3 全身参数表）                       |
| 智元官网 `agibot.com.cn`                           | 官网页      | 产品技术（远征 A2 参数表、吉尼斯纪录）                  |

**采集踩到并已处理的 6 个坑**（写进这里，下次直接避开）：

| # | 坑                          | 现象                                                                       | 处理                                                                  |
| - | -------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| 1 | **PDF 抽取乱码**               | 券商研报 PDF 抽出的中文变成 `成本占比分别为 32%�?2%`，汉字丢失成 `�?`                                | **整份弃用**。quote 要逐字，乱码文本无法证明原文——换同题的可读来源                            |
| 2 | **聚合站 / 内容农场**             | 搜索排名靠前的 `lankeji.com`、`aiot.csdn.net`、`sgpjbg.com` 都不是原始出处，是转抄/报告下载站           | **一律不引**，回到原始出处。判定标准：页面上是否写着"来源：XXX"（写了就去找 XXX 那一版）                   |
| 3 | **英文 quote 的 120 字上限**     | `len()` 按**字符**算，英文一句话就 130~380，校验器报 6 条超限                                 | **收成"能支撑答案的最短原句"**（截取、不改写），必要时把一问拆成一问一事；拆完 0 警告                     |
| 4 | **同一指标跨页数字不同，不能合并**        | 宇树 G1 官网现价 `¥8.5万元`，而 2024 年发布时报道是 `9.9万元起`——**是两个时点**，不是矛盾               | **各按各的来源单独出一条**，`source` 不混；output 里写清"发布时/官网现行"                     |
| 5 | **PDF 能把表格抽成"没有分隔符的一长串"**  | 绿的谐波年报抽出来是 `营业收入387,411,303.84356,165,776.908.77`，三个数字黏在一起，**人眼无法读、也没法当 quote** | **数值题改引正文句**（年报正文有"报告期内，公司实现营业收入…"），**不直接拷表格行**；同比增速这类只在表里的，改引权威媒体转述       |
| 6 | **港交所公告是繁体 + "排版空格"**      | 抽出来是 `較2023年 的43.5%增 加3.1個 百 分 點`（原文排版插了空格）                             | **去掉排版空格、保留繁体原字**（`較2023年的43.5%增加3.1個百分點`），**不改字、不转简体**——转简体会让"逐字"失效 |

> 心法补充：**"找不到干净的原始出处"≠"换个说法写上去"**。第 1 批为此直接砍掉 2 个来源（乱码研报、聚合站），条数从"看起来能凑 100"降到"扎实的 80"。第 2 批继续守这条线：**宁可换来源、宁可改题，也不把读不通的表格串当 quote**。

> 再补一条**数据诚实的范例**：三花智控这条，年报里写着"仿生机器人业务…构筑工艺和专利技术护城河"（听着很乐观），但**同一家公司的监管问询回复里明确写着"尚未有相关定点客户，亦尚未形成量产订单或收入"**。**两条都收进来了（id=92 / id=93）**——一份真实数据集应该同时保留"公司叙事"和"监管口径"，而不是只留好听的。

### 4.4 `build_real_qa.py`（**已建**：F2 校验器，~200 行，纯标准库）

> **文件已在** `第四周\day19\build_real_qa.py`（随教程一起生成，已实测两条路径）。**核心代码与设计说明如下**——你可以直接用它，也可以对照着读一遍（`.4.5` 的校验命令就是跑它）。

**和 F1 的差别**：F1 的 `build_paper_qa.py` 能把 `quote` **逐字回查**到 `chunks.json`（因为论文就在库里）；**行业资料不在库里**，所以改成：


| 检查项              | 说明                                                                      |
| ---------------- | ----------------------------------------------------------------------- |
| ① 条数门槛           | `--min`（默认 100）                                                         |
| ② 必填字段           | `instruction` / `output` / `source`                                     |
| ③ `source` 五要素齐全 | `org` / `title` / `url` / `date` / `quote`，且 `quote` 非空、`url` 以 http 开头 |
| ④ 去重             | `instruction` 归一化去重（防止"换个说法问同一个东西"凑条数）                                  |
| ⑤ 数据泄漏           | 与 `eval_questions.json` 逐题比相似度（阈值 0.70）**和 F1 同一把尺**                    |
| ⑥ 分布统计           | 五类分布 + 机构分布（**防止"一家公司占了 60 条"**）                                        |


```python
# -*- coding: utf-8 -*-
"""
build_real_qa.py —— F2 真实行业 QA 校验器（Day19）

与 day18 的 build_paper_qa.py 的分工：
    build_paper_qa.py ：论文 QA（quote 能逐字回查到 chunks.json）
    build_real_qa.py  ：行业 QA（来源在外部 → 校验"来源五要素 + 去重 + 泄漏 + 分布"）

用法：
    python build_real_qa.py --min 100 --show-categories
    python build_real_qa.py --example            # 打印一条样例（照格式写）
    python build_real_qa.py --allow-leak         # 临时调试：撞题只警告不报错
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_QA = os.path.join(BASE_DIR, "sft_data_real.json")
EVAL_JSON = os.path.join(REPO_ROOT, "第三周", "day13", "eval_questions.json")
LEAK_THRESHOLD = 0.70
SOURCE_KEYS = ["org", "title", "url", "date", "quote"]
CATEGORIES = ["公司经营", "产品技术", "行业标准与政策", "市场与产业链", "论文/研报结论"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(s):
    """归一化：去空白 + 去标点 + 转小写（中文标点也去掉）。"""
    s = str(s or "").lower()
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(r"[，。、；：！？“”‘’（）《》【】,.;:!?\"'()\[\]<>·\-—_/\\|]+", "", s)
    return s


def is_blank(v):
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def similarity(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def validate(records, args, eval_questions):
    errors, warnings = [], []
    stats = {"dup": 0, "leaks": 0}

    if len(records) < args.min:
        errors.append(f"条数不足：实到 {len(records)} 条，门槛 {args.min} 条（先补数据，别硬凑）")

    seen, orgs, cats = {}, Counter(), Counter()
    for i, r in enumerate(records, start=1):
        tag = f"第 {i} 条（id={r.get('id', '-')}）"

        for k in ("instruction", "output", "source"):
            if k not in r or is_blank(r[k]):
                errors.append(f"{tag}：缺必填字段 {k}")

        src = r.get("source") or {}
        if isinstance(src, dict):
            for k in SOURCE_KEYS:
                if k not in src or is_blank(src[k]):
                    errors.append(f"{tag}：source 缺 {k}")
            url = str(src.get("url", ""))
            if url and not url.lower().startswith("http"):
                errors.append(f"{tag}：source.url 不是合法链接（{url[:40]}）")
            if len(str(src.get("quote", ""))) > 120:
                warnings.append(f"{tag}：source.quote 超过 120 字（可能转载了整段）")
            if src.get("org"):
                orgs[src["org"]] += 1
        elif src:
            errors.append(f"{tag}：source 必须是对象（含 {'/'.join(SOURCE_KEYS)}）")

        key = norm(r.get("instruction"))
        if key in seen:
            stats["dup"] += 1
            errors.append(f"{tag}：instruction 与第 {seen[key]} 条重复（换个说法问同一件事也要算重复）")
        else:
            seen[key] = i

        cats[r.get("category", "未分类")] += 1

        for q in eval_questions:
            s = similarity(r.get("instruction", ""), q.get("question", ""))
            if s >= LEAK_THRESHOLD:
                stats["leaks"] += 1
                msg = f"{tag}：与评测题 Q{q['id']} 相似度 {s:.2f}（≥{LEAK_THRESHOLD}）= 数据泄漏"
                (warnings if args.allow_leak else errors).append(msg)
                break

    if orgs:
        top_org, top_n = orgs.most_common(1)[0]
        if len(records) and top_n / len(records) > 0.5:
            warnings.append(f"机构分布过于集中：{top_org} 占 {top_n}/{len(records)}（>50%），真实性存疑")

    return errors, warnings, stats, cats, orgs


def print_report(records, errors, warnings, stats, cats, orgs, args):
    print("=" * 68)
    print(f"F2 真实行业 QA 校验 ｜ 文件：{os.path.basename(args.json)}")
    print("=" * 68)
    print(f"① 条数          ：{len(records)} / 门槛 {args.min}")
    print(f"② 必填字段      ：instruction / output / source")
    print(f"③ source 五要素 ：{' / '.join(SOURCE_KEYS)}（url 必须 http 开头，quote ≤120 字）")
    print(f"④ 重复 instruction：{stats['dup']} 条")
    print(f"⑤ 数据泄漏      ：{stats['leaks']} 条（阈值 {LEAK_THRESHOLD}）")
    if args.show_categories:
        print(f"⑥ 类目分布      ：" + " / ".join(f"{k} {v}" for k, v in cats.most_common()))
        print(f"   机构 Top5    ：" + " / ".join(f"{k} {v}" for k, v in orgs.most_common(5)))
    print("-" * 68)
    if warnings:
        print(f"⚠ 警告 {len(warnings)} 条：")
        for w in warnings[:10]:
            print("  -", w)
    if errors:
        print(f"❌ 错误 {len(errors)} 条：")
        for e in errors[:20]:
            print("  -", e)
        print(f"\n[未通过] 共 {len(errors)} 个硬错误 —— 修完再来。")
        return 1
    print(f"✅ [通过] {len(records)} 条真实行业 QA，来源可追溯、无重复、无泄漏。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="F2 真实行业 QA 校验器（Day19）")
    ap.add_argument("--json", default=DEFAULT_QA, help="数据文件（默认同目录 sft_data_real.json）")
    ap.add_argument("--eval-questions", default=EVAL_JSON, help="20 条评测题（泄漏检查用）")
    ap.add_argument("--min", type=int, default=100, help="条数门槛（计划要求 ≥100）")
    ap.add_argument("--allow-leak", action="store_true", help="撞题只警告不报错（临时调试用）")
    ap.add_argument("--show-categories", action="store_true", help="打印类目/机构分布")
    ap.add_argument("--example", action="store_true", help="打印一条样例格式")
    args = ap.parse_args()

    if args.example:
        print(json.dumps({
            "instruction": "（问题，来自公开资料能回答的内容）",
            "input": "（可选上下文；行业 QA 一般留空）",
            "output": "（忠实来源的答案，不要加自己推断的数字）",
            "source": {"org": "（机构）", "title": "（标题）", "url": "https://...",
                       "date": "（年-月）", "quote": "（回答问题的那一两句原文，≤120 字）"},
            "category": f"（{'/'.join(CATEGORIES)} 之一）",
        }, ensure_ascii=False, indent=2))
        return 0

    if not os.path.exists(args.json):
        print(f"[错误] 找不到 {args.json} —— 先从种子文件复制起步：copy sft_data_real.seed.json sft_data_real.json")
        return 1

    records = load_json(args.json)
    eval_questions = load_json(args.eval_questions)
    if isinstance(eval_questions, dict):          # eval_questions.json = {"meta":..., "questions":[...]}
        eval_questions = eval_questions.get("questions", [])
    if not eval_questions:
        print(f"⚠ 评测题为空（{args.eval_questions}）→ 跳过泄漏检查（正式校验前请确认）")
    errors, warnings, stats, cats, orgs = validate(records, args, eval_questions)
    return print_report(records, errors, warnings, stats, cats, orgs, args)


if __name__ == "__main__":
    sys.exit(main())
```

**种子文件** `sft_data_real.seed.json` **已建**（3 条**格式示例**，已随教程生成、通过校验器）——**它的作用是"格式模板"，不是数据**：里面的 `url` 是 `example.com` 占位、内容都标了「（格式示例·请替换）」，**正式数据必须换成真实公开来源**。

```json
[
  {
    "instruction": "（格式示例·请替换）某公司 2024 年研发投入是多少、同比变化如何？",
    "input": "",
    "output": "（格式示例·请替换为来源里明写的内容，不加自己的推断）",
    "source": {
      "org": "（机构名）",
      "title": "（年报/公告/产品页标题）",
      "url": "https://example.com/xxx",
      "date": "2025-04",
      "quote": "（回答问题的那一两句原文，≤120 字）"
    },
    "category": "公司经营"
  }
]
```

### 4.5 开工与校验（四批节奏）

```powershell
# ① 起步：复制种子 → 正式集，先跑一次看到"条数不足"（这是正常的，别慌）
copy sft_data_real.seed.json sft_data_real.json
python build_real_qa.py --min 100 --show-categories

# ② 每写完一批（25 条）就校验一次
python build_real_qa.py --min 25  --show-categories
# ...补齐到 50 / 75 / 100...  门槛同步提高

# ③ 收尾：按计划门槛校验（≥100）
python build_real_qa.py --min 100 --show-categories
```

**校验通过的标准（写进报告的"可引用"口径）**：

- 条数 ≥100；
- **source 五要素 0 缺失**、`url` 全为 http(s)、`quote` 无超长；
- **重复 0 条**、**泄漏 0 条**；
- 类目五类都有、**机构不过度集中**（单机构 ≤50%）。

> **⚠ 三条如实标注的边界**（写报告时必须带）：
> ① **行业资料不在向量库里** → 无法像 F1 那样"逐字回查 chunk"，只能做到"**来源字段可追溯 + 人工复核**"，这一点必须写明，别冒充成"机器验证过"；
> ② **真实数据和合成数据的比例要如实写**（如"真实 190 + 合成 300"），**不假装全是真实数据**；
> ③ 若某批来源只找到 80 条，**就写 80 条**——**"没跑出来就不填"** 比"凑数"重要得多。

> **⚠ 一个实测踩到的坑（本教程两个脚本已内置兼容）**：`eval_questions.json` **不是 list**，而是 `{"meta": ..., "questions": [...]}`。自己新写脚本时若直接 `for q in load_json(path)`，拿到的会是 key 字符串 → 报 `AttributeError: 'str' object has no attribute 'get'` 或 `TypeError: unhashable type: 'slice'`。**统一按** `data["questions"]` **取**（`day18\build_paper_qa.py` 也是这么写的）。

### ✅ 第 4 步验收标准

- [x] `build_real_qa.py --min 100` **全绿**（或如实报告当前条数与补齐计划）
- [x] `sft_data_real.json` 每条含 `source{org,title,url,date,quote}`；能随机抽 3 条打开 URL 复核（**抽查动作要做**）
- [x] 类目/机构分布合理，能在报告里写出分布数字
- [x] 已如实标注"行业数据无法逐字回查、靠来源字段 + 人工复核"

---

## 🧠 第 5 步：F3 真实 vs 合成对照微调（60~90 分钟）⭐

### 5.1 ⚠ 显存纪律：训练前必须停服务（先做这一步，别抢显存）

**6G 显存一次只能装一个 3B**（服务约占 1.98GB，训练峰值约 4~5GB）。**训练与推理必须串行**：

```powershell
# ① 先停掉模型服务（终端 A 里 Ctrl+C；或按端口 kill）
Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { taskkill /PID $_.OwningProcess /F }
netstat -ano | findstr :8000        # 无输出 = 端口已释放 ✅

# ② 看显存余量（确认没有残留进程占着）
nvidia-smi
```

> **判断"真的停了"看端口，不看** `taskkill` **输出**（day18 的结论：`taskkill` 说"没找到进程"不代表失败，可能早就没了）。**Web 界面（7860）不影响显存，可以留着。**

### 5.2 数据配比方案（先定方案，再训练）—— 实际执行：**A（纯真实 201 条）**

F3 要回答的是：**"真实数据对领域适配有没有用？"** 但真实数据量（论文 90 + 行业 ~100 = ~190）**少于**合成的 300 条，所以配比本身就是个设计决策：


| 方案                   | 训练集                                | 能回答什么                         | 风险                                      |
| -------------------- | ---------------------------------- | ----------------------------- | --------------------------------------- |
| A. 只用真实              | 论文 90 + 行业 100 = **190**           | "纯真实数据能不能训出可用 adapter"        | 数据量小，可能欠拟合/不稳定                          |
| **B. 真实 + 合成混合（推荐）** | 真实 190 + 合成 300 = **490**（真实占 39%） | "真实数据加进来有没有增益"（**对照对象仍是 v1**） | 两次变量（加了数据 + 换了配比），**结论要说清是"混合 vs 纯合成"** |
| C. 真实过采样到 300        | 真实 190 → 重复采样到 300                 | 数据量对齐、只改"数据来源"这一个变量 ⭐         | 重复样本会让模型背诵                              |


> **今天的推荐 = C（最干净）+ B（最能说"真实有用"）二选一**；**时间只够一个就做 C**——因为 C 是**单变量对照**（同为 300 条，只有"来自哪"不同），归因最干净、最好写报告。**B 更好看但归因复杂**（要说明"这里同时改了两件事"）。不要B，就要C。

> **🧾 实测记录（9/21 收尾对账）**：本轮**实际跑的是 A（纯真实 201 条）**，不是 C。
> 原因：下面那条 one-liner 只做"论文 90 + 行业 111 合并"，**没有任何过采样代码**——即"把真实样本重复到 300"这一步**没有实现**，所以数据量就是 201。
> **这不算跑错**：A 同样能回答"纯真实数据能不能训出可用 adapter"，而且**结论更保守**（数据量更小、对结果更不利，负结果的说服力反而更强）。**报告里的"真实 201 条"是对的**。
> **要补的是文档口径**：§5.5 的结论必须写成「**真实 201 vs 合成 300：来源与数量两个变量同时变了**」——这与 A 完全吻合。
> **真正的未完成项 = B/C 的拆变量轮**（已被 §5.5 明确"留 D5"）：
> - **C（过采样到 300）**：单变量对照，"只改来源"，需**新写几行过采样代码**（本教程没提供，D5 要自己补）；
> - **B（真实 201 + 合成补到 300 = 501）**：才需要 `sft_data_v2_mix.json`（**今天没生成，表 0 里那行是"计划"不是"产出"**）。

**生成训练集（Python one-liner，注意编码）**：

```powershell
python -c "import json,io; a=json.load(io.open(r'..\day18\sft_data_paper.json',encoding='utf-8')); b=json.load(io.open('sft_data_real.json',encoding='utf-8')); out=[{'instruction':x['instruction'],'input':x.get('input',''),'output':x['output'],'origin':'real'} for x in a+b]; json.dump(out,io.open('sft_data_v2_real.json','w',encoding='utf-8'),ensure_ascii=False,indent=2); print('真实数据合计', len(out))"
```

> 训练脚本只认三字段（`instruction` / `input` / `output`）——`source` **/** `category` **/** `origin` **是给我们自己审计用的，训练时多余的字段会被** `load_dataset` **一起读进来但不影响训练**（若报错就去掉，用上面的 one-liner 只留三字段，`origin` 保留也无妨）。

### 5.3 训练 adapter v2（只改 3 个常量）

> **✅ 已改好**：`第四周\day19\train_lora_v2.py` 已经替你建好 = `第三周\day12\train_lora.py` 的**逐字复制件**，**只有这 3 行**和原件不同（其余一字未动，包括所有超参）：

```python
DATA_PATH = "sft_data_v2_real.json"      # 原：sft_data.json（300 条合成）
OUTPUT_ADAPTER = "lora_adapter_v2"       # 原：lora_adapter（v1 保留不动）
LOG_DIR = "./train_logs_v2"              # 原：./train_logs（免得覆盖 v1 日志）
```

**想自己核对一遍**（可选，确认"真的只改了这三行"）：

```powershell
Select-String -Path .\train_lora_v2.py -Pattern "^(DATA_PATH|OUTPUT_ADAPTER|LOG_DIR)"
# 想更严格：与原件逐行比，只应看到上面 3 行 + 头部 docstring 的差异
Compare-Object (Get-Content ..\..\第三周\day12\train_lora.py) (Get-Content .\train_lora_v2.py)
```

> **⚠ 注意相对路径**：`train_lora_v2.py` 里 `DATA_PATH` 是**相对当前工作目录**的名字 → **必须在 `day19\` 目录下运行**（`cd` 到 day19），否则会找不到数据。

**开训**（把 `Tee-Object` 存日志，方便事后回看峰值显存）：

```powershell
python train_lora_v2.py 2>&1 | Tee-Object -FilePath train_v2_console.log
```

**训练期间盯三件事**（10~30 分钟，看数据量）：


| 盯什么      | 正常                         | 异常与处置                                                      |
| -------- | -------------------------- | ---------------------------------------------------------- |
| 显存峰值     | `<6G`（`nvidia-smi` 看）      | 逼近 6G / 报 `os error 1455` → 降 `BATCH_SIZE`（2→1）或 `MAX_LEN` |
| loss 曲线  | 缓慢下降（不要求漂亮）                | 一直不降 → 学习率/数据格式问题（`input` 字段空值处理）                          |
| epoch 进度 | 300 条 × 3 epoch 约 10~30 分钟 | 卡住不动 → 检查是否还在跑（`nvidia-smi` 有进程 = 在跑）                      |


**训完确认产物**：

```powershell
dir lora_adapter_v2
# 期望看到 adapter_config.json / adapter_model.safetensors（约 20~30MB）
```

### 5.4 起 v2 服务（脚本已建好，只核对两处常量）

> **✅ 已改好**：`第四周\day19\local_api_lora_v2.py` 已经替你建好 = `第三周\day13\local_api_lora.py` 的**逐字复制件**，只改了这两行：

```python
ADAPTER_PATH = r"D:\Lan\研究生\技术学习\大模型算法\第四周\day19\lora_adapter_v2"
MODEL_NAME = "Qwen2.5-3B-Instruct-LoRA-v2"     # ⭐ 改成 -v2，防"服务在跑 ≠ 服务正确"
```

**想自己核对**（可选）：

```powershell
Select-String -Path .\local_api_lora_v2.py -Pattern "^(ADAPTER_PATH|MODEL_NAME)"
```

**为什么一定要改** `MODEL_NAME`：`eval_v2.py` 的 `check_service()` 会做**服务身份检查**（对比 `/v1/models` 返回的名字与 profile 要求的名字）。如果 v1/v2 服务都叫同一个名字，**你永远不知道自己刚跑的是哪个 adapter**——这正是第三周就立下的"防跑错模型"纪律。

**改 profile**：`第三周\day15\config.json` 里**已加** `lora_v2` profile（原有两个未动，`default_profile` 仍是 `lora`）：

```json
    "lora_v2": {
      "label": "微调模型 v2（真实/混合数据 adapter，第四周 Day19）",
      "api_url": "http://127.0.0.1:8000/v1/chat/completions",
      "model_name": "Qwen2.5-3B-Instruct-LoRA-v2",
      "top_k": 8,
      "temperature": 0.2,
      "service_note": "需启动 第四周/day19/local_api_lora_v2.py（务必先停 v1 服务：6G 显存一次只能装一个 3B）"
    }
```

> ⚠ `--top-k` 在 `eval_v2.py` 里是**独立参数**（默认 4），**不从 profile 读** → 跑 v2 对照时**照样要显式写 `--top-k 8`**，否则就不是"只换 adapter 一个变量"了。

起 v2 服务（**注意：起服务前先确认训练已经结束、显存已释放**）：

```powershell
# 终端 A
conda activate llm
cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day19"
python -m uvicorn local_api_lora_v2:app --host 127.0.0.1 --port 8000
# 等日志出现「模型已就绪」+「LoRA adapter 已加载：...day19\lora_adapter_v2」
```

> ⚠ **必须核对那一行路径**：`LoRA adapter 已加载：...\day19\lora_adapter_v2`。
> 只看到"模型已就绪"**不算**——若忘了改 `ADAPTER_PATH`，服务照样能起，但加载的是 v1 的 adapter，这一轮对照就全废了（这属于"服务在跑 ≠ 服务正确"）。

### 5.5 同评测集对照（F4 的 v2 版：v1 adapter vs v2 adapter）

```powershell
# 终端 B：v2 adapter 全量跑（同 k=8 / qi=off / runs=3 / 模板 v1）
python eval_v2.py --profile lora_v2 --top-k 8 --runs 3 --out result_lora_v2_runs3 `
  --exp-id F3v2 --date 填你的执行日 --note "F3 真实数据 adapter v2（与 v1 只差训练数据）" --append-log
```

**对照表**（**检索层两列应该完全一致**——因为库、k、查询都没变，**这也是自检信号**）：


| 层   | 指标                                | v1（合成 300 条）            | v2（真实/混合 **201** 条）    | 差异  |
| --- | --------------------------------- | ----------------------- | ---------------- | --- |
| 检索  | strict / loose                    | 4/10 / 5/10（= R3）       | 4/10 / 5/10（**完全相同 ✅**） | **0** |
| 生成  | in 正确率                            | 4/10                    | **5/10**             | +1  |
| 防幻觉 | out 正确率                           | 7/10                    | **2/10**             | **−5** |
| 防幻觉 | F1（回 `run_log.txt` 抄 TP/FP/FN/TN） | 0.824（TP7 FP0 FN3 TN10） | **0.333（TP2 FP0 FN8 TN10）** | **−0.491** |
| 总   | 总正确率                              | 11/20                   | **7/20 = 35%**        | **−4 题** |


> **✅ 自检信号通过**：检索两列**完全相同**（4/10、5/10）→ 换训练数据只落在生成层，符合单变量纪律。
>
> **⚠ 诚实提示（已写进报告 §四）**：**提升没有出现，反而是明显退化**——如实写。
> 但**不能据此说"真实数据不好"**：本轮是「**真实 201 条**」vs「**合成 300 条**」，**来源与数量两个变量同时变了**。
> 要拆开必须再跑一轮「真实 201 + 合成补到 300」的配比版（**留 D5**）。
> **HR 要的是"数据真实"，不是"数据一定带来分数提升"**；硬编一个"提升"才是自毁证据链。


### 5.6 领域问题集对比（可选，10 分钟，比评测题更"像真实使用"）

用 `..\day18\app.py` 或 `ask.py`，对 3~5 个**行业问题**分别问 v1 / v2 两个服务（串行切换），人工看**风格贴合度**（是否像行业专家、是否带出处、是否编数字）：

```powershell
# 先 v1 服务（第三周\day13\local_api_lora.py）→ 记答案
python "..\..\第三周\day15\ask.py" "Walker S 的负载能力大概是多少？"
# 停 v1 → 起 v2 服务 → 再问同一句
```

> 这一步的产物是**定性观察**（截图/摘录），可放报告"附录：双 adapter 定性对比"，**不要拿它当数字成绩**。

### ✅ 第 5 步验收标准

- [x] 训练前**服务已停**、`nvidia-smi` 确认显存已释放；训练全程峰值 **<6G**
- [x] `lora_adapter_v2/` 产出成功（`adapter_config.json` + `adapter_model.safetensors`）
- [x] `local_api_lora_v2.py` 起服务成功，`/v1/models` 返回 `Qwen2.5-3B-Instruct-LoRA-v2`，`eval_v2.py --profile lora_v2` 的**服务身份检查通过**
- [x] v1 vs v2 对照表填齐（**检索两列相同**）；结论如实写（升/平/降都写）
- [x] 训练日志（`train_v2_console.log`）留存；能说出"我这次只改了哪 3 个常量"

---

## 📝 第 6 步：O3《优化成果报告》初稿（40~50 分钟）

### 6.1 报告骨架（四张表 + 一节）

在 `第四周\day19\优化成果报告.md` 建骨架（**✅ 已建好并回填，见该文件**）：

```markdown
# 机器人领域 RAG 文档问答系统 · 优化成果报告

> 一句话：把首版基线（检索 20% / 生成 20% / 防幻觉 60% / F1 0.667 / 总 40%）
> 优化到（检索 __% / 生成 __% / 防幻觉 __% / F1 __ / 总 __%），每一步的贡献可归因；
> 数据从"300 条合成"整改为"真实数据（论文 + 行业，每条可追溯）"。
> 评测口径：同一 20 题固定评测集 ｜ 每题 3 次生成取多数投票（runs=3）｜ 双轨检索判据（strict/loose）

## 一、优化前后对比总表（必填）
## 二、逐步骤贡献表（每步单独重跑，归因）
## 三、失败尝试记录（改了但没用）
## 四、真实数据整改（HR 意见③）
## 五、结论与下一步
## 附：关键日志索引（每行数字回得到 run_log.txt）
```

> **⚠ 只有 `__` 是不许预填的**：那句话开头的"首版基线"四个数字是**已经定稿的历史数字**（day14），照写；**优化后那几个 `__` 必须等 D5 定稿**。
> **本节下面四张表（6.2~6.5）就是报告里那四张表的"填写现场"** —— 已按实跑数字回填，可从教程直接对照 `优化成果报告.md` 复核。


### 6.2 表一：优化前后对比总表


| 轮次              | 改了什么                | 检索 strict      | 检索 loose   | 生成（in）         | 防幻觉（out）       | 防幻觉 F1（TP/FP/FN/TN）       | 总正确率            | 一致率   | 结果目录                             |
| --------------- | ------------------- | -------------- | ---------- | -------------- | -------------- | ------------------------ | --------------- | ----- | -------------------------------- |
| 原始基线（第三周 day14） | ——（runs=1，旧口径）      | 2/10 = **20%** | 3/10 = 30% | 2/10 = **20%** | 6/10 = **60%** | **0.667**（6/2/4/8）       | 8/20 = **40%**  | —     | `第三周\day14\`                     |
| B0r3            | 判据校准 + 投票复跑（**新基线**） | 2/10           | 3/10       | 2/10           | 6/10           | 0.706（6/1/4/9）            | 8/20 = 40%      | 0.950 | `day17\result_lora_k4_runs3`     |
| R1              | + bge instruction 前缀 | 0/10           | 2/10       | 2/10           | 5/10           | 0.625（5/1/5/9）            | 7/20 = 35%      | 0.933 | `day17\result_lora_k4_qi_runs3`  |
| R1R2            | 前缀 + TOP_K 4→8       | 3/10           | 4/10       | 4/10           | 8/10           | **0.842**（8/1/2/9）        | 12/20 = 60%     | 0.967 | `day17\result_lora_k8_qi_runs3`  |
| **R3**（**定稿行**）    | 关前缀、保留 k=8          | **4/10 = 40%** | 5/10 = 50% | **4/10 = 40%** | **7/10 = 70%** | **0.824**（**7/0/3/10**）   | **11/20 = 55%** | 0.950 | `day17\result_lora_k8_runs3`     |
| R4 `term`         | + 查询改写（术语档）         | 4/10           | 6/10       | 3/10           | 6/10           | 0.706（6/**1**/4/9）        | 9/20 = 45%      | 0.983 | `result_rewrite_term_runs3`      |
| R4b `hyde`        | + 查询改写（假设答案档）       | 5/10           | **8/10**   | 6/10           | 4/10           | 0.571（4/0/6/10）           | 10/20 = 50%     | 0.950 | `result_rewrite_hyde_runs3`      |
| R4c `both`        | + 查询改写（术语+假设答案）     | **6/10**       | 7/10       | 6/10           | 5/10           | 0.625（5/**1**/5/9）        | 11/20 = 55%     | 0.933 | `result_rewrite_both_runs3`      |
| R5 `v2`           | 模板 09 v2（弱化拒答+禁编造）  | 4/10           | 5/10       | 5/10           | **3/10**       | **0.462**（3/0/7/10）       | **8/20 = 40%**  | 0.950 | `result_template_v2_runs3`       |
| R5a `v2a`         | 消融：只弱化拒答条件          | 4/10           | 5/10       | 3/10           | 4/10           | 0.571（4/0/6/10）           | **7/20 = 35%**  | 0.950 | `result_template_v2a_runs3`      |
| R5b `v2b`         | 消融：只加禁止编造编号/数字      | 4/10           | 5/10       | 4/10           | 6/10           | 0.706（6/**1**/4/9）        | 10/20 = 50%     | 0.967 | `result_template_v2b_runs3`      |
| R6 `strip`        | 引文强制白名单：**剪切**非法编号  | 4/10           | 5/10       | 5/10           | **7/10**       | **0.824**（**7/0/3/10**）   | **12/20 = 60%** | 0.933 | `result_cite_strip_runs3`        |
| R6b `retry`       | 引文兜底：**点名后重答一次**    | 4/10           | 5/10       | 3/10           | **7/10**       | **0.737**（7/**2**/3/**8**） | 10/20 = 50%     | 0.883 | `result_cite_retry_runs3`        |
| F3v2            | 真实数据 adapter v2      | 4/10           | 5/10       | 5/10           | **2/10**       | **0.333**（2/0/8/10）       | **7/20 = 35%**  | 0.933 | `result_lora_v2_runs3`           |

> **✅ 三次单变量自检全部通过**：R4 / R5 / F3v2 三组的 `检索 strict/loose` 与 R3 **一字不差（4/10、5/10）** → 只动了目标层（改写动检索、模板/数据动生成），没串味。
> **加粗的 `FP`** = **判据假阳性**（答案中间出现拒答词被误记"该答没答"，见 `实验日志.md` R4c / R5b 行），**报数时必须注明**。
> **R6 的 60% 比 R3 高 1 题 ≠ 兜底有效**：R6 是**重新生成**（T=0.2 重掷），且**轮内 raw vs delivered 逐题零改动** → 归**生成随机性**。
> **表一的完整版（含引文合法性补充表）见 `优化成果报告.md`**。



> **数字来源纪律**：每一行的 F1 都要**回该目录的** `run_log.txt` **抄 TP/FP/FN/TN**（day18 的笔误教训）。`__` **全部等跑完再填，绝不预填。**

### 6.3 表二：逐步骤贡献表（面试最看这一张）


| 步骤       | 变量                      | 对照              | 检索 strict 变化        | 总正确率变化              | 结论                       |
| -------- | ----------------------- | --------------- | ------------------- | ------------------- | ------------------------ |
| O0 校准尺子  | 判据补丁 + runs=3           | day14 基线 → B0r3 | 20% → 20%（不变）       | 40% → 40%           | 尺子没改坏；数字从此可引用（一致率 0.950） |
| O1-R2    | TOP_K 4→8               | B0r3 → R3       | **20% → 40%（+2 题）** | 40% → **55%（+3 题）** | **本项目最有效的一步**            |
| O1-R1    | bge instruction 前缀      | B0r3 → R1       | 20% → 0%（**−2 题**）  | 40% → 35%           | **负结果**（机制：排名位移，不是语义变差）  |
| O1-R4 `term` | 查询改写（术语档）               | R3 → R4         | 40% → 40%（平）｜ loose 5→**6** | 55% → **45%（−2 题）**   | **负结果**：v1 提示词的 `term` 档**会替换掉专名**（`PHC/ProtoMotions` → `approaches/techniques`） |
| O1-R4 `hyde` | 查询改写（假设答案档）             | R3 → R4b        | 40% → **50%**｜loose 5→**8** | 55% → 50%｜**out 7→4** | **检索/生成双涨、防幻觉崩**：HyDE 提升"对齐"，**但不区分"资料里到底有没有答案"**                     |
| O1-R4 `both` | 查询改写（两档合并）              | R3 → R4c        | 40% → **60%（三档最好）**        | 55% → 55%（**打平**）     | 总分打平但**结构变了**：in +2 / out −2；**FP=1 是判据假阳性**                  |
| （诊断）     | 人工理想查询                  | plain vs hyde   | 40% → **100%（上界）**  | ——                  | 只证明"瓶颈在查询侧"，**不是成绩**     |
| O2-G2    | 模板 09 v2                | R3 → R5         | 不变（**应相同**）         | 55% → **40%**        | **负结果**：F1 0.824→0.462；②"有依据就照实答"是祸首（见 §3.4） |
| O2-G2 消融 | 只弱化拒答（v2a）/ 只加引用约束（v2b） | R3 → R5a / R5b  | 不变                  | **35% / 50%**       | 三档全线回退；`v2b` 的 FP=1 是判据假阳性                              |
| O2-G3 兜底 | 引文强制白名单（`strip`）          | R3 → R6         | 不变（**应相同**）         | 55% → 60%           | **F1 逐格不变（0.824）**；交付口径非法 5→**0 处**。总正确率 +1 题**归因于生成重掷，不是兜底**（见 §3.5.7） |
| O2-G3 兜底 | 点名非法编号后重答（`retry`）        | R3 → R6b        | 不变                  | 55% → **50%**       | **负结果**：F1 0.824→**0.737**（FP 0→2）→ 重答提示词的"出口条款"被滥用 |
| **F3**   | **真实数据 adapter v2**     | R3 → F3v2       | 不变（✅ 自检通过）            | 55% → **35%**       | **负结果**：F1 0.824→**0.333**（FN 3→8）。**201 条真实数据在此规模下没有换来提升**；⚠ **"来源"与"数量"未拆开**（真实 201 vs 合成 300），留 D5 跑配比版（见 §5.5） |


### 6.4 表三：失败尝试记录（从 `实验日志.md`「三」搬过来，今天补新增的）


| 尝试                     | 数字变化                   | 归因                                                                |
| ---------------------- | ---------------------- | ----------------------------------------------------------------- |
| bge instruction 前缀（R1） | strict 20% → 0%        | 排名位移（chunk 3 第 4→第 5 名，掉出 k=4 窗口）                                 |
| k=8 下继续保留前缀（R1R2 局部）   | strict 3/10 < 无前缀 4/10 | chunk 6 被挤出前 8；**前缀的伤害不能靠加大 k 完全掩盖**                              |
| RRF 混合检索（O1-R5）        | 4/10 → **3/10**        | BM25 对中文近零信号 + `1/(60+rank)` 名次压缩 + 只认名次不认可信度；**反证**：hyde 下融合不再有害 |
| k=8 上下文变长（day16 记录）    | Q1 答错更甚                | 热词段稀释 → **但 R3 未复现**（属低概率抖动，不是稳定效应）                               |
| 模板 09 v2 系（R5/R5a/R5b） | 防幻觉 7/10 → 3/4/6；F1 0.824 → 0.462/0.571/0.706 | 归因锁死在一句提示语②"资料里有依据就照实回答"；**"误读型幻觉"提示词治不了** |
| 引文兜底 `retry`（R6b）      | F1 0.824 → **0.737**（FP 0→2）、总 55% → 50% | 重答指令给的"出口条款"被 3B 滥用：把 ✅ 的题改成拒答（Q3）、交出带拒答词的半截答案（Q1）、甚至**空答案**（Q5）；与 R5 规则②同型 |
| 查询改写三档（R4/R4b/R4c）   | 总 55% → 45% / 50% / 55%（**无一轮超过 R3**）；F1 0.824 → 0.706/0.571/0.625 | `term` 会替换专名（`redirected%` 那种脏查询）；`hyde` 让 out 题"有话说"（不区分资料里到底有没有答案）→ 改写无净增益（细节见 `实验日志.md` R4 三行） |
| **F3 真实数据微调（F3v2）**   | F1 0.824 → **0.333**（**FN 3→8**）、总 55% → **35%** | **201 条真实 QA 同配置微调后防幻觉大幅退化**：**6 条 out 题由"如实拒答"退回编造 —— Q13 / Q15 / Q16 / Q17 / Q19 / Q20**（另 Q11 反向变好、Q14/Q18 两轮都错；已逐题对账 `eval_results.json`）。⚠ **边界**：本轮**只用真实数据、没有加合成** → 退化也可能来自**数据量 300→201**，**不是"真实数据不好"**（拆变量留 D5） |


### 6.5 表四：真实数据整改（HR 意见③）


| 项                       | 数字                                             | 可追溯性                                                                        |
| ----------------------- | ---------------------------------------------- | --------------------------------------------------------------------------- |
| 合成数据（第三周遗留，**如实标注为合成**） | 300 条（语料池 + 模板生成）                              | 无外部来源                                                                       |
| 论文真实 QA（F1，day18）       | **90 条**（术语 18 / 方法 22 / 实验 17 / 结论 25 / 局限 8） | `source.page + chunk_id + quote`，**逐字回查 90/90、泄漏 0**                        |
| 行业真实 QA（F2，今天）          | **111 条**（市场与产业链 32 / 产品技术 27 / 公司经营 24 / 行业标准与政策 14 / 论文·研报结论 14；26 个机构、最大占比 10.8%）                                  | `source.org/title/url/date/quote`；**人工复核 + 泄漏检查**（行业资料不在库内，**无法逐字回查**，如实标注） |
| **真实数据合计**                | **201 条**（论文 90 + 行业 111，`origin=real`、`instruction` 去重 0）  | 全部带 `source`，每条可点开 URL 复核                                                 |
| **F3 对照（v1 vs v2 adapter）** | **检索两列完全相同（4/10、5/10 ✅）**；生成 4/10 → **5/10**；**防幻觉 7/10 → 2/10**；**F1 0.824 → 0.333**；总 **55% → 35%** | 同评测集、同配置、同 adapter 结构，**只改训练数据**；⚠ 真实 201 vs 合成 300，**来源与数量未拆开** |


### 6.6 收尾：`__` 纪律 + 下一步清单

- **所有** `__` **都不许预填**；D5 跑完"定稿数字 v2"（`--runs 3` 或 5）再回填；
- 数字一变，**同步三处**：`第三周\day15\README.md`、`效果评估报告.md`、`大模型算法学习成果汇报.md` 4.1；
- 报告末尾写"下一步"：D5 定稿 + 柱状图 + 双模型对比 + A/B 上线 + 简历结果句。

### ✅ 第 6 步验收标准

- [x] `优化成果报告.md` 骨架成型（四张表 + 结论与下一步 + 日志索引）→ **已建并回填**
- [x] 表一/表二里**已跑出的数字全部回填**（R3 及之后全部轮次都跑完了）；**仅"定稿数字 v2"留 D5**
- [x] 失败尝试记录 **≥4 条**（实际 **8 条**：含 RRF、改写三档、模板三档、`retry`、**F3 真实数据**）
- [x] 报告中明确写出"**HyDE 的 100% 是上界、不是成绩**"这一句（见 §5.3）
- [x] 额外补了两句防误读声明：「**非法 0 处 ≠ 答案干净**」「**FP 假阳性**」（报告 §5.3/§5.4）

---

## （可选）第 7 步：关掉 day18 的最后一条待补项（15~25 分钟）

day18 报告里"唯一仍未完全关闭"的是：`发布包\space_demo` **的 DeepSeek 端到端真冒烟**（要 key、花额度）。环境与 UI 冒烟已就绪（`conda activate demo` + `get_api_info()` 返回 2）。

```powershell
# ① 备好 key（不要写进代码、不要 submit 进 git；只在本机环境变量里）
$env:API_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
$env:MODEL_NAME   = "deepseek-chat"
$env:API_KEY      = "sk-****"          # 只在本终端会话里有效

# ② 在独立环境 demo 里跑（⚠ 千万别在 llm 环境里装发布包的 requirements）
conda activate demo
cd "D:\Lan\研究生\技术学习\大模型算法\第四周\发布包\space_demo"
python app.py
# 浏览器打开 7860 → 提问一句 → 看是否真的走 DeepSeek 返回，且页面顶部有"生成层 ≠ 微调模型"的标注
```

**验收**：能截一张"页面上真实返回答案 + 标注可见"的图；**跑完把 key 的临时环境变量清掉**（关终端即可）。
**若今天没 key / 没额度**：**如实挂账即可**（D5 上线前必做），不要为了"打勾"而跳过真冒烟——day18 的教训正是"`--check` 通过 ≠ 请求路径可用"。

---

## 🏃 零散时间任务（0.5~1 小时，穿插在等训练/等生成时做）

### A. 力扣 53. 最大子数组和（中等 · DP vs 贪心 ⭐）

**题目**：给一个整数数组 `nums`，找**和最大**的连续子数组，返回其和。

**两种思路，都要会（面试常被要求"你还能用别的方法吗"）**：

```python
# ① DP（Kadane）：dp[i] = 以 i 结尾的最大子数组和
#    转移：dp[i] = max(nums[i], dp[i-1] + nums[i])
#    直觉：要么"自己重新开始"，要么"接着前面那串"——这就是"选 / 不选"的变体
def max_sub_array_dp(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)     # 重新开始 or 延续
        best = max(best, cur)
    return best

# ② 贪心（本质与 DP 同构，但叙述是"累计和 <0 就清零"）
#    直觉：如果前面的累计和已经是负贡献，带着它只会拖累后面 → 直接扔掉
def max_sub_array_greedy(nums):
    best, cur = nums[0], 0
    for x in nums:
        cur += x
        best = max(best, cur)
        if cur < 0:
            cur = 0              # 负贡献，清零重来
    return best
```

**复杂度**：两者都是 **O(n) 时间 / O(1) 空间**；**DP 更"通用"**（能改造成求"最大子矩阵和"、带约束的版本），**贪心更"贴直觉"**（但清零的合法性需要一句话证明）。

**数学视角（写进题解注释）**：

> 最大子数组和 = 对**前缀和**求"最大差值"（`max_{j>i} (P[j] − P[i])`）→ 一趟扫描里维护"目前为止最小的前缀和"即可。**这就是 DP 与贪心在这里等价的根本原因**：`dp[i-1] + nums[i] < nums[i]` 等价于"前面的累计和是负的"。

**提交**：按 `动态规划` 分类推到 GitHub `leetcode` 仓库，注释里写清"DP 与贪心的等价性"。

### B. 统计八股：6 节串线②（10~15 分钟）

**主题：RAG 评测 ↔ 统计评估 的对照**（把工程口径翻译成统计学语言——这是你的差异化王牌）：


| RAG 里的东西      | 统计里的对应                    | 今天的具体数字                   |
| ------------- | ------------------------- | ------------------------- |
| 检索命中率 strict  | **Recall@K**（召回率）         | 4/10 = 40%                |
| 防幻觉 F1        | 分类问题的 **F1**（混淆矩阵口径）      | 0.824（TP7 FP0 FN3 TN10）   |
| 防幻觉 FP        | **假阳性**（该答却拒答 = 过度保守）     | R3 是 0；带前缀那轮是 1           |
| 防幻觉 FN        | **假阴性**（该拒答却编造）           | 3（Q11/Q14/Q18）            |
| `--runs 3` 投票 | **降方差**（蒙特卡洛重复 → 取众数）     | 一致率 0.950                 |
| n=20 的置信区间    | ±0.22 量级 → **只看量级，不看小数点** | "提升 20 个百分点"而不是"提升 20.0%" |
| 数据自检          | **测量误差**（先修仪器再量数据）        | 9/10 自洽                   |
| 数据泄漏检查        | **训练集/测试集独立性**            | 阈值 0.70，命中 0 条            |
| 同一评测集前后对比     | **配对实验**（同样的题、只改一个变量）     | R3 → R4 / R5              |


**一句话收尾（面试可讲）**：

> "我的评测不是'跑个分'，而是一次**受控实验**：固定卷子、每题多跑取多数降方差、判据双轨透明、每一步只改一个变量——**这套纪律跟做 A/B 测试和假设检验是同一套思路**。而且我知道 n=20 的置信区间大约 ±0.22，所以我只讲'提升几个百分点、提升几道题'，不在小数点后两位上做文章。"

### C. 牛客"银行/央企"真题 0.5 套（20~30 分钟）

按 day18 的节奏做**另外 0.5 套**（day18 做了一半，今天补齐一整套）：

- 行测部分**别恋战**；
- 编程题**先暴力拿分再优化**；
- 错题只记一行："为什么错 + 正确思路"。

> **⚠ 零散时间任务的纪律（day18 的教训）**：**没有可核验产物 = 不算完成**。做了就留下痕迹（力扣提交记录 / 笔记文件 / 错题一行），并在工作汇报里如实写；**没做就不填**。

---

## 🌙 睡前 15 分钟：收尾 + git 存档

### 7.1 汇总今天的产出（打勾核对）


| 产出                                      | 位置                                                   | 状态  |
| --------------------------------------- | ---------------------------------------------------- | --- |
| 查询改写脚本 + 缓存                             | `day19\rewrite_queries.py`、`queries_rewritten.json`  | ✅   |
| eval_v2（day19 版）                        | `day19\eval_v2.py`（含 `--rewrite-cache` / 模板 v2a/v2b / `--citation-policy`） | ✅   |
| 改写轮次结果（R4 / 三档 term·hyde·both）                  | `day19\result_rewrite_*`                             | ✅   |
| 模板消融结果（R5 / R5a / R5b）                  | `day19\result_template_*`                            | ✅   |
| 引文兜底结果（R6 / R6b）                        | `day19\result_cite_*`                                | ✅   |
| F2 校验器 + 真实行业数据（**111 条 / 门槛 100 ✅**）         | `day19\build_real_qa.py`、`sft_data_real.json`        | ✅   |
| F3 训练脚本 + adapter v2（**不进 git**）        | `day19\train_lora_v2.py`、`lora_adapter_v2\`          | ✅   |
| F3 对照结果（v2 adapter，**负结果已定性**）           | `day19\result_lora_v2_runs3`                         | ✅   |
| 优化成果报告（初稿，四张表 + 一节已回填）                  | `day19\优化成果报告.md`                                    | ✅   |
| 实验日志更新（R4/R5/R6/F3 行 + Day19 诊断结论 + 新失败尝试） | `第四周\实验日志.md`                                        | ✅   |


### 7.2 确认大文件不进 git（30 秒）

```powershell
git check-ignore -v 第三周\day13\chroma_db
git check-ignore -v 第三周\day12\lora_adapter
git check-ignore -v download
git check-ignore -v 第四周\day19\lora_adapter_v2      # ← 今天新增：必须被忽略（实测 57MB 权重）
```

> **今天必须新增检查** `lora_adapter_v2`。若它**没有输出**（= 没被忽略）→ 立刻加进 `.gitignore`（`第四周/day19/lora_adapter_v2/`、`第四周/day19/train_logs_v2/`），**别把权重推上去**。
> **要进 git 的**：`rewrite_queries.py` / `queries_rewritten.json` / `eval_v2.py` / `build_real_qa.py` / `sft_data_real.json` / `train_lora_v2.py` / `local_api_lora_v2.py` / `优化成果报告.md` / 各结果目录 / 教程 / 汇报——**都是纯文本、很小**。

#### ⚠ 实测记录：这条检查**真的没过**（9/21 收尾，已修好）

**发生了什么**：收尾时 `git add .` 一把梭，**`lora_adapter_v2/` 连权重一起进了提交**——`adapter_model.safetensors` **57.16 MB** + `tokenizer.json` **10.89 MB**。再跑 `check-ignore` 时它**没有任何输出**（❌），因为**已跟踪的文件无法被忽略**：`.gitignore` 只对"未跟踪文件"生效，**已经在索引里的文件加规则也拦不住**。

**为什么没被更早发现**：`git status` 里它**不显示为未跟踪**（因为已经提交了），看起来"很干净"——**这正是陷阱**：`.gitignore` 写对 ≠ 生效，**必须用 `check-ignore` 主动问一句**。

**修的过程（两个关键判断）**：

| 步骤 | 命令 | 为什么这么选 |
|---|---|---|
| ① 先看能不能安全改 | `git status -sb` → `## main...origin/main [ahead 17]` | **17 个提交全没 push** → 改历史不会影响远端，代价最小 |
| ② 加规则（防下次） | 在 `.gitignore` 加 `第四周/day19/lora_adapter_v2/`、`train_logs_v2/` | 光删索引不够，**规则不加、下次 `git add .` 还会进去** |
| ③ 移出索引（保磁盘） | `git rm -r --cached 第四周/day19/lora_adapter_v2` | **`--cached` 是关键**：只从 git 索引移除，**磁盘上的权重原样保留**（服务还能起） |
| ④ 从历史里抹掉 | `git reset --soft HEAD~1` → 重新 `git commit` | `--soft` 只动 HEAD、**工作区与索引都不动**，等于"重新写一次这个提交" |
| ⑤ 验证 | `git rev-list --objects main \| Select-String "adapter_model"` → **空** | 证明它**从 main 不可达** → 以后 `git push` 不会再传这 68 MB |

**⚠ 两个必须知道的边界**：

1. **`git rm --cached` 单独用是"不够"的**：它只让**未来**的提交不再包含这些文件，**旧提交里的 68 MB 仍然在历史里**、还会被 push 上去。**要么改写那个提交（本例），要么接受它永远在历史里**。
2. **改写只对"没 push 的提交"安全**：一旦 push 过，改写就必须 `push --force`，会打乱任何克隆。**所以"提交前先跑 `check-ignore`"不是形式主义——它决定了你有得选还是没得选。**
3. **磁盘与远端是两件事**：改写后 68 MB 变成"不可达对象"，**push 不会带它**；但**本地 `.git` 的 pack 要到 `git reflog expire --expire=now --all` + `git gc --prune=now` 才真正回收**。只想省远端流量的话，**不做也行**。
4. **本项目 `.git/lfs/objects` 还躺着 5.9 GB**（两个陈旧 LFS 对象：3.78 GB + 2.10 GB，**不在 pack 里、不影响 push**）。想清可以 `git lfs prune --dry-run` 先看一眼再决定——**本项目并没有 `.gitattributes`、权重没走 LFS**，这些是老早 `git lfs pull` 的残留。

### 7.3 提交今天的产出

```powershell
git add .
git commit -m "Day19: O1-R4 查询改写落地（R4 三档 strict 4/6/5，均未超 R3）+ 模板09 v2 与逐句消融（三档回退）+ O2-G3 引文白名单（strip 非法5→0处）+ F2 真实行业QA 111条可追溯（市场32/产品27/公司24/标准14/论文14） + F3 真实vs合成对照微调（F1 0.824→0.333，负结果）+ 优化成果报告初稿"
```

> **commit 信息里带数字**（本周纪律）。**数字没跑出来就留** `__`**，别编**。

#### 7.3.1 ⚠ `git push` 推不上去？三层坑，一层层过（9/21 实测，45 分钟踩完）

**症状**：`fatal: unable to access 'https://github.com/...': Failed to connect to github.com port 443 after 23146 ms`。
**先分清是"网络"还是"git"**——本例是**三层独立的坑叠在一起**，修掉一层马上露出下一层，**别以为修完一层还没好就是修错了**。

**第 0 步：先判断"能不能出去"**

```powershell
git remote -v                      # 确认推的是 https 还是 ssh
Test-NetConnection github.com -Port 443 -WarningAction SilentlyContinue | Select-Object TcpTestSucceeded
```

| 结果 | 含义 | 下一步 |
|---|---|---|
| `TcpTestSucceeded: False` | **连不出去**（国内直连 GitHub 常态） | 走第 1 层：挂代理 |
| `True` 但 push 仍报 443 | 能连但 TLS/认证有问题 | 跳到第 2 层 |

**第 1 层：git 没走你的代理**（最常见的根因）

代理软件（Clash 等）**开着 ≠ git 会用它**。先确认本地代理端口在不在听（Clash Verge 默认 **7897**，老版本 7890）：

```powershell
foreach ($p in 7890,7897,10809,1080) { "$p -> " + (Test-NetConnection 127.0.0.1 -Port $p -WarningAction SilentlyContinue).TcpTestSucceeded }
curl.exe -x http://127.0.0.1:7897 -sS -o NUL -w "HTTP %{http_code}`n" --max-time 25 https://github.com   # 200 = 代理通
```

代理通、直连不通 → **给 git 挂上代理**。**推荐只在本次命令里加**（代理一关 git 就不至于全废）：

```powershell
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push
```

> **为什么不写进全局配置**：写进 `--global` 后，**代理一关，所有仓库的 git 操作全部报错**。要持久化就写**仓库级**（`git config --local`），或者干脆每次带 `-c`。

**第 2 层：Git LFS 的"锁校验"把 push 打断**

挂上代理后如果报 `Post ".../info/lfs/locks/verify": EOF`、`Remote "origin" does not support the Git LFS locking API`：

**根因**：git-lfs 的 `pre-push` 钩子在推之前会调一个"文件锁校验"API，这个调用在代理下容易断。
**关键判断**：**本仓库并没有 `.gitattributes`、没有任何 LFS 文件**（`git lfs ls-files` 空）——所以这个校验对本项目**毫无意义**，关掉它不损失任何东西。

```powershell
# 只关"锁校验"这一项，且只对本仓库生效（.git/config 里，不进 git、不影响别的仓库）
git config "lfs.$(git remote get-url origin)/info/lfs.locksverify" false
```

> **⚠ 别用 `GIT_LFS_SKIP_PUSH=1` 一把梭**：那会把**整个 LFS pre-push 钩子**跳过（含真实对象上传）。本例没有 LFS 文件所以无害，但**换到真有 LFS 的仓库就会静默漏传权重**。只关 `locksverify` 是**范围最小**的修法。

**第 3 层：Windows 的 `schannel` TLS 后端扛不住代理**

再推如果报 `error: RPC failed; curl 35 schannel: failed to receive handshake, SSL/TLS connection failed` + `send-pack: unexpected disconnect while reading sideband packet`：

**根因**：Git for Windows 默认用**系统自带 TLS 后端 `schannel`**，经代理时握手不稳（`git config --get http.sslBackend` 返回 `schannel` 就是它）。
**修法**：换 git 自带的 `openssl` 后端 + 退回 HTTP/1.1（HTTP/2 在代理下的多路复用也常引发 sideband 断开）：

```powershell
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 -c http.sslBackend=openssl -c http.version=HTTP/1.1 push
```

**✅ 一次成功的完整命令**（三层一起带，实测 20 秒推完 21 个提交）：

```powershell
cd "D:\Lan\研究生\技术学习\大模型算法"
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 `
    -c http.sslBackend=openssl -c http.version=HTTP/1.1 push
# 成功标志：d9b0fd0..de5e153  main -> main
```

**推完自检**（**`ahead` 消失 = 真的推上去了**）：

```powershell
git status -sb            # 应只剩 "## main...origin/main"（没有 [ahead N]）
git log --oneline -1 origin/main   # 应等于你本地最新提交
```

**翻坑要点（可迁移到任何项目）**：

| 现象 | 别急着做 | 该做 |
|---|---|---|
| 连不上 | 以为是仓库/权限问题 | `Test-NetConnection` 分清"网络"还是"git"；**代理开着 ≠ git 在用** |
| 修好一层又报新的 | 以为修错了、开始乱改配置 | **这是正常的**：三层是串联的，**逐层推进**，每层都有独立的自检命令 |
| 想一步到位 | 直接改 `--global` / 关 `sslVerify` | **优先用 `-c` 一次性参数**，确认有效再决定要不要持久化；**`sslVerify=false` 是关安全检查，别碰** |
| LFS 报错 | 直接 `GIT_LFS_SKIP_PUSH=1` | 先 `git lfs ls-files` 看**到底有没有 LFS 文件**，再选**范围最小**的修法 |

### 7.4 写明日计划（Day20 = D5，本周收口）

**Day20 主线（照《第四周详细计划》D5 与路线图第五节）**：

1. **O3《优化成果报告》定稿** + `plot_compare.py` → `优化前后对比.png`（before/after 柱状图）；
2. **定稿数字 v2**：关键轮次用 `--runs 3`（或 5）重跑，双轨口径出最终行；
3. **F4 原模型 vs 微调模型**同评测集第二版对比（`--profile original` vs `lora`）；
4. **P 线**：个性化叙事 8 条 + 端到端复述脚本 + **现场彩排 3 次** + `面试追问自测表.md`（≥20 问，含 HR 新增 3 问）+ 简历改写成**结果句**（数字回填 O3 定稿值）；
5. **A/B 发布上线**：`发布包\blog\` 推博客 + `发布包\space_demo\` 推 HF Space（**Secrets 配 key，绝不进 git**）→ 拿到两个公网 URL；
6. **数字同步三处** + `week4工作汇报.md` + git 存档。

**零散**：力扣 322（零钱兑换，完全背包入门）＋ 6 节串线③（收尾）＋ 牛客真题复盘。

---

## 🚧 常见问题速查表（出问题先看这里）


| 现象                                                                                                            | 原因                                                                                                      | 解决办法                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `UnicodeEncodeError: 'gbk' codec can't encode ...`                                                            | 输出端 UTF-8 vs 显示端 GBK 不匹配                                                                                | `chcp 65001` + `$env:PYTHONIOENCODING='utf-8'`；或直接用 `& "D:\miniconda1\envs\llm\python.exe" ...`                      |
| `eval_v2.py` 报 `No module named 'langchain_chroma'`                                                           | 用了系统 python                                                                                             | `conda activate llm`                                                                                                 |
| `rewrite_queries.py` 报连接错误 / `ConnectionError`                                                                | 8000 端口的模型服务**没起**或还在加载                                                                                 | 先起服务、等日志出现「模型已就绪」；`curl http://127.0.0.1:8000/health` 应返回 `model_ready: true`                                        |
| `rewrite_queries.py` 报 `500 Server Error`（**只有一句 500，看不到原因**）                                                 | 服务端 `HTTPException(500, f"生成失败：{e}")` 把真实异常藏在**响应体** `detail` 里，`raise_for_status()` 不打印它               | 脚本已改成 `if r.status_code != 200: raise RuntimeError(r.text[:500])`，直接看到 `detail`；`detail` 不出现时去**服务端那个终端**看 traceback |
| `detail` 里是 `temperature must be a strictly positive float`                                                   | `call_model` 传了 `temperature=0.0`，而服务端 `do_sample=True`，`transformers` 的 `TemperatureLogitsWarper` 拒收 0 | 传 `0.01`（**要确定性就取极小正值**，不能用 0）；根因在服务端 `local_api_lora.py` 的 `generate()`，不是客户端网络问题                                   |
| 改写的 `term` 档里**混着中文/整句**                                                                                      | 模型没守指令（提示词不够硬）                                                                                          | 收紧 system（"只输出英文关键词"）+ 加 `clean()` 的兜底规则；仍不行就换 `--modes hyde`                                                        |
| 改写的 `hyde` 档**编出了一堆没见过的术语**                                                                                   | HyDE 机制本身允许"假设表述"                                                                                       | **这是预期行为但必须记进失败尝试**；对照 `chunks.json` 扫一眼，看这些术语是否真在库里；必要时在提示词里加"只能用你确信出现在论文中的术语"                                      |
| R4 的 `检索 strict` **没升反降**                                                                                     | 改写引入了系统性噪声（比随机更危险）                                                                                      | 先抽查改写内容；试 `term` 档（最稳）；并在报告里如实写"**上界 100% vs 真实现 X% 的差距 = 改写器能力瓶颈**"                                                 |
| R4 的 `retrieval_query` 全是原文（没生效）                                                                              | 缓存里的档位字段是空的 / `--rewrite-mode` 写错                                                                       | `python rewrite_queries.py --show 3` 看该档位有没有内容；确认 `--rewrite-mode` 与缓存字段名一致（`term`/`hyde`/`both`）                    |
| R5 的 `检索 strict` **和 R3 不一样**                                                                                 | 模板不该影响检索 → 说明**动了别的变量**                                                                                 | 检查命令是否漏了 `--top-k 8` / `--query-instruction off` / `--keyword-patch on`；**这是最好的自检信号**，别放过                            |
| 模板 v2 后 in 题**大量拒答**（新 FP 冒出来）                                                                                | "禁止编造"的约束太硬 → 模型宁可不答                                                                                    | 明确写"**精度-召回权衡**"；报告里 v2a（只弱化拒答）/ v2b（只加约束）两版都留着，按场景选                                                                 |
| `train_lora_v2.py` 报 `FileNotFoundError: sft_data_v2_real.json`                                               | **不在 day19 目录下运行**（`DATA_PATH` 是相对路径）                                                                   | `cd 第四周\day19` 再跑；或把 `DATA_PATH` 改成绝对路径                                                                              |
| 训练报 `os error 1455` / 显存爆 / 直接崩                                                                               | 6G 装不下：服务还在跑、`BATCH_SIZE` 太大                                                                            | ① 停服务（按端口 kill）；② `BATCH_SIZE` 2→1；③ 降 `MAX_LEN`；④ 确认 `low_cpu_mem_usage=True`                                       |
| 训练卡住在 `0%` 很久                                                                                                 | 正常：4bit 加载 + 首次前向会慢                                                                                     | 看 `nvidia-smi` 有没有进程在占显存 + 看 `train_v2_console.log` 是否有新行                                                            |
| v2 服务起来后 `eval_v2.py --profile lora_v2` 报"服务端模型 = X，但 profile 要求 = Y"                                         | 服务身份检查拦住了（**这是设计**）                                                                                     | 确认 `local_api_lora_v2.py` 的 `MODEL_NAME` 与 `config.json` 里 `lora_v2.model_name` **完全一致**；6G 一次只能跑一个 3B               |
| `local_api_lora_v2.py` 报"找不到 adapter"                                                                         | `ADAPTER_PATH` 没改 / 训练没产出                                                                               | `dir lora_adapter_v2` 确认有 `adapter_config.json`；路径用 **r"" 原始字符串 + 绝对路径**（day18 已记过这个坑）                               |
| `build_real_qa.py` 一直说"条数不足"                                                                                  | 正常：还没写够                                                                                                 | 按 `--min` 分批推进（25/50/75/100）；**别硬凑**                                                                                 |
| `build_real_qa.py` 报"source 缺 url / url 不是合法链接"                                                               | 手抄漏了 / 写成了域名                                                                                            | 补全 `https://…`；不确定来源就**换一条**（合规红线）                                                                                   |
| `build_real_qa.py` 报"重复 instruction"                                                                          | 换个说法问同一件事                                                                                               | **合并成一条**（真实性不是靠凑条数）                                                                                                 |
| `build_real_qa.py` 报"数据泄漏"                                                                                    | 与评测题太像                                                                                                  | **换一条题**；`--allow-leak` 只用于临时调试                                                                                      |
| 自己新写的脚本读评测题时报 `AttributeError: 'str' object has no attribute 'get'` / `TypeError: unhashable type: 'slice'` ⭐ | `eval_questions.json` **不是 list，而是** `{"meta":..., "questions":[...]}`——直接迭代拿到的是 key 字符串                | 统一按 `data["questions"]` 取（本教程两个脚本已内置兼容：`if isinstance(data, dict): data = data.get("questions", [])`）                |
| 传 `--eval-questions ..\day17\eval_questions.json` 报 `FileNotFoundError`                                       | **评测题全项目只有一份**：`第三周\day13\eval_questions.json`（day17 目录下**没有**它的副本）                                     | **不用传这个参数**——`rewrite_queries.py` 的默认值 `QUESTIONS` 已经指向 `第三周\day13\eval_questions.json`（见 1.4 的常量区）                  |
| 想拿 HyDE 的 100% 写进报告当成绩                                                                                        | **上界 ≠ 成绩**                                                                                             | 报告里只能写"上界 100%，说明瓶颈在查询侧"；成绩用 R4 的真实现                                                                                 |
| `git status` 里出现 `lora_adapter_v2/`                                                                           | 权重目录没被忽略                                                                                                | 加 `.gitignore` 条目；**push 前再** `git check-ignore` **一次**                                                              |
| 训练时忘了停服务 → 两个进程抢显存                                                                                            | 违反串行纪律                                                                                                  | 停掉其中一个；**下次训练前先** `nvidia-smi` **+ 查 8000 端口**                                                                       |
| **加了 `.gitignore` 但 `check-ignore` 仍没输出**（权重已进过提交）⭐                                                           | **`.gitignore` 只管"未跟踪文件"**；已进索引的文件加规则也拦不住，且 `git status` 里它**不显示**（看起来"很干净"）                                     | 先 `git status -sb` 看有没有 push：没 push → `git rm -r --cached <目录>` + 改写该提交（`git reset --soft HEAD~1` 重新 commit）；已 push → 只能接受。**验收**：`git rev-list --objects main \| Select-String "adapter_model"` 返回**空** |
| 把 API Key 写进 `发布包\space_demo\app.py`                                                                          | **本周最严重的翻车点**                                                                                           | key 只走环境变量 / HF Space Secrets；**push 前全局搜** `sk-` **/** `api_key`                                                    |


### ❗ 训练/服务切换速查（今天会反复切）

```powershell
# 看显存与进程
nvidia-smi

# 停模型服务（8000）
Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { taskkill /PID $_.OwningProcess /F }
netstat -ano | findstr :8000        # 无输出 = 已释放

# 起 v1 服务（合成数据 adapter）
cd "D:\Lan\研究生\技术学习\大模型算法\第三周\day13"
python -m uvicorn local_api_lora:app --host 127.0.0.1 --port 8000

# 起 v2 服务（真实数据 adapter）
cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day19"
python -m uvicorn local_api_lora_v2:app --host 127.0.0.1 --port 8000

# 确认"服务身份"（防跑错模型）
curl http://127.0.0.1:8000/v1/models
```

---

## ✅ 今日验收清单（完成一项打一个勾）

**第 1 步 · 查询改写落地（O1-R4）⭐⭐：**

- [x] `rewrite_queries.py` 跑通，`queries_rewritten.json` 含 `_meta`（模型 / 时间 / 提示词版本）
- [x] **人工抽查 ≥3 条**（Q1/Q3/Q5），能说出"实体保住没有、有没有幻觉术语"
- [x] `eval_v2.py --rewrite-cache` 接线生效（打印 `[改写] 已加载` + `rw=<档位>`，结果里有 `retrieval_query`；**且** `⑦ 改写生效` **不为 0**）
- [x] R4 全量（`--runs 3`）跑完并入表；**能说出"改写让 strict 从 40% 变成 __%"**
- [x] 报告/日志里写明"**上界 100% ≠ 成绩**"，以及真实现与上界的差距与原因

**第 2 步 · 改写后融合 / Rerank：**

- [x] `retrieval_lab.py --query-mode file` 跑通，得到真实改写后的三通道并集
- [x] 试过 `rrf_k=10/20` 至少两档，能说清"融合何时翻正"
- [x] 日志新增「Day19 诊断结论」（**标注诊断口径，不进成绩表**）

**第 3 步 · 模板 09 v2 + 消融：**

- [x] 从 R3 的 `run_log.txt` 对账出靶子 = **Q11/Q14/Q18**（且说得出 Q18 的"先编后拒答"形态）
- [x] `v2`（至少）跑完；有时间则 `v2a` / `v2b` 各跑一轮
- [x] 对照表填齐，**检索 strict 各轮一致**（自检信号）；结论按"检索层 vs 生成层分开报"
- [x] （加分）`citation_bad` 引文合法性后处理跑通

**第 4 步 · F2 真实行业数据：**

- [x] `build_real_qa.py --min 100` 全绿（或如实报告当前条数 + 补齐计划）
- [x] 随机抽 3 条**打开 URL 复核**（抽查动作真的做了）
- [x] 类目/机构分布合理，数字写进报告
- [x] 如实标注"行业数据无法逐字回查、靠来源字段 + 人工复核"

**第 5 步 · F3 对照微调：**

- [x] 训练前**服务已停**、`nvidia-smi` 确认显存释放；峰值 **<6G**
- [x] `lora_adapter_v2/` 产出成功；`train_v2_console.log` 留存
- [x] `local_api_lora_v2.py` 服务身份检查通过（`/v1/models` = `...-LoRA-v2`）
- [x] v1 vs v2 对照表填齐（**检索两列相同**），结论如实写

**第 6 步 · 成果报告初稿：**

- [x] `优化成果报告.md` 四张表骨架成型 + 失败尝试 ≥4 条
- [x] 已跑出的数字**全部回填**，未跑的留 `__`（**一个都没预填**）

**收尾：**

- [x] `第四周\实验日志.md` 新增 R4/R5/F3 行 + Day19 诊断结论 + 新失败尝试
- [x] `git check-ignore` 四条有输出（含 `lora_adapter_v2`）—— **首次检查没过**（权重已进提交），已按 §7.2「实测记录」修复：加规则 + `rm --cached` + 改写未 push 的提交 → 四条全绿 ✅
- [x] 训练/服务进程已停干净（`:8000` 已释放 ✅；训练结束后显存回落到无占用）
- [x] Day19 全部产出 git commit 成功（**两个提交**：`0faea36` 主体 + `06dc641` 收尾，commit 信息均含数字；`main` 仍 ahead、未 push）
- [ ] 零散时间三项**有产物才打勾**（力扣提交 / 笔记 / 错题一行），没有就如实留空

**全部打勾 = Day19 完成：把 day18 的"上界"兑现成了"成绩"（查询改写真实现）、把生成层的稳定靶子做成了可归因的模板消融、把"真实数据"从论文扩到行业并做了真实 vs 合成对照——HR 三条意见里的"优化成果"与"真实数据"今天都有了硬证据。**

---

## 📎 附录

### A. 今天与本周计划的逐条对应（对照 `01-第四周详细计划.md` D4）


| 计划原文（D4 要点）                                             | 今天怎么落地                                                          | 状态     |
| ------------------------------------------------------- | --------------------------------------------------------------- | ------ |
| F2 真实行业数据采集（100~200 条，带 URL/页码，只采公开资料）                  | `build_real_qa.py` + `sft_data_real.json`（第 4 步）→ **111 条 / exit 0** | ✅      |
| F3 真实 vs 合成对照微调（复用训练配置，训 `lora_adapter_v2/`；显存串行）       | `train_lora_v2.py` + 双 adapter 同评测集对照（第 5 步）→ **F1 0.824→0.333（负结果）**，见 §5.5 | ✅      |
| O2-G2 模板 09 v2 + 消融（治过拒答 FP / 编造引用 FN）                  | `v2 / v2a / v2b` 三档并排（第 3 步）→ **结论：三档全线回退**，见 §3.4 | ✅ 负结果已定性 |
| O2-G3 引文合法性（原"加分项"）                                    | 升级为**强制白名单**：`--citation-policy strip/retry` → R6 / R6b（§3.5）   | ✅ strip 有效 / retry 负结果 |
| O3《优化成果报告》初稿                                            | 四张表 + 一节已回填 → `day19\优化成果报告.md`（第 6 步）                            | ✅      |
| （day18 明日安排）把 HyDE 做成"模型自动改写"、计入成绩                      | `rewrite_queries.py` + `eval_v2.py --rewrite-cache` → R4 三档（第 1 步）→ **无净增益** | ✅ 负结果已定性 |
| （day18 明日安排）改写后再融合 / 评估 Rerank                          | `retrieval_lab.py --query-mode file` 复筛 + `rrf_k` 调参（第 2 步）→ **融合翻正（兑现率 3/7→6/7）** | ✅      |
| （day18 明日安排）拿 R3 当新对照基线                                 | 全程同 k=8 / qi=off / runs=3，只改一个变量                                | ✅ 纪律已立 |
| 验收：真实数据 ≥100 条可追溯；v2 adapter 训练完成且显存正常；模板 v2 出对比；报告初稿成型 | 见「今日验收清单」→ **四项全达成**（模板那条是负结果，但对比已出）                         | ✅      |


**今日结构上的"三个设计决策"**（面试可讲）：① **改写只作用于检索层**（生成仍用原问题）→ 保证单变量；② **改写结果缓存落盘** → 可人工复核、零成本换档位、成绩可追溯；③ **v2 模板拆成 v2a/v2b** → 提示词层面的"逐句消融"。

### B. 关键数字口径速查（今天所有结论统一用这套）


| 指标 / 结论              | 数值                                                                                                                       | 来源与口径                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| **今日对照基线 R3**        | strict **4/10 = 40%** ｜ loose 5/10 ｜ 生成 4/10 ｜ 防幻觉 7/10 ｜ **F1 0.824（TP7 FP0 FN3 TN10）** ｜ 总 **11/20 = 55%** ｜ 一致率 0.950 | `day17\result_lora_k8_runs3`（`--runs 3`，**可引用**）     |
| 原始基线（优化前）            | 检索 20% ｜ 生成 20% ｜ 防幻觉 60% ｜ F1 0.667 ｜ 总 40%                                                                             | `第三周\day14\`                                         |
| 本周量化目标               | 检索 20% → **≥40%（争取 60%）**；生成 20% → **≥40%**；F1 ≥0.667 不降（争取 ≥0.8）                                                        | `01-第四周详细计划.md`                                      |
| HyDE 人工理想查询（**上界**）  | 三通道 strict **10/10 = 100%**                                                                                              | `day18\retrieval_lab.py --query-mode hyde`（**不是成绩**） |
| 改写真实现（R4 三档）         | `term` strict 4/10 ｜ `hyde` **5/10（loose 8/10）** ｜ `both` **6/10**；总 **45% / 50% / 55%**（**无一超过 R3**）                        | `day19\result_rewrite_*`（**可引用**，`⑦ 改写生效=20/20`）      |
| 模板 v2 消融（R5/R5a/R5b） | 检索列与 R3 **完全一致（4/10、5/10 ✅）**；生成/防幻觉/F1 **5/3/4 ｜ 3/4/6 ｜ 0.462/0.571/0.706** → **全线回退**                            | `day19\result_template_*`                            |
| O2-G3 引文兜底（R6/R6b）    | `strip`：交付非法 **5→0 处**、F1 **逐格不变 0.824** ｜ `retry`：F1 **0.824→0.737（FP 0→2）**                                      | `day19\result_cite_*`                                |
| F2 真实行业数据            | **111 条** ｜ 泄漏 0 ｜ 重复 0 ｜ 警告 0 ｜ 类目分布 市场32/产品27/公司24/标准14/论文14 ｜ 机构 26 个、最大占比 10.8%                                                                                             | `build_real_qa.py --min 100`                         |
| F1 论文真实数据            | **90 条** ｜ 逐字回查 90/90 ｜ 泄漏 0 ｜ 五类 18/22/17/25/8                                                                          | `day18\build_paper_qa.py`                            |
| F3 v1 vs v2          | 检索两列**完全相同**；生成 **4/10 → 5/10**；防幻觉 **7/10 → 2/10**；F1 **0.824 → 0.333**；总 **55% → 35%**（**负结果**）                             | `day19\result_lora_v2_runs3`                         |


> **口径纪律（第三周沿用 + day18 强化，今天继续）**：
>
> 1. **"诊断"与"成绩"分开**（`retrieval_lab.py` 只进分析章节）；
> 2. **上界 ≠ 成绩**（HyDE 100% 只能用来定位瓶颈）；
> 3. 说"提升"一律用**百分点**；
> 4. **对外引用的数字必须** `--runs 3`；
> 5. **F1 必须回** `run_log.txt` **抄 TP/FP/FN/TN**（防笔误）；
> 6. **数字变了同步三处**（`第三周\day15\README.md`、`效果评估报告.md`、`大模型算法学习成果汇报.md` 4.1），`__` **绝不预填**。

### C. 今天新出现的名词速查


| 名词                            | 一句话解释                                                               |
| ----------------------------- | ------------------------------------------------------------------- |
| **query 改写（query rewriting）** | 把用户问句改写成"更贴语料"的检索查询；今天做的是"中文问题 → 英文术语/假设答案"                         |
| **oracle / 上界（upper bound）**  | 理想条件下的最好结果（今天=人工写的英文理想查询），**只能用来定位瓶颈**                              |
| **可行估计量**                     | 用真实组件（本地 3B 自动改写）能拿到的数字 → **这才是成绩**                                 |
| **HyDE**                      | Hypothetical Document Embeddings：先让模型"造一段假设答案"再拿去检索；今天从"冒烟"升级为"真实现" |
| **term 档 / hyde 档 / both 档**  | 今天的三种改写强度：只出术语 / 出假设答案片段 / 两者拼接                                     |
| **改写缓存（cache）**               | 把改写结果落盘成 json：只跑一次、可人工复核、换档位零成本                                     |
| **逐句消融（sentence ablation）**   | 一次只删/改提示词里的一句话，看是谁在起作用（v2a / v2b）                                   |
| **精度-召回权衡**                   | "禁止编造"会减少 FN 但可能增加 FP（该答却拒答）→ 要说清取舍                                 |
| **引文合法性校验（O2-G3）**            | 检查答案里的 `[资料§N]`/`[N]` 是否真在本轮 Top-K 里；`--citation-policy` 三档：`record` 只留痕（默认）/ `strip` 就地剪除 / `retry` 点名重答一次再剪 |
| **cross-encoder / Rerank**    | 让"问题+段落"一起过模型打分（比双塔准、比双塔慢）→ 两段式召回的第二步                               |
| **双塔（bi-encoder）**            | 问题与段落各自编码后算余弦（快，但无交互）                                               |
| **两段式召回**                     | 粗召回 k=20~50 → 精排 Top-4/8（精度与延迟的折中）                                  |
| **RRF 的** `rrf_k`             | `Σ 1/(k+rank)` 里的常数；**调小它 = 放大名次差**（对"确信的第一名"更有利）                   |
| **数据泄漏（leakage）**             | 训练数据里含测试题 → 评测数字作废（阈值 0.70 的相似度检查）                                  |
| **过采样（oversampling）**         | 把少量真实数据重复到与合成数据同量级，让"数据来源"成为唯一变量                                    |
| **配对实验**                      | 同一批题、同一把尺，只改一个变量做前后对比（今天的 R3→R4/R5/F3v2 都是）                         |
| **Kadane 算法**                 | 力扣 53 的 O(n) 解法：`dp[i] = max(nums[i], dp[i-1]+nums[i])`             |


### D. 停掉"跑飞的服务 / 训练"（终端关了也能停）

```powershell
# 按端口找 PID（推荐）
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# 一条命令（PowerShell，不用手抄 PID）
Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { taskkill /PID $_.OwningProcess /F }

# 怎么确认真的停了？（别只看 taskkill 输出）
netstat -ano | findstr :8000     # 无输出 = 端口已释放 ✅
nvidia-smi                       # 显存占用回落 = 模型真释放了 ✅
```


| 端口 / 进程                   | 是谁                                                            | 停掉它的意思                 |
| ------------------------- | ------------------------------------------------------------- | ---------------------- |
| **8000**                  | `local_api_lora.py` / `local_api_lora_v2.py` / `local_api.py` | 释放显存，好跑训练或换另一个 adapter |
| **7860**                  | `app.py`（day18 Web UI）                                        | 关掉网页界面（不占显存）           |
| `python train_lora_v2.py` | 训练进程                                                          | 停训练（**训练中别同时跑服务**）     |


> **今天新增一个判据**：切 adapter 时，**不只确认"端口起来了"，还要确认** `/v1/models` **返回的是你要的那个名字**（`-LoRA` vs `-LoRA-v2`）——"服务在跑 ≠ 服务正确"。

---

> **今日一句话总结**：**day18 证明了"问对了就能搜到"（上界 100%），day19 要回答"我的系统能不能问对"**——
> 用**本地 3B 自动改写**（缓存可复核）把上界兑现成**能进报告的成绩**；再把生成层的稳定靶子（Q11/Q14/Q18 编造）做成**可归因的模板消融**（v2a / v2b 分开跑）；最后把 HR 最在意的"真实数据"**两条腿都落地**（F2 行业 QA ≥100 条带来源、F3 真实 vs 合成对照微调）。
> **一句话记住今天的方法论**：**把"理想条件下的上界"和"真实组件的成绩"分开报，才是诚实的优化成果。**

