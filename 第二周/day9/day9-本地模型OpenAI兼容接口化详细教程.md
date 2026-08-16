# Day 9（第二周第 4 个工作日 · 本地模型 OpenAI 兼容接口化）· 超详细新手教程

> 时间安排：主线 3~~4 小时 + 零散时间 0.5~~1 小时 + 睡前 15 分钟
> 适用对象：已完成 Day 8——手里有 20 个**实测过**的提示词模板（含 RAG 标准版模板 09）、能讲清角色扮演 / 输出格式约束 / 迭代优化 / 抑制幻觉、本地 Qwen2.5-3B-Instruct（4bit）随时能跑。今天把"命令行里能跑的模型"升级成"任何程序都能调的 HTTP 服务"。
> 本教程的目标：① 讲清 **OpenAI 兼容 API / FastAPI / uvicorn** 是什么；② 用 FastAPI 写 `local_api.py`，把本地 Qwen 包成 `/v1/chat/completions` 接口；③ 用 **curl / requests / LangChain** 三种方式调通它，验证"无缝切换"；④ 为 day10 RAG 全链路准备好标准 HTTP 服务。
> 本教程的铁律：今天**不改模型、不训练**，只做"包装"——把模型变成一个可以被任何程序调用的服务；每个测试都要在你自己电脑上真实跑通，输出以实际为准。

---

## 📌 今天你要做什么（大白话版）

Day 8 结束时，你的模型是这样用的：**打开 Python 脚本 → 加载模型 → 写死一个问题 → 跑出答案**。问题写死在代码里，别人想用你的模型，得先学会你的脚本。

今天你要做的是给它装一个"**对外营业的窗口**"：

1. **把模型包成一家"网上商店"**：用 FastAPI 写一个服务，启动后一直在后台跑着，专门接待"对话请求"；
2. **店里挂一块"统一价目表"**：所有客人（任何程序）都用同一种格式来下单——这就是 **OpenAI 兼容接口**（和 OpenAI 官方 `/v1/chat/completions` 一模一样的格式）；
3. **三种"客人"来光顾**：
   - `curl`（命令行直接发请求）——最原始，验证服务真的活着；
   - Python `requests`（写脚本发请求）——以后你写应用最常用的方式；
   - LangChain（大模型应用框架）——**只改 3 个参数就把框架接到你的本地模型上**，这就是"无缝切换"。

**今天结束时的"验收标准"一句话：**

> curl 调通本地接口，返回 `{"choices":[...]}` 格式；能讲清"为什么接口化"（前后端分离、统一调用、为 RAG demo 打底）；LangChain 无缝切换测试通过。

**和你的数学背景接上的点（今天会反复出现）：**

- **接口化 = 把模型抽象成一个函数 `f(输入) → 输出`**：调用方根本不关心模型长什么样、显存多大、用什么量化，只关心"输入这个格式，一定能拿到那个格式的输出"——就像数学里你只用 `f(x)` 而不关心 `f` 的表达式；
- **OpenAI 兼容 = 全行业约定同一个"函数签名"**：不同供应商（OpenAI / 本地 Qwen / 别的）都是同一个接口形状。调用方代码零修改，换的只是"函数实现"——这是"抽象与多态"思想在 AI 应用层的体现；
- **接口里的 `temperature` / `top_p` 参数，就是直接透传给你 Day 6 学的那个生成概率分布**：调用方隔着 HTTP 也能调节"输出分布的熵"——分布参数从"脚本里的常量"变成了"网络上的参数"；
- **format = 压支持集（Day 8 的数学）**：HTTP 的 JSON 请求/响应格式，把"模型和应用之间怎么交流"的不确定性压成了一个标准结构——和"输出格式约束压条件熵"是同一个思想，只是作用在**模型外部**而不是**模型内部**。

---

## 🗺️ 今天的学习路线图（先看这里，心里有个数）

| 步骤 | 内容 | 预计时间 |
| --- | --- | --- |
| 第 0 步 | 准备：打开 day9 文件夹、装 3 个包、确认环境 | 15 分钟 |
| 第 1 步 | 概念先行：OpenAI 兼容 API / FastAPI / uvicorn / ASGI / HTTP + 为什么接口化 | 30~40 分钟 |
| 第 2 步 | 动手写 `local_api.py`（逐段讲解）⭐ 今日核心 | 50~70 分钟 |
| 第 3 步 | 启动服务（uvicorn） | 15~20 分钟 |
| 第 4 步 | curl 测试（命令行直连） | 20~30 分钟 |
| 第 5 步 | Python requests 测试（`api_client_test.py`） | 15~25 分钟 |
| 第 6 步 | LangChain 无缝切换测试（`langchain_switch_test.py`）⭐ 核心验收 | 20~30 分钟 |
| 零散时间 | 力扣 876、141（链表进阶）＋ 统计八股：偏差 / 方差 | 0.5~1 小时 |
| 睡前 15 分钟 | 收尾 + git commit 存档 + 写明日计划 | 15 分钟 |

> 主线合计约 3.5~4 小时。6 步主线 + 零散 + 睡前跑完即达标。

---

## 🔧 第 0 步：准备（15 分钟）

### 0.1 认识今天的学习素材

| 素材 | 位置 | 用途 |
| --- | --- | --- |
| 本文件夹 4 个文件 + 1 个 JSON | `local_api.py` / `api_client_test.py` / `langchain_switch_test.py` / `test_payload.json` / 本教程 | 今天全部主线 |
| 本地模型 | `download\Qwen2.5-3B-Instruct\`（Day 6 已下载） | 被包装的服务对象 |
| Day 8 模板库 | `第二周\day8\提示词模板库.md`（模板 09 RAG 标准版） | 今天的测试输入直接复用 |
| 第二周周计划 | `第二周\01-第二周详细计划.md`（8/11 章节） | 全周对照 |

### 0.2 打开今天的文件夹

打开终端，输入（照抄即可）：

```
cd "D:\Lan\研究生\技术学习\大模型算法\第二周\day9"
```

看到行首变成 `PS ...\day9>` 就成功了。✅

### 0.3 确认环境 + 安装 3 个包

确认行首有 `(llm)`，没有就敲：

```
conda activate llm
```

今天需要 3 个新包（Day 8 没用到）：**fastapi**（Web 框架）、**openai**（官方 SDK，LangChain 连本地端点也依赖它）、**langchain-openai**（LangChain 接 OpenAI 兼容端点的适配包）。安装：

```
pip install fastapi "openai>=1.0" langchain-openai
```

> 实测环境提醒：你机器上 **uvicorn / requests / langchain / pydantic 已经装好了**，所以只需补上面 3 个。`langchain-openai` 会自动带上 openai，一并显式安装是为了保证版本。

装完验证（5 秒钟）：

```
python -c "import fastapi, uvicorn, openai, langchain_openai, requests; print('所有依赖 OK')"
```

看到 `所有依赖 OK` 就说明第 0 步完成。✅

### ✅ 第 0 步验收标准

终端停在 day9 文件夹、行首有 `(llm)`、4 个新依赖能 import = 成功。

---

## 🧱 第 1 步：概念先行（30~40 分钟）

> 对应周计划 8/11 第 1 条："概念（新名词必带简介）：OpenAI 兼容 API / FastAPI。"今天先把 4 个词讲透，再动手。

### 1.1 三个核心名词（新名词必带简介）

- **HTTP（HyperText Transfer Protocol，超文本传输协议）** = 计算机之间通过网络"传消息"的通用规则：一方发"请求"，另一方回"响应"。它是最基础的通信协议，连网页也是走 HTTP 传的。
  - 请求里最重要的两样：**方法**（GET = 取数据，POST = 提交数据）和 **URL**（地址，如 `http://127.0.0.1:8000/v1/chat/completions`）。
- **JSON（JavaScript Object Notation）** = 一种通用数据格式，用 `{}` 装字段、用 `"键": 值` 表示数据。Python 的字典长啥样它就长啥样，所以 Python 程序天然和它合得来（`json.loads` 一行解析）。HTTP 传 JSON = 请求和响应都用这种格式，任何语言都能看懂。
- **OpenAI 兼容 API** = 与 OpenAI 官方的接口**格式完全一致**的接口：同样的 URL（`/v1/chat/completions`）、同样的请求体（`messages` 数组 + `temperature` 等参数）、同样的返回结构（`{"choices": [{"message": {"content": ...}}]}`）。
  - **一句话**：OpenAI 的接口长什么样，我们就把本地模型的服务也做成长什么样——这样所有"会调 OpenAI 的程序"改个地址就能调我们的本地模型，不用改任何代码。
- **FastAPI** = Python 的一个现代 **Web 框架**（专门用来快速写出"HTTP 服务"的工具包）。你只需要写一个普通 Python 函数，再用 `@app.post("/v1/chat/completions")` 标注一下，它就自动变成接口：收到请求 → 调用你的函数 → 把返回值转成 JSON 发给调用方。**FastAPI = 帮你把"函数"变成"网上服务"的胶水。**
- **uvicorn** = 跑 FastAPI 服务的**服务器**程序（ASGI 服务器）。FastAPI 本身不会"跑起来"，需要一个服务器在端口上监听；你敲 `uvicorn local_api:app --port 8000`，它就在 8000 端口上守株待兔，来了请求就转给 FastAPI 处理。
  - **ASGI（Asynchronous Server Gateway Interface）** = Python Web 服务的一个行业标准接口，说明 uvicorn 和 FastAPI 是"按同一套标准对接的"，可以随便换（还有 uvicorn 的替代品 hypercorn 等）。

### 1.2 一条请求的完整旅程（画出这个图，你就懂了接口）

```
你的浏览器 / Python 脚本 / LangChain
        │  ① 发 HTTP 请求（POST /v1/chat/completions，JSON 格式）
        ▼
    ┌──────────────────────────┐
    │  uvicorn（服务器，守在 8000 端口）│
    └──────────────────────────┘
        │  ② 把请求转给 FastAPI
        ▼
    ┌──────────────────────────┐
    │  FastAPI（Web 框架）        │
    │  ③ 校验 JSON 格式          │
    │  ④ 调用你写的函数           │
    └──────────────────────────┘
        │  ⑤ 函数里：messages → ChatML → 模型生成 → 返回文本
        ▼
    ┌──────────────────────────┐
    │  Qwen2.5-3B-Instruct（4bit）│
    └──────────────────────────┘
        │  ⑥ 结果一层层返回，最终拼成 OpenAI 格式 JSON
        ▼
你的脚本拿到 {"choices": [...]}，打印出来
```

**数学视角**：这整条链路就是在实现一个函数调用 `f(messages, temperature, ...) → content`。FastAPI + uvicorn 帮你把"函数调用"翻译成了"网络消息"，调用方不需要 `import` 你的模型代码——这就是**服务化**（把能力变成服务，任何语言、任何机器都能用）。

### 1.3 为什么接口化？（验收必答，现在就背下来）

面试官问"为什么要把本地模型包成接口"，分四点讲：

1. **前后端分离**：模型在独立进程/机器上跑，应用代码只发 HTTP 请求。模型挂了、换了、升级了，应用代码一行不改；
2. **统一调用**：OpenAI 格式是全行业事实标准。今天写了对接 OpenAI 的代码，明天想换成本地 Qwen / 别的模型，**只改地址和模型名**，这就是"无缝切换"；
3. **语言无关**：HTTP + JSON 任何编程语言都能调。模型不再只能被 Python 直接 import——网页前端、Java 后端、手机 App 都能用你的模型；
4. **为 RAG demo 打底**（下周重点）：day10 的 RAG 全链路是"**检索模块 + 生成模块**"两个独立部分：Chroma 负责检索，Qwen 负责生成。让"生成"走 HTTP 接口，两者就能松耦合地拼起来，最后前端（比如网页上传 PDF）只对接这个接口——这正是"上传 PDF 能问答"的 RAG demo 的骨架。

> **一句话记忆**：接口化 = 给模型装一个"标准插座"，任何插头（curl / requests / LangChain / 前端）插上就能用。

### ✅ 第 1 步验收标准

能口头讲清 3 个名词（OpenAI 兼容 API / FastAPI / uvicorn）+ 画出 1.2 的请求旅程图 + 说出"为什么接口化"的 4 个理由 = 完成。

---

## 🚀 第 2 步：动手写 `local_api.py`（50~70 分钟）⭐ 今日核心

> 对应周计划 8/11 第 2 条："方案 A（推荐）：用 FastAPI 写 local_api.py：启动时加载 Qwen 模型 → 提供 /v1/chat/completions 接口 → 收到请求后生成 → 按 OpenAI 格式返回 JSON。"

`local_api.py` 已经帮你写好了，在 day9 文件夹里。**不要直接跑，先把它从头到尾读一遍**，下面逐段带你读。读完你就拥有"自己能改接口"的能力了。

### 2.1 文件头：五件套（30 秒）

脚本第一段注释是五件套：

```
- 模型   ：Qwen2.5-3B-Instruct，4bit 量化（NF4）加载
- 数据   ：通过 HTTP 请求传入的对话消息（OpenAI 格式的 messages 数组）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事：启动时加载模型 → 提供 OpenAI 兼容接口 → 生成 → 返回 JSON
```

和 Day 8 脚本一样，一眼能看懂它在机器学习的哪一环——今天这脚本**不做训练**，它是把"会推理的模型"变成一个"服务"。

### 2.2 常量区（30 秒）

```
MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"
MODEL_NAME = "Qwen2.5-3B-Instruct"   # 对外暴露的模型名
DEFAULT_SYSTEM = "你是一个乐于助人、严谨可靠的中文助手。……"
DEFAULT_TEMPERATURE = 0.7
```

- `MODEL_PATH` 就是你 Day 6 下载的模型位置（和 Day 8 的 `prompt_test_v2.py` 同一个路径）；
- `MODEL_NAME` 是"对外名字"——调用方在请求里写 `"model": "Qwen2.5-3B-Instruct"`，服务就认这个名字。

### 2.3 `build_chatml()`：把消息拼成模型的提示词（2 分钟）

OpenAI 格式的 `messages` 是**结构化的消息列表**：

```
[
  {"role": "system", "content": "你是……"},
  {"role": "user", "content": "你好！"}
]
```

而 Qwen 认识的是 **ChatML 文本格式**。`build_chatml()` 就是把前者翻译成后者：

> **新名词必带简介**：**ChatML（Chat Markup Language，对话标记语言）** = Qwen 这类模型规定的一种"把对话写进一段文本"的格式：用 `<|im_start|>` 这类特殊标记区分 system / user / assistant 三段，模型读到 `<|im_start|>assistant` 就知道"该我接话了"。你可能没印象，因为它是**从 Day 6 起就一直藏在脚本里**的东西——`qwen_inference.py`、Day 7/8 的 `build_chat_prompt()` 拼的就是它，只是一直当"固定模板"直接用、没正式介绍过名字。它也可以理解成：**ChatML 就是 Qwen 版的"角色分配标签"，让模型知道每句话是谁说的。**

```
<|im_start|>system
你是……<|im_end|>
<|im_start|>user
你好！<|im_end|>
<|im_start|>assistant
```

注意最后一行：补了一个"助手开始说话"的标记，模型就从这里接着往下生成。**这个函数是"接口格式 → 模型格式"的翻译官**，是整个接口的核心桥接逻辑。

### 2.4 `generate()`：真正的生成函数（2 分钟）

和 Day 8 的 `generate()` 几乎一样：tokenizer 编码 → `model.generate()` → 截取新生成的 token → 解码成文字。

多干了一件事：判断 `finish_reason`（结束原因）：

- 生成的最后一个 token 是 `EOS`（结束符）→ `"stop"`（自然说完）；
- 否则 → `"length"`（被 `max_tokens` 截断了）。

这个字段是 OpenAI 格式的一部分，调用方靠它判断"模型是说完了还是被掐断了"。

### 2.5 `ChatRequest`：请求体模型（2 分钟）

```
class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage] = Field(...)
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS
    stream: bool = False
```

- `BaseModel` 来自 **pydantic**（Python 的数据校验库，已装）——它定义"合法的请求长什么样"；
- FastAPI 收到请求后**自动按这个模型校验**：`messages` 缺失 → 直接返回 400 错误；`temperature` 没传 → 用默认 0.7。
- **数学视角**：`ChatRequest` 就是"接口的函数签名"——调用方必须按这个 schema 传参，服务端保证按这个 schema 接收。这正是"格式约束"（Day 8）在**模型外部**的版本。

### 2.6 `lifespan`：启动时加载模型（3 分钟，重点）

```
@asynccontextmanager
async def lifespan(app: FastAPI):
    ...  # 启动时：加载 tokenizer + 4bit 加载模型 + 打印显存占用
    yield   # 服务运行期
    ...  # 关闭时：释放显存
```

- **lifespan（生命周期）** = 服务进程"出生"和"死亡"时自动执行的钩子。出生时加载模型（**只加载一次**，进程活着就一直用），关闭时释放显存；
- 这解决了 Day 6~8 的老问题：以前每次跑脚本都要重新加载模型（1~3 分钟），现在**加载一次，服务一直跑，所有请求共用这同一个模型实例**——这就是"服务化"的第一个好处；
- 里面用的 `BitsAndBytesConfig(load_in_4bit=True, ...)` 和你 Day 6/8 完全一样，模型加载代码原样复用。

> 如果启动时打印出 `[transformers] torch_dtype is deprecated!` 之类的警告——无害，是新版 transformers 的提示，可以忽略。

### 2.7 三个接口（5 分钟，核心）

**① `/health`（健康检查）**：返回 `{"status": "ok", "model_ready": true}`。就像体检中心——先确认"服务活着、模型加载完了"再去消费。

**② `/v1/models`（列出模型）**：返回可用模型列表。OpenAI 官方也有这个接口，作用相同。

**③ `/v1/chat/completions`（对话补全）★ 核心**：

```
@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    # 1. 检查模型就绪、拒绝 stream
    # 2. 补默认 system 消息
    # 3. 调 generate() 生成
    # 4. 按 OpenAI 格式组装返回 JSON
```

返回的 JSON 结构（**这是今天的"标准答案"，要能默写**）：

```
{
  "id": "chatcmpl-xxxxxxxxxxxx",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "Qwen2.5-3B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "模型生成的回答"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 30, "completion_tokens": 80, "total_tokens": 110}
}
```

**逐字段解释（所有英文名词都别跳过，每个都能对号入座才叫"默写得了"）：**

| 字段 | 英文名词解释 | 大白话 |
| --- | --- | --- |
| `id` | **id = identifier（标识符）**，全称其实是 `chatcmpl-xxx` 里的 **chatcmpl = chat completion（聊天补全）** 的缩写 | 这次请求的"快递单号"，唯一编号，排查问题时用它定位是哪一次响应 |
| `object` | **object（对象）**，表示"这个返回体的类型" | 类型标签：程序靠它判断"我拿到的是个聊天结果"。值 `chat.completion` = **chat（聊天）+ completion（补全）**，连起来就是"一次聊天补全的结果" |
| `created` | **created（创建）**，值是 **Unix timestamp（Unix 时间戳）** = 从 1970-01-01 起经过的秒数（`1750000000` ≈ 2025 年），这是计算机世界的通用"时间写法" | 这次响应是什么时候生成的 |
| `model` | **model（模型）**，回显你请求里填的模型名 | 告诉你"这次到底是哪个模型回答的"，方便核对 |
| `choices` | **choices = choice 的复数（候选/选择）**，是个数组，OpenAI 支持一次返回多个候选答案（请求里 `n=2` 就给 2 条），我们只生成 1 个 | 一个装"候选回答"的抽屉，里面有几条答案 |
| `index` | **index（索引/序号）**，数组元素的下标，从 0 开始 | 这是第几个候选（0 = 第一个，也是唯一一个） |
| `message` | **message（消息）**，装一条完整的对话消息（角色 + 内容） | 回答本体，里面有 role 和 content |
| `role` | **role（角色）**，消息是谁说的。值是 `assistant` = **assistant（助手）**，另外还有 `user`（用户）、`system`（系统指令） | 这条消息是"助手说的话" |
| `content` | **content（内容）**，消息的具体文本 | 真正要的答案文字，调用方取 `choices[0].message.content` 就是它 |
| `finish_reason` | **finish = 结束 + reason = 原因**，生成是怎么停下来的 | 值是 `stop`（自然说完，撞上结束符）或 `length`（被 `max_tokens` 上限截断）——排查"回答不完整"先看它 |
| `usage` | **usage（用量）**，本次请求的 token 消耗账本 | OpenAI 靠它计费，本地模型靠它做性能统计 |
| `prompt_tokens` | **prompt（提示词）+ tokens（分词后的最小单位）**，输入消耗的 token 数 | 你发的消息（输入）花了多少 token |
| `completion_tokens` | **completion（补全）+ tokens**，输出消耗的 token 数 | 模型生成的回答花了多少 token |
| `total_tokens` | **total（总计）+ tokens** | 输入 + 输出一共花了多少 |

> **token 是什么**（Day 7 已接触，这里再对齐一次）：模型不按"字"处理文本，而是切成 **token（词元/令牌）** 这种更小的单位——中文里一个字大约 1~2 个 token，一句话就是一堆 token 的序列。所以"用量"按 token 数而不是字数算。

**一句话记忆**（面试默写用）：整个 JSON 的骨架 = **"快递面单"（`id` / `created` / `model`）+ 一个装回答的数组 `choices`（里面第 0 个的 `message.content` 就是要的东西）+ 一张"用量小票" `usage`。**

> 注意两个细节：① 接口函数用的是普通 `def` 而不是 `async def`——FastAPI 会自动把这种函数放到**线程池**里跑，不会卡住服务处理其他请求（模型生成是"慢操作"，这样设计更稳）；② 请求里带 `stream: true` 会返回 501"暂未实现"——**流式输出**（一个字一个字往外吐）是进阶功能，今天不实现，不丢人。

### 2.8 读完整段代码后，回答 3 个自测问题

1. 模型什么时候被加载？加载几次？→（启动时一次，lifespan 里）
2. `build_chatml` 解决什么问题？→（把 OpenAI 的 messages 结构翻译成 Qwen 认识的 ChatML 文本）
3. 返回的 `choices[0].message.content` 是什么？→（模型生成的回答文本）

### ✅ 第 2 步验收标准

通读 `local_api.py` 并能回答上面 3 个自测题 = 完成。**不要求默写代码，但要求"改得动"**：比如把默认温度改成 0.3、把 `max_tokens` 默认改成 300，你都知道改哪一行。

---

## ▶️ 第 3 步：启动服务（15~20 分钟）

### 3.1 启动命令

```
uvicorn local_api:app --host 127.0.0.1 --port 8000
```

拆解这条命令：

| 部分 | 含义 |
| --- | --- |
| `uvicorn` | ASGI 服务器程序 |
| `local_api:app` | 加载 `local_api.py` 这个文件里的 `app` 对象（`local_api` 是文件名，`app` 是 FastAPI() 实例） |
| `--host 127.0.0.1` | 只在本机监听（`0.0.0.0` 才是局域网可访问，先别开） |
| `--port 8000` | 端口号（端口 = 一台机器上区分不同服务的"门牌号"） |

> 如果提示 `uvicorn` 不是可用的命令，改用：`python -m uvicorn local_api:app --host 127.0.0.1 --port 8000`
> 如果提示端口被占用，换个端口：`--port 8001`（下面所有测试的 URL 都要跟着改端口）。

### 3.2 观察启动日志

跑起来后，终端会依次出现：

```
INFO:     Started server process [xxxx]
INFO:     Waiting for application startup.
[1/2] 加载分词器与模型（4bit 量化），首次约需 1~3 分钟……
Loading weights: 100%|██████...| 434/434 [00:06<00:00, 65.xx it/s]
[OK] 模型已就绪：Qwen2.5-3B-Instruct，显存占用约 1.9x GB
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**看到 `Uvicorn running on http://127.0.0.1:8000` 就说明服务启动了**，它会一直跑着（别关这个终端）。

> ⚠️ 两个提醒：
> - **千万别加 `--reload`**：`--reload` 会在你每次保存文件时重启服务，而重启 = 重新加载 3B 模型（1~3 分钟）。调试代码时可以忍，正常使用时不要加；
> - 这个终端会被服务"占住"，之后所有测试请在**新开的终端**里做（`conda activate llm` 再进 day9 文件夹）。

### ✅ 第 3 步验收标准

终端出现 `Uvicorn running on http://127.0.0.1:8000` + 日志里有"模型已就绪" = 完成。

---

## 🐚 第 4 步：curl 测试（20~30 分钟）

> 对应周计划 8/11 第 4 条："测试：终端 curl 调用接口返回标准 JSON"。curl 是命令行里最原始的 HTTP 工具，它通了，说明"服务本身没问题"——后面所有花哨的测试都建立在它之上。

### 4.1 测试前先做一件事：解决 Windows 中文乱码

```
chcp 65001
```

（PowerShell 默认编码可能让中文显示乱码，先切换成 UTF-8。）

### 4.2 测试 `/v1/models`（30 秒）

```
curl.exe -s http://127.0.0.1:8000/v1/models
```

看到返回：

```
{"object":"list","data":[{"id":"Qwen2.5-3B-Instruct","object":"model","created":1750xxxxxx,"owned_by":"local"}]}
```

### 4.3 测试 `/v1/chat/completions`（核心）

**关键坑：PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名**（一个行为完全不同的命令），所以**必须写 `curl.exe`** 才调用真正的 curl。

请求体我们提前写进了 `test_payload.json`（POST 请求的"正文"）：

```
curl.exe -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "@test_payload.json"
```

拆解：

| 参数 | 含义 |
| --- | --- |
| `-s` | 静默模式（不显示进度条，只显示返回内容） |
| `-H "Content-Type: application/json"` | 声明发送的是 JSON 数据 |
| `-d "@test_payload.json"` | 请求正文从文件读取（`@` 表示"读文件"）。**用文件传 JSON 比在命令行里手打长 JSON 可靠得多，Windows 引号坑少** |

预期返回（**以实际跑出为准**，这里是示例）：

```
{"id":"chatcmpl-9f3a1c2b4d5e6f7a8b9c0d1e","object":"chat.completion","created":1755xxxxxx,"model":"Qwen2.5-3B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"大模型的应用方向包括智能客服、内容生成、代码辅助、金融风控、医疗辅助等，覆盖各行各业……"},"finish_reason":"stop"}],"usage":{"prompt_tokens":45,"completion_tokens":66,"total_tokens":111}}
```

**看到 `"choices"` 里有 `"message":{"content": ...}` 就说明接口通了。** 这就是周计划验收标准里说的"返回 `{"choices":[...]}` 格式"。

### 4.4 看懂返回 JSON 的"骨架"

对照 2.7 节默写的结构，在真实返回里把 `id / object / created / model / choices / usage` 一个个圈出来——**能把真实输出和接口规范对上号，才是真懂**。

### 4.5 加练（可选）：改问题再发一次

把 `test_payload.json` 里的 `content` 改成别的问题（比如"请用一句话介绍 FastAPI"），保存后再跑一次上面的 curl 命令。观察：`prompt_tokens` / `completion_tokens` 变了没有？为什么？→（换了问题，输入和输出长度都变了。）

### ✅ 第 4 步验收标准

`curl.exe` 调通 `/v1/chat/completions`，返回 JSON 含 `choices[0].message.content` = 完成。

---

## 🐍 第 5 步：Python requests 测试（15~25 分钟）

> curl 适合"验活"，但真正写应用时你要用 Python 的 **requests** 库（HTTP 客户端库 = 让 Python 程序发 HTTP 请求的现成工具）。我们准备好了 `api_client_test.py`。

### 5.1 运行测试脚本

**新开一个终端**（服务那个终端别关），执行：

```
conda activate llm
cd "D:\Lan\研究生\技术学习\大模型算法\第二周\day9"
python api_client_test.py
```

### 5.2 这个脚本测了什么（对应输出看）

| 测试 | 测什么 | 对应今天哪件事 |
| --- | --- | --- |
| 健康检查 + /v1/models | 服务活着、模型就绪 | 第 3 步 |
| 普通中文对话 | 最基础的单轮对话 | OpenAI 格式核心 |
| RAG 资料问答（system 角色） | `messages` 里带 `system` 指令，**直接复用 Day 8 模板 09 的资料结构** | 接口化与 Prompt 工程的衔接 |
| 温度 0 vs 0.9 对比 | 同一个问题两种温度各跑一次，观察输出差异 | Day 6 采样知识在接口层的复现 |

预期输出形态（**以实际跑出为准**）：

```
健康检查： {'status': 'ok', 'model_ready': True}
可用模型： ['Qwen2.5-3B-Instruct']

============================================================
测试：普通中文对话
HTTP 状态码： 200
返回顶层字段： ['choices', 'created', 'id', 'model', 'object', 'usage']
model / object / id 前缀： Qwen2.5-3B-Instruct / chat.completion / chatcmpl-xxxx
[模型回答]
你好！我是 Qwen2.5-3B-Instruct，一个由阿里云开发的轻量级中文大语言模型……
[finish_reason] stop
[token 用量] {'prompt_tokens': 26, 'completion_tokens': 48, 'total_tokens': 74}
```

### 5.3 温度对比观察点（这是"复习 Day 6"的好机会）

看脚本里"温度=0"和"温度=0.9"两次输出的对比：

- 温度 0：输出基本是"最可能的那条路"，每次跑都一样；
- 温度 0.9：输出更"散"、措辞更多样，同一问题重跑可能不一样。

**数学视角**：温度就是给生成分布开方再归一化的参数（Day 6 学过），今天你发现它还能**隔着 HTTP 传**——`temperature` 从"脚本里的常量"变成了"网络请求里的参数"，这就是接口化的日常价值：**把模型的"性格"暴露给调用方调节**。

### ✅ 第 5 步验收标准

`python api_client_test.py` 全部测试跑通（HTTP 200 + 模型正常回答），能说出 RAG 测试里 system 消息起什么作用 = 完成。

---

## 🔄 第 6 步：LangChain 无缝切换测试（20~30 分钟）⭐ 核心验收

> 对应周计划 8/11 第 4 条："把 LangChain 里的模型换成'OpenAI 兼容端点'验证无缝切换。"这一步是"为什么接口化"最有力的实证。

### 6.1 先认识 LangChain

- **LangChain** = 大模型应用开发框架（Day 8 模板 01 里你让它总结过"LangChain 是什么"）。它把"调模型、拼提示词、检索、工具调用"封装成标准组件，写 RAG 应用最常用。
- 今天只用到它两个组件：`ChatOpenAI`（模型封装）和 `ChatPromptTemplate`（提示词模板）。
- **重点**：LangChain 官方模型对象 `ChatOpenAI` 默认连的是 OpenAI 官方。**它支持传 `base_url` 参数**——指向任意"OpenAI 兼容端点"。我们把它指向本地服务，就成了"无缝切换"。

### 6.2 运行切换测试

```
python langchain_switch_test.py
```

脚本里做了三件事：

1. `ChatOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", model="Qwen2.5-3B-Instruct")`——指向本地端点；
2. `ChatPromptTemplate` 拼「system + 资料 + 问题」（资料还是 Day 8 那两段）；
3. `chain = prompt | llm`（LangChain 的"管道"写法：提示词 → 模型），`chain.invoke({...})` 出结果。

预期输出形态（**以实际跑出为准**）：

```
正在调用本地 Qwen（通过 OpenAI 兼容端点）……
[模型回答]
4bit 量化能把显存占用降到原来的四分之一左右[资料§1]。在本地部署中，这意味着低显存设备也能加载大模型，降低了部署门槛[资料§2]。
[返回对象类型] AIMessage
```

### 6.3 核心体验：三处切换点

看脚本里的注释——**如果要切回 OpenAI 官方，只需要改三处**：

| 参数 | 本地 Qwen | OpenAI 官方 |
| --- | --- | --- |
| `base_url` | `http://127.0.0.1:8000/v1` | `https://api.openai.com/v1` |
| `api_key` | `"EMPTY"`（本地不校验） | `"sk-你的key"` |
| `model` | `"Qwen2.5-3B-Instruct"` | `"gpt-4o-mini"` |

**其余代码一行不动。** 这就是"OpenAI 兼容接口"的意义：**你的应用代码永远只写一遍，模型供应商随便换。**

> 数学类比：这就像函数接口 `f(x)` ——调用方只依赖"签名"（参数和返回类型），不依赖"实现"。换 `f` 的实现（Qwen / GPT / 别的），调用方无感知。**兼容接口 = 面向接口编程，而不是面向实现编程。**

### 6.4 思考题

LangChain 返回的 `AIMessage` 是什么？（→ LangChain 对"模型输出消息"的统一封装，`.content` 是文本内容。）为什么它和直接调 HTTP 拿到的 JSON 不一样？（→ LangChain 帮你解析好了，这是框架的价值：少写胶水代码。）

### ✅ 第 6 步验收标准

`langchain_switch_test.py` 跑通并输出回答；能说清"三处切换点"（base_url / api_key / model）= 完成。

---

## 🏃 零散时间任务（0.5~1 小时，穿插在休息时做）

> 按 day8 工作汇报的安排：力扣 **876、141**（链表进阶·快慢指针）＋ 统计八股：**偏差 / 方差**。

### A. 力扣刷题 2 道（链表进阶·快慢指针）

1. **876. 链表的中间结点**（简单，高频）：快慢指针——`slow` 每次走 1 步、`fast` 每次走 2 步，`fast` 到链表末尾时，`slow` 正好在中间。**数学视角**：这就是追及/路程问题——速度比 1:2，同时同起点出发，时间相同时路程比 1:2，慢指针走的路程 = 快指针的一半 = 链表总长的一半。偶数个节点时注意"取后一个中间点"的边界。
2. **141. 环形链表**（简单，高频）：快慢指针——如果链表有环，`fast` 终会在环里追上 `slow`；如果没环，`fast` 先走到 `null`。**数学视角**：追及问题的"相对速度"思想——进入环后，`fast` 相对 `slow` 每步只靠近 1 格（2−1=1），环长 L，最多 L 步必追上；而"最多走 O(n) 步"由抽屉原理保证（节点就 n 个）。这题还能引出 Day 计划里的 142 环形链表 II（求环入口，需要数学推导）。

做题方法不变：先自己想 10 分钟 → 看题解 → 用自己的话讲一遍 → 提交到 GitHub `leetcode` 仓库（按 `链表` 分类），题解注释写"数学视角"思路。

### B. 统计八股 10 分钟：偏差 / 方差（Bias–Variance）

> 新名词必带简介：
> - **偏差（Bias）** = 模型预测的平均值偏离真实值的程度——系统性的"偏"。偏差大 = 欠拟合（连训练数据都学不好）；
> - **方差（Variance）** = 换一批训练数据，模型的预测值波动多大——"不稳"。方差大 = 过拟合（把训练数据的噪声都背下来了）；
> - **偏差-方差权衡（Bias–Variance Tradeoff）** = 模型复杂度升高时，偏差下降、方差上升，两者此消彼长，总误差存在一个最优复杂度。

**核心公式（用平方误差期望做一次推导，这是你数学的主场）：**

设真实规律是 `y = f(x) + ε`（`ε` 是零均值噪声），模型 `f̂(x)` 由训练集 `D` 决定。把"对训练集随机性和噪声取期望"的平方误差展开：

```
E[(y − f̂)²] = E[(f + ε − f̂)²]
            = E[(f − f̂)²] + E[ε²]        （交叉项 E[(f − f̂)ε] = 0，因为 ε 与 f̂ 独立且 E[ε]=0）
            = E[(f − f̂)²] + σ²
```

再把 `E[(f − f̂)²]` 展开，中间加一项减一项 `E[f̂]`：

```
E[(f − f̂)²] = E[( (E[f̂] − f) − (f̂ − E[f̂]) )²]
            = (E[f̂] − f)² + E[(f̂ − E[f̂])²]
            = 偏差² + 方差
```

**合起来（面试直接写这个式子）：**

```
MSE = 偏差² + 方差 + 噪声σ²
```

- 偏差² = `(E[f̂] − f)²`：平均预测离真值多远；
- 方差 = `E[(f̂ − E[f̂])²]`：预测值围绕自己均值抖多厉害；
- σ²：数据本身的噪声，谁都降不了（不可约误差）。

**和模型复杂度的关系（画一条 U 形曲线）：**

- 模型太简单（欠拟合）：偏差大、方差小 → 误差大；
- 模型太复杂（过拟合）：偏差小、方差大 → 误差又变大；
- 中间有个"最优点"。**正则化（L2 惩罚、dropout、早停）就是在用"一点偏差"换"很多方差"**，把过拟合压回去。

**和大模型/深度学习的联系（面试必提）：**

- 大模型参数几十上百亿 → 天然**低偏差**（拟合能力强）但**高方差**（容易背噪声）→ 所以要喂海量数据把方差压下来；
- **RLHF / 对齐 / 指令微调** = 一种"正则化"，把模型输出约束到人类偏好这个子集（Day 7/8 的"压支持集"思想在训练层的版本）；
- 你 Day 7/8 实测的**思维链降低方差**：把一步到位的估计拆成多步，每步误差独立，合成估计的方差更小——这就是"大数定律 + 独立误差平均"在推理时的体现；
- **面试话术**："我在本地 3B 模型上做过实验，同样的问题温度高方差大、few-shot/思维链能显著降低输出方差——这让我从实证上理解了偏差-方差权衡在 LLM 里的表现：模型的能力（低偏差）要靠数据和方法（低方差）来配平。"

**3 个面试追问备好：**

1. 过拟合的本质是什么？→ 方差过大：模型把训练集噪声当规律学进去了，换数据就不稳；
2. 为什么模型复杂度升到一定程度测试误差会上升？→ 偏差降的边际收益 < 方差涨的边际成本，总误差 = 偏差² + 方差 + σ² 走出 U 形；
3. 训练集更大为什么能同时降偏差和方差？→ 更多数据让 `E[f̂]` 更接近 `f`（偏差）也让 `f̂` 在不同数据子集下更一致（方差）。

---

## 🌙 睡前 15 分钟：收尾 + 存档（git commit）

### 8.1 汇总今天的产出

今天产出都在 `第二周\day9\` 里：

- `day9-本地模型OpenAI兼容接口化详细教程.md`（本教程，无需改动）；
- `local_api.py`（**FastAPI 服务**：OpenAI 兼容接口，核心产出）；
- `api_client_test.py`（requests 测试客户端）；
- `langchain_switch_test.py`（LangChain 无缝切换测试）；
- `test_payload.json`（curl 测试用请求体）；
- 运行截图（建议存 `day9_curl测试截图.png` + `day9_requests测试截图.png` + `day9_LangChain切换截图.png` 放本文件夹）。

### 8.2 存档到 git

先停掉服务终端（Ctrl+C），再在仓库根目录执行：

```
cd "D:\Lan\研究生\技术学习\大模型算法"
git add .
git commit -m "Day9: 本地模型 OpenAI 兼容接口化（FastAPI + /v1/chat/completions）+ curl/requests/LangChain 三种方式调通"
```

> 提醒：`download\` 已被 `.gitignore` 排除（模型不进 git）；`__pycache__/` 也不进 git。

---

## 🚧 常见问题速查表（出问题先看这里）

| 现象 | 原因 | 解决办法 |
| --- | --- | --- |
| `curl` 命令报错 / 行为奇怪 | PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名 | 一律用 `curl.exe`；或先 `chcp 65001` 切 UTF-8 |
| 提示 `uvicorn` 找不到 | 没在 (llm) 环境，或 PATH 没配 | `python -m uvicorn local_api:app --host 127.0.0.1 --port 8000` |
| 端口被占用 | 8000 被别的程序占了 | 换端口 `--port 8001`，所有测试 URL 同步改 |
| 请求返回 503 "模型还没加载完成" | 请求发得太早，模型还在加载 | 等日志出现 `Uvicorn running on ...` 和 `模型已就绪` 再发请求 |
| 请求很慢（20s~1min） | 3B 模型推理本来就要时间，第一次最慢 | 正常现象；把 `max_tokens` 调小可加快 |
| 中文乱码 | Windows 控制台编码问题 | 先 `chcp 65001` 再测试；或把输出重定向到文件看 |
| LangChain 报 `langchain_openai` 找不到 | 没装适配包 | `pip install langchain-openai` |
| LangChain 报 `openai` 相关错误 | openai 版本不对 | `pip install -U "openai>=1.0"` |
| 改代码后接口没变化 | 服务是旧代码，没重启 | Ctrl+C 停服务重新 `uvicorn`（不要用 --reload，会反复重载模型） |
| 显存不足 OOM | 开了太多占显存程序 | 关浏览器等；Ctrl+C 停服务释放显存再重启 |
| 启动报 `torch_dtype is deprecated` | 新版 transformers 提示 | 无害，可忽略（想消除可把 `torch_dtype` 改成 `dtype`） |
| 返回 JSON 里混着奇怪文字 | 用了 `print()` 往响应里塞东西 | 检查接口函数：返回的必须是 dict，不是 print 的结果 |
| 局域网其他电脑连不上 | 绑定了 `127.0.0.1`（仅本机） | 需要开放时用 `--host 0.0.0.0`（先别开，注意安全） |
| 服务一直起不来且报模型路径错 | `MODEL_PATH` 与本地实际路径不符 | 打开 `local_api.py` 确认第 1 个常量，改成你模型的真实路径 |

---

## ✅ 今日验收清单（完成一项打一个勾）

**主线：**
- [ ] 能讲清 3 个名词：OpenAI 兼容 API / FastAPI / uvicorn，各配一句大白话
- [ ] 能画出"一条请求的完整旅程"（curl → uvicorn → FastAPI → 模型 → 返回）
- [ ] 通读 `local_api.py`，能回答 3 个自测题（模型何时加载 / build_chatml 干什么 / choices 里是什么）
- [ ] 服务启动成功（日志出现 `Uvicorn running` + 模型已就绪）
- [ ] `curl.exe` 调通 `/v1/models` 和 `/v1/chat/completions`，返回 JSON 含 `choices`
- [ ] `python api_client_test.py` 跑通（中文对话 + RAG 资料问答 + 温度对比）
- [ ] `python langchain_switch_test.py` 跑通，能说清"三处切换点"
- [ ] 能讲清"为什么接口化"（前后端分离 / 统一调用 / 语言无关 / 为 RAG 打底）

**零散时间：**
- [ ] 力扣 876（链表的中间结点）、141（环形链表）完成并讲出快慢指针的数学视角
- [ ] 统计八股：偏差 / 方差定义、MSE = 偏差² + 方差 + 噪声的推导、与模型复杂度的 U 形关系能讲清

**睡前：**
- [ ] git commit 存档成功

**全部打勾 = Day 9 圆满结束。你现在有一个"对外营业"的本地模型服务：curl / requests / LangChain / OpenAI SDK 四种方式都能调它——这就是 day10 RAG 全链路的"生成层"，也是你简历上"本地大模型部署 + 服务化"最完整的一块拼图。** 🎉

---

## 📎 附录 A：local_api.py 怎么读（给想看懂代码的你）

结构总览（自上而下 6 块）：

1. **文件头注释**：五件套，说明"这个脚本在机器学习的哪一环"；
2. **常量区**：`MODEL_PATH`（模型位置）/ `MODEL_NAME`（对外名字）/ 默认生成参数——**日常改代码基本只改这里**；
3. **`build_chatml()`**：OpenAI 的 messages 结构 → Qwen 的 ChatML 文本（翻译官）；
4. **`generate()`**：真正的生成函数，返回文本 + 结束原因 + token 用量；
5. **pydantic 模型**（`ChatMessage` / `ChatRequest`）：定义接口的"函数签名"，FastAPI 自动校验；
6. **lifespan + 三个接口**：生命周期加载模型，`/health`、`/v1/models`、`/v1/chat/completions` 三个入口。

**想改默认温度/长度**：改常量区 `DEFAULT_TEMPERATURE` / `DEFAULT_MAX_TOKENS`。
**想加接口**：模仿 `/v1/models` 再写一个函数 + 装饰器即可。
**想加模型**：`list_models()` 的 `data` 列表里再加一个 id。

---

## 📎 附录 B：三种测试脚本的分工

| 脚本 | 用什么发请求 | 一句话定位 |
| --- | --- | --- |
| `api_client_test.py` | requests 库 | 验证接口 + 留 3 组测试样例（对话 / RAG / 温度对比） |
| `langchain_switch_test.py` | LangChain 框架 | 验证"无缝切换"——框架级对接 |

> 提醒：教程里所有"预期输出"都是**示例**，**以你实际跑出的结果为准**——跑出来不一样先别慌，看是不是方向性差异（比如 JSON 结构是否规范、回答是否合理），把截图发我一起看。
