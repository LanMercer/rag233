# -*- coding: utf-8 -*-
"""
LangChain 无缝切换测试（配合 Day9 教程第 6 步使用）

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：远端已启动的本地 Qwen2.5-3B-Instruct（OpenAI 兼容端点 http://127.0.0.1:8000/v1）
- 数据   ：一条手工编写的「资料 + 问题」（检验 LangChain 提示词模板 + 接口对接）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    ① 用 langchain_openai 的 ChatOpenAI 指向本地 OpenAI 兼容端点；
    ② 用 ChatPromptTemplate 拼「system + 资料 + 问题」→ chain = prompt | llm；
    ③ 打印回答，验证：LangChain 应用只需把 base_url / api_key / model 三处改掉，
       就能无缝切换供应商（OpenAI 官方 <-> 本地 Qwen）。

用法：
    1) 先启动服务（另一个终端）：
       uvicorn local_api:app --host 127.0.0.1 --port 8000
    2) 再在本终端执行：
       python langchain_switch_test.py
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 指向本地 OpenAI 兼容端点（三处切换点：base_url / api_key / model）
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",  # 本地服务不校验 key，填个占位即可
    model="Qwen2.5-3B-Instruct",
    temperature=0.3,
)

# 如果要切回 OpenAI 官方（需有 API key），只需把上面三处改成：
#   base_url = "https://api.openai.com/v1"
#   api_key  = "sk-你的key"
#   model    = "gpt-4o-mini"
# 其余代码一行都不用动——这就是"OpenAI 兼容接口"的意义。

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名严谨的资料问答助手。只依据资料回答，没有就说不知道，不要编造。"),
        ("human", "【资料】\n{context}\n\n【问题】\n{question}"),
    ]
)

chain = prompt | llm

context = (
    "[1] 4bit 量化把模型权重从 16 位压缩到 4 位，显存占用大约降到原来的四分之一。\n"
    "[2] Qwen2.5-3B-Instruct 用 4bit 加载后，在 6GB 显存的笔记本上实测占用约 1.92GB。"
)
question = "4bit 量化大概能省多少显存？它在本地部署里有什么意义？"

print("正在调用本地 Qwen（通过 OpenAI 兼容端点）……")
answer = chain.invoke({"context": context, "question": question})
print("[模型回答]")
print(answer.content)
print("\n[返回对象类型]", type(answer).__name__)
