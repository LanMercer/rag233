# -*- coding: utf-8 -*-
"""
本地 OpenAI 兼容接口测试客户端（配合 Day9 教程第 5 步使用）

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：远端已启动的本地 Qwen2.5-3B-Instruct（由 local_api.py 提供服务）
- 数据   ：几个手工编写的测试请求（问候对话 / RAG 资料问答 / 温度对比）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    ① GET /health 和 /v1/models 确认服务与模型就绪；
    ② POST /v1/chat/completions 依次发送多个测试请求；
    ③ 打印每个请求的返回内容，验证返回 JSON 符合 OpenAI 标准格式
       （顶层 id / object / choices[0].message.content / usage）。

用法：
    1) 先启动服务（另一个终端）：
       uvicorn local_api:app --host 127.0.0.1 --port 8000
    2) 再在本终端执行：
       python api_client_test.py
    如果改了端口，修改下面的 BASE_URL。
"""

import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def post_chat(messages, temperature=0.7, top_p=0.9, max_tokens=200, label=""):
    """发一次 /v1/chat/completions 请求并打印结果。"""
    payload = {
        "model": "Qwen2.5-3B-Instruct",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    print("\n" + "=" * 60)
    print(f"测试：{label}")
    print("请求 messages 前 120 字：", json.dumps(messages, ensure_ascii=False)[:120], "……")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=600)
    print("HTTP 状态码：", resp.status_code)

    if resp.status_code != 200:
        print("返回内容：", resp.text[:500])
        return None

    data = resp.json()
    print("返回顶层字段：", sorted(data.keys()))
    print("model / object / id 前缀：", data["model"], "/", data["object"], "/", data["id"][:12])
    print("[模型回答]")
    print(data["choices"][0]["message"]["content"])
    print("[finish_reason]", data["choices"][0]["finish_reason"])
    print("[token 用量]", data["usage"])
    return data


def main():
    # 0) 先检查服务是否活着
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print("健康检查：", r.json())
        r2 = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        print("可用模型：", [m["id"] for m in r2.json()["data"]])
    except Exception as e:
        print(f"[X] 连不上服务（{BASE_URL}）：{e}")
        print("[X] 请先启动 local_api.py 服务（uvicorn local_api:app --host 127.0.0.1 --port 8000），再运行本脚本。")
        return

    # 1) 普通中文对话
    post_chat(
        [{"role": "user", "content": "你好！请用一句话介绍你自己。"}],
        label="普通中文对话",
    )

    # 2) 带 system 角色的 RAG 资料问答（复用 Day8 模板 09 的资料结构）
    post_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是一名严谨的资料问答助手。只依据下面资料回答；"
                    "资料里没有就说「资料中没有提到」，不要编造。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "【资料】\n"
                    "[1] 4bit 量化把模型权重从16位压缩到4位，显存占用大约降到原来的四分之一。\n"
                    "[2] Qwen2.5-3B-Instruct 用 4bit 加载后，在 6GB 显存的笔记本上实测占用约 1.92GB。\n"
                    "[3] 推理时 temperature 控制输出的随机程度，top_p 控制候选词的范围。\n\n"
                    "【问题】\n4bit 量化大概能省多少显存？"
                ),
            },
        ],
        temperature=0.3,
        label="RAG 资料问答（system 角色）",
    )

    # 3) 温度对比：同一个问题，temperature=0 vs 0.9
    msg = [{"role": "user", "content": "用一句话夸夸数学这个学科。"}]
    post_chat(msg, temperature=0.0, label="温度=0（接近确定）")
    post_chat(msg, temperature=0.9, label="温度=0.9（更发散）")


if __name__ == "__main__":
    main()
