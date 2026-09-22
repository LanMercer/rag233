# -*- coding: utf-8 -*-
"""临时端到端冒烟 2（用完即删）：默认内置库 + 上传 PDF 覆盖 两条路径，都走真 HTTP。

验证点：
  ① 预热线程建好内置库后，首问是否快（不预热时约 31s）
  ② 默认路径：出处编号唯一、非空，回答带"本次使用内置语料"前缀
  ③ 上传 PDF：建库成功 → 提问的召回片段真的来自新 PDF（含 Transformer/attention 字样）
  ④ 上传不会破坏共享的内置库（105 段仍在）
  ⑤ 多问几轮的稳定性（内存库那个坑是间歇性的，只跑一次不算数）
"""
import io
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPACE = r"D:\Lan\研究生\技术学习\大模型算法\第四周\发布包\space_demo"
PDF = r"D:\Lan\研究生\技术学习\大模型算法\第一周\day2\attention is all you need_en.pdf"
sys.path.insert(0, SPACE)
os.environ["API_KEY"] = "dummy-for-smoke"
os.environ["API_BASE_URL"] = "http://127.0.0.1:9/v1/chat/completions"

import threading  # noqa: E402
import app  # noqa: E402

threading.Thread(target=app.build_default_store, daemon=True, name="warmup").start()
ui = app.build_ui()
ui.launch(server_name="0.0.0.0", server_port=7861, prevent_thread_lock=True,
          show_api=False, quiet=True)
time.sleep(2)

t0 = time.time()
while app._DEFAULT_STORE is None and time.time() - t0 < 240:
    time.sleep(1)
print(f"预热：{'完成' if app._DEFAULT_STORE else '超时'}，耗时 {time.time()-t0:.1f}s（端口已可用）")

from gradio_client import Client, handle_file  # noqa: E402
c = Client("http://127.0.0.1:7861", verbose=False)


def ask(q, tag):
    t = time.time()
    out = c.predict(question=q, api_name="/answer_question")
    ans, srcs, snips = out[0], out[1], out[2]
    cids = [s for s in str(srcs).replace("[资料§", "").replace("]", "").split("、") if s]
    print(f"   {tag}（{time.time()-t:.1f}s）出处={srcs} 唯一={len(cids)==len(set(cids))} "
          f"召回={str(snips).count('**[')}")
    return ans, srcs, snips


print("\n① 默认内置库路径")
a1, s1, sn1 = ask("论文的实验在哪个数据集上进行？", "Q1")
print("   回答前缀正确 :", str(a1).startswith("（本次使用内置示例语料"))
print("   召回含 GMR 内容 :", "gmr" in str(sn1).lower() or "humanoid" in str(sn1).lower())

print("\n⑤ 连续多轮（验间歇性跨线程问题）")
ok = 0
for i in range(5):
    a, s, sn = ask("GMR 的英文全称是什么？", f"Q{i+2}")
    ok += bool(s) and "[" in str(sn)
print(f"   成功 {ok}/5")

print("\n③ 上传 PDF 覆盖内置库")
res = c.predict(handle_file(PDF), 4, api_name="/_build")
print("   建库消息 :", str(res[0])[:120])
a2, s2, sn2 = ask("论文提出了什么模型结构？", "Q6(上传后)")
low = str(sn2).lower()
print("   召回真的换了库（含 transformer/attention）:", ("transformer" in low or "attention" in low))

print("   共享内置库是否还在（应 105 段）:",
      app._DEFAULT_STORE[0]._collection.count() if app._DEFAULT_STORE else "无")

print("\n④ 再连问两轮，确认稳定走 PDF 库")
for i in range(2):
    ask("位置编码的作用是什么？", f"Q{7+i}")

ui.close()
