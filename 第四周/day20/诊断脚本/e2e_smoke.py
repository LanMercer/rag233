# -*- coding: utf-8 -*-
"""临时端到端冒烟（用完即删）：真起 Gradio 服务 → 真发 HTTP 请求，验证：
  ① launch 是否真的绑 0.0.0.0（平台反代才连得上）
  ② 不上传 PDF 直接提问，回答是否走内置语料、出处编号是否唯一且非空
  ③ outputs 元数（4 个）与 State 回写是否被 Gradio 接受（元数不对会在这里报错）
  ④ 连问两次，第二次是否更快（复用进程内缓存的内置库）
注意：API_KEY 用假值 —— 生成层必然失败（返回 [生成失败]），但**检索层与出处是真实结果**，
      本脚本要验的正是检索层 + 界面接线，不烧额度。
"""
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPACE = r"D:\Lan\研究生\技术学习\大模型算法\第四周\发布包\space_demo"
sys.path.insert(0, SPACE)
os.environ["API_KEY"] = "dummy-for-smoke"      # 故意用假 key，只验检索层
os.environ["API_BASE_URL"] = "http://127.0.0.1:9/v1/chat/completions"  # 必然连不上，快速失败

import app  # noqa: E402

# 模仿 __main__ 的预热线程（本脚本是 import 起服务，不走 __main__）
import threading  # noqa: E402
threading.Thread(target=app.build_default_store, daemon=True, name="warmup").start()

ui = app.build_ui()
ui.launch(server_name="0.0.0.0", server_port=7861, prevent_thread_lock=True,
          show_api=False, quiet=True)
time.sleep(2)

# 等预热线程把内置库建好，并记录耗时（这决定"第一个访客"要不要等）
import time as _t  # noqa: E402
_t0 = _t.time()
while app._DEFAULT_STORE is None and _t.time() - _t0 < 180:
    _t.sleep(1)
print(f"内置库预热：{'已完成' if app._DEFAULT_STORE is not None else '超时未完成'}"
      f"，耗时 {_t.time() - _t0:.1f}s（预热期间端口已可用）")

# 端口可用性 + 绑定地址（平台反代能不能连进来的判据）
import subprocess  # noqa: E402
print(subprocess.run(["powershell", "-NoProfile", "-Command",
                      "Get-NetTCPConnection -LocalPort 7861 -State Listen | "
                      "Select-Object -ExpandProperty LocalAddress"],
                     capture_output=True, text=True).stdout.strip())

from gradio_client import Client  # noqa: E402

c = Client("http://127.0.0.1:7861", verbose=False)
print("可用接口 :", sorted(str(d.get("api_name")) for d in c.view_api(return_format="dict")["named_endpoints"].values()))

q = "论文的实验在哪个数据集上进行？"
for i in (1, 2):
    t0 = time.time()
    out = c.predict(question=q, api_name="/answer_question")
    dt = time.time() - t0
    print(f"\n--- 第 {i} 次提问（{dt:.1f}s）--- 返回共 {len(out)} 项")
    for j, v in enumerate(out):
        print(f"   out[{j}] = {v!r}"[:260])
    ans, srcs, snips = out[0], out[1], out[2]
    cids = [s for s in str(srcs).replace("[资料§", "").replace("]", "").split("、") if s]
    print("出处是否唯一:", len(cids) == len(set(cids)), "| 个数:", len(cids),
          "| 召回条数:", str(snips).count("**["))

print("\n提示：生成层报 [生成失败] 属预期（假 key + 假地址），检索层与出处为真实结果。")
ui.close()
