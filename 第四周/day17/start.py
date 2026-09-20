# -*- coding: utf-8 -*-
r"""
第四周 · 一键启动脚本（Day17 雏形）⭐ D 线交付物 D1

五件套说明（这个脚本在机器学习的哪一环）：
- 模型   ：**本脚本自己不加载任何模型**。它只是"启动器"——按需拉起
             ① 模型服务（8000 端口，local_api_lora.py / local_api.py）
             ② Web 界面（7860 端口，**第四周\day18\app.py**，Day18 完成版）
           为什么要拆成两个进程？本机 6G 显存一次只能跑一个 3B（约 1.9~2.0GB），
           界面必须是"纯 HTTP 客户端"才腾得出显存；想换模型只换服务、界面不用重启。
- 数据   ：第三周\day13\chroma_db（向量库）、第三周\day15\config.json（profile/embed/persist）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（工程/交付环节）：
           ① 体检：文件在不在 / 依赖装没装 / 8000 与 7860 端口谁在跑 / 服务身份对不对
           ② （可选 --start-service）开新窗口把模型服务起起来，并等 /health 的 model_ready=True
           ③ 拉起 Web 界面（app.py）
    ↑ 对应《第四周详细计划》D2（day17）第 5 条："`start.bat` / `start.py` 雏形"。
      Day18 会把它接到"上传 PDF → 建库 → 提问"的完整界面上，这里先做最小可用版。
      ✅ Day18 已接上：`APP_PATH` 现指向 `第四周\day18\app.py`（完成版；默认库开箱即用）。

用法（llm 环境；本文件所在目录下执行）：
    python start.py --check              # 只体检，不启动任何东西（最快，建议先跑这个）
    python start.py                      # 体检后直接拉起 Web 界面（模型服务需你自己在另一个终端起）
    python start.py --start-service      # 体检 → 自动起模型服务（新窗口）→ 等就绪 → 拉起界面
    python start.py --service original   # 换用原模型服务（第二周\day9\local_api.py）
    python start.py --no-launch          # 只做体检 + （可选）起服务，不拉起界面

退出方式：界面窗口里 Ctrl+C；模型服务窗口里 Ctrl+C。
    （注意：直接关窗口 ≠ 停进程，可能变孤儿进程占端口，见日教程「附录 D」）
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time

# 终端编码加固：Windows 控制台默认 GBK，中文/emoji 输出会崩
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 第 1 区：路径常量（相对本脚本定位，换机器也能跑）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))              # 第四周\day17
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))    # 仓库根

CONFIG_PATH = os.path.join(REPO_DIR, "第三周", "day15", "config.json")
APP_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "day18", "app.py"))   # 本地版界面（Day18 完成版；Day16 骨架已由它取代）
SERVICE_SCRIPTS = {
    "lora": os.path.join(REPO_DIR, "第三周", "day13", "local_api_lora.py"),
    "original": os.path.join(REPO_DIR, "第二周", "day9", "local_api.py"),
}

SERVICE_PORT = 8000    # 模型服务：OpenAI 兼容接口
UI_PORT = 7860         # Gradio Web 界面


# ---------------------------------------------------------------------------
# 第 2 区：体检小工具
# ---------------------------------------------------------------------------

def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")


def port_listening(port, host="127.0.0.1", timeout=1.0):
    """端口上有没有人在监听（有 = 服务可能已在跑）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def http_json(url, timeout=5):
    """GET 一个 JSON 接口，失败返回 None（体检不该因为服务没起就崩）。"""
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def read_config():
    """读 day15 config.json（profile / embed / persist 的唯一真相）。"""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        bad(f"读不到配置文件 {CONFIG_PATH}（{e}）")
        return None


def api_base(cfg, profile_name):
    """从 profile 的 api_url 截出服务根地址（拼 /health、/v1/models 用）。"""
    url = cfg["profiles"][profile_name]["api_url"]
    return url.rsplit("/v1/chat/completions", 1)[0]


# ---------------------------------------------------------------------------
# 第 3 区：四项体检
# ---------------------------------------------------------------------------

def check_files(cfg):
    """① 文件体检：向量库 / 界面脚本 / 服务脚本 / 评测题。"""
    print("\n[1/4] 文件体检")
    all_ok = True

    persist_rel = cfg["persist_dir"]
    cfg_dir = os.path.dirname(CONFIG_PATH)
    persist_dir = persist_rel if os.path.isabs(persist_rel) else os.path.normpath(os.path.join(cfg_dir, persist_rel))
    if os.path.isdir(persist_dir):
        ok(f"向量库：{persist_dir}")
    else:
        bad(f"向量库不存在：{persist_dir} → 到 第三周\\day13 跑 python build_index.py 复建")
        all_ok = False

    if os.path.exists(APP_PATH):
        ok(f"Web 界面：{APP_PATH}")
    else:
        bad(f"找不到界面脚本：{APP_PATH}")
        all_ok = False

    for name, path in SERVICE_SCRIPTS.items():
        if os.path.exists(path):
            ok(f"服务脚本（{name}）：{path}")
        else:
            warn(f"缺服务脚本（{name}）：{path}")

    return all_ok


def check_deps():
    """② 依赖体检：缺哪个人话提示装哪个。"""
    print("\n[2/4] 依赖体检")
    missing = []
    for mod, pip_name in [("requests", "requests"), ("gradio", "gradio"),
                          ("langchain_chroma", "langchain-chroma"),
                          ("langchain_huggingface", "langchain-huggingface")]:
        try:
            __import__(mod)
            ok(f"import {mod}")
        except Exception as e:
            bad(f"import {mod} 失败：{e} → pip install {pip_name}")
            missing.append(pip_name)
    return not missing


def check_ports(cfg, profile_name):
    """③ 端口体检：8000（模型服务）与 7860（界面）谁在。"""
    print("\n[3/4] 端口体检")
    service_up = port_listening(SERVICE_PORT)
    ui_up = port_listening(UI_PORT)
    if service_up:
        ok(f"{SERVICE_PORT} 端口有人在监听（模型服务已在跑）")
    else:
        warn(f"{SERVICE_PORT} 端口空着（模型服务没起）→ 界面能打开，但提问时会提示「连不上模型服务」")
    if ui_up:
        warn(f"{UI_PORT} 端口已被占用（可能上一个 app.py 还在跑）→ 见教程「附录 D」按端口找 PID 关掉")
    else:
        ok(f"{UI_PORT} 端口空闲（界面可以起）")
    return service_up


def check_service_identity(cfg, profile_name):
    """④ 服务身份体检：服务活着 ≠ 服务是我们想要的那个模型。"""
    print("\n[4/4] 服务身份体检")
    base = api_base(cfg, profile_name)
    health = http_json(base + "/health")
    if health is None:
        warn(f"读不到 {base}/health（服务没起或还没加载完）")
        return False
    if health.get("model_ready") is not True:
        warn("服务活着但模型还没加载完（model_ready=False）→ 再等等")
        return False
    models = http_json(base + "/v1/models")
    served = None
    try:
        served = models["data"][0]["id"]
    except Exception:
        pass
    expect = cfg["profiles"][profile_name]["model_name"]
    if served == expect:
        ok(f"服务身份正确：{served}（与 --service {profile_name} 一致）")
        return True
    warn(f"服务端模型 = {served}，但你选的是 {profile_name}（期望 {expect}）→ 换服务或改 --service")
    return False


# ---------------------------------------------------------------------------
# 第 4 区：（可选）起模型服务 + 等就绪
# ---------------------------------------------------------------------------

def start_service(profile_name):
    """开一个新控制台窗口启动模型服务（不阻塞本脚本）。"""
    script = SERVICE_SCRIPTS.get(profile_name)
    if not script or not os.path.exists(script):
        bad(f"找不到 {profile_name} 的服务脚本：{script}")
        return False
    workdir = os.path.dirname(script)
    mod = os.path.splitext(os.path.basename(script))[0]
    cmd = [sys.executable, "-m", "uvicorn", f"{mod}:app", "--host", "127.0.0.1", "--port", str(SERVICE_PORT)]
    print(f"\n[启动服务] 新窗口运行：{' '.join(cmd)}")
    print(f"           工作目录：{workdir}")
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)   # Windows：单开一个窗口，方便 Ctrl+C
    try:
        subprocess.Popen(cmd, cwd=workdir, creationflags=flags)
        ok("已在新窗口启动模型服务（加载 4bit 3B 约需 1~3 分钟，请耐心等）")
        return True
    except Exception as e:
        bad(f"启动服务失败：{e}")
        return False


def wait_service_ready(cfg, profile_name, timeout=300):
    """轮询 /health，等 model_ready=True（默认最多等 5 分钟）。"""
    base = api_base(cfg, profile_name)
    print(f"\n[等服务就绪] 最多等 {timeout}s：{base}/health")
    t0 = time.time()
    while time.time() - t0 < timeout:
        health = http_json(base + "/health", timeout=5)
        if health and health.get("model_ready") is True:
            ok(f"模型已就绪（等了 {time.time() - t0:.0f}s）")
            return True
        print(f"    ...等待中（{time.time() - t0:.0f}s）", end="\r")
        time.sleep(5)
    print()
    warn("等待超时：服务可能还在加载，或启动失败（去看那个新窗口的报错）")
    return False


# ---------------------------------------------------------------------------
# 第 5 区：主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="第四周一键启动（Day17 雏形）：体检 → 起服务 → 拉起 Web 界面",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--service", default="lora", choices=list(SERVICE_SCRIPTS),
                        help="用哪条生成链路：lora（微调，默认）| original（原模型）")
    parser.add_argument("--check", action="store_true", help="只体检，不启动任何东西")
    parser.add_argument("--start-service", action="store_true", help="自动在新窗口起模型服务，并等就绪")
    parser.add_argument("--no-launch", action="store_true", help="体检（+起服务）后不拉起 Web 界面")
    args = parser.parse_args()

    print("=" * 72)
    print("第四周一键启动（Day17 雏形）｜体检 → （可选）起服务 → 拉起 Web 界面")
    print("=" * 72)

    cfg = read_config()
    if cfg is None:
        sys.exit(1)

    files_ok = check_files(cfg)
    deps_ok = check_deps()
    service_up = check_ports(cfg, args.service)
    ready = check_service_identity(cfg, args.service) if service_up else False

    print("\n" + "=" * 72)
    print("体检结论")
    print("=" * 72)
    print(f"  文件：{'通过' if files_ok else '有缺失（见上）'}｜依赖：{'通过' if deps_ok else '有缺失（见上）'}"
          f"｜模型服务：{'已就绪' if ready else ('在跑但未就绪' if service_up else '未启动')}")

    if args.check:
        print("\n[--check] 只体检，不启动任何东西。去掉 --check 即可启动。")
        return

    if not files_ok or not deps_ok:
        print("\n先解决上面的 FAIL 项，再重新运行本脚本。")
        sys.exit(1)

    if not service_up and args.start_service:
        if start_service(args.service):
            ready = wait_service_ready(cfg, args.service)
    if not ready and not args.start_service:
        print("\n提示：模型服务没起（或身份不对）。两种做法——")
        print("  A) 本脚本自动起：python start.py --start-service")
        print(f"  B) 自己另开终端起：cd \"{os.path.dirname(SERVICE_SCRIPTS[args.service])}\"")
        print(f"     python -m uvicorn {os.path.splitext(os.path.basename(SERVICE_SCRIPTS[args.service]))[0]}:app "
              f"--host 127.0.0.1 --port {SERVICE_PORT}")
        print("  （界面仍会启动，只是提问时会给你一句「连不上模型服务」的人话提示）")

    if args.no_launch:
        print("\n[--no-launch] 不拉起界面。")
        return

    print("\n" + "=" * 72)
    print(f"[拉起界面] {APP_PATH}")
    print("  浏览器打开后：选模型 → 输入问题 → 看带 [资料§N] 出处的答案")
    print("  停止：在本窗口 Ctrl+C（不要把窗口直接关掉）")
    print("=" * 72)
    try:
        subprocess.call([sys.executable, APP_PATH], cwd=os.path.dirname(APP_PATH))
    except KeyboardInterrupt:
        print("\n[已停止] 界面退出。")


if __name__ == "__main__":
    main()
