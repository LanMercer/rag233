# -*- coding: utf-8 -*-
r"""
check_release.py —— 发布前自动检查（第四周 Day20 第 5 步 · A/B 上线前的"防翻车"闸门）
================================================================================
五件套说明（本脚本在机器学习的哪一环）：
- 模型   ：无（纯静态检查，不加载模型、不连网、不调 API）
- 数据   ：读 `第四周\发布包\` 下的全部文件 + `第四周\day20\定稿数字表.md`（若存在）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（发布/交付环节）：
    把 `发布包说明.md` 第六节那张"手工检查清单"变成**可执行的闸门**——六项机器可检的检查：
        ① 体积：发布包总体积是否超预算（默认 20 MB；超了多半是"把不该进的东西带进去了"）
        ② 禁入项：权重 / adapter / 训练日志 / 向量库 / 缓存等**不该外发**的文件
        ③ 密钥：代码与文档里有没有硬编码的 `sk-...` / `hf_...` / `api_key = "..."`（⭐ 推上去就是事故）
        ④ 依赖与上线自举：`space_demo\requirements.txt` 是否有 `pydantic==2.10.6`（gradio 4.44 的兼容坑）
                与 `starlette==0.46.2`（gradio 4.44 的另一个兼容坑）；
                `space_demo\app.py` 的 `launch()` 是否显式绑 `0.0.0.0`（绑 127.0.0.1 会让平台反代连不上）
        ④c 检索口径与建库自举：内置语料 `chunks.json` 是否 105 段且 id 连续；每处 `Chroma.from_documents`
                是否都给了唯一 `collection_name`（否则重复建库会追加 → 出处重复）；是否显式给了
                `hnsw:search_ef`（默认 10 会让 Top-K 与离线评测对不上）；是否落盘而不是内存库
                （内存库跨线程会报 `no such table: collections`）；有没有 `if progress:` 这种判空
                （gradio 4.44 会 IndexError → 首问 500）
        ⑤ 标注：三处（`app.py` / `space_demo\README.md` / `blog\项目展示页.md`）是否都写了
                "生成层 = DeepSeek API 且 ≠ 本项目微调模型"——**本周硬性规定**，漏一处就是误导
        ⑥ 占位符与数字：文章里还有没有 `__`（说明定稿数字没回填）；文章里出现的数字是否**都能在
                `定稿数字表.md` 里找到**（对不上的列出来给人看——不自动判错，因为字节数/条数之类本来就不在表里）

★ 为什么要有这个脚本？（面试可讲的"工程纪律"）
    发布是**不可逆**的：博客推上去、创空间一公开，**密钥泄露 / 标错口径 / 数字与报告不一致 / 上线起不来**
    都是"事后解释"型错误。人眼在"复制粘贴一小时后"一定会漏——所以把**能机器判的每一条都交给脚本**，
    人只负责机器判不了的部分（文案、截图、链接可用性）。
    这与项目里"数字只从 run_log.txt 抄""服务身份检查""引文白名单"是同一套思路：
    **凡是会被人手搬错的东西，就让它只有一条机器通道。**

用法（任意环境可跑，不需要 llm 环境；本文件所在目录下执行即可）：
    python check_release.py                    # 全量检查（有 FAIL 则退出码 1）
    python check_release.py --budget-mb 25     # 换体积预算
    python check_release.py --root ..\发布包    # 显式指定发布包目录（默认已指向 第四周\发布包）
================================================================================
"""

import argparse
import io
import json
import os
import re
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                # 第四周\day20
WEEK4_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))           # 第四周
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))      # 仓库根
DEFAULT_ROOT = os.path.join(WEEK4_DIR, "发布包")
FINALS_TABLE = os.path.join(SCRIPT_DIR, "定稿数字表.md")

# ---------------------------------------------------------------------------
# 规则表（想加检查项就在这里加一行，别的不用动）
# ---------------------------------------------------------------------------
BANNED_NAME_PATTERNS = [
    (r".*\.safetensors$", "模型/adapter 权重（几 MB ~ 几 GB）"),
    (r".*\.(bin|pt|pth|ckpt|gguf|onnx)$", "模型权重"),
    (r"^adapter_config\.json$", "LoRA adapter 配置（adapter 权重不进发布包）"),
    (r"^train_logs?(_v\d)?$", "训练日志目录"),
    (r"^lora_adapter(_v\d)?$", "LoRA adapter 目录"),
    (r"^__pycache__$", "Python 缓存"),
    (r"^_tmp.*", "临时文件（`_tmp*` 一律不该进发布包）"),
]
SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9_\-]{16,}", "疑似 DeepSeek/OpenAI API Key（`sk-` 开头）"),
    (r"hf_[A-Za-z0-9]{20,}", "疑似 HuggingFace Token（`hf_` 开头）"),
    (r"(?i)api[_-]?key\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']", "硬编码 API Key（`api_key = \"...\"`）"),
    (r"(?i)authorization\s*[:=]\s*[\"']Bearer\s+[A-Za-z0-9_\-\.]{16,}", "硬编码 Bearer Token"),
]
TEXT_EXT = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json", ".cfg", ".ini",
            ".sh", ".ps1", ".bat", ".html", ".css", ".js"}

# ⑤ 标注纪律：每条规则 = (说明, 备选写法列表, 缺失时是否算 FAIL)
#    备选写法是**为了不误杀**：同一个意思在不同文件里写法不同（"非本项目微调模型" / "并非本项目的微调模型"…），
#    但它们都必须出现——**漏一处就是误导**（详见 01-第四周详细计划.md 第 7.2 节硬性规定）。
DISCLAIM_RULES = [
    ("生成层 = DeepSeek API",
     ["DeepSeek"], True),
    ("生成层 ≠ 本项目微调模型",
     ["非本项目微调模型", "非本项目的微调模型", "并非本项目的微调模型", "不是本项目微调模型",
      "不是本项目的微调模型", "≠ 本项目微调模型", "非本地微调模型", "没有加载本项目的", "未加载本项目的"], True),
    ("评测数字只归属本地 Qwen 版本",
     ["只归属", "只算", "只属于", "只代表", "数字归属", "均来自本地", "不产出数字", "数字只", "不代表"], False),
]
DISCLAIM_FILES = [
    ("blog/项目展示页.md", "A 线文章"),
    ("space_demo/app.py", "B 线页面代码"),
    ("space_demo/README.md", "B 线 README"),
]


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} GB"


def main():
    ap = argparse.ArgumentParser(description="发布包上线前自动检查")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="发布包目录（默认 第四周\\发布包）")
    ap.add_argument("--budget-mb", type=float, default=20.0, help="总体积预算（默认 20 MB）")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    fails, warns = [], []

    print("=" * 88)
    print("发布包上线前检查（机器能判的都交给它；判不了的列在最下面给人看）")
    print("=" * 88)
    print(f"  目录：{root}")
    print(f"  时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if not os.path.isdir(root):
        raise SystemExit(f"[FAIL] 目录不存在：{root}\n       → 确认 第四周\\发布包 已建（或用 --root 指定）")

    # ---- 遍历 ----
    files, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            rel = os.path.relpath(p, root).replace("\\", "/")
            files.append((rel, sz, p))
            total += sz

    print(f"  文件数：{len(files)}｜总体积：{human(total)}（预算 {args.budget_mb} MB）")

    # ---- ① 体积 ----
    print("\n① 体积")
    if total > args.budget_mb * 1024 * 1024:
        fails.append(f"总体积 {human(total)} > 预算 {args.budget_mb} MB —— 先看下面「最大的 5 个文件」，"
                     f"多半是把不该进的东西带进来了")
        print(f"  ❌ 超预算：{human(total)} > {args.budget_mb} MB")
    else:
        print(f"  ✅ 在预算内（{human(total)} ≤ {args.budget_mb} MB）")
    for rel, sz, _ in sorted(files, key=lambda t: -t[1])[:5]:
        print(f"       {human(sz):>10s}  {rel}")

    # ---- ② 禁入项 ----
    print("\n② 禁入项（权重 / adapter / 训练日志 / 临时文件）")
    banned_hits = []
    for rel, sz, _ in files:
        for part in rel.split("/"):
            for pat, why in BANNED_NAME_PATTERNS:
                if re.match(pat, part):
                    banned_hits.append((rel, why, sz))
                    break
    if banned_hits:
        for rel, why, sz in banned_hits:
            fails.append(f"不该外发的文件：`{rel}`（{why}，{human(sz)}）")
            print(f"  ❌ {rel}  ← {why}（{human(sz)}）")
    else:
        print("  ✅ 没有发现权重 / adapter / 训练日志 / 临时文件")

    # ---- ③ 密钥 ----
    print("\n③ 密钥扫描（硬编码 API Key / Token）")
    secret_hits = []
    for rel, _, p in files:
        if os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        try:
            txt = io.open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for pat, why in SECRET_PATTERNS:
            for m in re.finditer(pat, txt):
                line_no = txt[:m.start()].count("\n") + 1
                secret_hits.append((rel, line_no, why, m.group(0)[:24]))
    if secret_hits:
        for rel, ln, why, snippet in secret_hits:
            fails.append(f"疑似硬编码密钥：`{rel}` 第 {ln} 行（{why}）：{snippet}…")
            print(f"  ❌ {rel}:{ln}  {why}  →  {snippet}…")
        print("     修法：代码只从环境变量读（`os.environ[\"API_KEY\"]`）；平台的 key 放「设置 → 环境变量 / Secrets」")
    else:
        print("  ✅ 未发现硬编码密钥（key 只从环境变量读）")

    # ---- ④ 依赖 + 上线自举 ----
    print("\n④ B 线依赖与上线自举（space_demo/requirements.txt + app.py）")
    req = os.path.join(root, "space_demo", "requirements.txt")
    if not os.path.exists(req):
        fails.append("缺少 `space_demo/requirements.txt`")
        print("  ❌ 文件不存在")
    else:
        req_txt = io.open(req, encoding="utf-8", errors="replace").read()
        pkgs = [x.strip() for x in req_txt.splitlines() if x.strip() and not x.strip().startswith("#")]
        print("  非注释依赖：" + " ｜ ".join(pkgs))
        req_flat = req_txt.replace(" ", "")
        if "pydantic==2.10.6" in req_flat:
            print("  ✅ pydantic==2.10.6 在（gradio 4.44 × pydantic≥2.11 的坑已钉住）")
        else:
            fails.append("`requirements.txt` 里没有 `pydantic==2.10.6` —— gradio 4.44 会因新版 pydantic"
                         "报错（day18 已实测），上线前必须钉住")
            print("  ❌ 没找到 pydantic==2.10.6")
        if "starlette==0.46.2" in req_flat:
            print("  ✅ starlette==0.46.2 在（gradio 4.44 × starlette≥0.47 的坑已钉住）")
        else:
            fails.append("`requirements.txt` 里没有 `starlette==0.46.2` —— starlette≥0.47 删了旧版"
                         "`TemplateResponse(name, context)` 签名，gradio 4.44 仍按旧签名调用 → "
                         "「服务能起但首屏 500」（day20 实测）。不能写成 `starlette<0.40`："
                         "fastapi 0.141.x 要求 starlette>=0.46.0，会变成 ResolutionImpossible")
            print("  ❌ 没找到 starlette==0.46.2")

    # ---- ④b 启动绑定：必须 0.0.0.0，否则平台反代 502 ----
    app_py = os.path.join(root, "space_demo", "app.py")
    if not os.path.exists(app_py):
        fails.append("缺少 `space_demo/app.py`")
        print("  ❌ app.py 不存在")
    else:
        app_txt = io.open(app_py, encoding="utf-8", errors="replace").read()
        if 'server_name="0.0.0.0"' in app_txt.replace("'", '"'):
            print("  ✅ launch() 显式绑 0.0.0.0（平台反代连得上）")
        else:
            fails.append("`space_demo/app.py` 的 `launch()` 没有显式绑 `server_name=\"0.0.0.0\"` —— "
                         "gradio 默认绑 127.0.0.1，平台反向代理在容器外连不进来 → "
                         "页面报「Could not load this space」/ 网关 502（day20 魔搭实测）")
            print("  ❌ launch() 未显式绑 0.0.0.0")

    # ---- ④c 检索口径与建库自举：四个"不报错也查不出来"的坑（2026-09-22 全部实测踩到）----
    #   为什么必须进闸门：这四条都是"改了也不会报错、但线上结果会悄悄变错"的类型，
    #   代码评审时最容易被当成无关细节删掉，所以在发布前用脚本卡住。
    print("\n④c 检索口径与建库自举（内置语料 / collection 唯一 / search_ef / 落盘 / progress 判空）")
    if not os.path.exists(app_py):
        print("  ⏭ 跳过（app.py 不存在）")
    else:
        # (1) 内置语料必须在，且内容对得上（105 段、字段齐全、id 连续）
        corpus = os.path.join(root, "space_demo", "chunks.json")
        if not os.path.exists(corpus):
            fails.append("缺少 `space_demo/chunks.json` —— 页面主打的\"不上传也能问\"会直接失效，"
                         "首问只能看到「请先上传 PDF」")
            print("  ❌ 缺少内置语料 chunks.json")
        else:
            try:
                items = json.loads(io.open(corpus, encoding="utf-8").read())
                n_item = len(items)
                ids = [c.get("id") for c in items]
                ok_item = (n_item == 105 and all("text" in c for c in items)
                           and ids == sorted(ids) and len(set(ids)) == n_item)
                if ok_item:
                    print(f"  ✅ 内置语料 chunks.json：{n_item} 段，id 连续且不重复（在线出处编号可对照评测表）")
                else:
                    fails.append(f"`space_demo/chunks.json` 内容异常：{n_item} 段（期望 105）、"
                                 f"id 是否连续不重复：{ids == sorted(ids) and len(set(ids)) == n_item} —— "
                                 f"它应当与 `第三周/day13/chunks.json` 完全一致，别手改")
                    print(f"  ❌ 内置语料异常：{n_item} 段")
            except Exception as e:
                fails.append(f"`space_demo/chunks.json` 解析失败：{type(e).__name__}: {e}")
                print(f"  ❌ 内置语料解析失败：{e}")

        # (2) collection_name 必须唯一：否则重复建库会往同一个 collection 追加 → 出处重复
        #     只统计**代码行**（去掉行内注释后仍出现才算），否则文件头注释里那句
        #     "`Chroma.from_documents(...)` 不传 collection_name 时…" 会被误算成一处调用。
        code_lines = [ln.split("#", 1)[0] for ln in app_txt.splitlines()]
        n_from_docs = sum(ln.count("Chroma.from_documents(") for ln in code_lines)
        n_uniq_name = sum(ln.count("collection_name=_new_collection_name(") for ln in code_lines)
        if n_from_docs and n_uniq_name >= n_from_docs:
            print(f"  ✅ 每处 from_documents（{n_from_docs} 处）都给了唯一 collection_name"
                  f"（否则第二次建库会追加，出处会出现 [资料§103]、[资料§103]… 的重复）")
        else:
            fails.append(f"`app.py` 里有 {n_from_docs} 处 `Chroma.from_documents(`，但只有 {n_uniq_name} 处"
                         f"带唯一 `collection_name` —— 不传就用默认名 \"langchain\"，而 chromadb 在同一进程"
                         f"共享同一个 in-memory system → 第二次建库是**追加**（实测 105→210 条），"
                         f"检索结果出现重复出处（day20 魔搭实测）")
            print(f"  ❌ from_documents {n_from_docs} 处 / 唯一 collection_name {n_uniq_name} 处")

        # (3) hnsw:search_ef 必须显式给：默认 10 会让 Top-K 与离线评测对不上
        has_ef = "hnsw:search_ef" in app_txt
        if has_ef:
            print("  ✅ 显式给了 hnsw:search_ef（chromadb 默认 ef=10，105 段会漏近邻 → 与离线评测对不上）")
        else:
            fails.append("`app.py` 的 collection_metadata 里没有 `hnsw:search_ef` —— chromadb "
                         "`hnsw_params.py` 里它默认只有 10，候选池小于库容量就会漏掉真近邻；"
                         "本语料 20 题实测只有 7 题与精确检索一致，且每次建库结果都不同"
                         "（提到 200 后与离线落盘 top_ids 20/20 一致）")
            print("  ❌ 没找到 hnsw:search_ef")

        # (4) 必须落盘而不是内存库：内存库是 threading.local + 弱引用，建库线程一退出就可能整体销毁
        if "persist_directory=" in app_txt:
            print("  ✅ 向量库落盘（persist_directory）：内存库跨线程会报 "
                  "`no such table: collections`（LockPool 用 threading.local + 弱引用）")
        else:
            fails.append("`app.py` 的 Chroma 建库没有 `persist_directory=` —— chromadb 的内存库用 "
                         "`file::memory:?cache=shared` + LockPool（连接在 threading.local 里、只存弱引用），"
                         "**建库线程退出后库可能被销毁** → Gradio worker 线程查询报 "
                         "`sqlite3.OperationalError: no such table: collections`（day20 实测，且是间歇性的）")
            print("  ❌ 没有 persist_directory=")

        # (5) 不能用 `if progress:` 判空：会触发 gradio Progress.__len__ 的 IndexError
        #     ⚠ 只认"行首就是这句"的代码行：`_tick()` 的注释里写着"不能写 if progress:"，
        #       用 search 会把它一起误判（闸门有误报就等于没有闸门）。
        bad_progress = [l for l in app_txt.splitlines()
                        if re.match(r"\s*if\s+progress\s*:", l)]
        if bad_progress:
            fails.append("`app.py` 里出现 `if progress:` —— gradio 4.44 的 `Progress.__len__` 实现是 "
                         "`self.iterables[-1].length`，还没报过进度时 iterables 为空 → 直接 "
                         "`IndexError: list index out of range`（首问就 500）。判空请写 "
                         "`progress is not None`（或统一走 `_tick()`）")
            print(f"  ❌ 出现 `if progress:`（{bad_progress[0].strip()[:60]}…）")
        else:
            print("  ✅ 没有 `if progress:` 这种判空（gradio 4.44 会 IndexError）")

        # (6) 预热线程：让"第一个访客"不用干等半分钟
        if "warmup-default-store" in app_txt or "Thread(target=build_default_store" in app_txt:
            print("  ✅ 内置库有启动后台预热（首问 31s → 2~3s，且不阻塞端口绑定）")
        else:
            warns.append("`app.py` 里没有内置库预热线程 —— 第一个访客要等约 30s 才出结果，"
                         "体感像页面坏了（不算 FAIL，只是体验损失）")
            print("  ⚠ 没找到预热线程（首问会慢）")

    # ---- ⑤ 三处标注 ----
    print("\n⑤ 标注纪律（检索层真实运行 ｜ 生成层 = DeepSeek API ｜ 非本项目微调模型）")
    for rel, label in DISCLAIM_FILES:
        p = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.exists(p):
            fails.append(f"缺少文件：`{rel}`（{label}）")
            print(f"  ❌ {label}：文件不存在（{rel}）")
            continue
        txt = io.open(p, encoding="utf-8", errors="replace").read()
        for rule, variants, is_fail in DISCLAIM_RULES:
            hit = next((v for v in variants if v in txt), None)
            if hit:
                print(f"  ✅ {label}（{rel}）· {rule}  ← 命中「{hit}」")
            else:
                msg = (f"`{rel}`（{label}）缺少「{rule}」的表述"
                       f"（可写：{' / '.join(variants[:3])} …）")
                (fails if is_fail else warns).append(msg)
                print(f"  {'❌' if is_fail else '⚠'} {label}（{rel}）· {rule}：未命中任何写法")

    # ---- ⑥ 占位符 + 数字对账 ----
    print("\n⑥ 占位符与数字对账")
    art = os.path.join(root, "blog", "项目展示页.md")
    if not os.path.exists(art):
        warns.append("找不到 `blog/项目展示页.md` → 无法检查占位符与数字")
        print("  ⚠ 找不到文章文件")
    else:
        art_txt = io.open(art, encoding="utf-8", errors="replace").read()
        n_blank = len(re.findall(r"__", art_txt))
        if n_blank:
            fails.append(f"`blog/项目展示页.md` 还有 {n_blank} 处 `__` 占位符 —— 定稿数字没回填就上线"
                         f"（数字取 `day20\\定稿数字表.md`）")
            print(f"  ❌ 文章里还有 {n_blank} 处 `__`")
        else:
            print("  ✅ 文章里没有 `__` 残留")

        art_nums = set(re.findall(r"\d+(?:\.\d+)?%|0\.\d{3}", art_txt))
        if os.path.exists(FINALS_TABLE):
            tbl = io.open(FINALS_TABLE, encoding="utf-8", errors="replace").read()
            tbl_nums = set(re.findall(r"\d+(?:\.\d+)?%|0\.\d{3}", tbl))
            unknown = sorted(art_nums - tbl_nums)
            print(f"  文章数字 {len(art_nums)} 个｜定稿表数字 {len(tbl_nums)} 个")
            if unknown:
                warns.append("文章里出现、但定稿数字表里没有的数字：" + "、".join(unknown) +
                             " —— 逐个确认（可能是字节数/条数/day14 旧口径基线，也可能真是抄错）")
                print("  ⚠ 需人工确认：" + "、".join(unknown))
            else:
                print("  ✅ 文章里的数字都能在定稿数字表里找到")
        else:
            warns.append("找不到 `day20\\定稿数字表.md` → 无法做数字对账（先跑 `day20\\collect_finals.py`）")
            print("  ⚠ 找不到定稿数字表，跳过数字对账")

    # ---- 汇总 ----
    print("\n" + "=" * 88)
    print("汇总")
    print("=" * 88)
    if fails:
        print(f"  ❌ FAIL {len(fails)} 条（**上线前必须修**）：")
        for i, f in enumerate(fails, 1):
            print(f"     {i}. {f}")
    else:
        print("  ✅ 六项检查全部通过，可以发布")
    if warns:
        print(f"  ⚠ 需人工确认 {len(warns)} 条：")
        for i, w in enumerate(warns, 1):
            print(f"     {i}. {w}")

    print("\n  机器判不了、上手之前自己还要看一眼的：")
    print("     - 手机/他人设备打开两个链接（**不需要本地模型服务开着**）")
    print("     - 演示 GIF 能播、体积可接受（>20 MB 就改传视频平台、博客里嵌链接）")
    print("     - 页面上「评测数字只归属本地 Qwen 版本」这句没被改掉")

    if fails:
        print("\n[FAIL] 发布包未过闸门（退出码 1）")
        sys.exit(1)
    print("\n[OK] 发布包过闸门")


if __name__ == "__main__":
    main()
