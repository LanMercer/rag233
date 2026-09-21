# -*- coding: utf-8 -*-
r"""
检索通道实验室（第四周 Day18 · O1-R4 HyDE 冒烟 + O1-R5 混合检索筛查）⭐

一句话：**在动生成层之前，先把"检索"这一个变量单独拎出来做受控对比**——
    同一套 20 条题、同一把双轨判据（与 eval_v2.py 一致），只换检索通道：
        ① vector：bge 向量 + Chroma 余弦 Top-K（现状，day17 默认配置）
        ② bm25  ：稀疏通道（关键词/专名命中）——治 LAFAN1 / BeyondMimic / Stanford / github 这类专名题
        ③ hybrid：vector + bm25 用 RRF（倒数排名融合）合成一路
    以及换"查询侧"：
        ④ --query-mode hyde：把中文问句换成手工写的"理想查询"（英文术语版，模拟 HyDE 的思想），
           验证"中英语义错位（假设②）到底占多少失败"。

为什么先做这一步（而不是直接跑 eval_v2.py）：
    eval_v2.py 一轮要起模型服务 + 生成 20 条（×runs），慢且带生成噪声；
    本脚本**不连模型服务**，只做检索排序，几秒钟出结果 →
    先用它**筛查**哪条通道有希望，再用 eval_v2.py 做**正式确认**（生成+判分）。
    这就是"先用便宜的实验排除假设，再用贵的实验确认结论"。

判据（与 eval_v2.py 严格一致，保证可比）：
    strict = Top-K 的 chunk_id 与题目 source_chunk 有交集
    loose  = 期望关键词是否**字面出现**在 Top-K 的 chunk 文本里（大小写不敏感）
    注：中文关键词在英文段落里字面匹配不到 → loose 天然偏低，这是已知口径，不是 bug。

运行方法（llm 环境；先 cd 到本文件所在目录）：
    conda activate llm
    cd "D:\Lan\研究生\技术学习\大模型算法\第四周\day18"

    python retrieval_lab.py                                  # 默认：vector/bm25/hybrid × k=8（现状口径）
    python retrieval_lab.py --top-k 4                         # 换 k 看窗口大小的影响
    python retrieval_lab.py --query-mode hyde                 # 换"理想查询"再跑一遍（O1-R4）
    python retrieval_lab.py --retrievers vector,bm25          # 只看两条通道
    python retrieval_lab.py --show-q 3 --top-k 8              # 单题钻取：看它到底召回什么
    python retrieval_lab.py --json                            # 落盘 retrieval_lab.json（报告可引用）

    # Day19 新增（O1-R4 真实现）：用【模型自动改写】的查询复筛
    python retrieval_lab.py --top-k 8 --query-mode file --rewrite-mode term --json
    python retrieval_lab.py --top-k 8 --query-mode file --rewrite-mode term --retrievers hybrid --rrf-k 10

依赖：langchain_chroma / langchain_huggingface（读第三周 day15 config.json 的库与模型）；
      BM25 为**手写实现**（不依赖 rank_bm25），只用标准库。

Day19 相对 Day18 的改动（只加"查询来源"这一档，**plain / hyde 的行为与数字一字不变**）：
    ① 【新增】--query-mode file：从 day19\queries_rewritten.json 读**真实改写结果**当查询。
       为什么要有它：plain 看的是"现状"、hyde 看的是"人类手写理想查询（上界）"，
       两者之间那个**真实现**（模型自动改写）才是能写进报告的数字 → 筛查与正式评测必须用同一批查询。
    ② 【新增】--rewrite-cache / --rewrite-mode（默认 term）。
    ③ 【修复】--json 的落盘名带模式：原先无论哪种 query-mode 都写 retrieval_lab.json，
       跑一次 hyde/file 就会**覆盖 day18 的 plain 证据**。现改为
       plain -> retrieval_lab.json（不改名，兼容历史）；其它模式 -> retrieval_lab_<mode>.json。
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# 第 1 区：终端编码加固
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 第 2 区：路径与配置（复用第三周 day15 config.json，避免路径写两份）
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))            # 第四周\day18
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))  # 仓库根
CONFIG_PATH = os.path.join(REPO_DIR, "第三周", "day15", "config.json")
CHUNKS_JSON = os.path.join(REPO_DIR, "第三周", "day13", "chunks.json")
QUESTIONS_JSON = os.path.join(REPO_DIR, "第三周", "day13", "eval_questions.json")

# Day19 新增：--query-mode file 的默认缓存路径（day19 的改写产物）
DEFAULT_REWRITE_CACHE = os.path.join(REPO_DIR, "第四周", "day19", "queries_rewritten.json")

# 判据校准补丁（与 day17\eval_v2.py 的 KEYWORD_PATCH 一字不差）
KEYWORD_PATCH = {
    5: {"expected_keywords": ["LAFAN1", "LAFAN"]},
}

# query-mode 的显示标签（只影响打印，不影响任何计算）
MODE_LABEL = {
    "plain": "",
    "hyde": "（理想查询/HyDE·上界）",
    "file": "（真实改写·模型自动生成）",
}
MODE_SUFFIX = {
    "plain": "",
    "hyde": "+hyde(上界)",
    "file": "+rw(真实现)",
}

# ---------------------------------------------------------------------------
# 第 3 区：手工"理想查询"表（O1-R4 HyDE 冒烟的核心）
# ---------------------------------------------------------------------------
# 思路（HyDE）：「先假设一个答案，拿答案去检索」。
#   本冒烟不调用任何模型，直接用**人工写的英文术语版查询**代替——
#   目的是**隔离变量**：如果换成英文术语检索，命中率大涨，
#   就证明失败主要来自"中文问句 ↔ 英文段落的语义错位"（假设②），而不是"库坏了"。
HYDE_QUERIES = {
    1: "GMR General Motion Retargeting full name definition",
    2: "embodiment gap between humans and humanoid robots fundamental challenge",
    3: "Stanford University author affiliation",
    4: "PHC ProtoMotions Unitree open-source retargeter comparison baselines",
    5: "LAFAN1 dataset experiments diverse subset",
    6: "BeyondMimic policy training reinforcement learning",
    7: "artifacts introduced during retargeting foot sliding self-penetration physically infeasible motion",
    8: "GMR outperforms open-source methods faithful to source motion close to closed-source baseline",
    9: "Code github.com repository for GMR",
    10: ("embodiment gap bone length joint range of motion kinematic structure "
         "body shape mass distribution actuation mechanisms"),
}


# ---------------------------------------------------------------------------
# 第 4 区：加载
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p, base_dir):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))


def load_questions():
    qs = load_json(QUESTIONS_JSON)["questions"]
    out = []
    for q in qs:
        q = json.loads(json.dumps(q))
        patch = KEYWORD_PATCH.get(q["id"])
        if patch:
            q.update(patch)
        out.append(q)
    return out


def load_chunks():
    return {c["id"]: c.get("text", "") for c in load_json(CHUNKS_JSON)}


# ---------------------------------------------------------------------------
# 第 5 区：BM25（手写，稀疏通道）
# ---------------------------------------------------------------------------
# BM25 直觉：TF-IDF 的升级版——① 词频饱和（一个词出现 100 次不该比 10 次强 10 倍）；
#   ② 文档长度归一（长文档天然容易命中）。公式：
#     score(q,d) = Σ_t idf(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )
#   idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
#   k1=1.5（词频饱和速度）、b=0.75（长度归一强度）——都是"默认经验值"，可调但不必调。

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text):
    """英文/数字 token（小写）。中文段落的 BM25 匹配不到英文库，故只取 ASCII token。"""
    return TOKEN_RE.findall(str(text).lower())


class BM25:
    def __init__(self, doc_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = doc_tokens          # 预先切好的 token 列表（避免每次打分重复切词）
        self.N = len(doc_tokens)
        self.avgdl = (sum(len(d) for d in doc_tokens) / self.N) if self.N else 0.0
        self.doc_freq = Counter()
        for toks in doc_tokens:
            self.doc_freq.update(set(toks))
        self.doc_len = [len(d) for d in doc_tokens]

    def idf(self, term):
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5)) if df else 0.0

    def score(self, query_tokens, doc_idx):
        """单个文档对 query 的 BM25 得分。注意：第 2 个参数是**文档下标**，不是文本。"""
        tokens = self.doc_tokens[doc_idx]
        tf = Counter(tokens)
        dl = self.doc_len[doc_idx]
        s = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            s += self.idf(t) * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def rank(self, query, ids, texts, pool):
        """返回按分数降序的 [(chunk_id, score), ...]，取前 pool 个（score>0 才算命中）。"""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scored = [(cid, self.score(q_tokens, i)) for i, cid in enumerate(ids)]
        scored.sort(key=lambda x: -x[1])
        return [(cid, sc) for cid, sc in scored[:pool] if sc > 0]


# ---------------------------------------------------------------------------
# 第 6 区：向量通道
# ---------------------------------------------------------------------------

def build_vector_store():
    """按 config.json 的 persist_dir / embed_model 建 Chroma 连接（只读不改库）。"""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    cfg = load_json(CONFIG_PATH)
    cfg_dir = os.path.dirname(CONFIG_PATH)
    persist_dir = resolve_path(cfg["persist_dir"], cfg_dir)
    print(f"  向量库：{persist_dir}")
    print(f"  向量模型：{cfg['embed_model']}")
    embedder = HuggingFaceEmbeddings(model_name=cfg["embed_model"])
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)


def vector_rank(store, query, pool):
    """返回 [(chunk_id, score), ...]，按余弦相似度降序。"""
    hits = store.similarity_search_with_relevance_scores(query, k=pool)
    out = []
    for doc, score in hits:
        cid = doc.metadata.get("chunk_id")
        out.append((cid, float(score)))
    return out


# ---------------------------------------------------------------------------
# 第 7 区：RRF 融合（混合检索）
# ---------------------------------------------------------------------------
# RRF（Reciprocal Rank Fusion）：不看分数、只看**名次**，把多路排名融合成一路。
#   公式：rrf_score(d) = Σ_over_lists 1 / (rrf_k + rank(d))   （rank 从 1 开始，rrf_k=60 是经验常数）
#   为什么用名次而不是分数：向量余弦（0~1）与 BM25（0~30+）**量纲完全不同**，
#   直接加权相加要调权重、还可能被某一路上限压死；RRF 天然免标定，是工程上的稳妥默认。


def rrf_fuse(rankings, pool, rrf_k=60):
    """rankings = [[(cid, score), ...], ...]（每路已降序）→ 融合后 [(cid, rrf_score), ...]。"""
    fused = Counter()
    for ranking in rankings:
        for pos, (cid, _score) in enumerate(ranking, start=1):
            fused[cid] += 1.0 / (rrf_k + pos)
    return sorted(fused.items(), key=lambda x: -x[1])[:pool]


# ---------------------------------------------------------------------------
# 第 8 区：双轨判据（与 eval_v2.py 一致）
# ---------------------------------------------------------------------------

def judge_retrieval(q, top_ids, chunks):
    if q["type"] != "in_material":
        return None, None
    strict = bool(set(top_ids) & set(q.get("source_chunk", [])))
    ctx = "\n".join(chunks.get(i, "") for i in top_ids).lower()
    kws = q.get("expected_keywords", [])
    loose = any(kw.lower() in ctx for kw in kws) if kws else None
    return strict, loose


# ---------------------------------------------------------------------------
# 第 9 区：主流程
# ---------------------------------------------------------------------------

def run(args):
    chunks = load_chunks()
    ids = sorted(chunks)                      # chunk_id 顺序
    texts = [chunks[i] for i in ids]
    questions = load_questions()

    print("=" * 96)
    print("检索通道实验室（O1-R4 HyDE 冒烟 + O1-R5 混合检索筛查）")
    print("=" * 96)
    print(f"  题数：{len(questions)} ｜ chunk：{len(chunks)} ｜ Top-K：{args.top_k} ｜ 候选池：{args.pool}")
    print(f"  通道：{args.retrievers} ｜ query-mode：{args.query_mode} ｜ RRF k={args.rrf_k}")
    print()

    store = None
    if "vector" in args.retrievers or "hybrid" in args.retrievers:
        print("加载向量通道（首次加载 bge 约需几秒）……")
        store = build_vector_store()
        print()

    # ---- Day19 新增：--query-mode file 时，从改写缓存读"真实改写结果" ----
    # 为什么不复用 eval_v2.py 的缓存读取逻辑：本脚本不 import eval_v2（那会把 langchain/service 一整套拖进来），
    # 干脆就地读一遍——只有 10 行，且两个脚本读的是同一份文件，数字天然可比。
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
        print()

    bm25 = BM25([tokenize(t) for t in texts])
    results = {}

    for retriever in args.retrievers:
        rows = []
        for q in questions:
            if q["type"] != "in_material":
                continue
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

            if retriever == "vector":
                ranked = vector_rank(store, query, args.pool)
            elif retriever == "bm25":
                ranked = bm25.rank(query, ids, texts, args.pool)
            else:  # hybrid
                v = vector_rank(store, query, args.pool)
                b = bm25.rank(query, ids, texts, args.pool)
                ranked = rrf_fuse([v, b], args.pool, args.rrf_k)

            top_ids = [cid for cid, _ in ranked[: args.top_k]]
            strict, loose = judge_retrieval(q, top_ids, chunks)
            rows.append({"id": q["id"], "question": q["question"], "query": query,
                         "expected": q.get("source_chunk", []), "top_ids": top_ids,
                         "strict": strict, "loose": loose,
                         "scores": {cid: round(float(s), 4) for cid, s in ranked[: args.top_k]}})
        results[retriever] = rows

        sn = sum(1 for r in rows if r["strict"])
        ln = sum(1 for r in rows if r["loose"])
        label = f"{retriever}" + MODE_LABEL[args.query_mode]
        print(f"【{label}】strict {sn}/{len(rows)} = {100*sn/len(rows):.1f}% ｜ "
              f"loose {ln}/{len(rows)} = {100*ln/len(rows):.1f}%")
        for r in rows:
            mark = "✅" if r["strict"] else ("◐" if r["loose"] else "❌")
            print(f"   Q{r['id']:>2} {mark}  期望{r['expected']} → top{r['top_ids']}")
        print()

    # ---- 总结对比表 ----
    print("=" * 96)
    print("汇总（同 10 条 in_material 题、同双轨判据）")
    print("=" * 96)
    print(f"{'通道':<28} | {'strict':>10} | {'loose':>10}")
    print("-" * 96)
    for retriever, rows in results.items():
        label = retriever + MODE_SUFFIX[args.query_mode]
        sn = sum(1 for r in rows if r["strict"])
        ln = sum(1 for r in rows if r["loose"])
        print(f"{label:<28} | {sn:>3}/{len(rows)}={100*sn/len(rows):>4.1f}% | {ln:>3}/{len(rows)}={100*ln/len(rows):>4.1f}%")
    print("-" * 96)
    print("判读：strict 涨 = 真把期望段落拉进窗口；loose 涨 = 关键词（英文）出现在上下文里。")
    print("      BM25 只吃 ASCII token，**中文问句对它几乎没有信号** → 专名题（LAFAN1/PHC/github）才是它的主场；")
    print("      hyde 模式把查询换成英文术语，BM25 与向量同时受益，可用来验证『中英语义错位』这个假设。")
    if args.query_mode == "file":
        print(f"      file 模式 = **模型自动改写**的真实查询（{args.rewrite_mode} 档）→ 这是可写进报告的『真实现』；")
        print("      hyde 是**人类手写理想查询**，只能当上界对照，别把两者混着说。")
        if rw_missing:
            print(f"      ⚠ 未取到改写的题号：{sorted(set(rw_missing))}（缓存里该档缺失/为空 → 已回落原问题）")
            print("        → 这几题的 strict 若变了，不能归因给改写（它们根本没被改写）。")

    if args.show_q:
        print("\n" + "=" * 96)
        for retriever, rows in results.items():
            r = next((x for x in rows if x["id"] == args.show_q), None)
            if r:
                print(f"[Q{r['id']}] {retriever}：查询={r['query']!r}")
                print(f"      期望={r['expected']} → top={r['top_ids']} ｜ scores={r['scores']}")
        print("=" * 96)

    if args.json:
        # Day19 修复：落盘名带模式，避免"跑一次 hyde/file 就把 day18 的 plain 证据覆盖掉"
        out_name = "retrieval_lab.json" if args.query_mode == "plain" else f"retrieval_lab_{args.query_mode}.json"
        out = os.path.join(SCRIPT_DIR, out_name)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 明细已落盘：{out}")


def main():
    parser = argparse.ArgumentParser(
        description="检索通道实验室：vector / bm25 / hybrid(RRF) × plain / hyde"
    )
    parser.add_argument("--retrievers", default="vector,bm25,hybrid",
                        help="要跑的通道，逗号分隔（默认全部）：vector,bm25,hybrid")
    parser.add_argument("--top-k", type=int, default=8, help="最终取前 K 条（默认 8 = day17 的默认配置）")
    parser.add_argument("--pool", type=int, default=20, help="每路候选池大小（RRF 融合前，默认 20）")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 常数（默认 60）")
    parser.add_argument("--query-mode", default="plain", choices=["plain", "hyde", "file"],
                        help="plain=原题；hyde=手工理想查询（上界）；file=读 day19 的真实改写缓存")
    parser.add_argument("--rewrite-cache", default=DEFAULT_REWRITE_CACHE,
                        help="改写缓存路径（query-mode=file 时用；默认 day19\\queries_rewritten.json）")
    parser.add_argument("--rewrite-mode", default="term", choices=["plain", "term", "hyde", "both"],
                        help="用缓存里的哪一档当查询（默认 term；plain 档=原题，可用于接线自检）")
    parser.add_argument("--show-q", type=int, default=None, help="单题钻取（如 --show-q 3）")
    parser.add_argument("--json", action="store_true",
                        help="落盘明细 json（plain -> retrieval_lab.json；其它模式 -> retrieval_lab_<mode>.json）")
    args = parser.parse_args()
    args.retrievers = [x.strip() for x in args.retrievers.split(",") if x.strip()]
    run(args)


if __name__ == "__main__":
    main()
