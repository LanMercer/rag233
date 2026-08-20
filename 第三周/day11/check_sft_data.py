# -*- coding: utf-8 -*-
"""
check_sft_data.py —— SFT 数据质量检查脚本（第三周 Day11 第 5 步验收）
========================================================================
【本脚本是什么】数据检查脚本，不涉及模型训练/测试。
【做什么】用 HuggingFace 的 datasets 库加载 sft_data.json，并检查：
         ① 三字段是否齐全（instruction / input / output）
         ② 总条数是否落在 200~500 区间
         ③ 是否存在空指令 / 空答案 / 重复条目
         ④ 抽样打印前 5 条 + 各字段长度统计
【为什么用 datasets】Datasets 是 HuggingFace 的数据集库，Day12 微调时
         就是用它加载数据训练模型。今天先验证"datasets 能正确加载自制数据"，
         明天才能顺利对接训练脚本。
【五件套说明】不训练不测试，没有模型/损失/优化器。
========================================================================
"""
import json
import sys

# Windows 控制台默认 GBK 打不出部分字符，强制用 UTF-8 输出（避免乱码报错）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "sft_data.json"
EXPECTED_FIELDS = ["instruction", "input", "output"]


def main():
    print("=" * 60)
    print("SFT 数据质量检查（datasets 加载验证）")
    print("=" * 60)

    # ① 用 datasets 加载 JSON（这就是 Day12 训练脚本加载数据的同款方式）
    try:
        from datasets import load_dataset
        ds = load_dataset("json", data_files=DATA_PATH, split="train")
        print(f"[1] datasets 加载成功：{len(ds)} 条")
    except Exception as e:
        print(f"[1] datasets 加载失败：{e}")
        # 降级用 json 加载，保证后续检查仍能跑
        with open(DATA_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        ds = None
        print(f"[1] 降级用 json 加载成功：{len(raw)} 条")

    # ② 条数区间检查
    n = len(ds) if ds is not None else len(raw)
    ok_count = 200 <= n <= 500
    print(f"[2] 总条数 = {n}，落在 200~500 区间：{'✅ 通过' if ok_count else '❌ 不通过'}")

    if ds is not None:
        # ③ 字段检查
        cols = ds.column_names
        missing = [c for c in EXPECTED_FIELDS if c not in cols]
        if missing:
            print(f"[3] 缺少字段：{missing}")
        else:
            print(f"[3] 三字段齐全：{cols}  ✅")

        # ④ 空值检查
        empty_inst = sum(1 for x in ds["instruction"] if not x or not str(x).strip())
        empty_out = sum(1 for x in ds["output"] if not x or not str(x).strip())
        print(f"[4] 空 instruction 条数 = {empty_inst}，空 output 条数 = {empty_out}")

        # ⑤ 重复检查（instruction+output 完全相同视为重复）
        seen = set()
        dup = 0
        for inst, out in zip(ds["instruction"], ds["output"]):
            key = (str(inst).strip(), str(out).strip())
            if key in seen:
                dup += 1
            seen.add(key)
        print(f"[5] 完全重复条目 = {dup}")

        # ⑥ 抽样打印
        print("-" * 60)
        print("[6] 抽样前 5 条：")
        for i in range(min(5, n)):
            print(f"  --- 第 {i+1} 条 ---")
            print(f"  instruction: {ds['instruction'][i]}")
            inp = ds["input"][i]
            print(f"  input     : {inp[:80] + '...' if inp and len(inp) > 80 else (inp or '(空)')}")
            out = ds["output"][i]
            print(f"  output    : {out[:80] + '...' if len(out) > 80 else out}")

        # ⑦ 输入长度统计（帮判断数据是否太长，超过 2048 token 的微调时会被截断）
        inputs = [str(x) for x in ds["input"] if x]
        outputs = [str(x) for x in ds["output"]]
        avg_in = sum(len(s) for s in inputs) / max(len(inputs), 1)
        avg_out = sum(len(s) for s in outputs) / max(len(outputs), 1)
        print("-" * 60)
        print(f"[7] 平均 input 字符数 ≈ {avg_in:.0f}，平均 output 字符数 ≈ {avg_out:.0f}")

    print("=" * 60)
    print("[完成] 数据检查完毕，全部通过即可进入 Day12 微调训练。")


if __name__ == "__main__":
    main()
