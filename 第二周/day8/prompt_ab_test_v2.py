# -*- coding: utf-8 -*-
"""
提示词进阶 A/B 对比实验 v2（配合 Day8 加餐使用）

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：Qwen2.5-3B-Instruct，4bit 量化（NF4）加载（复用 Day6 的加载配置）
- 数据   ：三组对照实验的提示词 + 题目/合同文本（自动构造，重复抽样 N 次）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    ① 4bit 加载本地模型；
    ② 三组 A/B 对照（控制变量、只改一个条件、重复抽样 N 次、频率≈概率）：
        实验 A：zero-shot vs few-shot，任务=3 层嵌套 JSON 抽取（合同信息）
        实验 B：直接回答 vs 一步步思考（思维链），任务=小学数学题（N 加大复测 Day7 结论）
        实验 C：temperature=0.2 vs 0.9，任务=3 层嵌套 JSON 抽取
    ③ 自动判分（JSON 嵌套字段校验 / 答案数字匹配）+ 统计合规率/正确率并打印。

设计说明（对应 Day7 加餐复盘的三条改进方向）：
- Day7 实验 A 用"词分类"太简单、两组都 100% -> 换成"3 层嵌套 JSON"这种困难输出任务；
- Day7 实验 C N=5 样本太少、prompt 约束太死 -> N 默认加到 20，并放宽 prompt 让温度差异暴露；
- Day7 实验 B 已显著（60% -> 84%）-> 加大到 N=20 复测巩固结论。

用法：
    python prompt_ab_test_v2.py        # 默认每组每条件 20 次，跑全部三个实验
    python prompt_ab_test_v2.py 3      # 每组每条件 3 次（先小跑验证逻辑）
    python prompt_ab_test_v2.py 20 A C # 只跑实验 A 和 C（每组 20 次）
"""

import os
import re
import sys

# 如果走 hf-mirror 下载方案，加载前也设一下镜像，避免自动联网找模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 模型所在本地路径（Day6 download_qwen.py 下载的位置；如果实际路径不同请修改这里）
MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"

TOP_P = 0.9
MAX_NEW_TOKENS = 250

# ---------------------------------------------------------------------------
# 实验素材
# ---------------------------------------------------------------------------

# 实验 A / C 共用：一份合同文本（要求抽 3 层嵌套 JSON）
CONTRACT_TEXT = (
    "合同编号 HT20250801。甲方：杭州云启科技有限公司，乙方：上海智算软件有限公司。"
    "合同总金额 100000 元，签订日期 2025 年 8 月 1 日。"
    "若乙方逾期交付，每逾期 1 天按合同总金额的 10% 支付违约金，逾期超过 30 天甲方可单方解约。"
)

# zero-shot：只下命令，不给示例
EXPERIMENT_A_ZERO_SHOT_PROMPT = (
    "请从下面的合同信息中抽取结构化字段，只输出一个 JSON 对象，不要任何解释：\n\n"
    "字段结构：\n"
    "{ \"contract_no\": 合同编号, \"parties\": {\"buyer\": 甲方, \"seller\": 乙方}, "
    "\"amount\": 合同总金额, \"date\": 签订日期, "
    "\"penalty\": {\"rate\": 违约金比例, \"days\": 宽限天数} }\n\n"
    "合同信息：\n" + CONTRACT_TEXT
)

# few-shot：zero-shot 基础上加 1 个完整示例
EXPERIMENT_A_FEW_SHOT_PROMPT = (
    "请从下面的合同信息中抽取结构化字段，只输出一个 JSON 对象，不要任何解释。\n\n"
    "示例：\n"
    "合同信息：合同编号 HT20240101。甲方：北京一公司，乙方：深圳二公司。"
    "合同总金额 50000 元，签订日期 2024 年 1 月 1 日。违约金比例 5%，宽限 15 天。\n"
    "输出："
    "{\"contract_no\": \"HT20240101\", "
    "\"parties\": {\"buyer\": \"北京一公司\", \"seller\": \"深圳二公司\"}, "
    "\"amount\": 50000, \"date\": \"2024年1月1日\", "
    "\"penalty\": {\"rate\": 0.05, \"days\": 15}}\n\n"
    "现在请抽取下面的合同：\n"
    "合同信息：\n" + CONTRACT_TEXT
)

# 实验 B：5 道小学数学题（直接问 vs 加"一步步思考"）
MATH_QUESTIONS = [
    ("小明有 12 个苹果，分给 3 个朋友每人 2 个，又买了 5 个，现在有多少个苹果？", "11"),
    ("一本书 240 页，第一天看了 1/4，第二天看了剩下的 1/3，还剩多少页没看？", "120"),
    ("某商品原价 200 元，打八折后再降价 10 元，现价多少元？", "150"),
    ("一项工程甲单独做 6 天完成，乙单独做 12 天完成，两人合作几天完成？", "4"),
    ("火车每小时行 90 公里，行驶 2.5 小时后还剩全程的 1/4，全程多少公里？", "300"),
]


def build_chat_prompt(system_text: str, user_text: str) -> str:
    """把 system / user 文字包装成 Qwen 使用的对话格式（ChatML）。"""
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate(user_text: str, temperature: float) -> str:
    """让模型生成一段文本。"""
    prompt = build_chat_prompt(
        "你是一个乐于助人、严谨可靠的中文助手。请严格按用户提示词的要求输出。",
        user_text,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 判分函数（写简单、方向性即可，和 Day7 一样为"感受变数字"服务）
# ---------------------------------------------------------------------------

def _extract_json(text: str):
    """截取文本中第一对平衡的花括号并尝试解析为 dict；失败返回 None。"""
    start = text.find("{")
    if start < 0:
        return None
    # 从 { 开始数括号配平，找到配平的 }
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    import json
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def is_nested_json_ok(text: str) -> bool:
    """3 层嵌套 JSON 校验：解析成功 + 各层字段齐全即算合规（值不精确核对）。"""
    data = _extract_json(text)
    if not isinstance(data, dict):
        return False
    parties = data.get("parties")
    penalty = data.get("penalty")
    return (
        "contract_no" in data
        and "amount" in data
        and "date" in data
        and isinstance(parties, dict)
        and "buyer" in parties
        and "seller" in parties
        and isinstance(penalty, dict)
        and "rate" in penalty
        and "days" in penalty
    )


def is_correct(text: str, answer: str) -> bool:
    """判断输出里是否包含正确数字（简单方向性判分）。"""
    return answer in text


# ---------------------------------------------------------------------------
# 三个实验
# ---------------------------------------------------------------------------

def run_experiment_a(n: int) -> None:
    """实验 A：few-shot 是否提升复杂 JSON 合规率？（zero-shot vs few-shot，各 n 次）"""
    print("\n" + "=" * 60)
    print(f"===== 实验 A：few-shot 是否提升复杂 JSON 合规率？（各 {n} 次）=====")
    print("=" * 60)
    zero_ok = 0
    few_ok = 0
    for i in range(n):
        zero_ok += int(is_nested_json_ok(
            generate(EXPERIMENT_A_ZERO_SHOT_PROMPT, temperature=0.3)))
        few_ok += int(is_nested_json_ok(
            generate(EXPERIMENT_A_FEW_SHOT_PROMPT, temperature=0.3)))
        print(f"  进度：{i + 1}/{n}  zero={zero_ok} few={few_ok}")
    print(f"  zero-shot（0 示例）：{zero_ok}/{n} = {zero_ok / n * 100:.1f}%")
    print(f"  few-shot（1 示例） ：{few_ok}/{n} = {few_ok / n * 100:.1f}%")


def run_experiment_b(n: int) -> None:
    """实验 B：思维链复测（直接回答 vs 一步步思考，各 n 次，5 题轮流）"""
    print("\n" + "=" * 60)
    print(f"===== 实验 B：思维链复测（直接回答 vs 一步步思考，各 {n} 次，5 题轮流）=====")
    print("=" * 60)
    direct_ok = 0
    cot_ok = 0
    for i in range(n):
        question, answer = MATH_QUESTIONS[i % len(MATH_QUESTIONS)]
        direct_prompt = f"请回答：{question}（只输出最终答案数字）"
        cot_prompt = f"请一步步思考，然后回答：{question}（最后以「答案：」开头只写数字）"
        direct_ok += int(is_correct(generate(direct_prompt, temperature=0.3), answer))
        cot_ok += int(is_correct(generate(cot_prompt, temperature=0.3), answer))
        print(f"  进度：{i + 1}/{n}  direct={direct_ok} cot={cot_ok}")
    print(f"  直接回答    ：{direct_ok}/{n} = {direct_ok / n * 100:.1f}%")
    print(f"  一步步思考  ：{cot_ok}/{n} = {cot_ok / n * 100:.1f}%")


def run_experiment_c(n: int) -> None:
    """实验 C：温度 × 复杂 JSON 合法率（temperature=0.2 vs 0.9，各 n 次）"""
    print("\n" + "=" * 60)
    print(f"===== 实验 C：低温是否让复杂 JSON 更稳？（各 {n} 次）=====")
    print("=" * 60)
    low_ok = 0
    high_ok = 0
    for i in range(n):
        low_ok += int(is_nested_json_ok(
            generate(EXPERIMENT_A_FEW_SHOT_PROMPT, temperature=0.2)))
        high_ok += int(is_nested_json_ok(
            generate(EXPERIMENT_A_FEW_SHOT_PROMPT, temperature=0.9)))
        print(f"  进度：{i + 1}/{n}  low={low_ok} high={high_ok}")
    print(f"  temperature=0.2：{low_ok}/{n} = {low_ok / n * 100:.1f}%")
    print(f"  temperature=0.9：{high_ok}/{n} = {high_ok / n * 100:.1f}%")


def main():
    global model, tokenizer

    print("=" * 60)
    print("第 1 步：4bit 量化加载 Qwen2.5-3B-Instruct")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[X] 找不到模型文件夹：{MODEL_PATH}")
        print("[X] 请先运行第二周\\day6\\download_qwen.py 下载模型，或修改脚本里的 MODEL_PATH。")
        sys.exit(1)

    print("[OK] GPU 可用：", torch.cuda.is_available())
    print("[OK] 显卡：", torch.cuda.get_device_name(0))

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("[OK] 正在加载分词器（Tokenizer）……")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("[OK] 正在加载模型（4bit 量化），首次约需 1~3 分钟……")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # 解析命令行参数：python prompt_ab_test_v2.py [N] [A B C]
    n = 20
    experiments = ["A", "B", "C"]
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        n = int(sys.argv[1])
        experiments = sys.argv[2:] if len(sys.argv) > 2 else experiments
    elif len(sys.argv) > 1:
        experiments = sys.argv[1:]

    print(f"第 2 步：开始实验（每组每条件 {n} 次）")
    if "A" in experiments:
        run_experiment_a(n)
    if "B" in experiments:
        run_experiment_b(n)
    if "C" in experiments:
        run_experiment_c(n)

    print("=" * 60)
    print("实验结束。请把统计表截图保存（day8_进阶AB对比实验截图.png），")
    print("并把数字填进《提示词模板库.md》v2 末尾的 A/B 实验结果表。")
    print("=" * 60)


if __name__ == "__main__":
    main()
