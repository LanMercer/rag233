# -*- coding: utf-8 -*-
"""
提示词 A/B 对比实验脚本（配合 Day7 教程"加餐任务"使用）

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：Qwen2.5-3B-Instruct，4bit 量化（NF4）加载（复用 Day6/Day7 的加载配置）
- 数据   ：三组手工构造的对照实验（few-shot / 思维链 / 温度 × JSON）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    实验 A：zero-shot vs few-shot，统计"按指定格式输出"的合规率；
    实验 B：直接回答 vs 思维链（一步步思考），统计数学题答案正确率；
    实验 C：temperature=0.2 vs 0.9 做 JSON 抽取，统计"合法 JSON 比例"。
    每个条件重复 N 次（默认 5 次），用频率估计概率（大数定律）。

用法：
    python prompt_ab_test.py          # 三组实验各跑 5 次（约 10~20 分钟）
    python prompt_ab_test.py 3        # 三组实验各跑 3 次（快一点）
    python prompt_ab_test.py 3 A C    # 每组 3 次，只跑实验 A 和 C
"""

import json
import os
import re
import sys

# 如果走 hf-mirror 下载方案，加载前也设一下镜像，避免自动联网找模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 模型所在本地路径（Day6 download_qwen.py 下载的位置；如果实际路径不同请修改这里）
MODEL_PATH = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"

# 生成参数（Day6 实验过的最稳组合）
TOP_P = 0.9
MAX_NEW_TOKENS = 120

# 词分类实验的四个候选类别
CATEGORIES = ["水果", "动物", "交通工具", "植物"]

# 思维链实验的题目（question, 期望答案数字）。设计上避免干扰数字与答案重合。
MATH_QUESTIONS = [
    ("小明有 3 个苹果，又买了 5 个，吃掉 2 个，还剩几个？", "6"),
    ("一辆车每小时行驶 60 公里，行驶 3 小时，一共行驶多少公里？", "180"),
    ("书架上有 7 本书，借出去 3 本，又还回来 1 本，现在有几本？", "5"),
    ("一根绳子长 12 米，剪成每段 3 米，可以剪成几段？", "4"),
    ("班级有 25 人，其中女生 12 人，男生有多少人？", "13"),
]

# JSON 抽取实验的文本
JSON_TEXT = (
    "3月15日，王小明在上海浦东发展银行办理了一笔10万元的定期存款业务，"
    "理财经理李芳为其推荐了两款稳健型理财产品。"
)


def build_chat_prompt(system_text: str, user_text: str) -> str:
    """把 system / user 文字包装成 Qwen 使用的对话格式（ChatML）。"""
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate(user_text: str, temperature: float = 0.7) -> str:
    """让模型生成一段文本。temperature 作为实验变量传入。"""
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 判分规则（写得简单，只求"方向性"结论）
# ---------------------------------------------------------------------------
def is_format_ok(output: str) -> bool:
    """实验 A：输出是否为四个类别词之一（去掉句号/空白后精确匹配）。"""
    cleaned = re.sub(r"[。．.!！\s]", "", output)
    return cleaned in CATEGORIES


def is_correct(output: str, target: str) -> bool:
    """实验 B：输出里是否出现正确答案数字。"""
    return str(target) in re.findall(r"\d+", output)


def try_parse_json(text: str):
    """实验 C：截取第一个 { 到最后一个 } 之间的内容尝试 json 解析。"""
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1:
        return False
    try:
        json.loads(text[s:e + 1])
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 三个实验
# ---------------------------------------------------------------------------
def run_experiment_a(n: int) -> None:
    """zero-shot vs few-shot：词分类，统计格式合规率。"""
    print("\n" + "=" * 60)
    print("实验 A：few-shot 是否提升格式合规率？（词分类）")
    print("=" * 60)

    zero_prompt = (
        "请判断下面这个词属于哪一类，只输出类别两个字，不要解释。"
        f"可选类别：{'、'.join(CATEGORIES)}。\n\n输入：苹果"
    )
    few_prompt = (
        "请判断下面这个词属于哪一类，只输出类别两个字，不要解释。\n"
        f"可选类别：{'、'.join(CATEGORIES)}。\n\n"
        "示例：\n老虎 → 动物\n飞机 → 交通工具\n\n现在判断：苹果"
    )

    ok_z = sum(is_format_ok(generate(zero_prompt)) for _ in range(n))
    ok_f = sum(is_format_ok(generate(few_prompt)) for _ in range(n))

    print(f"  zero-shot（0 示例）：{ok_z}/{n} = {ok_z / n * 100:.1f}%")
    print(f"  few-shot（2 示例） ：{ok_f}/{n} = {ok_f / n * 100:.1f}%")
    print("  期望方向：few-shot 的合规率 ≥ zero-shot（示例把输出分布收窄）")


def run_experiment_b(n: int) -> None:
    """直接回答 vs 思维链：5 道数学题，统计正确率。"""
    print("\n" + "=" * 60)
    print("实验 B：思维链是否提升数学题正确率？")
    print("=" * 60)

    correct_direct = 0
    correct_cot = 0
    for question, answer in MATH_QUESTIONS:
        direct_prompt = f"请回答下面的数学题，只给出最终答案。\n\n题目：{question}"
        cot_prompt = (
            "请解决下面的数学题。请一步步思考，并把计算过程写出来，"
            f"最后单独一行给出最终答案。\n\n题目：{question}"
        )
        for _ in range(n):
            if is_correct(generate(direct_prompt), answer):
                correct_direct += 1
            if is_correct(generate(cot_prompt), answer):
                correct_cot += 1

    total = n * len(MATH_QUESTIONS)
    print(f"  直接回答：{correct_direct}/{total} = {correct_direct / total * 100:.1f}%")
    print(f"  一步步思考：{correct_cot}/{total} = {correct_cot / total * 100:.1f}%")
    print("  期望方向：思维链的正确率 ≥ 直接回答（把高方差估计拆成多步低方差）")


def run_experiment_c(n: int) -> None:
    """temperature=0.2 vs 0.9：JSON 抽取，统计合法 JSON 比例。"""
    print("\n" + "=" * 60)
    print("实验 C：低温是否让 JSON 输出更稳定？")
    print("=" * 60)

    json_prompt = (
        "请从下面的文本中抽取「人物 / 时间 / 地点 / 金额」四类信息。"
        "只输出 JSON 对象，字段为 person / date / location / amount，"
        "没有的信息填 null，不要输出任何解释：\n\n" + JSON_TEXT
    )

    ok_low = sum(try_parse_json(generate(json_prompt, temperature=0.2)) for _ in range(n))
    ok_high = sum(try_parse_json(generate(json_prompt, temperature=0.9)) for _ in range(n))

    print(f"  temperature=0.2：{ok_low}/{n} = {ok_low / n * 100:.1f}%")
    print(f"  temperature=0.9：{ok_high}/{n} = {ok_high / n * 100:.1f}%")
    print("  期望方向：低温的合法 JSON 比例 ≥ 高温（低温压熵 → 输出更稳）")


def main():
    global model, tokenizer

    # 命令行解析：第一个纯数字参数 = 每组次数 N；字母 A/B/C = 只跑对应实验
    args = sys.argv[1:]
    n = 5
    experiments = []
    for a in args:
        if a.isdigit():
            n = int(a)
        elif a.upper() in ("A", "B", "C"):
            experiments.append(a.upper())
    if not experiments:
        experiments = ["A", "B", "C"]

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

    print(f"\n[OK] 每组每条件重复 {n} 次，实验：{', '.join(experiments)}")
    print("[OK] 开始跑实验……（每个条件跑完才出下一行，请耐心等待）")

    for exp in experiments:
        if exp == "A":
            run_experiment_a(n)
        elif exp == "B":
            run_experiment_b(n)
        elif exp == "C":
            run_experiment_c(n)

    print("\n" + "=" * 60)
    print("全部实验完成。请把统计表截图保存，并把三组数字填进《提示词模板库.md》")
    print("的『A/B 实验结果』小节，作为 Day7 加餐验收材料。")
    print("=" * 60)


if __name__ == "__main__":
    main()
