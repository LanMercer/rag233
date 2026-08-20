# -*- coding: utf-8 -*-
"""
make_sft_data.py —— 自制 SFT 微调数据生成脚本（第三周 Day11 第 3 步）
========================================================================
【本脚本是什么】纯数据准备脚本，不涉及模型训练/测试（为 Day12 的 LoRA 训练铺路）。
【做什么】用"模板 + 随机组合"的方式，自动生成 200~500 条中文"指令-回复"数据，
         题材与项目领域一致（机器人行业公告 / 研报问答），保存为 JSON 三字段格式：
         instruction（指令）/ input（输入材料）/ output（标准答案）。
【为什么这么写】SFT = Supervised Fine-Tuning（监督微调），需要"问题-答案"配对数据。
        手写 300 条太累，用代码模板批量生成，保证字段格式统一、数量足、可复现
        （固定随机种子，每次运行结果一致）。
【五件套说明】本脚本不训练也不测试模型，所以没有"模型 / 损失函数 / 优化器"。
        Day12 的 train_lora.py 才是真正训练脚本，五件套会写在那里。
========================================================================
"""
import json
import random
from collections import Counter

# 固定随机种子：保证每次运行生成的数据完全一致（可复现、可验收）
random.seed(42)

# ----------------------------------------------------------------------
# 语料池：机器人产业链公司 / 指标 / 公告类型 / 研报评级 / 观点 / 风险 等
# ----------------------------------------------------------------------
COMPANIES = [
    "埃斯顿", "汇川技术", "绿的谐波", "拓斯达", "新时达",
    "埃夫特", "优必选", "鸣志电器", "禾川科技", "双环传动",
    "雷赛智能", "兆威机电", "中大力德", "江苏北人", "凯尔达",
    "奥普特", "凌云光", "大族激光", "五洲新春", "巨轮智能",
]

# 三类指标：财务（亿元）/ 数量（台、万套）/ 比例（%）
MONEY_INDICATORS = ["营业收入", "净利润", "归母净利润", "研发投入"]
QUANTITY_INDICATORS = [("机器人整机出货量", "台"), ("谐波减速器销量", "万套"),
                       ("伺服电机销量", "万台"), ("控制器销量", "万台")]
RATIO_INDICATORS = ["毛利率", "净利率"]

GROWTH_RATES = ["3.2%", "8.5%", "15.6%", "-5.1%", "22.3%", "11.8%", "0.9%", "18.7%"]

ANNOUNCE_TYPES = [
    "年度报告", "半年度报告", "业绩预告", "重大订单公告",
    "新品发布公告", "产能扩建公告", "战略合作公告", "股权激励公告",
]

RATING_WORDS = [
    "买入", "增持", "中性", "推荐", "强烈推荐",
]

OUTLOOK_WORDS = [
    "人形机器人产业化加速，核心零部件国产替代空间大",
    "工业机器人需求回暖，出货量同比高增",
    "具身智能带动伺服/减速器需求向上",
    "海外市场拓展顺利，全球化布局深化",
    "政策利好持续释放，行业景气度上行",
]

RISK_WORDS = [
    "下游资本开支不及预期", "原材料价格波动", "行业竞争加剧",
    "技术迭代风险", "地缘政治与供应链风险",
]

# 概念解释题：机器人技术/统计概念池（用概率与数学语言讲，衔接用户数学背景）
CONCEPT_POOL = [
    ("谐波减速器", "一种靠弹性变形传动的精密减速装置，机器人关节'减速增扭'的核心零件"),
    ("RV减速器", "摆线针轮减速器，承载大、精度高，常用于机器人重载大关节，与谐波减速器互补"),
    ("伺服电机", "能精确控制转速和位置的电机，机器人每个关节的'发动机'，配合编码器形成闭环"),
    ("控制器", "机器人的'大脑'，负责运动规划、轨迹插补和伺服指令下发"),
    ("六轴机器人", "有6个旋转关节的工业机器人，自由度=6，可到达工作空间内任意位姿，是工业机器人主流形态"),
    ("SCARA机器人", "水平关节型机器人，垂直方向刚度高、水平面灵活，常用于装配、贴片等平面内作业"),
    ("协作机器人", "能与人在同一空间安全协作、自带力控的机器人，无需安全围栏"),
    ("AGV", "自动导引车，沿预设路径自动搬运物料，是智能工厂里的'移动搬运工'"),
    ("末端执行器", "装在机器人手腕上直接接触作业对象的部件，如夹爪、吸盘、焊枪，类比人类的'手'"),
    ("运动规划", "给定起点终点求一条可行且平滑的轨迹，常建模为最优化/搜索问题——数学优化在机器人里的落地"),
    ("轨迹插补", "在给定路径点之间'补点'生成连续运动，类比数值分析里的插值"),
    ("重复定位精度", "机器人多次回到同一指令位置时实际位置的离散程度，用标准差衡量——数学上就是精度散布"),
    ("负载能力", "机器人末端能承受的最大载荷（含手爪与工件），决定能搬多重的物件"),
    ("自由度(DOF)", "机器人可独立运动的关节数量，自由度越多动作越灵活，机械臂一般6自由度"),
    ("减速比", "输入转速与输出转速之比，减速比越大输出越慢但扭矩越大，是传动系统的核心参数"),
    ("PID控制", "比例-积分-微分控制器，按误差及其积分、微分调节输出，让关节角度收敛到目标——经典反馈控制"),
    ("机器视觉", "用相机'看'工件位置与姿态并引导机器人抓取，相当于给机器人装上眼睛"),
    ("具身智能", "把大模型能力装进机器人本体，让机器人在物理世界感知、规划并执行——AI与大模型的下一个主战场"),
    ("传感器", "把物理量（位置/力/温度）变成电信号的器件，机器人'感知世界'的感官"),
    ("马尔可夫决策过程(MDP)", "机器人决策的数学骨架：当前状态+动作→下一状态+奖励，目标是最大化累计回报，是强化学习的建模框架"),
    ("置信区间", "用样本估计参数时给出一个以一定概率覆盖真值的区间，机器人标定/测量中常用95%置信区间表示精度"),
    ("泊松过程", "描述'随机到达'事件的计数过程，如产线工件随机到达机器人工位的间隔——排队论的数学基础"),
    ("方差与标准差", "衡量数据离散程度，机器人重复定位精度就是位置误差的标准差，是'稳定性'的量化"),
    ("正态分布", "大量独立微小误差叠加后近似服从正态分布，机器人定位误差常假设为正态——中心极限定理的体现"),
]

# 合规判断题池：证券合规 + 机器人行业特色
JUDGE_POOL = [
    ("订单", "与某客户签订重大订单合同", "需要披露。重大订单达到披露标准的须及时公告，说明金额、交付周期及风险。"),
    ("减持", "控股股东计划减持股份", "需要。控股股东减持需按规定提前披露，并遵守比例和时间限制。"),
    ("回购", "回购股份用于股权激励", "合规。回购用于股权激励需经董事会/股东大会审议，符合条件即可实施。"),
    ("出口管制", "拟向特定地区出口工业机器人", "需要评估。涉及出口管制物项的须办理出口许可，并评估合规风险。"),
    ("安全认证", "新产品取得安全认证证书", "合规。通过认证并取得证书即可投放市场。"),
    ("专利", "核心专利被提起无效宣告", "需要披露。涉及重大专利纠纷达到披露标准的须公告并评估影响。"),
    ("产能扩张", "拟新建减速器生产基地", "需要披露。重大投资达到标准的须公告资金来源、建设周期及预期产能。"),
    ("数据合规", "机器人采集的数据跨境传输", "需要评估。涉及个人信息/重要数据出境的须依法进行安全评估或备案。"),
]

# ----------------------------------------------------------------------
# 6 类任务模板函数：每个返回 (instruction, input, output)
# ----------------------------------------------------------------------
def gen_extract():
    """任务1：公告信息抽取——给一段公告，抽出指定指标"""
    comp = random.choice(COMPANIES)
    a_type = random.choice(ANNOUNCE_TYPES)
    year = random.choice(["2024", "2023"])
    growth = random.choice(GROWTH_RATES)
    # 按指标类型生成对应的"数值 + 单位"和句子
    kind = random.choice(["money", "quantity", "ratio"])
    if kind == "money":
        ind = random.choice(MONEY_INDICATORS)
        value = round(random.uniform(5, 300), 2)
        unit = "亿元"
    elif kind == "quantity":
        ind, unit = random.choice(QUANTITY_INDICATORS)
        value = round(random.uniform(1, 50), 2)
    else:
        ind = random.choice(RATIO_INDICATORS)
        value = round(random.uniform(10, 60), 2)
        unit = "%"
    instruction = f"请从以下{comp}的{a_type}中，提取{ind}的数据。"
    input_text = (
        f"{comp}发布{a_type}。报告期内公司{ind}为 {value} {unit}，"
        f"同比{'' if growth.startswith('-') else '增长'}{growth}。"
        f"截至{year}年末，公司订单金额达到 {round(value * 8, 2)} 亿元。"
    )
    output = f"{comp}{year}年{ind}为 {value} {unit}，同比{'' if growth.startswith('-') else '增长'}{growth}。"
    return instruction, input_text, output


def gen_qa():
    """任务2：研报观点问答——根据研报摘要，回答公司投资建议"""
    comp = random.choice(COMPANIES)
    rating = random.choice(RATING_WORDS)
    outlook = random.choice(OUTLOOK_WORDS)
    risk = random.choice(RISK_WORDS)
    target = round(random.uniform(10, 200), 1)
    instruction = f"根据以下研报摘要，{comp}的投资评级和建议是什么？"
    input_text = (
        f"某券商发布关于{comp}的深度研报，给予\"{rating}\"评级，目标价 {target} 元。"
        f"核心观点：{outlook}。风险提示：{risk}。"
    )
    output = f"研报给予{comp}\"{rating}\"评级，目标价 {target} 元。{outlook}。"
    return instruction, input_text, output


def gen_explain():
    """任务3：概念解释——用大白话解释机器人技术/统计概念（数学视角）"""
    concept, explain = random.choice(CONCEPT_POOL)
    instruction = f"请用通俗的语言解释什么是{concept}？"
    return instruction, "", f"{concept}：{explain}。"


def gen_calc():
    """任务4：计算题——给数据算结果（数学视角，计算过程要写出来）"""
    mode = random.choice(["growth", "margin", "roe", "shipment", "utilization"])
    comp = random.choice(COMPANIES)
    if mode == "growth":
        # 已知去年与今年净利润，算同比增速
        last = round(random.uniform(5, 100), 2)
        rate = round(random.uniform(-0.2, 0.6), 3)
        cur = round(last * (1 + rate), 2)
        rate_pct = round(rate * 100, 1)
        instruction = f"{comp}去年净利润 {last} 亿元，今年净利润 {cur} 亿元，请计算同比增长率。"
        input_text = ""
        output = (
            f"同比增长率 = (今年 - 去年) / 去年 × 100% = ({cur} - {last}) / {last} × 100% "
            f"= {round(cur - last, 2)} / {last} × 100% ≈ {rate_pct}%。"
        )
    elif mode == "margin":
        # 已知营收与净利润，算净利率
        rev = round(random.uniform(20, 300), 2)
        profit = round(rev * random.uniform(0.05, 0.3), 2)
        margin = round(profit / rev * 100, 2)
        instruction = f"{comp}营业收入 {rev} 亿元，净利润 {profit} 亿元，请计算净利率。"
        input_text = ""
        output = (
            f"净利率 = 净利润 / 营业收入 × 100% = {profit} / {rev} × 100% ≈ {margin}%。"
        )
    elif mode == "roe":
        # 已知净利润与净资产，算 ROE
        equity = round(random.uniform(50, 300), 2)
        profit = round(equity * random.uniform(0.08, 0.25), 2)
        roe = round(profit / equity * 100, 2)
        instruction = f"{comp}净利润 {profit} 亿元，股东权益 {equity} 亿元，请计算净资产收益率(ROE)。"
        input_text = ""
        output = (
            f"ROE = 净利润 / 股东权益 × 100% = {profit} / {equity} × 100% ≈ {roe}%。"
        )
    elif mode == "shipment":
        # 已知去年与今年机器人出货量（台），算同比增速
        last = round(random.uniform(1000, 20000), 0)
        rate = round(random.uniform(-0.1, 0.8), 3)
        cur = round(last * (1 + rate), 0)
        rate_pct = round(rate * 100, 1)
        instruction = f"{comp}去年机器人整机出货 {int(last)} 台，今年出货 {int(cur)} 台，请计算出货量同比增长率。"
        input_text = ""
        output = (
            f"出货量同比增速 = (今年 - 去年) / 去年 × 100% = ({int(cur)} - {int(last)}) / {int(last)} × 100% "
            f"= {round(cur - last, 0)} / {int(last)} × 100% ≈ {rate_pct}%。"
        )
    else:
        # 已知实际产量与设计产能，算产能利用率
        cap = round(random.uniform(5000, 50000), 0)
        out = round(cap * random.uniform(0.5, 1.0), 0)
        util = round(out / cap * 100, 1)
        instruction = f"{comp}减速器生产基地设计产能 {int(cap)} 万套/年，实际产量 {int(out)} 万套，请计算产能利用率。"
        input_text = ""
        output = (
            f"产能利用率 = 实际产量 / 设计产能 × 100% = {int(out)} / {int(cap)} × 100% ≈ {util}%。"
        )
    return instruction, input_text, output


def gen_summary():
    """任务5：公告要点总结——总结一段公告的核心信息"""
    comp = random.choice(COMPANIES)
    a_type = random.choice(ANNOUNCE_TYPES)
    ind = random.choice(MONEY_INDICATORS)
    value = round(random.uniform(5, 300), 2)
    growth = random.choice(GROWTH_RATES)
    n1 = random.randint(1, 20)
    n2 = random.randint(1, 10)
    instruction = f"请总结以下{comp}公告的核心要点。"
    input_text = (
        f"{comp}今日发布{a_type}。公司全年{ind}实现 {value} 亿元，同比{'' if growth.startswith('-') else '增长'}{growth}；"
        f"报告期内公司新增 {n1} 项发明专利，完成 {n2} 项重大业务布局。公司表示将持续加大研发投入，推进国产替代。"
    )
    output = (
        f"① {ind}实现 {value} 亿元，同比{'' if growth.startswith('-') else '增长'}{growth}；"
        f"② 新增专利 {n1} 项，完成业务布局 {n2} 项；③ 公司持续加大研发投入、推进国产替代。"
    )
    return instruction, input_text, output


def gen_judge():
    """任务6：合规判断——判断公告中的行为是否符合常见合规要求"""
    comp = random.choice(COMPANIES)
    action, desc, answer = random.choice(JUDGE_POOL)
    instruction = f"{comp}{desc}，请问是否合规/是否需要披露？"
    return instruction, "", f"{comp}{desc}：{answer}"


# 任务函数列表（权重表示该类数据生成多少条）
TASKS = [
    (gen_extract, "公告信息抽取"),
    (gen_qa, "研报观点问答"),
    (gen_explain, "概念解释"),
    (gen_calc, "数据计算"),
    (gen_summary, "公告要点总结"),
    (gen_judge, "合规判断"),
]
WEIGHTS = [60, 60, 40, 40, 50, 50]  # 共 300 条

def main():
    total = 300  # 落在 200~500 区间
    items = []
    categories = []
    seen = set()  # 去重集合：instruction+input 相同视为重复
    attempts = 0
    max_attempts = total * 30  # 防止死循环上限

    # 按权重分配条数，生成到无重复为止
    while len(items) < total and attempts < max_attempts:
        attempts += 1
        fn, cat = random.choices(TASKS, weights=WEIGHTS, k=1)[0]
        instruction, input_text, output = fn()
        key = (instruction.strip(), input_text.strip())
        if key in seen:
            continue  # 重复则跳过，重新生成
        seen.add(key)
        items.append({
            "instruction": instruction,
            "input": input_text,
            "output": output,
        })
        categories.append(cat)

    # 保存 JSON（ensure_ascii=False 保证中文正常，utf-8 编码）
    out_path = "sft_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("SFT 数据生成完成")
    print("=" * 60)
    print(f"保存路径     : {out_path}")
    print(f"总条数       : {len(items)} 条")
    print(f"字段         : instruction / input / output（三字段 SFT 标准格式）")
    print("-" * 60)
    print("类别分布：")
    for cat, cnt in Counter(categories).most_common():
        print(f"  {cat:<10s}: {cnt} 条")
    print("-" * 60)
    print("抽样前 3 条：")
    for i, item in enumerate(items[:3]):
        print(f"  [{i+1}] instruction: {item['instruction']}")
        print(f"      input     : {item['input'][:50]}..." if item["input"] else "      input     : (空)")
        print(f"      output    : {item['output'][:60]}...")
    print("=" * 60)
    print("[完成] 数据已生成，接下来运行 check_sft_data.py 检查数据质量。")

if __name__ == "__main__":
    main()
