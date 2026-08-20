# -*- coding: utf-8 -*-
"""
check_env.py —— QLoRA 微调环境检查脚本（第三周 Day11 第 4 步）
========================================================================
【本脚本是什么】环境体检脚本，不涉及模型训练/测试。
【做什么】检查 QLoRA 微调 Qwen2.5-3B 需要的 5 个库版本 + torch/CUDA/显存，
         并输出"6G 显存下 QLoRA 方案是否可行"的结论。
【为什么做这个】Day12 就要跑真正微调训练，今天先把环境核清楚：
         ① 库版本对不对（transformers/peft 有官方兼容矩阵，版本不匹配会报错）；
         ② bitsandbytes（4bit 量化库）在 Windows 上是否可用（这是常见坑）；
         ③ 显存够不够（Day5 算过账：QLoRA 只训练 adapter，6G 可行）。
【五件套说明】不训练不测试，没有模型/损失/优化器。
========================================================================
"""
import sys

# Windows 控制台默认 GBK 打不出部分字符，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# QLoRA 方案目标（Day12 要用）：
#   模型 = Qwen2.5-3B-Instruct + 4bit（nf4）量化底座 + LoRA adapter（r=8, alpha=16）
#   参考兼容版本：transformers>=4.40, peft>=0.10, bitsandbytes>=0.43,
#                accelerate>=0.30, datasets>=2.19
MIN_VERSIONS = {
    "transformers": "4.40.0",
    "peft": "0.10.0",
    "bitsandbytes": "0.43.0",
    "accelerate": "0.30.0",
    "datasets": "2.19.0",
}


def main():
    print("=" * 60)
    print("QLoRA 微调环境检查（Windows + RTX 3060 6G 目标）")
    print("=" * 60)

    # ① 5 个关键库版本核对
    import importlib
    all_ok = True
    for lib, min_v in MIN_VERSIONS.items():
        try:
            mod = importlib.import_module(lib)
            ver = getattr(mod, "__version__", "未知")
            ok = _version_ge(ver, min_v)
            all_ok = all_ok and ok
            flag = "✅" if ok else "⚠️ 偏低"
            print(f"  {lib:<14s} 当前 {ver:<12s} 要求≥{min_v:<10s} {flag}")
        except ModuleNotFoundError:
            all_ok = False
            print(f"  {lib:<14s} ❌ 未安装（要求≥{min_v}）—— 用 pip install {lib} 安装")

    # ② torch / CUDA
    print("-" * 60)
    try:
        import torch
        print(f"  torch            {torch.__version__}")
        print(f"  CUDA 可用        : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU 名称         : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  显存总容量       : {props.total_memory / 1024**3:.1f} GB")
            free = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"  当前可用显存     : {free:.1f} GB")
        else:
            print("  ❌ CUDA 不可用！QLoRA 微调必须在 GPU 上跑")
            all_ok = False
    except ImportError:
        print("  torch ❌ 未安装")

    # ③ bitsandbytes 在 Windows 的兼容性说明
    print("-" * 60)
    print("  bitsandbytes 说明：Windows 需 0.43+ 才稳定支持 QLoRA 4bit 量化")
    print("  （本机已装，Day12 若 import 报错，先 pip install --upgrade bitsandbytes）")

    # ④ 结论
    print("=" * 60)
    if all_ok:
        print("结论：✅ 环境基本就绪，6G 显存 + QLoRA 方案可行！")
        print("      Day12 将用 Qwen2.5-3B-Instruct + 4bit + LoRA(r=8) 训练 adapter。")
    else:
        print("结论：❌ 环境还有缺口，先按上面提示补齐，再进入 Day12。")
    print("=" * 60)


def _version_ge(cur, req):
    """简单的版本号比较（只比数字段，忽略 alpha/beta 后缀）"""
    try:
        def nums(v):
            out = []
            for part in str(v).split("."):
                d = ""
                for ch in part:
                    if ch.isdigit():
                        d += ch
                    else:
                        break
                out.append(int(d) if d else 0)
            return out
        return nums(cur) >= nums(req)
    except Exception:
        return True  # 解析失败时放行，交给 Day12 实际报错来暴露


if __name__ == "__main__":
    main()
