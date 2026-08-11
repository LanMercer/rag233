# -*- coding: utf-8 -*-
"""
Qwen2.5-3B-Instruct 模型下载脚本（配合 Day6 教程第 2 步使用）

五件套说明（脚本在机器学习里的哪一环）：
- 模型   ：Qwen2.5-3B-Instruct（阿里开源的中文大模型，今天要部署的主角）
- 数据   ：无（只是把模型文件从网上下载到本地，不涉及训练数据）
- 损失   ：无（不训练）
- 优化器 ：无（不训练）
- 本脚本做的事（测试环节）：
    ① 优先尝试用 ModelScope（魔搭，阿里出品的模型社区，国内下载快）下载；
    ② 若没装 modelscope 则提示安装；
    ③ 下载完成后打印本地保存路径，供 qwen_inference.py 加载使用。
"""

import os
import sys

# 模型在网上的名字（ModelScope 与 HuggingFace 通用这个名字）
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# 下载到本地哪个文件夹（建议放在仓库根目录的 download 文件夹，方便管理）
LOCAL_DIR = r"D:\Lan\研究生\技术学习\大模型算法\download\Qwen2.5-3B-Instruct"


def download_with_modelscope():
    """方案 A：用 ModelScope（魔搭）下载，国内速度快，最推荐。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[X] 未检测到 modelscope 库，请先安装：pip install modelscope")
        print("    安装命令：conda activate llm 之后执行 pip install modelscope")
        sys.exit(1)

    print(f"[OK] 开始用 ModelScope 下载：{MODEL_ID}")
    print(f"[OK] 保存位置：{LOCAL_DIR}")
    print("[OK] 模型约 6GB，请耐心等待……（下载速度取决于你的网速）")
    snapshot_download(
        model_id=MODEL_ID,
        local_dir=LOCAL_DIR,
    )
    print(f"[OK] 下载完成！模型已保存在：{LOCAL_DIR}")
    print("[OK] 下一步：运行 python qwen_inference.py")


def download_with_hf_mirror():
    """方案 B：用 HuggingFace 国内镜像（hf-mirror）下载，作为备选。"""
    print("[OK] 开始用 hf-mirror 镜像下载（需要 huggingface_hub）……")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[X] 未检测到 huggingface_hub，请先执行：pip install huggingface_hub")
        sys.exit(1)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
    )
    print(f"[OK] 下载完成！模型已保存在：{LOCAL_DIR}")
    print("[OK] 下一步：运行 python qwen_inference.py")


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen2.5-3B-Instruct 下载工具")
    print("=" * 60)
    if os.path.exists(LOCAL_DIR) and os.listdir(LOCAL_DIR):
        print(f"[OK] 检测到 {LOCAL_DIR} 下已有文件，可能之前已下载过。")
        print("[OK] 建议直接运行 python qwen_inference.py 试试加载；")
        print("    若加载报错文件缺失，再重新执行本脚本补全下载。")
        sys.exit(0)

    print("[1] 方案 A：ModelScope（推荐，国内快）")
    print("[2] 方案 B：hf-mirror 镜像（备选）")
    choice = input("请选择下载方案（输入 1 或 2，默认 1）：").strip() or "1"
    if choice == "2":
        download_with_hf_mirror()
    else:
        download_with_modelscope()
