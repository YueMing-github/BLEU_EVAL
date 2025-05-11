#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="运行Qwen模型翻译评测")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/Users/bytedance/Downloads/qwen2.5-7b-cpt-policy",
        help="模型路径"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10, 
        help="评估样本数量，设置为-1表示使用全部样本"
    )
    parser.add_argument(
        "--direction", 
        type=str, 
        choices=["zh-en", "en-zh", "both"], 
        default="zh-en",
        help="翻译方向: zh-en (中译英), en-zh (英译中), both (两个方向都评测)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="调试模式，显示更多信息"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="评测全部样本，等同于设置 --num_samples=-1"
    )
    parser.add_argument(
        "--recreate_data", 
        action="store_true",
        help="重新创建测试数据集，覆盖已有数据"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 如果设置了--all参数，则使用全部样本
    if args.all:
        args.num_samples = -1
    
    # 获取当前Python解释器路径
    python_executable = sys.executable
    
    # 1. 首先安装依赖
    print("步骤1: 安装所需依赖")
    subprocess.call([python_executable, "install_deps.py"])
    
    # 2. 确认测试数据集
    print("步骤2: 确认测试数据集")
    test_data_dir = "./test_data"
    zh_en_test_path = os.path.join(test_data_dir, "zh_en_test.json")
    en_zh_test_path = os.path.join(test_data_dir, "en_zh_test.json")
    
    # 创建测试数据目录（如果不存在）
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir, exist_ok=True)
        
    # 检查是否需要创建测试数据
    need_create_data = args.recreate_data or not (os.path.exists(zh_en_test_path) and os.path.exists(en_zh_test_path))
    
    if need_create_data:
        print("正在创建测试数据集...")
        subprocess.call([python_executable, "create_test_dataset.py"])
    else:
        print(f"使用已存在的测试数据集: {test_data_dir}")
        print(f"- 中译英数据: {zh_en_test_path}")
        print(f"- 英译中数据: {en_zh_test_path}")
        print("如需重新创建数据集，请使用 --recreate_data 参数")
    
    # 3. 需要评测的翻译方向
    print("\n步骤3: 准备评测...")
    
    # 如果选择了both，先评测中译英，再评测英译中
    if args.direction == "both":
        directions = ["zh-en", "en-zh"]
    else:
        directions = [args.direction]
    
    for direction in directions:
        # 配置评测参数
        if direction == "zh-en":
            src_lang = "zh"
            tgt_lang = "en"
            dataset_path = zh_en_test_path
            print(f"\n开始评测 中译英 翻译能力...")
        else:  # en-zh
            src_lang = "en"
            tgt_lang = "zh"
            dataset_path = en_zh_test_path
            print(f"\n开始评测 英译中 翻译能力...")
        
        # 构建评测命令
        cmd = [
            python_executable, "evaluate.py",
            "--model_path", args.model_path,
            "--dataset_path", dataset_path,
            "--src_lang", src_lang,
            "--tgt_lang", tgt_lang,
            "--num_samples", str(args.num_samples)
        ]
        
        # 如果是调试模式，添加debug参数
        if args.debug:
            cmd.append("--debug")
        
        # 执行评测
        subprocess.call(cmd)
        
        # 重命名结果文件以区分不同方向
        if os.path.exists("translation_results.txt"):
            new_filename = f"translation_results_{direction}.txt"
            os.rename("translation_results.txt", new_filename)
            print(f"翻译结果已保存到 {new_filename}")
    
    # 4. 输出评测完成信息
    if args.direction == "both":
        print("\n中英双向评测完成！结果分别保存在 translation_results_zh-en.txt 和 translation_results_en-zh.txt")
    else:
        print(f"\n评测完成！结果保存在 translation_results_{args.direction}.txt")

if __name__ == "__main__":
    main() 