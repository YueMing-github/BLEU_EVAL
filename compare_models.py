#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="比较不同模型在翻译任务上的表现")
    parser.add_argument(
        "--model_list", 
        type=str, 
        nargs='+',
        required=True,
        help="要比较的模型路径列表，用空格分隔多个模型路径"
    )
    parser.add_argument(
        "--model_names", 
        type=str, 
        nargs='+',
        help="模型的显示名称，顺序与model_list对应，如不提供则使用模型路径"
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
        default="both",
        help="翻译方向: zh-en (中译英), en-zh (英译中), both (两个方向都评测)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="调试模式，显示更多信息"
    )
    parser.add_argument(
        "--recreate_data", 
        action="store_true",
        help="重新创建测试数据集，覆盖已有数据"
    )
    return parser.parse_args()

def extract_bleu_score(result_file):
    """从结果文件中提取BLEU分数"""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找BLEU分数行
            for line in content.split('\n'):
                if '分数:' in line:
                    # 提取数字
                    bleu_score = float(line.split('分数:')[1].strip().split()[0])
                    return bleu_score
    except Exception as e:
        print(f"提取BLEU分数时出错: {e}")
    return None

def main():
    args = parse_args()
    
    # 确保模型名称列表长度与模型路径列表相同
    if args.model_names is None or len(args.model_names) != len(args.model_list):
        args.model_names = [os.path.basename(model_path) for model_path in args.model_list]
    
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
    
    # 3. 创建结果目录
    results_dir = "./comparison_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    # 记录当前时间，用于生成唯一的结果文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 4. 设置要评测的翻译方向
    if args.direction == "both":
        directions = ["zh-en", "en-zh"]
    else:
        directions = [args.direction]
    
    # 5. 创建结果汇总
    comparison_results = {
        "timestamp": timestamp,
        "num_samples": args.num_samples,
        "models": args.model_names,
        "results": {}
    }
    
    for direction in directions:
        print(f"\n开始评测 {direction} 翻译方向...")
        comparison_results["results"][direction] = {}
        
        # 配置评测参数
        if direction == "zh-en":
            src_lang = "zh"
            tgt_lang = "en"
            dataset_path = zh_en_test_path
        else:  # en-zh
            src_lang = "en"
            tgt_lang = "zh"
            dataset_path = en_zh_test_path
        
        # 对每个模型进行评测
        for i, (model_path, model_name) in enumerate(zip(args.model_list, args.model_names)):
            print(f"\n[{i+1}/{len(args.model_list)}] 评测模型: {model_name}")
            
            # 构建评测命令
            cmd = [
                python_executable, "evaluate.py",
                "--model_path", model_path,
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
            
            # 重命名结果文件以区分不同模型
            if os.path.exists("translation_results.txt"):
                result_filename = f"translation_results_{direction}_{model_name}.txt"
                result_path = os.path.join(results_dir, result_filename)
                # 如果文件已存在，先删除
                if os.path.exists(result_path):
                    os.remove(result_path)
                os.rename("translation_results.txt", result_path)
                print(f"翻译结果已保存到 {result_path}")
                
                # 提取BLEU分数
                bleu_score = extract_bleu_score(result_path)
                if bleu_score is not None:
                    comparison_results["results"][direction][model_name] = bleu_score
                    print(f"BLEU分数: {bleu_score:.2f}")
    
    # 6. 保存比较结果
    comparison_summary_file = os.path.join(results_dir, f"comparison_summary_{timestamp}.json")
    with open(comparison_summary_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    # 7. 生成比较报告
    report_file = os.path.join(results_dir, f"comparison_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 模型翻译能力比较报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数量: {args.num_samples}\n\n")
        
        # 打印每个方向的结果
        for direction in directions:
            f.write(f"## {direction} 翻译方向\n\n")
            f.write("| 模型 | BLEU分数 |\n")
            f.write("|------|------|\n")
            
            # 对结果按分数排序
            sorted_results = sorted(
                comparison_results["results"][direction].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for model_name, score in sorted_results:
                f.write(f"| {model_name} | {score:.2f} |\n")
            
            f.write("\n")
    
    # 8. 输出完成信息
    print(f"\n评测完成!")
    print(f"详细结果保存在 {results_dir} 目录下")
    print(f"比较报告: {report_file}")
    
    # 打印比较结果表格
    print("\n模型翻译能力比较结果:")
    for direction in directions:
        print(f"\n{direction} 翻译方向:")
        print("模型\t\tBLEU分数")
        print("-" * 30)
        
        # 对结果按分数排序
        sorted_results = sorted(
            comparison_results["results"][direction].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for model_name, score in sorted_results:
            print(f"{model_name}\t\t{score:.2f}")

if __name__ == "__main__":
    main() 