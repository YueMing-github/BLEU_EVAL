#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize
import re

# 全局变量定义
USE_SACREBLEU = False
HAS_JIEBA = False

# 尝试导入sacrebleu，如果失败则使用nltk的BLEU计算
try:
    from sacrebleu.metrics import BLEU
    USE_SACREBLEU = True
    print("使用sacrebleu计算BLEU分数")
except ImportError:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    print("使用nltk计算BLEU分数")
    # 确保nltk数据已下载
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# 尝试导入jieba用于中文分词
try:
    import jieba
    HAS_JIEBA = True
    print("使用jieba进行中文分词")
except ImportError:
    print("jieba未安装，使用字符级分词")

def parse_args():
    parser = argparse.ArgumentParser(description="评测Qwen模型在机器翻译任务上的表现")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/Users/bytedance/Desktop/BLEU_EVAL/models/Qwen2.5-7B-Instruct",
        help="模型路径"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="./test_data/zh_en_test.json",
        help="本地数据集路径，JSON格式"
    )
    parser.add_argument(
        "--src_lang", 
        type=str, 
        default="zh", 
        help="源语言"
    )
    parser.add_argument(
        "--tgt_lang", 
        type=str, 
        default="en", 
        help="目标语言"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10, 
        help="评估样本数量，设置为-1表示使用全部样本"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="批处理大小"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="最大序列长度"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="调试模式，显示更多信息"
    )
    return parser.parse_args()

def prepare_translation_prompt(src_text, src_lang, tgt_lang):
    """准备翻译提示"""
    if src_lang == "zh" and tgt_lang == "en":
        prompt = f"""你是一个专业的中译英翻译专家。请将以下中文文本准确翻译成英文，只输出翻译结果，不要有任何其他内容。

中文文本：{src_text}

英文翻译："""
    elif src_lang == "en" and tgt_lang == "zh":
        prompt = f"""你是一个专业的英译中翻译专家。请将以下英文文本准确翻译成中文，只输出翻译结果，不要有任何其他内容。

英文文本：{src_text}

中文翻译："""
    else:
        prompt = f"""你是一个专业的翻译专家。请将以下{src_lang}文本准确翻译成{tgt_lang}，只输出翻译结果，不要有任何其他内容。

原文：{src_text}

翻译："""
    return prompt

def extract_translation(output_text, src_lang, tgt_lang, debug=False):
    """从模型输出中提取翻译结果"""
    if debug:
        print("\n==== 原始输出 ====")
        print(output_text)
        print("==================\n")
    
    translation = ""
    
    # 根据不同的翻译方向尝试提取结果
    if src_lang == "zh" and tgt_lang == "en":
        if "英文翻译：" in output_text:
            translation = output_text.split("英文翻译：")[-1].strip()
        else:
            # 尝试找到英文文本
            lines = output_text.split('\n')
            for line in lines:
                if re.match(r'^[A-Za-z]', line.strip()):
                    translation = line.strip()
                    break
    elif src_lang == "en" and tgt_lang == "zh":
        if "中文翻译：" in output_text:
            translation = output_text.split("中文翻译：")[-1].strip()
        else:
            # 尝试找到中文文本
            lines = output_text.split('\n')
            for line in lines:
                if re.match(r'[\u4e00-\u9fa5]', line.strip()):
                    translation = line.strip()
                    break
    else:
        if "翻译：" in output_text:
            translation = output_text.split("翻译：")[-1].strip()
    
    # 如果上述方法都无法提取，则返回最后一行非空文本
    if not translation:
        lines = [line.strip() for line in output_text.split('\n') if line.strip()]
        if lines:
            translation = lines[-1]
    
    # 清理翻译结果，移除引号等
    translation = translation.strip('"\'')
    
    if debug:
        print("\n==== 提取的翻译 ====")
        print(translation)
        print("=====================\n")
    
    return translation

def tokenize_text(text, lang):
    """根据语言对文本进行分词"""
    if lang == "zh":
        if HAS_JIEBA:
            return list(jieba.cut(text))
        else:
            # 字符级分词
            return [char for char in text]
    else:  # 英文和其他语言
        return word_tokenize(text.lower())

def compute_bleu_nltk(translations, references, tgt_lang="en"):
    """使用NLTK计算BLEU分数"""
    # 分词处理
    tokenized_translations = [tokenize_text(t, tgt_lang) for t in translations]
    tokenized_references = [[tokenize_text(r, tgt_lang)] for r in references]
    
    # 计算BLEU分数
    smoothing = SmoothingFunction().method1
    return corpus_bleu(
        tokenized_references, 
        tokenized_translations,
        smoothing_function=smoothing
    ) * 100  # 转换为0-100的分数范围以匹配sacrebleu

def load_local_dataset(dataset_path, num_samples=None):
    """加载本地数据集"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}\n请先运行create_test_dataset.py创建测试数据，或者确认数据集路径是否正确。")
    
    print(f"正在加载本地数据集: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查数据格式
    if 'translation' not in data:
        raise ValueError("数据集格式错误，应包含'translation'字段")
    
    # 选择指定数量的样本
    samples = data['translation']
    if num_samples is not None and num_samples > 0 and num_samples < len(samples):
        samples = samples[:num_samples]
    
    print(f"已加载 {len(samples)} 个翻译样本")
    
    # 创建数据集对象
    return Dataset.from_dict({
        'translation': samples
    })

def main():
    args = parse_args()
    
    # 如果没有jieba但是在做英译中，尝试安装jieba
    global HAS_JIEBA
    if args.tgt_lang == "zh" and not HAS_JIEBA:
        try:
            print("检测到英译中评测，但缺少jieba库，尝试安装...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
            import jieba
            print("jieba安装成功！")
            HAS_JIEBA = True
        except:
            print("jieba安装失败，将使用字符级分词")
    
    # 加载本地数据集
    dataset = load_local_dataset(args.dataset_path, args.num_samples)
    
    print(f"加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 定义评估函数
    def translate(batch):
        src_texts = [item[args.src_lang] for item in batch["translation"]]
        tgt_texts = [item[args.tgt_lang] for item in batch["translation"]]
        
        translations = []
        raw_outputs = []  # 存储原始输出用于调试
        
        for src_text in tqdm(src_texts, desc="翻译进度"):
            prompt = prepare_translation_prompt(src_text, args.src_lang, args.tgt_lang)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 使用贪婪解码进行生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_length,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            raw_outputs.append(output_text)
            
            # 提取翻译结果
            translation = extract_translation(output_text, args.src_lang, args.tgt_lang, args.debug)
            translations.append(translation)
        
        return {"translations": translations, "references": tgt_texts, "raw_outputs": raw_outputs}
    
    # 批处理数据集
    results = []
    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i:i+args.batch_size]
        batch_results = translate(batch)
        results.append(batch_results)
    
    # 汇总结果
    all_translations = []
    all_references = []
    all_raw_outputs = []
    for result in results:
        all_translations.extend(result["translations"])
        all_references.extend(result["references"])
        all_raw_outputs.extend(result["raw_outputs"])
    
    # 计算BLEU分数
    global USE_SACREBLEU
    if USE_SACREBLEU:
        # 对于中文目标语言，需要特殊处理
        if args.tgt_lang == "zh":
            # 使用sacrebleu的zh tokenizer
            bleu = BLEU(tokenize='zh')
        else:
            bleu = BLEU()
        
        bleu_score = bleu.corpus_score(all_translations, [all_references])
        bleu_value = bleu_score.score
        bleu_details = str(bleu_score)
    else:
        bleu_value = compute_bleu_nltk(all_translations, all_references, args.tgt_lang)
        bleu_details = f"NLTK BLEU: {bleu_value:.2f}"
    
    print(f"\n评估结果:")
    print(f"样本数量: {len(all_translations)}")
    print(f"BLEU分数: {bleu_value:.2f}")
    print(f"详细信息: {bleu_details}")
    
    # 保存结果到文件
    with open("translation_results.txt", "w", encoding="utf-8") as f:
        f.write(f"评估结果摘要:\n")
        f.write(f"样本数量: {len(all_translations)}\n")
        f.write(f"BLEU分数: {bleu_value:.2f}\n")
        f.write(f"详细信息: {bleu_details}\n\n")
        f.write("="*50 + "\n\n")
        
        for i, (src, ref, trans, raw) in enumerate(zip(
            [item[args.src_lang] for item in dataset["translation"]], 
            all_references, 
            all_translations,
            all_raw_outputs
        )):
            f.write(f"样本 {i+1}:\n")
            f.write(f"源文本: {src}\n")
            f.write(f"参考译文: {ref}\n")
            f.write(f"模型译文: {trans}\n")
            if args.debug:
                f.write(f"\n原始输出:\n{raw}\n")
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"翻译结果和详细对比已保存到 translation_results.txt")

if __name__ == "__main__":
    main() 