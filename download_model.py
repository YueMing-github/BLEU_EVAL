#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import getpass
from huggingface_hub import snapshot_download, login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="下载Hugging Face上的模型")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct",
        help="要下载的模型ID"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./models",
        help="模型保存路径"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="下载后验证模型是否可以加载"
    )
    parser.add_argument(
        "--token", 
        type=str,
        help="Hugging Face认证令牌，如果不提供将会提示输入"
    )
    parser.add_argument(
        "--use_auth", 
        action="store_true",
        help="是否使用Hugging Face认证"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 处理认证
    if args.use_auth:
        token = args.token
        if not token:
            # 如果没有提供令牌，提示用户输入
            print("需要登录Hugging Face账户下载此模型")
            token = getpass.getpass("请输入您的Hugging Face令牌 (https://huggingface.co/settings/tokens): ")
        
        if token:
            print("正在使用提供的令牌登录Hugging Face...")
            login(token=token)
        else:
            print("未提供令牌，尝试使用已保存的凭据")
    
    # 创建输出目录
    model_output_dir = os.path.join(args.output_dir, os.path.basename(args.model_id))
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"开始下载模型 {args.model_id}...")
    print(f"模型将保存到 {model_output_dir}")
    
    try:
        # 下载模型
        snapshot_download(
            repo_id=args.model_id,
            local_dir=model_output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"模型下载完成: {model_output_dir}")
        
        # 验证模型是否可以正确加载
        if args.verify:
            print("验证模型加载...")
            
            try:
                # 加载模型和分词器
                tokenizer = AutoTokenizer.from_pretrained(model_output_dir, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_output_dir,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                # 简单测试
                prompt = "Hello, how are you?"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=10)
                
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"模型测试输出: {output_text}")
                
                print("模型验证成功!")
            except Exception as e:
                print(f"模型验证失败: {e}")
    
    except Exception as e:
        print(f"模型下载失败: {e}")
        print("\n可能的解决方案:")
        print(f"1. 确保您已在Hugging Face网站上接受了模型许可 (https://huggingface.co/{args.model_id})")
        print("2. 使用 --use_auth 参数并提供有效的Hugging Face令牌")
        print("3. 尝试直接从浏览器下载模型，并放到指定目录")
        print("4. 如果您的网络连接不稳定，请尝试使用VPN或代理")

    print("\n如果下载失败，您也可以尝试以下替代方法：")
    print("1. 直接使用transformers加载模型:")
    print("   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{args.model_id}', trust_remote_code=True)")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{args.model_id}', device_map='auto', trust_remote_code=True)")
    print(f"2. 在浏览器中手动下载并放入目录: https://huggingface.co/{args.model_id}")

if __name__ == "__main__":
    main() 