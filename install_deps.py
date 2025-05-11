#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
import importlib.util

required_packages = [
    "transformers",
    "torch",
    "datasets",
    "sacrebleu",
    "nltk",
    "tqdm",
    "jieba",
    "huggingface_hub"
]

def check_and_install():
    print("检查依赖项...")
    packages_to_install = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is not None:
            print(f"✓ {package} 已安装")
        else:
            print(f"✗ {package} 未安装")
            packages_to_install.append(package)
    
    if packages_to_install:
        print("\n安装缺失的依赖项...")
        try:
            for package in packages_to_install:
                print(f"正在安装 {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--index-url", "https://pypi.org/simple", package],
                    check=False,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✓ {package} 安装成功")
                else:
                    print(f"✗ {package} 安装失败: {result.stderr}")
                    print("尝试不指定版本号安装...")
                    simple_result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    if simple_result.returncode == 0:
                        print(f"✓ {package} 安装成功")
                    else:
                        print(f"✗ {package} 安装失败，请手动安装")
        except Exception as e:
            print(f"安装过程中发生错误: {e}")
            print("请尝试手动安装依赖: pip install transformers torch datasets sacrebleu nltk tqdm jieba huggingface_hub")
    else:
        print("所有依赖项已安装！")
        
    # 检查是否成功安装sacrebleu
    try:
        import sacrebleu
        print("sacrebleu 已成功安装和导入")
    except ImportError:
        print("sacrebleu 导入失败，尝试替代安装方式...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "sacrebleu"],
            check=False
        )
        
    # 确保nltk punkt已下载
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("nltk punkt 数据已存在")
    except LookupError:
        print("下载nltk punkt数据...")
        nltk.download('punkt')

if __name__ == "__main__":
    check_and_install() 