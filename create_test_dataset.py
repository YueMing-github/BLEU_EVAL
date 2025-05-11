#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse

# 中英翻译测试样本
ZH_EN_SAMPLES = [
    # 基础样本
    {
        "zh": "人工智能正在改变我们的世界。",
        "en": "Artificial intelligence is changing our world."
    },
    {
        "zh": "科技的进步带来了新的机遇和挑战。",
        "en": "Technological advances bring new opportunities and challenges."
    },
    {
        "zh": "数据是人工智能发展的关键因素。",
        "en": "Data is a key factor in the development of artificial intelligence."
    },
    {
        "zh": "大语言模型可以理解和生成人类语言。",
        "en": "Large language models can understand and generate human language."
    },
    {
        "zh": "中国在人工智能领域取得了显著进展。",
        "en": "China has made significant progress in the field of artificial intelligence."
    },
    {
        "zh": "深度学习是机器学习的一个重要分支。",
        "en": "Deep learning is an important branch of machine learning."
    },
    {
        "zh": "计算机视觉技术可以识别图像中的对象。",
        "en": "Computer vision technology can identify objects in images."
    },
    {
        "zh": "自然语言处理使机器能够理解人类语言。",
        "en": "Natural language processing enables machines to understand human language."
    },
    {
        "zh": "强化学习是一种通过尝试和错误来学习的方法。",
        "en": "Reinforcement learning is a method of learning through trial and error."
    },
    {
        "zh": "机器学习算法从数据中学习模式。",
        "en": "Machine learning algorithms learn patterns from data."
    },
    
    # 长句和复杂句式
    {
        "zh": "随着人工智能技术的不断发展，它已经在医疗、金融、教育等多个领域展现出巨大的应用潜力，但同时也引发了关于隐私、就业和伦理等方面的争议。",
        "en": "With the continuous development of artificial intelligence technology, it has shown tremendous application potential in various fields such as healthcare, finance, and education, but has also raised controversies regarding privacy, employment, and ethics."
    },
    {
        "zh": "数字化转型不仅意味着技术升级，更重要的是思维方式、组织结构和商业模式的全面变革，企业需要重新思考如何创造和交付价值。",
        "en": "Digital transformation means not only technological upgrades, but more importantly, a comprehensive change in mindset, organizational structure, and business models, requiring companies to rethink how they create and deliver value."
    },
    
    # 专业术语
    {
        "zh": "量子计算利用量子叠加和量子纠缠原理，可以在特定问题上实现指数级的计算加速。",
        "en": "Quantum computing leverages the principles of quantum superposition and quantum entanglement to achieve exponential computational speedup for specific problems."
    },
    {
        "zh": "区块链是一种分布式账本技术，通过密码学保证交易的安全性和透明性，无需中央权威机构的参与。",
        "en": "Blockchain is a distributed ledger technology that ensures transaction security and transparency through cryptography, without the involvement of a central authority."
    },
    
    # 日常用语和口语表达
    {
        "zh": "这部电影太精彩了，情节扣人心弦，演员的表演也非常出色。",
        "en": "This movie is fantastic, with a gripping plot and excellent performances by the actors."
    },
    {
        "zh": "周末我打算和朋友们一起去郊外野餐，希望天气能够配合。",
        "en": "I plan to go on a picnic with friends in the suburbs this weekend, and I hope the weather cooperates."
    },
    
    # 习语和文化表达
    {
        "zh": "这个项目困难重重，我们需要迎难而上，方能成功。",
        "en": "This project is full of difficulties; we need to face the challenges head-on to succeed."
    },
    {
        "zh": "俗话说，万事开头难，但只要坚持不懈，终会有所成就。",
        "en": "As the saying goes, the beginning is always the hardest, but with persistence, one will eventually achieve something."
    },
    
    # 时事和新闻
    {
        "zh": "全球气候变化引发的极端天气事件日益频繁，各国政府正采取行动减少碳排放。",
        "en": "Extreme weather events caused by global climate change are becoming increasingly frequent, and governments around the world are taking action to reduce carbon emissions."
    },
    {
        "zh": "随着远程工作的普及，许多企业正在重新评估其办公空间需求和工作模式。",
        "en": "With the popularization of remote work, many companies are reassessing their office space requirements and work patterns."
    },
    
    # 命令和指令
    {
        "zh": "请在提交申请前仔细阅读所有说明，并确保所有必填信息都已完整填写。",
        "en": "Please read all instructions carefully before submitting your application and ensure that all required information is completed."
    },
    {
        "zh": "系统更新将在今晚10点开始，请确保在此之前保存所有工作并关闭应用程序。",
        "en": "The system update will begin at 10 PM tonight. Please make sure to save all work and close applications before then."
    },
    
    # 问题和疑问句
    {
        "zh": "人工智能是否会在未来取代人类大部分工作？这个问题一直备受争议。",
        "en": "Will artificial intelligence replace most human jobs in the future? This question has been highly controversial."
    },
    {
        "zh": "如何在保持创新的同时确保技术发展符合伦理标准？",
        "en": "How can we ensure that technological development adheres to ethical standards while maintaining innovation?"
    },
    
    # 数字和统计数据
    {
        "zh": "根据最新统计，该平台日活跃用户已超过5亿，同比增长25%。",
        "en": "According to the latest statistics, the platform's daily active users have exceeded 500 million, a 25% year-on-year increase."
    },
    {
        "zh": "2023年全球人工智能市场规模预计将达到1500亿美元，比去年增长约35%。",
        "en": "The global artificial intelligence market is expected to reach $150 billion in 2023, an increase of about 35% compared to last year."
    },
    
    # 专有名词和品牌
    {
        "zh": "谷歌的DeepMind开发的AlphaFold系统在蛋白质结构预测方面取得了突破性进展。",
        "en": "AlphaFold, a system developed by Google's DeepMind, has made breakthrough progress in protein structure prediction."
    },
    {
        "zh": "特斯拉发布了最新版本的全自动驾驶软件，进一步提升了车辆的自主导航能力。",
        "en": "Tesla has released the latest version of its Full Self-Driving software, further enhancing the vehicle's autonomous navigation capabilities."
    }
]

# 英文技术文档和论文摘要样本
EN_TECHNICAL_SAMPLES = [
    {
        "en": "The transformer architecture has revolutionized natural language processing through its self-attention mechanism, enabling parallel processing of sequential data.",
        "zh": "Transformer架构通过其自注意力机制彻底改变了自然语言处理，使得顺序数据的并行处理成为可能。"
    },
    {
        "en": "Deep reinforcement learning combines neural networks with reinforcement learning principles to enable agents to learn optimal policies through interaction with their environment.",
        "zh": "深度强化学习将神经网络与强化学习原理相结合，使智能体能够通过与环境的交互学习最优策略。"
    },
    {
        "en": "Federated learning allows model training across multiple decentralized devices while keeping the training data local, thereby enhancing privacy and security.",
        "zh": "联邦学习允许模型在多个去中心化设备上进行训练，同时保持训练数据的本地化，从而增强隐私和安全性。"
    }
]

# 英文新闻和报道样本
EN_NEWS_SAMPLES = [
    {
        "en": "Scientists have discovered a new method to efficiently convert carbon dioxide into ethanol, potentially offering a solution to reduce greenhouse gas emissions while producing useful chemicals.",
        "zh": "科学家们发现了一种将二氧化碳高效转化为乙醇的新方法，这可能为减少温室气体排放同时生产有用化学品提供解决方案。"
    },
    {
        "en": "The latest smartphone model features a revolutionary camera system that can capture detailed images in extremely low light conditions, setting a new standard for mobile photography.",
        "zh": "最新的智能手机型号配备了一个革命性的相机系统，能够在极低光照条件下捕捉详细图像，为移动摄影设立了新标准。"
    }
]

# 英文文学和表达样本
EN_LITERATURE_SAMPLES = [
    {
        "en": "The ancient oak tree stood as a silent witness to centuries of history, its gnarled branches reaching towards the sky like arthritic fingers seeking warmth.",
        "zh": "这棵古老的橡树作为历史的无声见证者屹立着，它粗糙扭曲的枝干像关节炎的手指一样朝向天空，寻求温暖。"
    },
    {
        "en": "In the quiet moments before dawn, when the world holds its breath in anticipation of the sun's return, one can often find clarity amidst life's complexities.",
        "zh": "在黎明前的宁静时刻，当世界屏息以待太阳的归来，人们常常能在生活的复杂性中找到清晰。"
    }
]

# 英文商业和营销样本
EN_BUSINESS_SAMPLES = [
    {
        "en": "Our innovative solution streamlines workflow processes, reducing operational costs by 30% while simultaneously improving customer satisfaction metrics across all key performance indicators.",
        "zh": "我们的创新解决方案简化了工作流程，在降低30%运营成本的同时，提高了所有关键绩效指标的客户满意度。"
    },
    {
        "en": "The merger between the two industry giants is expected to create significant market synergies, potentially reshaping the competitive landscape while delivering enhanced shareholder value.",
        "zh": "这两家行业巨头的合并预计将创造显著的市场协同效应，可能重塑竞争格局，同时提升股东价值。"
    }
]

def parse_args():
    parser = argparse.ArgumentParser(description="创建本地翻译测试数据集")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./test_data",
        help="输出目录"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建中译英数据集
    zh_en_data = {
        "translation": [
            {"zh": item["zh"], "en": item["en"]} for item in ZH_EN_SAMPLES
        ]
    }
    
    # 保存中译英数据集到JSON文件
    with open(os.path.join(args.output_dir, "zh_en_test.json"), "w", encoding="utf-8") as f:
        json.dump(zh_en_data, f, ensure_ascii=False, indent=2)
    
    print(f"已创建中译英测试数据集: {os.path.join(args.output_dir, 'zh_en_test.json')}")
    print(f"中译英数据集包含 {len(ZH_EN_SAMPLES)} 个翻译样本")
    
    # 合并所有英译中样本
    en_zh_samples = []
    en_zh_samples.extend(EN_TECHNICAL_SAMPLES)
    en_zh_samples.extend(EN_NEWS_SAMPLES)
    en_zh_samples.extend(EN_LITERATURE_SAMPLES)
    en_zh_samples.extend(EN_BUSINESS_SAMPLES)
    
    # 创建英译中数据集
    en_zh_data = {
        "translation": [
            {"en": item["en"], "zh": item["zh"]} for item in en_zh_samples
        ]
    }
    
    with open(os.path.join(args.output_dir, "en_zh_test.json"), "w", encoding="utf-8") as f:
        json.dump(en_zh_data, f, ensure_ascii=False, indent=2)
    
    print(f"已创建英译中测试数据集: {os.path.join(args.output_dir, 'en_zh_test.json')}")
    print(f"英译中数据集包含 {len(en_zh_samples)} 个翻译样本")

if __name__ == "__main__":
    main()