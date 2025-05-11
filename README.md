# Qwen模型评测

这个项目用于评测微调后的Qwen2.5-7B模型在机器翻译任务上的表现，通过计算BLEU分数来进行性能评估。

## 文件说明
- `evaluate.py`: 主要评测脚本，用于加载模型并计算BLEU分数
- `install_deps.py`: 检查并安装所需的依赖项
- `run_evaluation.py`: 简易启动脚本，自动配置参数运行评测
- `create_test_dataset.py`: 创建本地测试数据集（中译英和英译中两个方向）
- `requirements.txt`: 依赖库列表
- `test_data/`: 包含测试数据集的目录
  - `zh_en_test.json`: 中译英测试数据
  - `en_zh_test.json`: 英译中测试数据
- `download_model.py`: 下载Hugging Face上的模型
- `compare_models.py`: 比较多个模型在翻译任务上的表现

## 使用方法

### 快速启动
最简单的方法是直接运行：

```bash
python3 run_evaluation.py
```

这将使用默认参数启动评测（中译英，评测10个样本）。

### 自定义评测参数

您可以自定义评测参数：

```bash
# 评测英译中翻译能力
python3 run_evaluation.py --direction en-zh

# 评测中英双向翻译能力
python3 run_evaluation.py --direction both

# 自定义模型路径
python3 run_evaluation.py --model_path /path/to/your/model

# 设置较小的样本数以加速测试
python3 run_evaluation.py --num_samples 5

# 启用调试模式，显示更多信息
python3 run_evaluation.py --debug

# 评测全部样本
python3 run_evaluation.py --all

# 或者使用 num_samples=-1 评测全部样本
python3 run_evaluation.py --num_samples -1

# 重新创建测试数据集（会覆盖现有数据）
python3 run_evaluation.py --recreate_data
```

### 下载模型

如果您需要下载Hugging Face上的模型进行评测，可以使用：

```bash
# 下载默认模型 (Qwen/Qwen2.5-7B-Instruct)
python3 download_model.py

# 下载指定模型
python3 download_model.py --model_id Qwen/Qwen2.5-7B-Instruct   

# 自定义下载目录
python3 download_model.py --output_dir ./my_models

# 下载后验证模型
python3 download_model.py --verify
```

### 比较多个模型

要比较多个模型在翻译任务上的表现，可以使用：

```bash
# 比较两个模型
python3 compare_models.py --model_list /path/to/model1 /path/to/model2

# 指定模型显示名称
python3 compare_models.py --model_list /path/to/model1 /path/to/model2 --model_names "原始模型" "微调模型"

# 比较中英双向翻译能力
python3 compare_models.py --model_list /path/to/model1 /path/to/model2 --direction both

# 使用更多样本进行评测
python3 compare_models.py --model_list /path/to/model1 /path/to/model2 --num_samples 20
```

### 直接使用评测脚本

您也可以直接使用评测脚本 `evaluate.py` 进行更细粒度的配置：

```bash
# 中译英评测
python3 evaluate.py --model_path /path/to/model --dataset_path ./test_data/zh_en_test.json --src_lang zh --tgt_lang en --num_samples 10

# 英译中评测
python3 evaluate.py --model_path /path/to/model --dataset_path ./test_data/en_zh_test.json --src_lang en --tgt_lang zh --num_samples 10

# 评测全部样本
python3 evaluate.py --model_path /path/to/model --dataset_path ./test_data/zh_en_test.json --src_lang zh --tgt_lang en --num_samples -1
```

### 管理测试数据

如果您想创建或更新测试数据集，可以使用：

```bash
# 创建新的测试数据集
python3 create_test_dataset.py

# 指定输出目录
python3 create_test_dataset.py --output_dir ./my_test_data
```

您也可以手动编辑 `test_data` 目录下的JSON文件，添加或修改测试样本。

## 数据集

本项目使用本地自定义的测试数据集，包含两个方向的翻译样本：

1. **中译英数据集**(`zh_en_test.json`): 包含30个多样化的中文样本及其对应的英文参考译文，涵盖：
   - 基础句子和常见表达
   - 长句和复杂句式
   - 专业术语和技术描述
   - 日常用语和口语表达
   - 习语和文化表达
   - 时事新闻和报道
   - 指令和命令句
   - 问题和疑问句
   - 数字和统计数据
   - 专有名词和品牌

2. **英译中数据集**(`en_zh_test.json`): 包含9个英文样本及其对应的中文参考译文，涵盖：
   - 技术文档和论文摘要
   - 新闻和报道
   - 文学和表达
   - 商业和营销

您可以通过修改`create_test_dataset.py`文件来添加更多测试样本，或直接编辑JSON文件。

## 评测结果

评测完成后，结果将保存在不同的文件中：
- 中译英评测结果: `translation_results_zh-en.txt`
- 英译中评测结果: `translation_results_en-zh.txt`

当进行模型比较时，结果将保存在 `comparison_results` 目录下，包括：
- 各模型的详细评测结果文件
- 比较报告文件
- 比较摘要JSON文件

结果文件包含：
- 源文本、参考译文和模型生成的译文对比
- BLEU分数和详细评测信息

## 注意事项

- 本评测脚本支持本地测试数据集，无需网络连接
- 如果无法安装sacrebleu库，脚本会自动使用nltk实现的BLEU评分
- 在Mac环境下，模型加载可能会使用MPS或CPU进行推理
- 增强的提示词和结果提取方法使评测结果更加准确
- 使用`--all`参数或`--num_samples -1`可以评测数据集中的全部样本
- 默认情况下，脚本会使用已存在的测试数据，除非使用`--recreate_data`参数
- 模型比较功能可以方便地对比原始模型和微调模型的差异 