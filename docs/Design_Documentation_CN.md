# 摘要Adaptor设计文档

本项目通过对**Llama-3.2-3B-Instruct**进行高效微调，开发了一个**长文档摘要Adaptor**，用于生成高质量的**科学文献**摘要。

## 目录
- [1. Adaptor架构](#1-Adaptor架构)
   - [1.1 技术方法](#11-技术方法)
   - [1.2 系统组件](#12-系统组件)
   - [1.3 主要优势](#13-主要优势)
- [2. 输入和输出格式](#2-输入和输出格式)
   - [2.1 输入格式](#21-输入格式)
   - [2.2 输出格式](#22-输出格式)
- [3. 评估指标](#3-评估指标)
   - [3.1 主要指标](#31-主要指标)
   - [3.2 训练损失](#32-训练损失)
   - [3.3 指标使用](#33-指标使用)
- [4. 数据集](#4-数据集)
   - [4.1 训练数据集](#41-训练数据集)
   - [4.2 评估数据集](#42-评估数据集)
- [5. 实施计划](#5-实施计划)
   - [5.1 环境准备](#51-环境准备)
   - [5.2 数据预处理](#52-数据预处理)
   - [5.3 Adaptor设计与实现](#53-Adaptor设计与实现)
   - [5.4 训练过程](#54-训练过程)
   - [5.5 后处理](#55-后处理)
   - [5.6 评估与优化](#56-评估与优化)
   - [5.7 挑战与解决方案](#57-挑战与解决方案)
- [6. 进一步工作](#6-进一步工作)
   - [6.1 评估指标扩展](#61-评估指标扩展)
   - [6.2 超参数优化](#62-超参数优化)
   - [6.3 块处理增强](#63-块处理增强)
   - [6.4 后处理增强](#64-后处理增强)
   - [6.5 跨语言泛化](#65-跨语言泛化)
- [7. 源代码和模型](#7-源代码和模型)
   - [7.1 源代码](#71-源代码)
   - [7.2 Adaptor](#72-Adaptor)
   - [7.3 使用方法](#73-使用方法)

## 1. Adaptor架构

### 1.1 技术方法

#### 1.1.1 核心方法
- **基础模型**：Llama-3.2-3B-Instruct（30亿参数）具有8,192个token的上下文窗口
- **微调方法**：使用LoRA的参数高效微调（PEFT）
- **目标组件**：关键注意力机制和前馈网络

#### 1.1.2 集成机制
Adaptor通过**非入侵式权重修改方法**与基础模型集成：
- 基础模型的权重在训练和推理过程中完全冻结
- Adaptor引入小型、可训练的矩阵，修改特定Transformer层的输出
- 在推理过程中，Adaptor权重可以与基础权重并行运行（使用PEFT的推理模式），或者与基础权重数学合并以获得最佳性能
- 集成不需要对底层模型和数据集进行架构更改，保留了所有原始功能

### 1.2 系统组件

Adaptor系统集成了四个基本组件：

1. **基础语言模型**（Llama-3.2-3B-Instruct）

2. **LoRAAdaptor模块**
   - 通过低秩矩阵实现有针对性的权重更新
   - 专注于关键层：
     * Attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`
     * Feed-forward networks：`gate_proj`、`up_proj`、`down_proj`

3. **数据处理管道**
   - **领域提示**：优化科学文献处理的指令
   - **智能分块**：保留更多有用信息

4. **综合评估指标**
   - 使用ROUGE、BERTScore和摘要长度比率来评估词汇重叠度、语义相似度和压缩效率

### 1.3 主要优势

Adaptor架构提供了几个显著的优势：

1. **计算效率**：仅训练总参数的0.1%，同时保持基础模型冻结，极大减少GPU内存使用和计算需求。

2. **集成灵活性**：可以作为单独模块部署，或与基础模型权重合并以获得最佳推理性能。

3. **领域专业化**：专门为科学文献优化摘要生成，同时保留基础模型的通用能力。

4. **质量导向设计**：结合专业领域提示、智能分块策略和综合评估指标，确保科学摘要的高质量。

5. **模型和数据集通用性**：支持在不同基础模型（超出Llama-3.2）之间灵活切换，并适应各种微调和评估数据集，实现特定用例和领域的定制化。

## 2. 输入和输出格式

### 2.1 输入格式

- 文章、报告、论文等的纯文本
- 示例：
```
{
"article": "Artificial Intelligence (AI) is a subfield of computer science dedicated to developing systems and software that can simulate human intelligence.
It includes multiple research directions such as machine learning, deep learning, natural language processing, and computer vision.
Machine learning uses statistical techniques to enable computer systems to learn from data and gradually improve performance without explicit programming.
Deep learning is a branch of machine learning that uses multi-layer neural networks to process data, particularly suitable for handling unstructured data such as images, sound, and text.
..."
}
```

### 2.2 输出格式

- 纯文本摘要

- 示例：
```
Artificial intelligence is a branch of computer science that simulates human intelligence through technologies such as machine learning, deep learning, natural language processing, and computer vision. It has been widely applied in medical, financial, autonomous driving and other fields. Despite facing challenges in ethics and privacy, researchers continue to develop more advanced AI systems.
```

## 3. 评估指标

### 3.1 主要指标
1. **ROUGE评分** (↑) 
   - ROUGE-1：词重叠
   - ROUGE-2：二元组重叠
   - ROUGE-L：最长公共子序列
   - 使用`rouge_score`库实现
   
2. **BERTScore** (↑) 
   - 提供精确度、召回率和F1分数
   - 使用`bert_score`库实现
   - 支持多语言评估，默认为英语("en")

3. **摘要长度比率** (↔)
   - 生成摘要长度与参考摘要长度的比率
   - 最佳值接近100%（既不太高也不太低）

### 3.2 训练损失 (↓)
训练使用标准**交叉熵损失**，计算预测和目标Token之间的负对数似然。它通过PyTorch的CrossEntropyLoss实现，仅应用于摘要Token，并作为主要优化目标。

### 3.3 指标使用
这些指标用于：
1. **监控训练进度**：使用交叉熵损失实时跟踪进度，通过TensorBoard可视化，并基于验证损失保存检查点。

2. **模型选择和比较**：使用ROUGE和BERTScore_F1作为主要选择标准。

3. **基线比较**：与原始Llama-3.2-3B-Instruct和其他开源摘要模型（如PEGASUS、BART、T5）进行性能比较。

4. **超参数优化**：根据指标反馈调整LoRA参数、模板设计和生成参数（温度、top-p、最大长度）。

5. **错误分析和调试**：分析低分样本，监控不同文档类型的性能差异，并调整训练策略。

## 4. 数据集

### 4.1 训练数据集

#### 4.1.1 数据集描述
在本项目中，对于长文档摘要，我们选择了**PubMed摘要数据集**（[ccdv/pubmed-summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization)）。该数据集包含生物医学领域的科学文章及其对应的摘要，作为参考摘要。在处理管道中，"article"字段被重命名为"text"，"abstract"字段被重命名为"summary"，以保持一致性。

**注意**：由于项目时间限制，我们的实验使用了5,000个训练样本、500个验证样本和500个测试样本的子集，同时保持了下面描述的完整数据集的分布特征。

**数据集特点**：
- **规模**：133,215对文章-摘要配对（119,924训练、6,633验证、6,658测试样本）
- **领域**：生物医学和医疗保健研究
- **文档类型**：科学研究论文和临床研究
- **来源**：来自PubMed数据库的文章，一个综合性的生物医学文献档案
- **语言**：英语
- **出版时间**：跨越生物医学研究出版物的多个年份

**长度统计**：
- **文档长度**：
  - 平均值：4,069 tokens（18,140 characters）
  - 中位数：3,376 tokens（15,424 characters）
  - 95百分位：9,450 tokens（43,212 characters）
  - 最大值：45,840 tokens（159,114 characters）

- **摘要长度**：
  - 平均值：268 tokens（1,254 characters）
  - 中位数：271 tokens（1,307 characters）
  - 95百分位：434 tokens（1,923 characters）
  - 最大值：532 tokens（2,325 characters）

- **Token分布**：
  - 0-1kToken：4.9%的文档
  - 1k-2kToken：20.0%的文档
  - 2k-4kToken：36.0%的文档
  - 4k-8kToken：30.9%的文档
  - 8k-16kToken：7.7%的文档
  - 16k+Token：0.5%的文档

- **压缩比**：摘要平均压缩文档15.18倍

#### 4.1.2 数据集质量分析

质量评估揭示了几个重要特征：

- **文本质量**：
  - 特殊字符比例：2.98%
  - 平均句子长度：121.18 characters
  - 每个文档的平均句子数：145.49
  - 词汇多样性（唯一词/总词数）：30.46%

- **摘要质量**：
  - 摘要完整性（摘要长度/文档长度）：6.92%
  - 摘要唯一性（摘要中的唯一词）：61.21%

- **数据集平衡性**：
  - 长度标准差归一化均值：0.73（表示中等变异性）

详细的数据集质量分析是使用`src/data/analysis.py`中的代码执行的，该代码实现了一个全面的评估管道，用于评估文本质量、摘要特征和数据集平衡指标。

总体而言，PubMed数据集提供了高质量的科学文章，并配有专业撰写的摘要，保持一致的压缩比，使其成为训练摘要Adaptor的理想候选数据集。

#### 4.1.3 主要参数建议

基于这些统计数据，为模型推导出以下参数建议：

- **输入上下文长度**：9,510 tokens（包括带提示的输入文档）
  - 这能容纳数据集中95%的文档

- **最大生成长度**：454 tokens
  - 这涵盖了参考摘要的第95百分位加上一小部分缓冲区

- **采样参数**：
  - Temperature：0.5（在创造性和连贯性之间取得平衡）
  - Top-p：0.9（控制多样性的标准值）

### 4.2 评估数据集

为了全面评估模型，使用了两个不同的测试数据集来评估领域内性能和跨领域泛化能力：

#### 4.2.1 PubMed测试集

- **描述**：来自PubMed数据集的标准测试分割。
- **规模**：6,658对文章-摘要配对
- **领域**：生物医学和医疗保健研究
- **文档类型**：生物医学文献中的科学研究论文、临床研究和医学案例报告

#### 4.2.2 ArXiv测试集

- **描述**：来自ArXiv摘要数据集的测试分割（[ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization)），用于评估模型在不同领域科学论文上的跨领域泛化能力。

- **规模**：6,440对文章-摘要配对

- **领域**：多学科科学研究，侧重于物理学、数学、计算机科学和工程学

- **文档类型**：提交到ArXiv存储库的科学论文、技术报告和研究预印本

使用这两个数据集提供了对模型能力更强健的评估：
- PubMed测试集测量模型在专门训练的领域上的性能
- ArXiv测试集评估模型向相邻但不同科学领域泛化的能力
- 两者之间的比较揭示了领域适应能力和局限性

## 5. 实施计划

### 5.1 环境准备
- 安装必要的依赖项（transformers、peft、accelerate等）
- 配置用于一致性日志记录、可重现随机种子和指标计算的实用模块

### 5.2 数据预处理

数据预处理为模型训练奠定了基础，将原始文档和摘要转换为最佳格式。管道专注于三个核心方面：数据质量保证、格式标准化和Token管理优化。

#### 步骤1：数据集加载和准备
1. **数据集获取**：从Hugging Face下载ccdv/pubmed-summarization数据集
2. **分割识别**：保留原始训练/验证/测试分割
3. **初始化设置**：配置分词器、最大长度和处理参数

#### 步骤2：数据清洗和过滤
1. **空值处理**：过滤掉文本或摘要为空的样本
2. **字段标准化**：统一字段命名（article → text, abstract → summary）
3. **抽样控制**：为大型数据集实现可配置的随机抽样

#### 步骤3：模板应用和格式化
1. **提示模板选择**：根据数据集类型选择适当的领域特定提示模板
2. **格式构建**：将文本与提示模板组合，创建模型就绪的输入格式
3. **分词器配置**：确保分词器正确处理特殊Token（如在必要时使用EOS Token作为PAD Token）


> **引入：领域优化的提示模板选择**
>
> 在内部处理时，输入文本使用领域特定的模板系统转换为提示格式。这增强了模型为不同内容类型生成适当摘要的能力。
>
> | Dataset | Prompt Template |
> |---------|----------------|
> | Default | `Please generate a concise summary for the following text:`<br>`[Input Text]`<br>`Summary: ` |
> | PubMed/ArXiv | `Generate a comprehensive abstract for the following scientific article:`<br>`[Input Text]`<br>`Summary: ` |

#### 步骤4：智能Token管理

在训练过程中，输入由连接的文档文本和摘要组成，其中摘要部分至关重要，必须完整保留。由于Llama-3.2-3B-Instruct的8,192 tokens上下文窗口限制，以及大约10%的科学论文超过8,000 tokens的长度，简单连接经常导致信息丢失。

**为什么需要智能Token管理**：
- 科学论文+摘要经常超过8,192 tokens限制
- 文档开头比后面部分包含更关键的信息
- 完整的摘要保存对有效训练至关重要

**智能Token分配方法**：

1. **优先编码摘要**：优先保留摘要
   - 不截断地编码完整摘要
   - 计算考虑特殊Token的Token预算：
     ```python
     # 通过比较有无特殊Token的编码文本来计算特殊Token数
     sample_encoded = tokenizer(sample_text, return_tensors="pt")
     sample_text_only = tokenizer("SAMPLESAMPLE", return_tensors="pt")
     special_tokens_count = sample_encoded.input_ids.size(1) - sample_text_only.input_ids.size(1) + 2
     ```
   - 为摘要部分保留必要的Token

2. **动态文档截断**：
   - 根据每个样本的摘要长度创建自定义截断阈值
   - 当文档超过可用Token预算时，在句子边界应用精确截断
   - 优先考虑文档截断而非摘要压缩

3. **验证**：最终编码检查，确保输入不超过上下文窗口，并跟踪Token分配指标。

与固定长度截断策略相比，这种自适应方法确保了源文档信息的最大保留，同时保留完整的目标摘要，显著提高了训练效率。

### 5.3 Adaptor设计与实现

Adaptor实现采用模块化方法，在最小化计算和内存需求的同时有效地扩展Llama-3.2-3B-Instruct的摘要能力。该设计使用LoRA有选择地适应关键模型组件。

#### 步骤1：模型和分词器初始化
使用Hugging Face的AutoModelForCausalLM加载基础模型，并配置分词器，正确处理填充Token。

#### 步骤2：LoRA Adaptor配置
1. **目标模块选择**：确定注意力和前馈网络中特定权重矩阵进行适应
   ```python
   # 来自adapter.py
   target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"]
   ```
   
   为摘要任务选择这些特定层进行适应是策略性和任务特定的：
   
   - **Attention** (`q_proj`, `k_proj`, `v_proj`, `o_proj`)：
     - 这些层控制模型如何关注输入文本的不同部分
     - 对于摘要，修改注意力模式帮助模型识别长文档中的突出信息
     - `q_proj` 决定要寻找什么信息；适应这一点帮助模型学习什么内容值得摘要
     - `k_proj`，`v_proj` 帮助建立文本不同部分之间的关系，对于连贯摘要至关重要
     - `o_proj` 整合关注的信息，影响模型如何综合内容
   
   - **Feed-Forward Networks** (`gate_proj`, `up_proj`, `down_proj`)：
     - 这些层负责更高级别的特征转换和推理
     - `up_proj` 扩展表示维度，允许抽象摘要所需的更复杂转换
     - `gate_proj` 控制信息流，帮助过滤相关与不相关内容
     - `down_proj` 压缩信息，与摘要的基本压缩性质一致

2. **超参数设置**：
   ```python
   # 来自adapter.py
   peft_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       inference_mode=False,
       r=16,
       lora_alpha=32,
       lora_dropout=0,
       target_modules=target_modules,
       bias="none"
   )
   ```
   - 秩（`r=16`）：此秩是基于经验测试选择的，显示它在适应能力和参数效率之间提供最佳平衡
   - Alpha（`lora_alpha=32`）：Alpha = 2 × 秩提供训练期间稳定的梯度流
   - Dropout（`lora_dropout=0`）：
     - 选择零dropout是因为摘要受益于确定性注意力模式
     - PubMed数据集的高质量和规模提供了足够的正则化，无需dropout
   - 偏置设置（`bias="none"`）：控制是否训练偏置项
     - 偏置项主要影响Token级基线偏好而非上下文理解
     - 对于摘要，Token之间的上下文关系比单 tokens偏置更重要

### 5.4 训练过程

训练过程利用TRL（Transformer强化学习）库的SFTTrainer来简化微调，同时应用最佳实践进行高效学习。该过程专注于Adaptor参数优化，同时保持基础模型冻结。

#### 步骤1：训练配置和参数设置
- 使用SFTConfig配置全面训练参数
- 建立优化策略和调度
- 设置评估协议和输出管理
- **数据集配置**：由于项目时间限制，我们使用了5,000个训练样本、500个验证样本和500个测试样本的子集，同时保持原始分布特征。

```python
training_args = SFTConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # 有效批量大小：每次更新16个样本
    num_train_epochs=5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=1,
    optim="adamw_8bit",
    save_steps=50,
    eval_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    max_grad_norm=0.7,
    max_seq_length=8192,
    dataloader_drop_last=True,
)
```

配置包括几个优化组件：

1. **优化器策略**
   - **AdamW 8位**优化器（`optim="adamw_8bit"`）提高内存效率
   - 默认学习率：Adaptor参数为1e-5
   - 权重衰减：0.01（应用于非偏置参数）
   - Beta参数：β₁=0.9，β₂=0.999（默认AdamW动量系数，未在代码中显式设置）
   - Epsilon：1e-8（默认数值稳定性值，继承自优化器实现）

2. **梯度裁剪策略**
   - 最大梯度范数：0.7（`max_grad_norm=0.7`）
     * 比一般语言任务中使用的1.0默认值更保守
     * 防止长文本处理中常见的梯度爆炸
     * 帮助在处理复杂科学术语和嵌套关系时稳定训练

3. **学习率调度**
   - 前10%步骤线性预热（`warmup_ratio=0.1`）- 明确配置
   - 剩余步骤线性衰减（调度器默认行为）
   - 最小学习率：峰值学习率的10%（线性调度器的固有行为，未明确配置）

#### 步骤2：硬件配置和训练时间

训练使用以下硬件设置：

- **GPU**：8× NVIDIA H100 GPU（每个80GB内存）
- **系统**：基于Linux的高性能计算服务器

在5,000样本子集上训练Adaptor大约花费10小时（5个周期），展示了LoRA方法的效率，与完整模型微调相比，后者在类似硬件上通常需要数天而非数小时。

#### 步骤3：训练器设置和执行
- 通过设置带有模型、数据集和LoRA配置的SFTTrainer初始化并执行训练过程，然后运行带有自动评估和检查点的训练循环。

```python
# 初始化SFTTrainer
trainer = SFTTrainer(
    model=adapter.model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    processing_class=adapter.tokenizer,
    args=training_args,
)

# 开始训练
trainer.train()
```

#### 步骤4：模型保存
- 基于验证性能保存最佳Adaptor权重
- 创建可与基础模型分开分发的便携式Adaptor
```python
# 保存Adaptor模型
trainer.save_model(output_dir)
adapter.tokenizer.save_pretrained(output_dir)
```

### 5.5 后处理

输出清理和标准化的后处理工作流程：
- 从生成的文本中删除任何残留的提示工件或分隔符Token
- 标准化空格、标点和格式
- 修复常见的分词工件（分割词、不规则间距）
- 规范化科学符号、单位和数字表示

### 5.6 评估与优化

#### 步骤1：指标实施和计算
- 部署第3.1节中描述的多维评估指标

#### 步骤2：测试集评估
- 评估PubMed测试集（6,658个样本）进行领域内评估
- 在ArXiv测试集（6,440个样本）上进行跨领域评估

#### 步骤3：超参数优化
- 根据评估反馈调整生成参数：
  * 温度设置：0.3 ~ 0.8
  * Top-p值：评估0.85 ~ 0.95
  * 最大生成长度：使用摘要长度比率校准
- 基于加权指标性能选择最佳配置

#### 初步评估结果

以下表格展示了我们的Adaptor与基线模型相比的初步评估结果，使用仓库中的默认参数（5000个样本用于训练，500个用于验证，500个用于测试）。

##### PubMed测试集（领域内）

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | 长度比率 |
|-------|---------|---------|---------|--------------|--------------|
| 我们的Adaptor | 41.46 | 18.02 | 24.41 | 85.70 | 338.08% |
| 基础Llama-3.2-3B | 37.18 | 14.05 | 19.88 | 83.34 | 381.19% |

##### ArXiv测试集（跨领域）

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | 长度比率 |
|-------|---------|---------|---------|--------------|--------------|
| 我们的Adaptor | 30.34 | 12.12 | 17.90 | 84.81 | 1347.67% |
| 基础Llama-3.2-3B | 28.53 | 10.33 | 16.10 | 83.30 | 1356.81% |

这些初步结果证明了Adaptor方法的有效性。尽管受到时间和计算资源的限制，只使用5,000个样本进行微调，Adaptor模型在所有评估指标上仍然显著优于基础Llama-3.2-3B-Instruct模型。

在ArXiv跨领域测试集上，Adaptor模型在ROUGE-1上提高了1.81个百分点（30.34对28.53），在ROUGE-2上提高了1.79个百分点（12.12对10.33），在ROUGE-L上提高了1.80个百分点（17.90对16.10）。这些改进表明，即使使用有限的训练数据，Adaptor模型也能生成更准确和相关的摘要内容。

同样值得注意的是BERTScore F1的提升（84.81对83.30），这表明模型在语义理解和表达方面取得了进步，更好地捕捉了原文的核心含义。

然而，两个模型的长度比率仍然过高（分别为1347.67%和1356.81%），远超理想的摘要长度比例。这表明配置的最大Token长度可能过于宽松，导致模型生成冗长的摘要。虽然Adaptor模型在这方面有轻微改进（1347.67%对1356.81%），但差异很小，表明长度控制仍然是未来优化的关键领域。

总体而言，这些结果证明，即使使用有限的训练样本，Adaptor方法也能有效提升大型语言模型的摘要性能。未来工作应该专注于增加训练数据规模，并通过调整最大Token长度来优化摘要简洁性，进一步提高模型性能。

### 5.7 挑战与解决方案

#### 挑战1：长文本处理
**问题**：科学论文经常超过8000 tokens的上下文窗口。即使在上下文窗口限制内，大型语言模型在处理长文本时表现出性能退化，表现为注意力分散、关键信息遗漏和生成质量不一致，特别是处理文档后半部分的内容时。

**实施的解决方案**：
- 设计专门的提示，增强长文本处理能力：
  * 感知段落的提示，引导模型注意文档结构
  * 增强记忆的提示，包括明确指令以在长跨度中保持连贯性

#### 挑战2：科学领域适应

**问题**：科学文本包含专业术语、复杂关系和领域特定惯例，一般语言模型可能难以准确表示。

**实施的解决方案**：
- 开发针对科学内容优化的领域特定提示模板
- 将Adaptor训练集中在科学出版物上，以优化领域特定语言模式
- 针对控制信息优先级的特定注意力组件（`q_proj`、`k_proj`、`v_proj`、`o_proj`）

#### 挑战3：评估方法

**问题**：自动指标通常无法捕捉摘要质量的细微方面，特别是对于科学内容，事实准确性至关重要。

**实施的解决方案**：
- 平衡自动指标（ROUGE、BERTScore）与有针对性的人工评估协议

#### 挑战4：摘要质量平衡

**问题**：在全面信息覆盖和简洁表达之间取得最佳平衡具有挑战性，特别是对于复杂的科学内容。

**实施的解决方案**：
- 开发结合多个质量维度的加权评估指标
- 实验控制生成参数以找到最佳设置：
  * 较低温度（0.3-0.5）产生更确定性和精确的摘要
  * 较高top-p值（0.9-0.95）保持一些创造性灵活性
  * 基于文档特征校准最大生成长度

#### 挑战5：跨语言泛化

**问题**：Adaptor主要在英语科学文献上训练，限制了其对不断增长的非英语科学出版物和国际研究社区的有效性。

**实施的解决方案**：
- 利用从基础Llama-3.2-3B-Instruct模型继承的多语言能力

#### 挑战6：计算资源约束

**问题**：训练和部署科学文献的高效摘要模型需要平衡模型质量和计算可行性，特别是在GPU资源有限的研究环境中。

**实施的解决方案**：
- 优化LoRA超参数（秩、alpha）以最大限度提高质量，同时最小化参数数量
- 实施梯度检查点和混合精度训练，减少内存需求

## 6. 进一步工作

本节按实施顺序概述未来工作方向。

### 6.1 评估指标扩展

开发专门针对科学文献的评估指标：

- **术语精确度**：测量科学术语的正确使用和保留
- **事实一致性**：评估生成摘要与原文事实的匹配度
- **引用保留**：评估保留关键引用信息的能力

实施简单直接，并能增强我们选择更好模型的能力。

### 6.2 超参数优化

- **层特定LoRA配置**：为不同层类型实施差异化LoRA参数：
  * 注意力组件（q_proj、k_proj、v_proj、o_proj）具有定制的秩/alpha值
  * 前馈网络（gate_proj、up_proj、down_proj）具有优化的参数
  * 这种有针对性的方法可以用相同的参数数量实现更好的精度

- **生成参数调整**：优化温度（0.3-0.5）和top-p值（0.9-0.95）

### 6.3 块处理增强

改进超出上下文窗口的文档处理：

- **两阶段摘要**：
  * 第1阶段：生成具有更高压缩比的段落级摘要
  * 第2阶段：将段落摘要合成为连贯的最终摘要
- **交叉引用保留**：维持文档段落之间的联系

### 6.4 后处理增强

实施输出优化技术：

- **科学内容验证**：使用大型语言模型比较源文本和摘要的事实一致性
- **语言优化**：应用思维链提示来改善语法和流畅性
- **格式适应**：用领域特定规则实施基于模板的转换

### 6.5 跨语言泛化

扩展模型对非英语科学文献的能力：

- **多语言提示模板**：为不同语言结构设计模板
- **特定语言Adaptor**：为主要科学语言训练专门的LoRAAdaptor
- **模块化Adaptor系统**：支持语言Adaptor之间的动态切换

实施顺序优先考虑指标和超参数的基础改进，然后再进展到分块、后处理和多语言能力的更复杂增强。这些增强代表了一个结构化路线图，在保持其高效架构的同时，推进我们科学摘要Adaptor的能力。

## 7. 源代码和模型

### 7.1 源代码

- **GitHub仓库**：https://github.com/amd-zhaofeng/SummarizationAdaptor
- **源代码组织**：

```
.
├── models/                # Adaptor检查点（保存的模型权重）
│   └── summarization_adapter_20250408_030703* # 微调Adaptor路径
├── src/                   # 源代码
│   ├── data/
│   │   ├── analysis.py    # 数据集分析工具
│   │   ├── processor.py   # 数据集处理
│   │   └── prompt.py      # 提示模板
│   ├── model/
│   │   └── processor.py   # 模型处理函数
│   ├── utils/
│   │   ├── logging.py     # 日志设置
│   │   ├── metrics.py     # 评估指标实现
│   │   └── seed.py        # 随机种子工具
│   ├── adapter.py         # Adaptor模型定义
│   ├── evaluate.py        # 评估模型
│   ├── inference.py       # 推理脚本
│   └── train.py           # 训练模型
├── requirements.txt       # 依赖项
└── README.md              # 项目描述
```

### 7.2 Adaptor

- **Adaptor模型路径**：`./models/summarization_adapter_20250408_030703`
- **使用方法**：在仓库的`evaluate.py`中将文件夹路径设置为`adapter_path`。更多详情请参阅`README.md`。

### 7.3 使用方法

- **训练模型**

  ```bash
  python src/train.py --base_model meta-llama/Llama-3.2-3B-Instruct --dataset ccdv/pubmed-summarization --output_dir models/
  ```

  输出Adaptor将保存在`./models`目录中，带有基于时间戳的名称，如`./models/summarization_adapter_20250408_030703`，这被称为Adaptor路径。

- **评估基础模型**

  ```bash
  python src/evaluate.py --dataset ccdv/pubmed-summarization --output_file results_base_model_evaluation.json
  ```

- **评估Adaptor模型**

  ```bash
  python src/evaluate.py --adapter_path ./models/summarization_adapter_20250408_030703 --dataset ccdv/pubmed-summarization --output_file results_adapter_model_evaluation.json
  ```

- **使用基础模型进行推理**

  ```bash
  python src/inference.py --input_file ./test_case/input.txt --output_file test_case/output_base.txt
  ```

- **使用Adaptor模型进行推理**

  ```bash
  python src/inference.py --adapter_path ./models/summarization_adapter_20250408_030703 --input_file ./test_case/input.txt --output_file test_case/output.txt
  ```
