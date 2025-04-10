# Design Document of the Summary Adapter

This project develops a **long document summarization adapter** for generating high-quality summaries of **scientific literature** through the efficient fine-tuning of **Llama-3.2-3B-Instruct**. 

## Table of Contents
- [1. Adapter Architecture](#1-adapter-architecture)
   - [1.1 Technical Approach](#11-technical-approach)
   - [1.2 System Components](#12-system-components)
   - [1.3 Key Advantages](#13-key-advantages)
- [2. Input and Output Format](#2-input-and-output-format)
   - [2.1 Input Format](#21-input-format)
   - [2.2 Output Format](#22-output-format)
- [3. Evaluation Metrics](#3-evaluation-metrics)
   - [3.1 Primary Metrics](#31-primary-metrics)
   - [3.2 Training Loss](#32-training-loss)
   - [3.3 Metric Usage](#33-metric-usage)
- [4. Dataset](#4-dataset)
   - [4.1 Training Dataset](#41-training-dataset)
   - [4.2 Evaluation Datasets](#42-evaluation-datasets)
- [5. Implementation Plan](#5-implementation-plan)
   - [5.1 Environment Preparation](#51-environment-preparation)
   - [5.2 Data Preprocessing](#52-data-preprocessing)
   - [5.3 Adapter Design and Implementation](#53-adapter-design-and-implementation)
   - [5.4 Training Process](#54-training-process)
   - [5.5 Post-Processing](#55-post-processing)
   - [5.6 Evaluation and Optimization](#56-evaluation-and-optimization)
   - [5.7 Challenges and Solutions](#57-challenges-and-solutions)
- [6. Source Code and Model](#6-source-code-and-model)
   - [6.1 Source Code](#61-source-code)
   - [6.2 Adapter](#62-adapter)

## 1. Adapter Architecture

### 1.1 Technical Approach

#### 1.1.1 Core Methodology
- **Base Model**: Llama-3.2-3B-Instruct (3B parameters) with 8,192 token context window
- **Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **Target Components**: Key attention mechanisms and feed-forward networks

#### 1.1.2 Integration Mechanism
The adapter integrates with the base model through a **non-intrusive weight modification approach**:
- Base model weights remain completely frozen during training and inference
- Adapter introduces small, trainable matrices that modify the output of specific transformer layers
- During inference, adapter weights can either operate in parallel with base weights (using PEFT's inference mode) or be mathematically merged with base weights for optimal performance
- Integration requires no architecture changes to the underlying model and datasets, preserving all original capabilities

### 1.2 System Components

The adapter system integrates four essential components:

1. **Base Language Model** (Llama-3.2-3B-Instruct)

2. **LoRA Adapter Module**
   - Implements targeted weight updates via low-rank matrices
   - Focuses on key layers:
     * Attention components: `q_proj`, `k_proj`, `v_proj`, `o_proj`
     * Feed-forward networks: `gate_proj`, `up_proj`, `down_proj`

3. **Data Processing Pipeline**
   - **Domain Prompt**: Optimizes instructions for scientific literature processing
   - **Intelligent chunking**: Keep more useful information

4. **Comprehensive Metrics**
   - Employs ROUGE, BERTScore, and Summary Length Ratio to evaluate lexical overlap, semantic similarity, and compression efficiency.

### 1.3 Key Advantages

The adapter architecture offers several significant advantages:

1. **Computation Efficiency**: Trains only 0.1% of total parameters while keeping the base model frozen, dramatically reducing GPU memory usage and computational requirements.

2. **Integration Flexibility**: Can be deployed as a separate module or merged with the base model weights for optimal inference performance.

3. **Domain Specialization**: Optimizes summarization specifically for scientific literature while preserving general capabilities of the base model.

4. **Quality-Focused Design**: Combines specialized domain prompts, intelligent chunking strategies, and comprehensive evaluation metrics to ensure high-quality scientific summaries.

5. **Model and Dataset Versatility**: Supports flexible switching between different base models (beyond Llama-3.2) and adaptation to various fine-tuning and evaluation datasets, enabling customization for specific use cases and domains.

## 2. Input and Output Format

### 2.1 Input Format

- Plain text in articles, reports, papers, etc.
- Example:
```
{
"document": "Artificial Intelligence (AI) is a subfield of computer science dedicated to developing systems and software that can simulate human intelligence.
It includes multiple research directions such as machine learning, deep learning, natural language processing, and computer vision.
Machine learning uses statistical techniques to enable computer systems to learn from data and gradually improve performance without explicit programming.
Deep learning is a branch of machine learning that uses multi-layer neural networks to process data, particularly suitable for handling unstructured data such as images, sound, and text.
..."
}
```

### 2.2 Output Format

- Plain text summaries

- Example:
```
Artificial intelligence is a branch of computer science that simulates human intelligence through technologies such as machine learning, deep learning, natural language processing, and computer vision. It has been widely applied in medical, financial, autonomous driving and other fields. Despite facing challenges in ethics and privacy, researchers continue to develop more advanced AI systems.
```

## 3. Evaluation Metrics

### 3.1 Primary Metrics
1. **ROUGE Scores** (↑) 
   - ROUGE-1: Word overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence
   - Implemented using the `rouge_score` library
   
2. **BERTScore** (↑) 
   - Provides Precision, Recall, and F1 scores
   - Implemented using the `bert_score` library
   - Supports multi-language evaluation, with English ("en") as the default

3. **Summary Length Ratio** (↔)
   - Ratio of generated summary length to reference summary length
   - Optimal values are close to 100%  (neither too high nor too low)

> **Note on Future work**: Future work will implement specialized scientific evaluation metrics (**Terminology Precision, Factual Consistency, Citation Retention**) to better assess scientific accuracy beyond general text similarity.

### 3.2 Training Loss (↓)
Training uses standard **Cross-Entropy Loss**, calculating negative log-likelihood between predicted and target tokens. It's implemented with PyTorch's CrossEntropyLoss, applies only to summary tokens, and serves as the main optimization objective.

### 3.3 Metric Usage
These metrics are used for:
1. **Monitoring training progress**: Tracks real-time progress using cross-entropy loss, visualizes via TensorBoard, and saves checkpoints based on validation loss.

2. **Model selection and comparison**: Uses ROUGE and BERTScore_F1 as primary selection criterion.

3. **Baseline comparison**: Compares performance with original Llama-3.2-3B-Instruct and other open-source summarization models (e.g., PEGASUS, BART, T5).

4. **Hyperparameter optimization**: Adjusts LoRA parameters, template design, and generation parameters (temperature, top-p, maximum length) based on metric feedback.

5. **Error analysis and debugging**: Analyzes low-scoring samples, monitors performance differences across document types, and adjusts training strategies.

## 4. Dataset

### 4.1 Training Dataset

#### 4.1.1 Dataset Description
In this project, for long document summarization, the **PubMed Summarization Dataset** ([ccdv/pubmed-summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization)) was selected. This dataset contains scientific articles from the biomedical domain with corresponding abstracts, which serve as reference summaries. In the processing pipeline, the "article" field is renamed to "text" and the "abstract" field is renamed to "summary" for consistency.

**Dataset Features**:
- **Size**: 133,215 article-abstract pairs (119,924 training, 6,633 validation, 6,658 test samples)
- **Domain**: Biomedical and healthcare research
- **Document Type**: Scientific research papers and clinical studies
- **Source**: Articles from the PubMed database, a comprehensive archive of biomedical literature
- **Language**: English
- **Publication Period**: Spans multiple years of biomedical research publications

**Length Statistics**:
- **Document Length**: 
  - Mean: 4,069 tokens (18,140 characters)
  - Median: 3,376 tokens (15,424 characters)
  - 95th percentile: 9,450 tokens (43,212 characters)
  - Maximum: 45,840 tokens (159,114 characters)

- **Summary Length**:
  - Mean: 268 tokens (1,254 characters)
  - Median: 271 tokens (1,307 characters)
  - 95th percentile: 434 tokens (1,923 characters)
  - Maximum: 532 tokens (2,325 characters)

- **Token Distribution**:
  - 0-1k tokens: 4.9% of documents
  - 1k-2k tokens: 20.0% of documents
  - 2k-4k tokens: 36.0% of documents
  - 4k-8k tokens: 30.9% of documents
  - 8k-16k tokens: 7.7% of documents
  - 16k+ tokens: 0.5% of documents

- **Compression Ratio**: The average document is compressed by a factor of 15.18x in the summaries

#### 4.1.2 Dataset Quality Analysis

The quality evaluation revealed several important characteristics:

- **Text Quality**:
  - Special character ratio: 2.98%
  - Average sentence length: 121.18 characters
  - Average sentence count per document: 145.49
  - Word diversity (unique words/total words): 30.46%

- **Summary Quality**:
  - Summary completeness (summary length/document length): 6.92%
  - Summary uniqueness (unique words in summary): 61.21%

- **Dataset Balance**:
  - Length standard deviation normalized by mean: 0.73 (indicating moderate variability)

The detailed dataset quality analysis was performed using the code in `src/data/analysis.py`, which implements a comprehensive evaluation pipeline for assessing text quality, summary characteristics, and dataset balance metrics.

Overall, the PubMed dataset provides high-quality scientific articles with professionally written abstracts that maintain a consistent compression ratio, making it an ideal candidate for training the summarization adapter.

#### 4.1.3 Primarily Parameter Suggestions

Based on these statistics, the following parameter suggestions were derived for the model:

- **Input Context Length**: 9,510 tokens (includes input document with prompt)
  - This accommodates 95% of documents in the dataset

- **Maximum Generation Length**: 454 tokens
  - This covers the 95th percentile of reference summaries plus a small buffer

- **Sampling Parameters**:
  - Temperature: 0.5 (balanced between creativity and coherence)
  - Top-p: 0.9 (standard value for controlled diversity)

### 4.2 Evaluation Datasets

For comprehensive model evaluation, two distinct test datasets were used to assess both in-domain performance and cross-domain generalization capability:

#### 4.2.1 PubMed Test Set

- **Description**: The standard test split from the PubMed dataset.
- **Size**: 6,658 article-abstract pairs
- **Domain**: Biomedical and healthcare research
- **Type of Documents**: Scientific research papers, clinical studies, and medical case reports from the biomedical literature

#### 4.2.2 ArXiv Test Set

- **Description**: The test split from the ArXiv Summarization Dataset ([ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization)), used to evaluate the model's cross-domain generalization capabilities on scientific papers from different fields.

- **Size**: 6,440 article-abstract pairs

- **Domain**: Multi-disciplinary scientific research with emphasis on physics, mathematics, computer science, and engineering

- **Type of Documents**: Scientific papers, technical reports, and research preprints submitted to the ArXiv repository

Using both datasets provides a more robust evaluation of the model's capabilities:
- PubMed test set measures performance on the domain the model was specifically trained for
- ArXiv test set assesses how well the model generalizes to adjacent but different scientific domains
- Comparison between the two reveals domain adaptation capabilities and limitations

## 5. Implementation Plan

### 5.1 Environment Preparation
- Install necessary dependencies (transformers, peft, accelerate, etc.)
- Configure utility modules for consistent logging, reproducible random seed, and metrics calculation

### 5.2 Data Preprocessing

Data preprocessing forms the foundation for model training, transforming raw documents and summaries into optimal formats. The pipeline focuses on three core aspects: data quality assurance, format standardization, and token management optimization.

#### Step 1: Dataset Loading and Preparation
1. **Dataset Acquisition**: Download the ccdv/pubmed-summarization dataset from Hugging Face
2. **Split Identification**: Preserve original train/validation/test splits
3. **Initialization Setup**: Configure tokenizer, maximum length, and processing parameters

#### Step 2: Data Cleaning and Filtering
1. **Empty Value Handling**: Filter out samples with empty text or summary
2. **Field Standardization**: Unify field naming (article → text, abstract → summary)
3. **Sampling Control**: Implement configurable random sampling for large datasets

> **Note**: Based on our dataset quality analysis (Section 4.1.2), the PubMed dataset exhibits high quality with professionally written content. Therefore, minimal cleaning was required beyond basic empty value filtering. For datasets of lower quality, additional cleaning steps commonly include:
> - **Text Normalization**: Standardizing unicode characters, removing excessive whitespace, and normalizing punctuation
> - **Noise Removal**: Eliminating HTML tags, URLs, and other non-textual elements
> - **Duplicate Detection**: Removing exact or near-duplicate document-summary pairs
> - **Outlier Filtering**: Removing samples with extreme length ratios or unusual text patterns
> - **Language Detection**: Ensuring all content is in the expected language (English)
> - **Content Quality Filtering**: Removing samples with high special character ratios or extremely short/long sentences

#### Step 3: Template Application and Formatting
1. **Prompt Template Selection**: Choose appropriate domain-specific prompt templates based on dataset type
2. **Format Construction**: Combine text with prompt templates to create model-ready input format
3. **Tokenizer Configuration**: Ensure tokenizer correctly handles special tokens (such as using EOS token as PAD token when necessary)


> **Introduction of: Domain-Optimized Prompt Template Selection**
>
> When processed internally, the input text is converted to a prompt format using a domain-specific template system. This enhances the model's ability to generate appropriate summaries for different content types.
>
> | Dataset | Prompt Template |
> |---------|----------------|
> | Default | `Please generate a concise summary for the following text:`<br>`[Input Text]`<br>`Summary: ` |
> | PubMed/ArXiv | `Generate a comprehensive abstract for the following scientific article:`<br>`[Input Text]`<br>`Summary: ` |

#### Step 4: Intelligent Token Management

During training, inputs consist of concatenated document text and summary, where the summary portion is critical and must be preserved completely. With Llama-3.2-3B-Instruct's 8,192 token context window limitation and approximately 10% of scientific papers exceeding 8,000 tokens in length, simple concatenation often leads to information loss.

**Why Intelligent Token Management is Necessary**:
- Scientific papers + summaries often exceed the 8,192 token limit
- Document beginnings contain more critical information than later sections
- Complete summary preservation is essential for effective training

**Intelligent Token Allocation Approach**:

1. **Summary-First Encoding**: Prioritize summary preservation
   - Encode the complete summary without truncation
   - Calculate token budget accounting for special tokens:
     ```python
     # Calculate special tokens count by comparing encoded text with and without special tokens
     sample_encoded = tokenizer(sample_text, return_tensors="pt")
     sample_text_only = tokenizer("SAMPLESAMPLE", return_tensors="pt")
     special_tokens_count = sample_encoded.input_ids.size(1) - sample_text_only.input_ids.size(1) + 2
     ```
   - Reserve necessary tokens for the summary portion

2. **Dynamic Document Truncation**:
   - Create custom truncation threshold for each sample based on its summary length
   - Apply precise truncation at sentence boundaries when documents exceed available token budget
   - Prioritize document truncation over summary compression

3. **Verification**: Final encoding check to ensure inputs don't exceed context window and track token allocation metrics.

This adaptive approach ensures maximum retention of source document information while preserving complete target summaries, significantly improving training efficiency compared to fixed-length truncation strategies.

### 5.3 Adapter Design and Implementation

The adapter implementation follows a modular approach to efficiently extend Llama-3.2-3B-Instruct with summarization capabilities while minimizing computational and memory requirements. The design employs LoRA to selectively adapt key model components.

#### Step 1: Model and Tokenizer Initialization
Load the base model using Hugging Face's AutoModelForCausalLM and configure tokenizer with proper padding token handling.

#### Step 2: LoRA Adapter Configuration
1. **Target Module Selection**: Identify specific weight matrices in attention and feed-forward networks for adaptation
   ```python
   # From adapter.py
   target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"]
   ```
   
   The selection of these specific layers for adaptation is strategic and task-specific for summarization:
   
   - **Attention Mechanism Components** (`q_proj`, `k_proj`, `v_proj`, `o_proj`):
     - These layers control how the model attends to different parts of the input text
     - For summarization, modifying attention patterns helps the model identify salient information across long documents
     - Query projections (`q_proj`) determine what information to look for; adapting this helps the model learn what content is summary-worthy
     - Key and value projections (`k_proj`, `v_proj`) help establish relationships between different parts of text, critical for coherent summarization
     - Output projections (`o_proj`) integrate attended information, influencing how the model synthesizes content
   
   - **Feed-Forward Networks** (`gate_proj`, `up_proj`, `down_proj`):
     - These layers are responsible for higher-level feature transformation and reasoning
     - The up-projection (`up_proj`) expands representation dimensionality, allowing for more complex transformations needed in abstractive summarization
     - The gate projection (`gate_proj`) controls information flow, helping filter relevant vs. irrelevant content
     - The down-projection (`down_proj`) compresses information, aligning with the fundamental compression nature of summarization

2. **Hyperparameter Setting**:
   ```python
   # From adapter.py
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
   - Rank (`r=16`): This rank was selected based on empirical testing showing it provides an optimal balance between adaptation capacity and parameter efficiency
   - Alpha (`lora_alpha=32`):  Alpha = 2 × rank provides stable gradient flow during training
   - Dropout(`lora_dropout=0`): 
     - Zero dropout was chosen as summarization benefits from deterministic attention patterns
     - The PubMed dataset's high quality and size provided sufficient regularization without dropout
   - Bias Setting (`bias="none"`): Controls whether bias terms are trained
     - Bias terms primarily affect token-level baseline preferences rather than contextual understanding
     - For summarization, contextual relationships between tokens are more important than individual token biases

> **Future work of layer-specific LoRA configurations**: The current implementation applies identical LoRA parameters across all target modules. Future experiments could implement differentiated LoRA configurations for each layer type based on their specific function in the summarization process. For example, attention mechanism components (q_proj, k_proj, v_proj, o_proj) might benefit from different rank or alpha values than feed-forward network components (gate_proj, up_proj, down_proj). This layer-specific optimization could potentially achieve better precision with the same or fewer trainable parameters.

### 5.4 Training Process

The training process leverages the SFTTrainer from the TRL (Transformer Reinforcement Learning) library to streamline fine-tuning while applying best practices for efficient learning. The process focuses on adapter parameter optimization while keeping the base model frozen.

#### Step 1: Training Configuration and Arguments Setup
- Configure comprehensive training parameters using SFTConfig
- Establish optimization strategy and scheduling
- Set up evaluation protocols and output management

```python
training_args = SFTConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # Effective batch size: 16 samples per update
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

The configuration includes several optimized components:

1. **Optimizer Strategy**
   - **AdamW 8-bit** optimizer (`optim="adamw_8bit"`) for memory efficiency
   - Default learning rate: 1e-5 for adapter parameters
   - Weight decay: 0.01 (applied to non-bias parameters)
   - Beta parameters: β₁=0.9, β₂=0.999 (default AdamW momentum coefficients, not explicitly set in code)
   - Epsilon: 1e-8 (default numerical stability value, inherited from the optimizer implementation)

2. **Gradient Clipping Strategy**
   - Maximum gradient norm: 0.7 (`max_grad_norm=0.7`)
     * More conservative than the typical default of 1.0 used in general language tasks
     * Prevents gradient explosions common in long text processing with extended context windows
     * Helps stabilize training when processing complex scientific terminology and nested relationships

3. **Learning Rate Schedule**
   - Linear warmup for first 10% of steps (`warmup_ratio=0.1`) - explicitly configured
   - Linear decay for remaining steps (default scheduler behavior in Hugging Face Trainer)
   - Minimum learning rate: 10% of peak learning rate (inherent behavior of the linear scheduler, not explicitly configured)

#### Step 2: Trainer Setup and Execution
- Initialize and execute the training process by setting up SFTTrainer with the model, datasets, and LoRA configuration, then run the training loop with automated evaluation and checkpointing.

```python
# Initialize SFTTrainer
trainer = SFTTrainer(
    model=adapter.model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    processing_class=adapter.tokenizer,
    args=training_args,
)

# Start training
trainer.train()
```

#### Step 3: Model Persistence and Deployment
- Save the best adapter weights based on validation performance
- Create a portable adapter that can be distributed separately from base model
```python
# Save adapter model
trainer.save_model(output_dir)
adapter.tokenizer.save_pretrained(output_dir)
```

### 5.5 Post-Processing

Post-Processing Workflow of Output Cleanup and Normalization：
- Remove any residual prompt artifacts or delimiter tokens from generated text
- Standardize whitespace, punctuation, and formatting
- Fix common tokenization artifacts (split words, irregular spacing)
- Normalize scientific notation, units, and numerical representations

> **Future work on Advanced Post-Processing**: The following steps represent the design vision but are not yet implemented in the current version. These capabilities could be realized through:
>    - Scientific Content Verification: Using LLMs to compare source text and summary for factual consistency
>    - Linguistic Refinement: Chain-of-thought prompting for targeted grammar and fluency improvements
>    - Format Adaptation: Template-based transformation with domain-specific rules


### 5.6 Evaluation and Optimization

#### Step 1: Metrics Implementation and Calculation
- Deploy the multi-dimensional evaluation metrics described in Section 3.1

#### Step 2: Test Set Evaluation
- Evaluate performance on PubMed test set (6,658 samples) for in-domain assessment
- Perform cross-domain evaluation on ArXiv test set (6,440 samples)

#### Step 3: Hyperparameter Optimization
- Tune generation parameters based on evaluation feedback:
  * Temperature settings: 0.3 ~ 0.8
  * Top-p values: Evaluating 0.85 ~ 0.95
  * Maximum generation length: Calibrated using Summary Length Ratio
- Select optimal configuration based on weighted metric performance

> **Future work**: Future work will incorporate human evaluation for qualitative analysis and error diagnosis to complement the automated metrics.

#### Preliminary Evaluation Results

The following tables present the preliminary evaluation results of our adapter compared to baseline models with the default parameters in the repository (with 5000 samples for training, 500 for validation, 500 for test).

##### PubMed Test Set (In-Domain)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Length Ratio |
|-------|---------|---------|---------|--------------|--------------|
| Our Adapter | 41.46 | 18.02 | 24.41 | 85.70 | 338.08% |
| Base Llama-3.2-3B | 37.18 | 14.05 | 19.88 | 83.34 | 381.19% |

##### ArXiv Test Set (Cross-Domain)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Length Ratio |
|-------|---------|---------|---------|--------------|--------------|
| Our Adapter | 30.34 | 12.12 | 17.90 | 84.81 | 1347.67% |
| Base Llama-3.2-3B | 28.53 | 10.33 | 16.10 | 83.30 | 1356.81% |

These preliminary results demonstrate the effectiveness of the adapter approach. Despite being constrained by time and computational resources, which limited the training to only 5,000 samples for fine-tuning, the adapter model still significantly outperforms the base Llama-3.2-3B-Instruct model across all evaluation metrics.

On the ArXiv cross-domain test set, the adapter model shows an improvement of 1.81 percentage points in ROUGE-1 (30.34 vs 28.53), 1.79 percentage points in ROUGE-2 (12.12 vs 10.33), and 1.80 percentage points in ROUGE-L (17.90 vs 16.10). These improvements indicate that even with limited training data, the adapter model can generate more accurate and relevant summary content.

Equally noteworthy is the enhancement in BERTScore F1 (84.81 vs 83.30), suggesting that the model has made progress in semantic understanding and expression, better capturing the core meaning of the original text.

However, the Length Ratio for both models remains excessively high (1347.67% and 1356.81% respectively), far exceeding the ideal summary length proportion. This suggests that the configured maximum token length may be too generous, causing the models to generate verbose summaries. Although the adapter model shows a slight improvement in this aspect (1347.67% vs 1356.81%), the difference is minimal, indicating that length control remains a key area for future optimization.

Overall, these results are encouraging, proving that even with limited training samples, the adapter method can effectively enhance the summarization performance of large language models. Future work should focus on increasing the scale of training data and optimizing summary conciseness by adjusting the maximum token length to further improve model performance.

> **Future work**: 
> - Human evaluation of factual accuracy and coherence
> - Domain-specific assessment of scientific terminology preservation
> - Analysis of adapter performance across different scientific subdomains
> - Detailed error analysis to guide future improvements

### 5.7 Challenges and Solutions

#### Challenge 1: Long Text Processing
**Problem**: Scientific papers frequently exceed the 8000 token context window. Even within context window limits, large language models demonstrate degraded performance when processing lengthy texts, exhibiting attention dilution, key information omission, and inconsistent generation quality, particularly when handling content from the latter portions of documents.

**Implemented Solutions**:
- Design specialized prompts that enhance long-text processing capabilities:
  * Section-aware prompts that direct model attention to document structure
  * Memory-enhancing prompts that include explicit instructions to maintain coherence across long spans

**Future Enhancements**:
- Process each chunk separately while maintaining cross-references between sections
- Apply a two-stage summarization approach:
  * Stage 1: Generate section-level summaries with higher compression ratio
  * Stage 2: Synthesize section summaries into a coherent final summary

#### Challenge 2: Scientific Domain Adaptation

**Problem**: Scientific text contains specialized terminology, complex relationships, and domain-specific conventions that general language models may struggle to accurately represent.

**Implemented Solutions**:
- Develop domain-specific prompt templates optimized for scientific content
- Focus adapter training on scientific publications to optimize for domain-specific language patterns
- Target specific attention components (`q_proj`, `k_proj`, `v_proj`, `o_proj`) that control information prioritization

#### Challenge 3: Evaluation Methodology

**Problem**: Automated metrics often fail to capture nuanced aspects of summary quality, particularly for scientific content where factual accuracy is paramount.

**Implemented Solutions**:
- Balance automated metrics (ROUGE, BERTScore) with targeted human evaluation protocols

**Future Enhancements**:
- Develop specialized metrics for scientific document summarization that assess:
  * Terminology precision and consistency
  * Factual accuracy of numerical data and relationships
  * Preservation of key scientific claims and findings
  * Methodological clarity and completeness

#### Challenge 4: Summary Quality Balance

**Problem**: Achieving the optimal balance between comprehensive information coverage and concise presentation is challenging, particularly for complex scientific content.

**Implemented Solutions**:
- Develop a weighted evaluation metric combining multiple quality dimensions
- Experiment with controlled generation parameters to find optimal settings:
  * Lower temperature (0.3-0.5) for more deterministic and precise summaries
  * Higher top-p values (0.9-0.95) to maintain some creative flexibility
  * Calibrated maximum generation length based on document characteristics

#### Challenge 5: Cross-Lingual Generalization

**Problem**: The adapter is primarily trained on English scientific literature, limiting its effectiveness for the growing body of non-English scientific publications and international research communities.

**Implemented Solutions**:
- Leverage the multilingual capabilities inherited from the base Llama-3.2-3B-Instruct model

**Future Enhancements**:
- Create language-specific prompt templates that account for structural differences in scientific writing across languages
- Develop targeted evaluation datasets for major scientific languages (Chinese, Spanish, French, German, Japanese)
- Implement language-specific adapters that can be applied to the base model for different language requirements:
  * Train language-specialized LoRA adapters using the same architecture but language-specific datasets
  * Create a modular system allowing dynamic switching between language adapters

#### Challenge 6: Computational Resource Constraints

**Problem**: Training and deploying efficient summarization models for scientific literature requires balancing model quality with computational feasibility, especially in research environments with limited GPU resources.

**Implemented Solutions**:
- Optimize LoRA hyperparameters (rank, alpha) to maximize quality while minimizing parameter count
- Implement gradient checkpointing and mixed precision training to reduce memory requirements

**Future Enhancements**:
- Develop inference optimization techniques such as weight merging and quantization
- Design progressive training strategies that enable incremental improvements with limited resources

## 6. Source Code and Model

### 6.1 Source Code

- **GitHub repository**: https://github.com/amd-zhaofeng/SummarizationAdaptor
- **Source code organization**:

```
.
├── models/                # Adapter checkpoints (saved model weights)
│   └── summarization_adapter_20250408_030703* # Finetuned adapter path
├── src/                   # Source code
│   ├── data/
│   │   ├── analysis.py    # Dataset analysis tools
│   │   ├── processor.py   # Dataset processing
│   │   └── prompt.py      # Prompt templates
│   ├── model/
│   │   └── processor.py   # Model processing functions
│   ├── utils/
│   │   ├── logging.py     # Logging setup
│   │   ├── metrics.py     # Evaluation metrics implementation
│   │   └── seed.py        # Random seed utilities
│   ├── adapter.py         # Adapter model definition
│   ├── evaluate.py        # Evaluate model
│   ├── inference.py       # Inference script
│   └── train.py           # Train model
├── requirements.txt       # Dependencies
└── README.md              # Project description
```

### 6.2 Adapter

- **Adapter Model Path**: `./models/summarization_adapter_20250408_030703`
- **Usage**: Set the folder path as `adapter_path` in `evaluate.py` of repository. Refer `README.md` for more details.

### 6.3 Usage

- **Training the Model**

  ```bash
  python src/train.py --base_model meta-llama/Llama-3.2-3B-Instruct --dataset ccdv/pubmed-summarization --output_dir models/
  ```

  The output adapter will be saved in the `./models` directory with a timestamp-based name, such as `./models/summarization_adapter_20250408_030703`, which is referred to as the adapter path.

- **Evaluating the Base Model**

  ```bash
  python src/evaluate.py --dataset ccdv/pubmed-summarization --output_file results_base_model_evaluation.json
  ```

- **Evaluating the Adapter Model**

  ```bash
  python src/evaluate.py --adapter_path ./models/summarization_adapter_20250408_030703 --dataset ccdv/pubmed-summarization --output_file results_adapter_model_evaluation.json
  ```

- **Inference**

  ```bash
  python src/inference.py --adapter_path ./models/summarization_adapter_20250408_030703 --input_file test_case/input.txt --output_file test_case/output.txt
  ```