![# OneSpace2RuleThemAll](asset/logo.jpg)
# Subspace Intersection Guidance for Hallucination Mitigation
The core idea is to guide large language models (LLMs) to focus on the **intersection of capability subspaces** so that the model can mitigate **two types of hallucinations** *without* introducing a trade\-off between them.

The codebase provides:

\- Methods to obtain **activation representations** of LLMs  
\- An editing procedure in **capability subspace** 
\- An **evaluation pipeline** to quantify hallucination mitigation

---

## 1\. Project Overview

Modern LLMs are powerful but prone to multiple kinds of hallucinations (e.g., factual hallucinations, reasoning hallucinations).  
This project implements an approach that:

1. Extracts model activations corresponding to different capabilities.
2. Identifies and operates on the **intersection** of these capability subspaces.
3. Edits the model in this intersection to jointly reduce:
   - hallucinations measured by **DiSQ\-Score**, and  
   - hallucinations measured by **TruthfulQA**,  
   while avoiding a performance trade-off between them.

The overall pipeline is:

1. [Get Activations](#41-get-activations) - collect and store model activations. 
2. [Space Edit](#42-space-edit) - perform capability subspace intersection and model editing.  
3. [Evaluation](#43-evaluation) - run benchmarks to evaluate hallucination mitigation

---

## 2\. Environment Setup

This project uses `conda` for environment management.

```bash
# Create environment from file
conda env create -f environment.yaml

# Activate environment
conda activate space

```
## 3\. Data

This project uses two public datasets:

1. **DiSQ\-Score**  
   - Repository: https://github.com/YisongMiao/DiSQ-Score  
   - Purpose: Evaluating understanding errors.

2. **TruthfulQA**  
   - Repository: https://github.com/sylinrl/TruthfulQA  
   - Purpose: Evaluating whether model outputs are truthful and non\-misleading.

### 3\.1 Data Preparation

Please download the datasets from the respective repositories.
## 4\. Usage

The workflow has three main stages:

1. **Get activations**
2. **Edit in capability subspace**
3. **Evaluate**

### 4\.1 Get Activations

This step extracts model activations needed to construct capability subspaces.

Example:

```bash
    python get_activation/get_activation.py \
    --model <MODEL_NAME_OR_PATH> \
    --data_dir data/ \
    --output_dir outputs/activations
```

### 4.2 Space Edit
This step performs the editing of the model in the intersection of capability subspaces.
Example:
```bash
    python scripts/space_edit.py \
    --model <MODEL_NAME_OR_PATH> \
    --activation_dir outputs/activations \
    --output_dir outputs/edited_model \
```
### 4.3 Evaluation
This step evaluates the edited model on hallucination benchmarks.
Example:
```bash
    python scripts/evaluate.py \
    --model <EDITED_MODEL_PATH> \
    --data_dir data/ \
    --output_dir outputs/evaluation_results
```

## 5\. Citation
If you find this code useful for your research, please cite our paper:

```
@inproceedings{wangone,
  title={One SPACE to Rule Them All: Jointly Mitigating Factuality and Faithfulness Hallucinations in LLMs},
  author={Wang, Pengbo and Li, Chaozhuo and Wang, Chenxu and Zheng, Liwen and Zhang, Litian and Zhang, Xi},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```