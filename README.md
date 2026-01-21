# Leveraging Protein Language Models for Rapid and Robust Disulfide Bond Prediction

## Abstract

**Motivation:**
Disulfide bonds are fundamental to protein stability and the design of therapeutics. While structural folding models like AlphaFold2 have revolutionized biology, they remain computationally prohibitive for high-throughput screening and intractable for extremely large proteins due to memory constraints. Additionally, these models often underperform on orphan sequences lacking evolutionary profiles. To address these bottlenecks, we introduce a lightweight, sequence-only predictor that captures the intrinsic propensity for bond formation without the computational cost of full 3D folding.

**Results:**
We introduce **DeepDisulfide**, a deep learning framework that predicts disulfide bonds directly from protein sequences using embeddings from pre-trained Protein Language Models. By leveraging the inherent pairwise modeling capabilities of attention mechanisms, our model captures long-range cysteine dependencies without requiring Multiple Sequence Alignments. To ensure scalability, we employ a transfer learning strategy, pre-training on standard lengths before fine-tuning on long sequences. This allows DeepDisulfide to process proteins exceeding 9,000 residues on consumer hardware, a regime inaccessible to 3D folding methods, while maintaining high accuracy on short peptides. DeepDisulfide achieves a precision-recall area under curve of 0.91, outperforming state-of-the-art folding models (around 0.75), with a 100x reduction in inference time.

---

## Repository Contents

This repository contains the source code for the DeepDisulfide framework, including scripts for:
* Training
* Fine-tuning
* Testing and Inference

## Usage

### 1. Test with Your Own Sequence
If you only want to run inference on your own protein sequences to predict disulfide bonds, please refer to the Jupyter Notebook:
* `example.ipynb`

### 2. Reproduce Results
To reproduce the training process and results presented in the paper, please follow the steps below.

**Step 1: Prepare Data**
Download the preprocessed data from the following link:
* [Link TBD - Data upload in progress]

**Step 2: Training**
After downloading the data, run the initial training script:
```bash
python training.py
```

**Step 3: Fine-tuning**
Once the initial training is complete, run the fine-tuning script for long sequences:
```bash
python longseq_finetune.py
```
### 3. Performance Evaluation
If you only want to reproduce the performance metrics on our testing set, please refer to the testing notebook:
* `test.ipynb`
