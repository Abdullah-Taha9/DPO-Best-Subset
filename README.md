# IQP-Based Subset Selection for DPO Training

Efficient preference data selection for Direct Preference Optimization (DPO) using Integer Quadratic Programming (IQP).

🚀 Achieves **0.50 alignment score using only 550 samples (11%)** from the HH-RLHF dataset, outperforming baseline DPO trained on the full dataset (0.40).

---

## Overview

Training DPO on large preference datasets is computationally expensive and often limited by noisy or redundant samples.

This project addresses that by selecting a **small, high-quality subset** of training data that:

* Maximizes **informativeness** (uncertain preference pairs)
* Ensures **diversity** (coverage of different behaviors)
* Filters out **low-quality or irrelevant samples**

The result is a more efficient and effective training pipeline for alignment.

---

## Method

We formulate subset selection as an **Integer Quadratic Programming (IQP)** problem:

* **Utility**:
  Measures uncertainty using the inverse log-probability margin between chosen and rejected responses.

* **Diversity**:
  Computed using Gaussian similarity over **delta embeddings**:
  (chosen embedding − rejected embedding)

* **Optimization**:

  * LP-based initialization (heuristic scoring)
  * Truncated power iteration for refinement

* **Preprocessing**:
  Safety-focused filtering to remove noisy and irrelevant samples.

---

## Pipeline

```
HH-RLHF (5000 samples)
        ↓
Safety Filtering (~1400)
        ↓
Utility + Diversity Computation
        ↓
IQP Optimization
        ↓
Subset Selection (k ≈ 550)
        ↓
DPO Training
        ↓
Evaluation
```

---

## Results

| Method             | Samples |    Score |
| ------------------ | ------: | -------: |
| SFT Baseline       |       — |     0.10 |
| DPO (Full Dataset) |    5000 |     0.40 |
| IQP Subset (Ours)  |     550 | **0.50** |

---

## Repository Structure

```
.
├── main.ipynb        # Full pipeline (filtering, selection, evaluation)
├── best_subset.csv   # Selected subset used for training
├── requirements.txt  # Dependencies
└── README.md         # Project documentation
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

1. Open the notebook:

   ```bash
   main.ipynb
   ```

2. Run all cells sequentially.

The notebook performs:

* Dataset filtering
* Utility and diversity computation
* IQP-based subset selection
* Evaluation

### Output

* `best_subset.csv`: Selected subset of ~550 samples
* Evaluation metrics printed in notebook

---

## Dataset

This project uses the **Anthropic HH-RLHF dataset** (5,000 preference pairs).

Each sample contains:

* Prompt
* Chosen response (preferred)
* Rejected response

---

## Key Design Choices

* **Inverse-margin utility**
  Prioritizes uncertain samples where the model struggles

* **Delta embeddings**
  Capture preference differences rather than raw semantics

* **Low diversity weight (λ = 0.01)**
  Focus on informative samples over broad coverage

* **Safety-focused filtering**
  Removes noisy and irrelevant data before optimization

---

## Limitations

* Evaluation metric relies on pattern matching (may misclassify refusals)
* Results may vary slightly across hardware
* Dataset contains noisy or incorrectly labeled samples
* GPT-2 limits overall performance ceiling

---

## References

* Banerjee, S., Chakraborty, S. (2020)
  *Budgeted subset selection for fine-tuning deep learning architectures*
  IEEE IJCNN

* Rafailov, R. et al. (2023)
  *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
  NeurIPS

* Snowflake Arctic Embeddings
  https://huggingface.co/Snowflake/snowflake-arctic-embed-m

---

## Summary

This project demonstrates that **careful data selection can outperform full-dataset training**.

Instead of scaling data, we:

* filter noise,
* prioritize uncertainty,
* enforce diversity,
* and optimize selection.

Result: **better alignment with 9× less data**.
