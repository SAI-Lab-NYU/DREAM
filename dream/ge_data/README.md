# Training Data Preparation

## Quick Overview

DREAM training requires ~61,000 samples:

- 55,000 from LLaVA v1.5 mix665k
- 1,000 from each of 6 benchmark datasets

## Prerequisites

Download the [LLaVA v1.5 mix665k dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) and place it in your data directory.

## Benchmark Datasets

| Dataset | HuggingFace Path |
|---------|------------------|
| MMT-Bench | `OpenGVLab/MMT-Bench` |
| SEED-Bench-2 | `AILab-CVC/SEED-Bench-2` |
| ScienceQA | `derek-thomas/ScienceQA` |
| MathVista | `AI4Math/MathVista` |
| OCRBench | `lmms-lab/OCRBench-v2` |
| ChartQA | `ahmed-masry/ChartQA` |

## Steps

### 1. Generate base training data (55K samples)

```bash
python -m dream.ge_data.allocation_mix665
```

### 2. Split MMT-Bench and SEED-Bench-2

```bash
python -m dream.ge_data.train_test_split.mmt_ge
python -m dream.ge_data.train_test_split.seed_ge
```

Output: `processed_data/` and `processed_seed_data/` folders with train/test splits

### 3. Generate benchmark training data

For each dataset, modify line 301 in `dream/ge_data/allocation_suppliments.py`:

Then run:

```bash
python -m dream.ge_data.allocation_suppliments.py
```

Repeat 6 times (once per dataset).

## Verification

You should have:

- 55,000 samples from LLaVA mix665k
- 1,000 training samples × 6 benchmarks = 6,000 samples
- Test splits for evaluation

**Total: ~61,000 training samples**
