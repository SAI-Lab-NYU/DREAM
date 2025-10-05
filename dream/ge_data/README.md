<div align="center">
  <img src="figs/logo.png" alt="DREAM Logo" width="80" />
  <h1 align="center">DREAM</h1>
  <p align="center">
    <strong>D</strong>rafting with <strong>R</strong>efined Target Features and <strong>E</strong>ntropy-<strong>A</strong>daptive Cross-Attention Fusion for <strong>M</strong>ultimodal Speculative Decoding
  </p>
  <p align="center">
    An open-source framework to accelerate Vision Language Model (VLM) inference by up to 3x with no quality loss.
  </p>
</div>

<p align="center">
  <a href="https://github.com/SafeAILab/EAGLE/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache_2.0-blue.svg"></a>
  <a href="https://pypi.org/project/dream-llm/"><img alt="Version" src="https://img.shields.io/badge/version-1.2.1-brightgreen.svg"></a>
  <a href="https://arxiv.org/abs/2505.19201"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2505.19201-b31b1b.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue.svg">
</p>

<div align="center" style="display: flex; gap: 5px; justify-content: center;">
  <p>
  <b>🔥  Our work is accepted by NeurIPS 2025. Welcome to star and cite our work! ✨</b> 
  </p>
</div>

---

## 🚀 Overview

DREAM is a cutting-edge framework designed to significantly accelerate the inference speed of Vision Language Models (VLMs), such as LLaVA. By employing a novel speculative decoding mechanism, DREAM achieves up to a 3x speedup over traditional autoregressive methods without compromising the quality of the output.

The core of DREAM is its innovative approach: **D**rafting with **R**efined Target Features and **E**ntropy-**A**daptive Cross-Attention Fusion for **M**ultimodal Speculative Decoding. This allows the model to generate multiple tokens in parallel and validate them efficiently, leading to substantial gains in performance.

## ✨ Key Features

- **High-Performance Inference:** Up to 3x faster inference for Vision Language Models (VLMs) compared to standard methods.
- **Zero Quality Loss:** Maintains the same output distribution as the original model.
- **Multimodal Support:** Fully compatible with multimodal models like LLaVA.
- **Efficient Training:** Includes scripts for training the auto-regression head using DeepSpeed.
- **Interactive Web UI:** Comes with a Gradio-based web interface for easy testing and demonstration.
- **Comprehensive Tooling:** Provides scripts for training data generation and performance evaluation.

## 🎥 Demo

<table align="center">
  <tr>
    <td align="center"><b>Vanilla</b></td>
    <td align="center"><b>DREAM</b></td>
  </tr>
  <tr>
    <td><img src="figs/vanila_demp.gif" alt="Vanilla Demo"></td>
    <td><img src="figs/dream_demo.gif" alt="DREAM Demo"></td>
  </tr>
</table>

## 🛠️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SAI-Lab-NYU/DREAM.git
    cd DREAM
    ```

2.  **Install dependencies:**
    We recommend creating a virtual environment first.
    ```bash
    pip install -e .
    ```
    *Note: `-e` installs the project in editable mode.*

3.  **Download Model Weights:**
    See the [Model Weights](#-model-weights) section below for links to the available models.

## ⚡ Quick Start

### 1. Inference with Web UI

Run our Gradio-based web interface for an interactive experience. The command automatically handles model allocation across multiple GPUs.

```bash
python -m dream.application.webui \
    --ea-model-path [PATH_TO_DREAM_WEIGHTS] \
    --base-model-path [PATH_TO_BASE_MODEL]
```

-   `[PATH_TO_DREAM_WEIGHTS]`: Path to the downloaded DREAM weights (e.g., `./DREAM-llava-v1.6-vicuna-7b`).
-   `[PATH_TO_BASE_MODEL]`: Path to the original base model weights (e.g., the original `vicuna-7b-v1.3`).
-   `total-token`: Number of draft tokens. Adjust this based on your hardware for optimal performance. Set to `-1` for auto-configuration.

Once the model is loaded, a URL will be displayed in the terminal.

### 2. Preparing Training Data

First, download all required dataset for [llava_v1_5_mix665k](https://github.com/haotian-liu/LLaVA) and generate training data:
```bash
python -m dream.ge_data.allocation_mix665 # replace running program in line 50 with python -m dream.ge_data.ge_data_all_llava_vicuna_llava_mix665k.py 
```

Second, split train (1000) / test (3000) data for each test dataset, for example:
```bash
python -m dream.ge_data.train_test_split.mmt_ge # train.jsonl and test.jsonl will be placed in processed_data/ folder
python -m dream.ge_data.train_test_split.seed_ge # train.jsonl and test.jsonl will be placed in processed_seed_data/ folder
```

Then, generate training data for each test dataset:
```bash
python -m dream.ge_data.llava_allocation_suppliments.py # need to modify line 301 in ge_data_suppliments.py for different dataset
```

### 3. Training the Auto-regression Head

Use the following DeepSpeed command to start training:
```bash
cd dream/train
deepspeed main_deepspeed.py \
    --deepspeed_config ./ds_config.json \
    --tmpdir [PATH_TO_TRAINING_DATA] \
    --cpdir [PATH_TO_SAVE_CHECKPOINTS] \
    --configpath ./vicuna_7B_config.json
```

### 4. Evaluation

Test the inference speed of DREAM on benchmarks like MT-Bench.
```bash
python -m dream.evaluation.eval_llava \
    --ea-model-path [PATH_TO_DREAM_WEIGHTS] \
    --base-model-path [PATH_TO_BASE_MODEL]
```
This will generate a `.jsonl` file containing the generation results and wall time.

## 📦 Model Weights

| Model                                     | Base Model        | Download                                                                          |
| ----------------------------------------- | ----------------- | --------------------------------------------------------------------------------- |
| `DREAM-llava-v1.6-vicuna-7b`              | `vicuna-7b-v1.6`  | [🤗 HideonBed12138/DREAM-llava-v1.6-vicuna-7b](https://huggingface.co/HideonBed12138/DREAM-llava-v1.6-vicuna-7b) |

## 📄 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@misc{hu2025dreamdraftingrefinedtarget,
  title={DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding},
  author={Yunhai Hu and Tianhua Xia and Zining Liu and Rahul Raman and Xingyu Liu and Bo Bao and Eric Sather and Vithursan Thangarasa and Sai Qian Zhang},
  year={2025},
  eprint={2505.19201},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.19201},
}
```

## 🙏 Acknowledgements

This project is built upon the incredible work of the open-source community. We are especially grateful to the developers of [Medusa](https://github.com/FasterDecoding/Medusa), [EAGLE](https://github.com/SafeAILab/EAGLE), and [FastChat](https://github.com/lm-sys/FastChat).

## 📜 License

DREAM is licensed under the [Apache 2.0 License](LICENSE)
