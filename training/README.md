# Specialist Model Training

This directory contains the training scripts for the domain-tuned specialist models described in the paper. Each script corresponds to a specific task and implements the training procedures from Section 3.1 and Appendix A.2.

## Scripts

| Script | Task | Classes | Loss | Post-hoc Calibration |
|---|---|---|---|---|
| `train_task_a.py` | EDOS Task A (binary) | 2 | CB-CE | Temperature scaling + threshold tuning |
| `train_task_b.py` | EDOS Task B (4-way) | 4 | CB-Focal | Tau logit adjustment |
| `train_task_c.py` | EDOS Task C (11-way) | 11 | CB-Focal | Tau logit adjustment |
| `train_exist_task1_1.py` | EXIST 2025 Task 1.1 (binary) | 2 | CB-CE | Temperature scaling + threshold tuning |

All scripts use QLoRA fine-tuning of LLaMA-3.2-3B with Class-Aware Batch Sampling (CAB) for multi-class tasks. Hyperparameters match Table A1 in the paper.

## Setup

```bash
pip install -r requirements.txt
```

Access to `meta-llama/Llama-3.2-3B` requires a Hugging Face token with the Meta LLaMA license accepted:

```bash
huggingface-cli login
```

## Data

**EDOS** (Tasks A, B, C): Download from the [SemEval-2023 Task 10 repository](https://github.com/rewire-online/edos). The scripts expect a single CSV/TSV file with columns: `text` (or `comment_text`), `label_sexist`, `label_category` (Task B), `label_vector` (Task C), and `split` (`train`/`dev`/`test`).

**EXIST 2025** (Task 1.1): Available through the [EXIST 2025 evaluation campaign](https://nlp.uned.es/exist2025/). The script expects separate JSON files for train, dev, and test splits.

## Usage

### EDOS Task A (Binary)

```bash
python train_task_a.py \
  --train_path data/edos.csv \
  --output_dir runs/task_a \
  --predictions_path predictions/task_a_test.csv
```

Supports a separate merged training file (for multi-dataset training as in C2):

```bash
python train_task_a.py \
  --train_path data/merged_train.csv \
  --eval_path data/edos.csv \
  --output_dir runs/task_a
```

### EDOS Task B (4-way)

```bash
python train_task_b.py \
  --data_path data/edos.csv \
  --output_dir runs/task_b \
  --predictions_path predictions/task_b_test.csv
```

### EDOS Task C (11-way)

```bash
python train_task_c.py \
  --data_path data/edos.csv \
  --output_dir runs/task_c \
  --predictions_path predictions/task_c_test.csv
```

### EXIST 2025 Task 1.1 (Binary)

```bash
python train_exist_task1_1.py \
  --train_path data/exist2025_train.json \
  --dev_path data/exist2025_dev.json \
  --test_path data/exist2025_test.json \
  --output_predictions predictions/exist_task1_1_hard.json
```

The output is an EvALL-format JSON file for submission to the [EvALL evaluation platform](https://evall.uned.es/).

## Hyperparameters

To override any hyperparameter:

```bash
python train_task_b.py \
  --data_path data/edos.csv \
  --learning_rate 3e-5 \
  --epochs 10 \
  --lora_r 128
```

Run `python <script> --help` for the full list of arguments.

## Hardware

Experiments were conducted on NVIDIA A100 GPUs. With 4-bit quantization (enabled by default), a single GPU with 24 GB VRAM is sufficient for all tasks.

To disable quantization:

```bash
python train_task_a.py --no_4bit --data_path data/edos.csv
```
