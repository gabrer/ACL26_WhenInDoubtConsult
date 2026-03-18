import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer as HFTrainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate


LABEL_MAP = {
    "sexist": 1, "sexism": 1, "1": 1, "true": 1, "yes": 1,
    "not sexist": 0, "non-sexist": 0, "non sexist": 0,
    "0": 0, "false": 0, "no": 0, "none": 0, "neutral": 0,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="EDOS Task A: binary sexism detection with QLoRA + CB-CE + temperature scaling"
    )
    p.add_argument("--train_path", type=str, required=True,
                    help="Path to training CSV/TSV (columns: text, label OR label_sexist; optionally split)")
    p.add_argument("--eval_path", type=str, default=None,
                    help="Path to evaluation CSV/TSV. If omitted, dev/test splits are read from train_path")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--output_dir", type=str, default="runs/task_a")
    p.add_argument("--predictions_path", type=str, default="predictions/task_a_test.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size_train", type=int, default=16)
    p.add_argument("--batch_size_eval", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--early_stopping_patience", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--cb_beta", type=float, default=0.999)
    p.add_argument("--use_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="use_4bit", action="store_false")
    return p.parse_args()


def map_label(v):
    s = str(v).strip().lower()
    if s in LABEL_MAP:
        return LABEL_MAP[s]
    return np.nan


def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"http\S+|www\S+|https\S+", " <URL> ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def detect_text_column(df: pd.DataFrame) -> str:
    for c in ["text", "comment_text", "content", "raw_text"]:
        if c in df.columns:
            return c
    raise ValueError("No recognized text column found in the dataset.")


def detect_label_column(df: pd.DataFrame) -> str:
    for c in ["label", "label_sexist"]:
        if c in df.columns:
            return c
    raise ValueError("No recognized label column found in the dataset.")


def read_csv(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    return pd.read_csv(path, sep="\t") if ext == ".tsv" else pd.read_csv(path)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    text_col = detect_text_column(df)
    label_col = detect_label_column(df)
    out = df[[text_col, label_col]].copy()
    out = out.rename(columns={text_col: "text", label_col: "label"})
    out["text"] = out["text"].apply(clean_text)
    out["label"] = out["label"].map(map_label)
    out = out[out["label"].notna()].copy()
    out["label"] = out["label"].astype(int)
    return out


def load_splits(train_path, eval_path=None):
    train_raw = read_csv(train_path)

    if eval_path is not None:
        eval_raw = read_csv(eval_path)
        if "split" in eval_raw.columns:
            normalize = lambda x: str(x).strip().lower()
            dev_df = prepare_df(eval_raw[eval_raw["split"].map(normalize).eq("dev")])
            test_df = prepare_df(eval_raw[eval_raw["split"].map(normalize).eq("test")])
        else:
            dev_df = prepare_df(eval_raw)
            test_df = dev_df
        train_df = prepare_df(train_raw)
    elif "split" in train_raw.columns:
        normalize = lambda x: str(x).strip().lower()
        train_df = prepare_df(train_raw[train_raw["split"].map(normalize).eq("train")])
        dev_df = prepare_df(train_raw[train_raw["split"].map(normalize).eq("dev")])
        test_df = prepare_df(train_raw[train_raw["split"].map(normalize).eq("test")])
    else:
        raise ValueError("No split column found and no separate eval_path provided.")

    return train_df, dev_df, test_df


def compute_cb_weights(counts, beta=0.999, w_min=0.25, w_max=4.0):
    """
    Class-balanced weights via effective number of samples (Cui et al., 2019).
    w_y = (1 - beta) / (1 - beta^{n_y}), normalized to unit mean and clamped.
    """
    effective_num = [
        (1.0 - beta) / (1.0 - beta ** max(1, int(n))) for n in counts
    ]
    w = torch.tensor(effective_num, dtype=torch.float)
    w = w / w.mean()
    w = torch.clamp(w, w_min, w_max)
    w = w / w.mean()
    return w


def tokenize_split(df, tokenizer, max_length):
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )
    ds = ds.rename_column("label", "labels")
    ds.set_format(
        type="torch",
        columns=[c for c in ["input_ids", "attention_mask", "labels"] if c in ds.column_names],
    )
    return ds


def build_model(base_model_name, num_labels, use_4bit=True, lora_r=32, lora_dropout=0.1):
    lora_alpha = 2 * lora_r
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
        )

    id2label = {0: "not sexist", 1: "sexist"}
    label2id = {"not sexist": 0, "sexist": 1}

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=quant_config,
        device_map="auto" if use_4bit else None,
    )
    base.config.use_cache = False
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    return get_peft_model(base, lora_config)


class CBCETrainer(HFTrainer):
    """
    Trainer with Class-Balanced Cross-Entropy (Cui et al., 2019) and label smoothing.
    Used for binary classification (Task A / EXIST Task 1.1) where focal loss
    is not applied (paper Section 3.1, Appendix A.2.2).
    """

    def __init__(self, *args, class_weights, smooth_eps=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.smooth_eps = float(smooth_eps)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_classes = logits.size(-1)

        log_probs = F.log_softmax(logits, dim=-1)

        if self.smooth_eps > 0:
            eps = self.smooth_eps
            with torch.no_grad():
                y = F.one_hot(labels, num_classes=num_classes).to(logits.dtype)
                y = y * (1 - eps) + eps / num_classes
            per_example = -(y * log_probs).sum(dim=-1)
        else:
            per_example = F.nll_loss(log_probs, labels, reduction="none")

        cw = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = (per_example * cw[labels]).mean()
        return (loss, outputs) if return_outputs else loss


def fit_temperature(logits, labels, lr=0.01, max_iter=200):
    """
    Learn a single temperature T that minimizes NLL on the development set (Eq. 11).
    Calibrated probabilities: p_T(y|x) = softmax(z(x) / T*) (Eq. 12).
    """
    temperature = nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    labels_t = torch.tensor(labels, dtype=torch.long, device=logits.device)

    def closure():
        optimizer.zero_grad()
        scaled = logits / temperature
        loss = F.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def tune_threshold(probs_positive, y_true, n_steps=181):
    """
    Tune decision threshold t on calibrated probabilities to maximize
    development set macro-F1 (Eq. 13).
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, n_steps):
        preds = (probs_positive >= t).astype(int)
        score = f1_score(y_true, preds, average="macro")
        if score > best_f1:
            best_t, best_f1 = t, score
    return best_t, best_f1


def train(train_df, dev_df, tokenizer, model, class_weights, args):
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.asarray(logits).argmax(axis=1)
        labels = np.asarray(labels)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_eval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        eval_accumulation_steps=8 if args.use_4bit else None,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    train_ds = tokenize_split(train_df, tokenizer, args.max_length)
    dev_ds = tokenize_split(dev_df, tokenizer, args.max_length)

    trainer = CBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience, 0.0)],
        class_weights=class_weights,
        smooth_eps=args.label_smoothing,
    )
    trainer.train()
    return trainer


def predict_logits(trainer, df, tokenizer, max_length):
    ds = tokenize_split(df, tokenizer, max_length)
    with torch.inference_mode():
        out = trainer.predict(ds)
    logits = torch.tensor(out.predictions)
    labels = np.asarray(out.label_ids)
    return logits, labels


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    train_df, dev_df, test_df = load_splits(args.train_path, args.eval_path)
    print(f"Rows: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(
        args.base_model,
        num_labels=2,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
    )

    counts = Counter(train_df["label"].tolist())
    class_weights = compute_cb_weights(
        [max(1, counts.get(i, 0)) for i in range(2)], beta=args.cb_beta
    )
    print(f"Class counts: {dict(counts)}")
    print(f"CB weights: {class_weights.tolist()}")

    trainer = train(train_df, dev_df, tokenizer, model, class_weights, args)

    logits_dev, y_dev = predict_logits(trainer, dev_df, tokenizer, args.max_length)
    logits_test, y_test = predict_logits(trainer, test_df, tokenizer, args.max_length)

    temperature = fit_temperature(logits_dev, y_dev)
    print(f"Fitted temperature T*: {temperature:.4f}")

    probs_dev = torch.softmax(logits_dev / temperature, dim=-1).numpy()
    probs_test = torch.softmax(logits_test / temperature, dim=-1).numpy()

    threshold, dev_f1 = tune_threshold(probs_dev[:, 1], y_dev)
    print(f"Tuned threshold: {threshold:.3f} (dev macro-F1={dev_f1:.4f})")

    for split_name, probs, y_true in [("DEV", probs_dev, y_dev), ("TEST", probs_test, y_test)]:
        pred_default = probs.argmax(axis=1)
        pred_tuned = (probs[:, 1] >= threshold).astype(int)
        print(f"\n=== {split_name} ===")
        print(
            f"  Default (t=0.5): acc={accuracy_score(y_true, pred_default):.4f}  "
            f"macro-F1={f1_score(y_true, pred_default, average='macro'):.4f}"
        )
        print(
            f"  Tuned (t={threshold:.3f}): acc={accuracy_score(y_true, pred_tuned):.4f}  "
            f"macro-F1={f1_score(y_true, pred_tuned, average='macro'):.4f}"
        )

    print(f"\n TEST classification report")
    pred_test_tuned = (probs_test[:, 1] >= threshold).astype(int)
    print(classification_report(y_test, pred_test_tuned, digits=4,
                                target_names=["not sexist", "sexist"]))

    os.makedirs(os.path.dirname(args.predictions_path) or ".", exist_ok=True)

    pred_test_default = probs_test.argmax(axis=1)
    out_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred_default": pred_test_default,
        "y_pred_tuned": pred_test_tuned,
        "prob_not_sexist": probs_test[:, 0],
        "prob_sexist": probs_test[:, 1],
    })
    out_df.to_csv(args.predictions_path, index=False)
    print(f"\nSaved predictions to {args.predictions_path}")

    meta = {
        "labels": ["not sexist", "sexist"],
        "temperature": temperature,
        "threshold": float(threshold),
        "base_model": args.base_model,
        "cb_beta": args.cb_beta,
    }
    meta_path = os.path.join(args.output_dir, "task_a_meta.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
