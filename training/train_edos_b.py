import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, Sampler
from sklearn.metrics import f1_score, accuracy_score

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


TASK_B_LABELS = [
    "threats, plans to harm and incitement",
    "derogation",
    "animosity",
    "prejudiced discussions",
]

_NUM_PREFIX = re.compile(r"^\s*\d+(?:\.\d+)*\s*[\.\-:]*\s*")

CATEGORY_ALIASES = {
    "animosity": "animosity",
    "hostility": "animosity",
    "derogation": "derogation",
    "disparagement": "derogation",
    "insult": "derogation",
    "insults": "derogation",
    "prejudiced discussions": "prejudiced discussions",
    "prejudiced discussion": "prejudiced discussions",
    "prejudice discussions": "prejudiced discussions",
    "threats, plans to harm and incitement": "threats, plans to harm and incitement",
    "threats": "threats, plans to harm and incitement",
    "threat": "threats, plans to harm and incitement",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="EDOS Task B: 4-way sexism categorization with QLoRA + CB-Focal + CAB"
    )
    p.add_argument("--data_path", type=str, required=True,
                    help="Path to CSV/TSV with columns: text, label_sexist, label_category, split")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--output_dir", type=str, default="runs/task_b")
    p.add_argument("--predictions_path", type=str, default="predictions/task_b_test.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size_train", type=int, default=16)
    p.add_argument("--batch_size_eval", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=6e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=96)
    p.add_argument("--lora_dropout", type=float, default=0.2)
    p.add_argument("--cb_beta", type=float, default=0.999)
    p.add_argument("--use_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="use_4bit", action="store_false")
    p.add_argument("--tau_grid", type=float, nargs="+",
                    default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    p.add_argument("--cab_with_replacement", action="store_true", default=False)
    p.add_argument("--cab_k_per_class", type=int, default=None)
    return p.parse_args()


def strip_num_prefix(s: str) -> str:
    return _NUM_PREFIX.sub("", str(s)).strip().lower()


def canonicalize_category(s: str) -> str:
    t = strip_num_prefix(s)
    return CATEGORY_ALIASES.get(t, t)


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


def load_and_split(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    text_col = detect_text_column(df)
    normalize = lambda x: str(x).strip().lower()

    sex_mask = df["label_sexist"].astype(str).str.lower().eq("sexist")

    splits = {}
    for split_name in ["train", "dev", "test"]:
        mask = df["split"].map(normalize).eq(split_name) & sex_mask
        part = df[mask].copy()
        part = part.rename(columns={text_col: "text"})
        part["text"] = part["text"].apply(clean_text)
        part["label_name"] = part["label_category"].apply(canonicalize_category)
        splits[split_name] = part

    return splits["train"], splits["dev"], splits["test"]


class ClassAwareBatchSampler(Sampler):
    """
    Class-Aware Batch Sampling (CAB) (Henning et al., 2023).
    Ensures uniform exposure to all classes by sampling k = floor(B / C)
    instances per class per batch.
    """

    def __init__(self, labels, batch_size, num_classes,
                 k_per_class=None, with_replacement=False):
        self.labels = np.asarray(labels)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.k = k_per_class if k_per_class else max(1, batch_size // num_classes)
        self.with_replacement = with_replacement
        self.idx_by_cls = [np.where(self.labels == c)[0] for c in range(num_classes)]
        self.pos_by_cls = np.zeros(num_classes, dtype=int)
        if not with_replacement:
            for c in range(num_classes):
                np.random.shuffle(self.idx_by_cls[c])

    def __iter__(self):
        while True:
            batch = []
            for c in range(self.num_classes):
                pool = self.idx_by_cls[c]
                if len(pool) == 0:
                    continue
                if self.with_replacement:
                    pick = np.random.choice(
                        pool, size=min(self.k, len(pool)),
                        replace=len(pool) < self.k,
                    )
                else:
                    pos = self.pos_by_cls[c]
                    if pos + self.k <= len(pool):
                        pick = pool[pos : pos + self.k]
                        self.pos_by_cls[c] = pos + self.k
                    else:
                        left = len(pool) - pos
                        pick = pool[pos:]
                        self.pos_by_cls[c] = 0
                        extra_needed = self.k - left
                        if extra_needed > 0 and len(pool) > 0:
                            pick = np.concatenate([pick, pool[: min(extra_needed, len(pool))]])
                batch.extend(pick.tolist())
            if not batch:
                break
            yield batch[: self.batch_size]
            if not self.with_replacement and all(
                self.pos_by_cls[c] == 0 for c in range(self.num_classes)
            ):
                break

    def __len__(self):
        per_batch = max(1, self.k * self.num_classes)
        return max(1, len(self.labels) // per_batch)


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


def build_model(base_model_name, num_labels, id2label, label2id,
                use_4bit=True, lora_r=96, lora_dropout=0.2):
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


class CBFocalTrainer(HFTrainer):
    """
    Trainer with Class-Balanced Focal Loss (Cui et al., 2019) and
    Class-Aware Batch Sampling (Henning et al., 2023).

    CB-Focal: L(x,y) = -w_y * (1 - p_y)^gamma * log(p_y)
    Combined with optional label smoothing (epsilon).
    """

    def __init__(self, *args, class_weights, focal_gamma=2.0, smooth_eps=0.05,
                 k_per_class=None, cab_with_replacement=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = float(focal_gamma)
        self.smooth_eps = float(smooth_eps)
        self.k_per_class = k_per_class
        self.cab_with_replacement = cab_with_replacement

    def get_train_dataloader(self):
        labels = np.array(self.train_dataset["labels"])
        num_classes = int(labels.max()) + 1
        kpc = self.k_per_class or max(1, self.args.per_device_train_batch_size // num_classes)
        sampler = ClassAwareBatchSampler(
            labels,
            self.args.per_device_train_batch_size,
            num_classes,
            k_per_class=kpc,
            with_replacement=self.cab_with_replacement,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_classes = logits.size(-1)

        probs = F.softmax(logits, dim=-1)
        true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - true_class_probs) ** self.focal_gamma

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
        loss = (focal_weight * per_example * cw[labels]).mean()
        return (loss, outputs) if return_outputs else loss


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
            "f1_micro": f1_metric.compute(predictions=preds, references=labels, average="micro")["f1"],
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

    trainer = CBFocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience, 0.0)],
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        smooth_eps=args.label_smoothing,
        k_per_class=args.cab_k_per_class,
        cab_with_replacement=args.cab_with_replacement,
    )
    trainer.train()
    return trainer


def predict_logits(trainer, df, tokenizer, max_length):
    ds = tokenize_split(df, tokenizer, max_length)
    with torch.inference_mode():
        out = trainer.predict(ds)
    logits = torch.tensor(out.predictions)
    labels = np.asarray(out.label_ids)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return logits, probs, labels


def find_best_tau(logits, y_true, grid):
    num_classes = logits.shape[1]
    prior = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    prior /= max(1, prior.sum())
    log_prior = torch.tensor(
        np.log(np.clip(prior, 1e-12, 1.0)), dtype=logits.dtype
    )
    best_tau, best_f1 = 0.0, -1.0
    for tau in grid:
        preds = (logits - tau * log_prior.to(logits.device)).argmax(-1).cpu().numpy()
        score = f1_score(y_true, preds, average="macro")
        if score > best_f1:
            best_tau, best_f1 = tau, score
    return best_tau


def apply_tau(logits, y_reference, tau):
    num_classes = logits.shape[1]
    prior = np.bincount(y_reference, minlength=num_classes).astype(np.float64)
    prior /= max(1, prior.sum())
    log_prior = torch.tensor(
        np.log(np.clip(prior, 1e-12, 1.0)),
        dtype=logits.dtype,
        device=logits.device,
    )
    return logits - tau * log_prior


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(args.seed)

    label2id = {c: i for i, c in enumerate(TASK_B_LABELS)}
    id2label = {i: c for c, i in label2id.items()}
    num_labels = len(TASK_B_LABELS)

    train_df, dev_df, test_df = load_and_split(args.data_path)

    for df in (train_df, dev_df, test_df):
        df["label"] = df["label_name"].map(label2id).astype(int)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(
        args.base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
    )

    counts = Counter(train_df["label"].tolist())
    class_weights = compute_cb_weights(
        [counts.get(i, 0) for i in range(num_labels)], beta=args.cb_beta
    )
    print(f"Class counts: {dict(counts)}")
    print(f"CB weights: {class_weights.tolist()}")

    trainer = train(
        train_df[["text", "label"]],
        dev_df[["text", "label"]],
        tokenizer,
        model,
        class_weights,
        args,
    )

    logits_dev, probs_dev, y_dev = predict_logits(
        trainer, dev_df[["text", "label"]], tokenizer, args.max_length
    )
    logits_test, probs_test, y_test = predict_logits(
        trainer, test_df[["text", "label"]], tokenizer, args.max_length
    )

    tau = find_best_tau(logits_dev, y_dev, args.tau_grid)
    print(f"Best tau: {tau:.3f}")

    logits_dev_adj = apply_tau(logits_dev, y_dev, tau)
    logits_test_adj = apply_tau(logits_test, y_dev, tau)
    probs_dev_adj = torch.softmax(logits_dev_adj, -1).cpu().numpy()
    probs_test_adj = torch.softmax(logits_test_adj, -1).cpu().numpy()

    for split_name, probs_raw, probs_adj, y_true in [
        ("DEV", probs_dev, probs_dev_adj, y_dev),
        ("TEST", probs_test, probs_test_adj, y_test),
    ]:
        pred_raw = probs_raw.argmax(axis=1)
        pred_adj = probs_adj.argmax(axis=1)
        print(f"\n=== {split_name} ===")
        print(
            f"  Unadjusted: acc={accuracy_score(y_true, pred_raw):.4f}  "
            f"macro-F1={f1_score(y_true, pred_raw, average='macro'):.4f}"
        )
        print(
            f"  tau-adjusted: acc={accuracy_score(y_true, pred_adj):.4f}  "
            f"macro-F1={f1_score(y_true, pred_adj, average='macro'):.4f}"
        )

    os.makedirs(os.path.dirname(args.predictions_path) or ".", exist_ok=True)

    pred_test_raw = probs_test.argmax(axis=1)
    pred_test_adj = probs_test_adj.argmax(axis=1)

    out_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred_unadjusted": pred_test_raw,
        "y_pred_tau": pred_test_adj,
    })
    for i, name in enumerate(TASK_B_LABELS):
        out_df[f"prob_{name}"] = probs_test[:, i]
    for i, name in enumerate(TASK_B_LABELS):
        out_df[f"prob_adj_{name}"] = probs_test_adj[:, i]

    out_df.to_csv(args.predictions_path, index=False)
    print(f"\nSaved predictions to {args.predictions_path}")

    meta = {
        "labels": TASK_B_LABELS,
        "label2id": label2id,
        "tau": float(tau),
        "base_model": args.base_model,
        "focal_gamma": args.focal_gamma,
        "cb_beta": args.cb_beta,
    }
    meta_path = os.path.join(args.output_dir, "task_b_meta.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
