
import argparse
import json
import os
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Confidence-aware routing: merge specialist + CEJ judge predictions"
    )
    p.add_argument("--specialist_csv", type=str, required=True,
                    help="Path to specialist predictions CSV (with probability columns)")
    p.add_argument("--judge_json", type=str, required=True,
                    help="Path to CEJ judge output JSON (from cej_judge.py)")
    p.add_argument("--output_path", type=str, required=True,
                    help="Path to save routed predictions CSV")
    p.add_argument("--tau_conf", type=float, required=True,
                    help="Confidence threshold: route if max prob < tau_conf")
    p.add_argument("--tau_margin", type=float, required=True,
                    help="Margin threshold: route if (top1 - top2) < tau_margin")
    p.add_argument("--prob_prefix", type=str, default=None,
                    help="Prefix for probability columns (e.g. 'prob_', 'probB_', 'probC_'). "
                         "Auto-detected if omitted.")
    p.add_argument("--pred_col", type=str, default=None,
                    help="Name of the specialist prediction column. Auto-detected if omitted.")
    p.add_argument("--id_col", type=str, default=None,
                    help="Name of the ID column. Auto-detected if omitted.")
    return p.parse_args()


def detect_id_column(df):
    for c in ["id", "rewire_id", "comment_id", "id_EXIST", "tweet_id"]:
        if c in df.columns:
            return c
    raise KeyError(f"No ID column found. Available: {df.columns.tolist()}")


def detect_prob_prefix(df):
    prefixes = ["prob_adj_", "probC_adj_", "probB_adj_", "prob_", "probC_", "probB_"]
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if len(cols) >= 2:
            return prefix
    raise KeyError(
        f"No probability columns found. "
        f"Available columns: {df.columns.tolist()}"
    )


def detect_pred_column(df):
    candidates = [
        "y_pred_tau", "y_pred_tuned",
        "yB_pred_tau", "yC_pred_tau",
        "y_pred_unadjusted", "yB_pred_unadj", "yC_pred_unadj",
        "y_pred_default", "y_pred",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No prediction column found. Available: {df.columns.tolist()}")


def compute_confidence_and_margin(df, prob_prefix):
    prob_cols = sorted([c for c in df.columns if c.startswith(prob_prefix)])
    if len(prob_cols) < 2:
        raise ValueError(
            f"Need at least 2 probability columns for margin computation, "
            f"found {len(prob_cols)} with prefix '{prob_prefix}'"
        )

    probs = df[prob_cols].astype(float).values

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    confidence = sorted_probs[:, 0]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]

    return (
        pd.Series(confidence, index=df.index),
        pd.Series(margin, index=df.index),
    )


def load_judge_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())

    judge_map = {}
    for rec in data:
        rid = str(rec.get("id", ""))
        label = rec.get("label")
        if rid and label is not None:
            judge_map[rid] = label
    return judge_map


def route(specialist_pred, confidence, margin, tau_conf, tau_margin, judge_pred):
    if judge_pred is None:
        return specialist_pred, False
    if confidence < tau_conf and margin < tau_margin:
        return judge_pred, True
    return specialist_pred, False


def main():
    args = parse_args()

    df = pd.read_csv(args.specialist_csv)
    id_col = args.id_col or detect_id_column(df)
    prob_prefix = args.prob_prefix or detect_prob_prefix(df)
    pred_col = args.pred_col or detect_pred_column(df)

    print(f"Loaded {len(df)} specialist predictions")
    print(f"  ID column: {id_col}")
    print(f"  Prob prefix: {prob_prefix}")
    print(f"  Pred column: {pred_col}")

    df["_confidence"], df["_margin"] = compute_confidence_and_margin(df, prob_prefix)

    judge_map = load_judge_predictions(args.judge_json)
    print(f"Loaded {len(judge_map)} judge predictions")

    routed_preds = []
    routed_flags = []

    for _, row in df.iterrows():
        rid = str(row[id_col])
        specialist_pred = row[pred_col]
        conf = row["_confidence"]
        marg = row["_margin"]
        judge_pred = judge_map.get(rid)

        pred, was_routed = route(
            specialist_pred, conf, marg,
            args.tau_conf, args.tau_margin, judge_pred,
        )
        routed_preds.append(pred)
        routed_flags.append(int(was_routed))

    df["routed_pred"] = routed_preds
    df["routed"] = routed_flags

    n_routed = sum(routed_flags)
    print(f"Routed {n_routed}/{len(df)} predictions "
          f"({n_routed/len(df)*100:.1f}%)")

    out_df = pd.DataFrame({
        id_col: df[id_col],
        "specialist_pred": df[pred_col],
        "routed_pred": df["routed_pred"],
        "confidence": df["_confidence"],
        "margin": df["_margin"],
        "routed": df["routed"],
    })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
