# Confidence-Aware Routing

This directory implements the confidence-aware routing mechanism. The routing module acts as a gatekeeper between the specialist classifier and the CEJ reasoning module, forwarding only low-confidence predictions for deliberation.

## Scripts

| Script | Tasks | Routing Condition | 
|---|---|---|
| `route_predictions_binary.py` | Task A, EXIST Task 1.1 | `c_s(x) < tau_conf` |
| `route_predictions.py` | Task B, Task C | `c_s(x) < tau_conf AND m(x) < tau_margin` | 

For binary classification, the margin between two classes is fully determined by the confidence score, so only the confidence threshold is needed. For multi-class settings, the margin condition prevents routing instances where the model is confident in its top prediction even if the absolute probability is moderate.

## Usage

### Binary tasks (Task A / EXIST Task 1.1)

```bash
python route_predictions_binary.py \
  --specialist_csv predictions/task_a_test.csv \
  --judge_json outputs/judgments.json \
  --tau_conf 0.6 \
  --output_path predictions/task_a_routed.csv
```

### Multi-class tasks (Task B / Task C)

```bash
python route_predictions.py \
  --specialist_csv predictions/task_b_test.csv \
  --judge_json outputs/judgments.json \
  --tau_conf <threshold> \
  --tau_margin <threshold> \
  --output_path predictions/task_b_routed.csv
```

## Inputs

**Specialist CSV**: Output of any training script (`train_task_a.py`, `train_task_b.py`, etc.). Must contain an ID column, a prediction column, and probability columns. All are auto-detected.

**Judge JSON**: Output of `cej_judge.py` (stage 4). Each record must have `id` and `label` fields.

## Output

CSV with one row per sample:

| Column | Description |
|---|---|
| `id` | Sample identifier |
| `specialist_pred` | Original specialist prediction |
| `routed_pred` | Final prediction (specialist or judge) |
| `confidence` | Max probability from specialist  |
| `margin` | Top-1 minus top-2 probability (multi-class only) |
| `routed` | 1 if prediction was replaced by judge, 0 otherwise |

## Auto-detection

Both scripts auto-detect columns from the specialist CSV:

- **ID column**: looks for `id`, `rewire_id`, `comment_id`, `id_EXIST`, `tweet_id`
- **Probability prefix**: looks for `prob_adj_`, `probC_adj_`, `probB_adj_`, `prob_`, `probC_`, `probB_`
- **Prediction column**: looks for `y_pred_tau`, `y_pred_tuned`, `yB_pred_tau`, `yC_pred_tau`, etc.

Override any of these with `--id_col`, `--prob_prefix`, or `--pred_col`.
