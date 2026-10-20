"""Per-subject result aggregation and group-level CSV export."""
import numpy as np
import pandas as pd


def aggregate_subject_results(subject, fold_df, cm_total, n_samples, n_features, tr_subject):
    mean_acc = float(fold_df["accuracy"].mean())
    std_acc = float(fold_df["accuracy"].std(ddof=0))
    mean_bal = float(fold_df["balanced_accuracy"].mean())
    std_bal = float(fold_df["balanced_accuracy"].std(ddof=0))
    per_class_recall = np.diag(cm_total) / np.maximum(cm_total.sum(axis=1), 1)

    return {
        "subject": subject,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_balanced_accuracy": mean_bal,
        "std_balanced_accuracy": std_bal,
        "macro_recall": float(np.mean(per_class_recall)),
        "per_class_recall": str(per_class_recall.tolist()),
        "n_samples": int(n_samples),
        "n_voxels": int(n_features),
        "tr_subject": float(tr_subject),
    }


def save_group_results(all_subject_rows, results_csv_path):
    if not all_subject_rows:
        raise RuntimeError("No subject produced valid results.")
    results_df = pd.DataFrame(all_subject_rows).sort_values("subject").reset_index(drop=True)
    results_df.to_csv(results_csv_path, index=False)
    return results_df