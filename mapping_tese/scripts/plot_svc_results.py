import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from nilearn.image import load_img, index_img

from mapping_tese.modules import constants as C
from mapping_tese.modules.paths import resolve_bold_path
from mapping_tese.modules.visualization import (
    plot_dual_confusion_matrix,
    plot_fold_performance,
    plot_group_balanced_accuracy,
    plot_somatotopic_map,
    plot_stat_maps_per_class,
)


def plot_subject_results(subject, results_dir, scheme="s1_hrf6_win1"):
    results_dir = Path(results_dir)
    subject_results_dir = results_dir / subject
    subject_figures_dir = subject_results_dir / "figures"
    subject_models_dir = subject_results_dir / "models"
    subject_figures_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = C.PROJECT_ROOT / "data" / "extracted" / scheme
    masker = joblib.load(cache_dir / f"{subject}_masker.joblib")
    run1_path = resolve_bold_path(subject, run=1, space_name=C.SPACE)
    ref_img = index_img(load_img(str(run1_path)), 0)

    fold_df = pd.read_csv(subject_results_dir / "fold_balanced_accuracy.csv")
    cm = np.load(subject_results_dir / "confusion_matrix.npy")
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.where(row_sum == 0, 1, row_sum))

    full_pipe = joblib.load(subject_models_dir / "full_pipe.joblib")
    scaler = full_pipe.named_steps["scale"]
    svm = full_pipe.named_steps["clf"]
    class_labels = svm.classes_
    weight_voxel = svm.coef_ / scaler.scale_[np.newaxis, :]

    chance_level = 1.0 / len(class_labels)
    mean_bacc = float(fold_df["balanced_accuracy"].mean())

    plot_fold_performance(
        fold_df["accuracy"].to_numpy(), fold_df["balanced_accuracy"].to_numpy(),
        chance_level, mean_bacc, f"4-Way Region Classification ({subject})",
        subject_figures_dir / "4way_performance.png",
    )
    plot_dual_confusion_matrix(cm, cm_norm, class_labels, subject_figures_dir / "4way_confusion_matrix.png")
    plot_stat_maps_per_class(np.abs(weight_voxel), class_labels, masker, ref_img, "SVM Weights", subject_figures_dir)
    plot_somatotopic_map(
        weight_voxel, class_labels, masker, ref_img,
        "Somatotopic Map (" + ", ".join(f"{i+1}={class_labels[i]}" for i in range(len(class_labels))) + ")",
        subject_figures_dir / "somatotopic_map.png", subject_figures_dir / "somatotopic_map_surface.html",
    )

    shap_abs_path = subject_results_dir / "shap_abs_svc.npy"
    if shap_abs_path.exists():
        shap_abs = np.load(shap_abs_path)
        plot_stat_maps_per_class(shap_abs, class_labels, masker, ref_img, "SHAP Importance", subject_figures_dir)
        plot_somatotopic_map(
            shap_abs, class_labels, masker, ref_img,
            "SHAP Somatotopic Map (" + ", ".join(f"{i+1}={class_labels[i]}" for i in range(len(class_labels))) + ")",
            subject_figures_dir / "somatotopic_map_shap.png", subject_figures_dir / "somatotopic_map_shap_surface.html",
        )


def plot_group_summary(results_dir):
    results_dir = Path(results_dir)
    summary_csv = results_dir / "svc_multi_subject_summary.csv"
    if not summary_csv.exists():
        print(f"No summary CSV found at {summary_csv}")
        return
    summary_df = pd.read_csv(summary_csv)
    chance_level = float(summary_df["chance_level"].iloc[0]) * 100 if not summary_df.empty else 25.0
    plot_group_balanced_accuracy(
        summary_df, "Per-Subject SVC Performance (LORO)",
        results_dir / "multi_subject_balanced_accuracy.png", chance_level=chance_level,
    )


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "notebooks" / "SVC" / "results"
    plot_subject_results(sys.argv[1] if len(sys.argv) > 1 else "sub-p0001", results_dir)
    plot_group_summary(results_dir)