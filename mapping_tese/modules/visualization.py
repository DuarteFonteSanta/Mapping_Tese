"""Confusion-matrix and group-level balanced-accuracy plotting (generalizes scripts/open_cm.py)."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
from pathlib import Path


def plot_confusion_matrix(cm, class_labels, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_labels, yticklabels=class_labels,
        cbar_kws={"label": "Count"}, linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_group_balanced_accuracy(results_df, title, save_path, chance_level):
    fig, ax = plt.subplots(figsize=(max(7, len(results_df) * 1.0), 4))
    x = np.arange(len(results_df))
    ax.bar(
        x, results_df["mean_balanced_accuracy"] * 100,
        yerr=results_df["std_balanced_accuracy"] * 100,
        capsize=5, color="#3b82f6", ecolor="black",
    )
    ax.axhline(chance_level, color="red", linestyle="--", label=f"Chance ({chance_level:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["subject"], rotation=45, ha="right")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()



def plot_fold_performance(fold_acc, fold_bacc, chance_level, mean_bacc, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(fold_acc) + 1)
    width = 0.38
    ax.bar(x - width / 2, fold_acc, width=width, color="#1f77b4", alpha=0.85, edgecolor="black", label="Accuracy")
    ax.bar(x + width / 2, fold_bacc, width=width, color="#ff7f0e", alpha=0.85, edgecolor="black", label="Balanced accuracy")
    ax.axhline(chance_level, linestyle=":", linewidth=2, color="gray", label=f"Chance = {chance_level:.3f}")
    ax.axhline(mean_bacc, linestyle="--", linewidth=2, color="red", alpha=0.5, label=f"Mean bacc = {mean_bacc:.3f}")
    ax.set_xlabel("Fold (left-out run)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_ylim(0, max(0.40, float(np.max(fold_bacc)) * 1.15))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dual_confusion_matrix(cm, cm_norm, class_labels, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(cm, cmap="Blues", aspect="auto")
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    count_threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            axes[0].text(j, i, f"{val}", ha="center", va="center", fontsize=8,
                         color="white" if val > count_threshold else "black")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = float(cm_norm[i, j])
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if val > 0.5 else "black")

    for ax in axes:
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=90)
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("Predicted Region")
        ax.set_ylabel("True Region")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stat_maps_per_class(values_per_class, class_labels, masker, ref_img, title_prefix, save_dir, percentile):
    save_dir = Path(save_dir)
    for class_index, class_name in enumerate(class_labels):
        threshold_value = np.percentile(values_per_class[class_index], percentile)
        stat_img = masker.inverse_transform(values_per_class[class_index].reshape(1, -1))
        stat_display = plotting.plot_stat_map(
            stat_img, bg_img=ref_img, title=f"{title_prefix}: {class_name}",
            threshold=threshold_value, colorbar=True, display_mode="ortho", cut_coords=(0, -20, 60),
        )
        stat_display.savefig(save_dir / f"{title_prefix.lower().replace(' ', '_')}_{class_name}.png", dpi=300)
        stat_display.close()


def plot_somatotopic_map(values_per_class, class_labels, masker, ref_img, title, png_path, html_path, top_percentile):
    abs_values = np.abs(values_per_class)
    winner_map = np.argmax(abs_values, axis=0)
    max_value = np.amax(abs_values, axis=0)
    threshold_value = np.percentile(max_value, top_percentile)
    winner_values = np.where(max_value >= threshold_value, winner_map + 1, 0).astype(np.float64)
    winner_img = masker.inverse_transform(winner_values.reshape(1, -1))

    winner_display = plotting.plot_roi(
        winner_img, bg_img=ref_img, title=title,
        display_mode="ortho", cut_coords=(0, -20, 60), cmap="tab10",
    )
    winner_display.savefig(png_path, dpi=300)
    winner_display.close()

    surf_view = plotting.view_img_on_surf(
        winner_img, surf_mesh="fsaverage", title=title,
        symmetric_cmap=False, vmin=1, vmax=len(class_labels), threshold=1,
        vol_to_surf_kwargs={"interpolation": "nearest_most_frequent"},
    )
    surf_view.save_as_html(str(html_path))