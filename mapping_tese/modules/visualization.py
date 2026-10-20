"""Confusion-matrix and group-level balanced-accuracy plotting (generalizes scripts/open_cm.py)."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_group_balanced_accuracy(results_df, title, save_path, chance_level=25.0):
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