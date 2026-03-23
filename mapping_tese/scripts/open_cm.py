from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "notebooks" / "results" / "4_Classes_ANN_DeepRF"

MODEL_NAME = "model_h128-32_wd0.5_dr0.6_pca50"

CLASS_LABELS = ['Middle_Finger', 'Hand', 'Forearm', 'Arm']

cm_file = RESULTS_DIR / f"{MODEL_NAME}_confusion_matrix.npy"
cm = np.load(cm_file)

print(f"Confusion Matrix for {MODEL_NAME}:")
print(cm)
print(f"\nTotal samples: {cm.sum()}")
print(f"Accuracy: {np.trace(cm) / cm.sum() * 100:.2f}%")

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
            cbar_kws={'label': 'Count'}, linewidths=0.5)

plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix: {MODEL_NAME}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()