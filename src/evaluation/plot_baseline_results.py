import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PLOTS_DIR = os.path.join(REPO_ROOT, "results", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------------------------------
# Baseline results from Parinitha's reports
# --------------------------------------------------
df = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "precision": 0.9951,
        "recall": 0.7798,
        "f1_score": 0.8744,
        "fpr": 0.0023,
        "fnr": 0.2202,
        "tn": 2637,
        "fp": 6,
        "fn": 342,
        "tp": 1211,
    },
    {
        "model": "KNN",
        "precision": 0.9548,
        "recall": 0.5441,
        "f1_score": 0.6932,
        "fpr": 0.0151,
        "fnr": 0.4559,
        "tn": 2603,
        "fp": 40,
        "fn": 708,
        "tp": 845,
    },
    {
        "model": "SVM",
        "precision": 0.9893,
        "recall": 0.8339,
        "f1_score": 0.9050,
        "fpr": 0.0053,
        "fnr": 0.1661,
        "tn": 2629,
        "fp": 14,
        "fn": 258,
        "tp": 1295,
    },
])

print(df)

# --------------------------------------------------
# 1) Baseline comparison plot
# --------------------------------------------------
metrics = ["precision", "recall", "f1_score"]
x = np.arange(len(df["model"]))
width = 0.25

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, df[metric], width=width, label=metric)

plt.xticks(x + width, df["model"], rotation=15)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Baseline Models Comparison on SKAB Test Set")
plt.legend()
plt.tight_layout()

comparison_path = os.path.join(PLOTS_DIR, "baseline_model_comparison.png")
plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {comparison_path}")

# --------------------------------------------------
# Helper for confusion matrix plot
# --------------------------------------------------
def save_confusion_matrix_plot(row, filename):
    cm = np.array([
        [row["tn"], row["fp"]],
        [row["fn"], row["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Fault"])
    ax.set_yticklabels(["Actual Normal", "Actual Fault"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f'Confusion Matrix - {row["model"]}')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# --------------------------------------------------
# 2) Per-model confusion matrices
# --------------------------------------------------
for _, row in df.iterrows():
    model_name = row["model"]
    if model_name == "Logistic Regression":
        save_confusion_matrix_plot(row, "lr_confusion_matrix.png")
    elif model_name == "KNN":
        save_confusion_matrix_plot(row, "knn_confusion_matrix.png")
    elif model_name == "SVM":
        save_confusion_matrix_plot(row, "svm_confusion_matrix.png")

print("\nDone. Baseline plots saved in results\\plots")