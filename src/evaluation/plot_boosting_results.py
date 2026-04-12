import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

CSV_PATH = os.path.join(REPO_ROOT, "results", "boosting_models_test_summary.csv")
PLOTS_DIR = os.path.join(REPO_ROOT, "results", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Summary CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("Loaded summary CSV:")
print(df)

# --------------------------------------------------
# 1) Bar chart for key metrics
# --------------------------------------------------
metrics = ["recall", "f1_score", "pr_auc"]
x = np.arange(len(df["model"]))
width = 0.25

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, df[metric], width=width, label=metric)

plt.xticks(x + width, df["model"], rotation=15)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Boosting Models Comparison on SKAB Test Set")
plt.legend()
plt.tight_layout()

bar_chart_path = os.path.join(PLOTS_DIR, "boosting_model_comparison.png")
plt.savefig(bar_chart_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {bar_chart_path}")

# --------------------------------------------------
# Helper to plot confusion matrix from summary counts
# --------------------------------------------------
def save_confusion_matrix_plot(row, output_name):
    cm = np.array([
        [row["tn"], row["fp"]],
        [row["fn"], row["tp"]]
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f'Confusion Matrix - {row["model"]}')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, output_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")

# --------------------------------------------------
# 2) Confusion matrix for best F1 model
# --------------------------------------------------
best_f1_idx = df["f1_score"].idxmax()
best_f1_row = df.loc[best_f1_idx]
save_confusion_matrix_plot(best_f1_row, "best_f1_confusion_matrix.png")

# --------------------------------------------------
# 3) Confusion matrix for best Recall model
# --------------------------------------------------
best_recall_idx = df["recall"].idxmax()
best_recall_row = df.loc[best_recall_idx]
save_confusion_matrix_plot(best_recall_row, "best_recall_confusion_matrix.png")

print("\nDone. Plots saved in results\\plots")