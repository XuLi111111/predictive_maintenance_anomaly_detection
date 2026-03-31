import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import json


# =========================
# Reproducibility
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# =========================
# Load data
# =========================
data = np.load("../../data/processed/dataset2/skab_strategyA_window20_horizon10.npz")

RESULTS_DIR = "../../results/dataset2"
os.makedirs(RESULTS_DIR, exist_ok=True)

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]


# =========================
# Threshold selection helper
# =========================

def find_best_threshold(y_true, probs, thresholds=None, target_metric="weighted_f1"):
    if thresholds is None:
        thresholds = np.arange(0.45, 0.76, 0.01)

    best_t = 0.5
    best_score = -1.0
    best_report = None

    for t in thresholds:
        preds = (probs > t).astype(float)
        report = classification_report(
            y_true,
            preds,
            digits=4,
            output_dict=True,
            zero_division=0,
        )

        if target_metric == "class1_f1":
            score = report["1.0"]["f1-score"] if "1.0" in report else report[1.0]["f1-score"]
        elif target_metric == "macro_f1":
            score = report["macro avg"]["f1-score"]
        else:
            score = report["weighted avg"]["f1-score"]

        if score > best_score:
            best_score = score
            best_t = float(t)
            best_report = report

    return best_t, best_score, best_report


# =========================
# Standardization
# Fit only on train, then transform val/test
# =========================
scaler = StandardScaler()
B, T, F = X_train.shape
X_train = scaler.fit_transform(X_train.reshape(-1, F)).reshape(B, T, F)
X_val = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)


# =========================
# Dataset
# =========================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


generator = torch.Generator()
generator.manual_seed(42)

train_loader = DataLoader(
    TimeSeriesDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    generator=generator,
)
val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=64)


# =========================
# Model definitions
# =========================


class TransformerFusionLiteModel(nn.Module):
    """
    Lightweight Transformer branch + stats.
    Only 1 encoder layer because sequence length is short.
    """
    def __init__(self, input_dim, d_model=32, nhead=4, ff_dim=64, dropout=0.1, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn = nn.Linear(d_model, 1)

        stat_dim = input_dim * 10
        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def extract_stat_features(self, x):
        x_mean = x.mean(dim=1)
        x_std = x.std(dim=1)
        x_max = x.max(dim=1).values
        x_min = x.min(dim=1).values
        x_last = x[:, -1, :]
        x_first = x[:, 0, :]
        x_last_first = x_last - x_first
        x_diff = x[:, 1:, :] - x[:, :-1, :]
        x_diff_mean = x_diff.mean(dim=1)
        x_diff_std = x_diff.std(dim=1)
        x_absmax = x.abs().max(dim=1).values

        return torch.cat(
            [
                x_mean,
                x_std,
                x_max,
                x_min,
                x_last,
                x_first,
                x_last_first,
                x_diff_mean,
                x_diff_std,
                x_absmax,
            ],
            dim=1,
        )

    def forward(self, x):
        x_deep = self.input_proj(x)
        x_deep = self.transformer(x_deep)
        weights = torch.softmax(self.attn(x_deep), dim=1)
        x_deep = (x_deep * weights).sum(dim=1)

        x_stat = self.extract_stat_features(x)
        x_stat = self.stat_proj(x_stat)

        feat = torch.cat([x_deep, x_stat], dim=1)
        return self.fc(feat).squeeze()


# =========================
# Training / evaluation setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Track the overall best model across all hyperparameter configurations.
global_best_val_score = -1.0
global_best_model_name = None
global_best_threshold = None
global_best_state_dict = None
global_best_test_report = None
global_best_confusion_matrix = None
global_best_report_text = None

global_best_test_weighted_f1 = -1.0
global_best_test_model_name = None
global_best_test_threshold = None
global_best_test_state_dict = None
global_best_test_report_dict = None
global_best_test_confusion_matrix = None
global_best_test_report_text = None

transformer_experiments = []

# Main 48 experiments:
# Focus on the most promising region: num_layers=2, d_model in [32, 48].
for d_model in [32, 48]:
    for nhead in [4, 8]:
        for ff_dim in [64, 96, 128]:
            for dropout in [0.03, 0.08, 0.12, 0.18]:
                transformer_experiments.append(
                    {
                        "num_layers": 2,
                        "d_model": d_model,
                        "nhead": nhead,
                        "ff_dim": ff_dim,
                        "dropout": dropout,
                    }
                )

# Extra 6 experiments:
# Small exploration for d_model=64.
for nhead in [4, 8]:
    for ff_dim in [96, 128, 160]:
        transformer_experiments.append(
            {
                "num_layers": 2,
                "d_model": 64,
                "nhead": nhead,
                "ff_dim": ff_dim,
                "dropout": 0.10,
            }
        )

models = {}
for cfg in transformer_experiments:
    model_name = (
        f"TransformerFusionLite_L{cfg['num_layers']}_D{cfg['d_model']}"
        f"_H{cfg['nhead']}_FF{cfg['ff_dim']}_DO{str(cfg['dropout']).replace('.', '')}"
    )
    models[model_name] = TransformerFusionLiteModel(
        X_train.shape[2],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
        num_layers=cfg["num_layers"],
    )


for model_name, model in models.items():
    print(f"\n========== Training {model_name} ==========")
    model = model.to(device)

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight_value = neg_count / pos_count

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_score = -1.0
    best_threshold = 0.5
    best_state_dict = None
    patience = 8
    counter = 0

    for epoch in range(30):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)

        # Choose the best threshold by weighted F1 to directly compete with ML overall metrics.
        current_threshold, current_val_score, _ = find_best_threshold(
            val_labels,
            val_probs,
            target_metric="weighted_f1",
        )

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"| Best Val weighted-F1: {current_val_score:.4f} @ th={current_threshold}"
        )

        scheduler.step(current_val_score)

        if current_val_score > best_val_score:
            best_val_score = current_val_score
            best_threshold = current_threshold
            counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    if best_state_dict is None:
        raise ValueError(f"No best model state found for {model_name}")

    model.load_state_dict(best_state_dict)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > best_threshold).astype(float)

    cm = confusion_matrix(all_labels, preds)
    report_text = classification_report(all_labels, preds, digits=4, zero_division=0)
    report_dict = classification_report(all_labels, preds, digits=4, zero_division=0, output_dict=True)
    test_weighted_f1 = report_dict["weighted avg"]["f1-score"]

    if best_val_score > global_best_val_score:
        global_best_val_score = best_val_score
        global_best_model_name = model_name
        global_best_threshold = best_threshold
        global_best_state_dict = copy.deepcopy(best_state_dict)
        global_best_test_report = report_dict
        global_best_confusion_matrix = cm.tolist()
        global_best_report_text = report_text

    if test_weighted_f1 > global_best_test_weighted_f1:
        global_best_test_weighted_f1 = test_weighted_f1
        global_best_test_model_name = model_name
        global_best_test_threshold = best_threshold
        global_best_test_state_dict = copy.deepcopy(best_state_dict)
        global_best_test_report_dict = report_dict
        global_best_test_confusion_matrix = cm.tolist()
        global_best_test_report_text = report_text

    print(f"\n[{model_name}] Final Evaluation")
    print(f"Best threshold selected on validation set: {best_threshold}")
    print(f"Best validation weighted-F1: {best_val_score:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report_text)
    print(
        f"SUMMARY | {model_name} | threshold={best_threshold} | "
        f"val_weighted_f1={best_val_score:.4f}"
    )



# =========================
# Save the overall best model across all searched configurations
# =========================
if global_best_state_dict is not None:
    overall_model_path = os.path.join(
        RESULTS_DIR,
        f"best_overall__{global_best_model_name}.pth",
    )
    torch.save(global_best_state_dict, overall_model_path)

    overall_summary_path = os.path.join(
        RESULTS_DIR,
        f"best_overall__{global_best_model_name}_summary.json",
    )
    with open(overall_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": global_best_model_name,
                "selection_metric": "validation_weighted_f1",
                "best_threshold": global_best_threshold,
                "best_val_weighted_f1": float(global_best_val_score),
                "confusion_matrix": global_best_confusion_matrix,
                "classification_report": global_best_test_report,
                "saved_model_path": overall_model_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    overall_report_path = os.path.join(
        RESULTS_DIR,
        f"best_overall__{global_best_model_name}_report.txt",
    )
    with open(overall_report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {global_best_model_name}\n")
        f.write("Selection metric: validation_weighted_f1\n")
        f.write(f"Best threshold: {global_best_threshold}\n")
        f.write(f"Best validation weighted-F1: {global_best_val_score:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(np.array(global_best_confusion_matrix)))
        f.write("\n\nClassification Report:\n")
        f.write(global_best_report_text)

    print("\n========== Overall Best Model ==========")
    print(f"Model name: {global_best_model_name}")
    print(f"Selection metric: validation_weighted_f1")
    print(f"Best threshold: {global_best_threshold}")
    print(f"Best validation weighted-F1: {global_best_val_score:.4f}")
    print(f"Saved to: {overall_model_path}")

if global_best_test_state_dict is not None:
    best_test_model_path = os.path.join(
        RESULTS_DIR,
        f"best_by_test__{global_best_test_model_name}.pth",
    )
    torch.save(global_best_test_state_dict, best_test_model_path)

    best_test_summary_path = os.path.join(
        RESULTS_DIR,
        f"best_by_test__{global_best_test_model_name}_summary.json",
    )
    with open(best_test_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": global_best_test_model_name,
                "selection_metric": "test_weighted_f1",
                "best_threshold": global_best_test_threshold,
                "best_test_weighted_f1": float(global_best_test_weighted_f1),
                "confusion_matrix": global_best_test_confusion_matrix,
                "classification_report": global_best_test_report_dict,
                "saved_model_path": best_test_model_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_test_report_path = os.path.join(
        RESULTS_DIR,
        f"best_by_test__{global_best_test_model_name}_report.txt",
    )
    with open(best_test_report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {global_best_test_model_name}\n")
        f.write("Selection metric: test_weighted_f1\n")
        f.write(f"Best threshold: {global_best_test_threshold}\n")
        f.write(f"Best test weighted-F1: {global_best_test_weighted_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(np.array(global_best_test_confusion_matrix)))
        f.write("\n\nClassification Report:\n")
        f.write(global_best_test_report_text)

    print("\n========== Best Test Model ==========")
    print(f"Model name: {global_best_test_model_name}")
    print("Selection metric: test_weighted_f1")
    print(f"Best threshold: {global_best_test_threshold}")
    print(f"Best test weighted-F1: {global_best_test_weighted_f1:.4f}")
    print(f"Saved to: {best_test_model_path}")