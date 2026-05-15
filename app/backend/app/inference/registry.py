"""Metadata for the 8 pre-trained models exposed via /api/models (FR-08).

Headline metrics are the SKAB held-out test-set numbers from the
classical baseline run. The TransformerFusionLite entry has `None`
metrics until Xu Li delivers `model_transformer.pt` — the front-end
uses that to render a disabled "Coming soon" state. Once the artifact
is on disk, fill in the metrics here from his summary JSON.

The transformer's `requires_scaling` flag is unused at inference time:
the inference path detects `is_dl=True` and switches to the per-feature
`transformer_scaler.pkl` regardless. The flag is retained for clarity.
"""
from dataclasses import asdict, dataclass


@dataclass
class ModelMeta:
    id: str
    name: str
    family: str
    artifact: str  # filename under settings.artifact_dir
    is_dl: bool
    recall: float | None
    precision: float | None
    f1: float | None
    # Whether the model was trained on `scaler.transform(...)` features.
    # Tree-based ensembles (rf/et/gb/xgb) are scale-invariant and were
    # trained on raw flattened windows; feeding them scaled inputs at
    # inference time silently destroys their predictions.
    requires_scaling: bool = False

    def public_dict(self) -> dict:
        d = asdict(self)
        # Front-end uses this flag to grey-out the card without having
        # to inspect each metric individually.
        d["unavailable"] = self.recall is None
        return d


MODEL_REGISTRY: dict[str, ModelMeta] = {
    # Tree / boosting / DL models — trained on raw flattened windows.
    "xgb": ModelMeta("xgb", "XGBoost", "Ensemble", "model_xgb.pkl", False,
                     0.8049, 0.9944, 0.8897, requires_scaling=False),
    "rf": ModelMeta("rf", "Random Forest", "Tree", "model_rf.pkl", False,
                    0.8429, 0.8635, 0.8530, requires_scaling=False),
    "et": ModelMeta("et", "Extra Trees", "Tree", "model_et.pkl", False,
                    0.7669, 0.8989, 0.8277, requires_scaling=False),
    "gb": ModelMeta("gb", "Gradient Boosting", "Boosting",
                    "model_gb.pkl", False,
                    0.8397, 0.8811, 0.8599, requires_scaling=False),

    # Linear / kernel / instance models — trained on scaled features.
    "lr": ModelMeta("lr", "Logistic Regression", "Linear",
                    "model_lr.pkl", False,
                    0.8455, 0.9755, 0.9058, requires_scaling=True),
    "knn": ModelMeta("knn", "KNN", "Instance", "model_knn.pkl", False,
                     0.5544, 0.8768, 0.6793, requires_scaling=True),
    "svm": ModelMeta("svm", "SVM", "Kernel", "model_svm.pkl", False,
                     0.3690, 0.9409, 0.5301, requires_scaling=True),

    # Deep-learning model — TransformerFusionLite (Xu Li, T7).
    # Selected from a 54-config sweep; the chosen artifact is the one with
    # the best test weighted-F1 (D32_H8_FF64_DO018, val_F1=0.9395, test F1=0.9460).
    # Test class-1 metrics below come from
    # results/skab/best_by_test__..._summary.json.
    # `requires_scaling` is not consulted for is_dl models — the inference
    # path uses transformer_scaler.pkl unconditionally.
    "transformer": ModelMeta(
        "transformer", "TransformerFusionLite",
        "Deep Learning", "model_transformer.pt", True,
        0.8822, 0.9709, 0.9244, requires_scaling=False,
    ),
}
