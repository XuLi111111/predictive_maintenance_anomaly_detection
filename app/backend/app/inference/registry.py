"""Metadata for the 8 pre-trained models exposed via /api/models (FR-08).

Headline metrics are the SKAB held-out test-set numbers from the classical
baseline run. Update the transformer entry once the .pt artifact is in place.
"""
from dataclasses import asdict, dataclass


@dataclass
class ModelMeta:
    id: str
    name: str
    family: str
    artifact: str  # filename under settings.artifact_dir
    is_dl: bool
    recall: float
    precision: float
    f1: float

    def public_dict(self) -> dict:
        return asdict(self)


MODEL_REGISTRY: dict[str, ModelMeta] = {
    "xgb": ModelMeta("xgb", "XGBoost", "Ensemble", "model_xgb.pkl", False,
                     0.8049, 0.9944, 0.8897),
    "rf": ModelMeta("rf", "Random Forest", "Tree", "model_rf.pkl", False,
                    0.8429, 0.8635, 0.8530),
    "et": ModelMeta("et", "Extra Trees", "Tree", "model_et.pkl", False,
                    0.7669, 0.8989, 0.8277),
    "gb": ModelMeta("gb", "Gradient Boosting", "Boosting", "model_gb.pkl", False,
                    0.8397, 0.8811, 0.8599),
    "lr": ModelMeta("lr", "Logistic Regression", "Linear", "model_lr.pkl", False,
                    0.8455, 0.9755, 0.9058),
    "knn": ModelMeta("knn", "KNN", "Instance", "model_knn.pkl", False,
                     0.5544, 0.8768, 0.6793),
    "svm": ModelMeta("svm", "SVM", "Kernel", "model_svm.pkl", False,
                     0.3690, 0.9409, 0.5301),
    "transformer": ModelMeta("transformer", "TransformerFusionLite",
                             "Deep Learning", "model_transformer.pt", True,
                             0.0, 0.0, 0.0),  # TODO: fill once artifact lands
}
