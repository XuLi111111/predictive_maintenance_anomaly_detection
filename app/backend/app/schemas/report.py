from pydantic import BaseModel, ConfigDict


class ReportRequest(BaseModel):
    # Allow `model_id` field name despite Pydantic's reserved `model_` namespace.
    model_config = ConfigDict(protected_namespaces=())

    file_id: str
    model_id: str
    prediction_id: str | None = None
