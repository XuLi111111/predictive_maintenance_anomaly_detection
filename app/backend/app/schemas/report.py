from pydantic import BaseModel


class ReportRequest(BaseModel):
    file_id: str
    model_id: str
    prediction_id: str | None = None
