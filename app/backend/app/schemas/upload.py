from pydantic import BaseModel


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    rows: int
    columns: list[str]
    has_label: bool
    message: str