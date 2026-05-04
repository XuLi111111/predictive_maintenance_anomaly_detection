from fastapi import APIRouter, HTTPException

from app.schemas.report import ReportRequest

router = APIRouter()


@router.post("/report")
def generate_report(req: ReportRequest):
    """FR-17/18 — generate a PDF report (chart, model, metrics, summary panel,
    detected columns, preprocessing applied, plain-language explanation).

    TODO (teammate): build PDF with ReportLab, return as application/pdf.
    """
    raise HTTPException(status_code=501, detail="report endpoint not implemented")
