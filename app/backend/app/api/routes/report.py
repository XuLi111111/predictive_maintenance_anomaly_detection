"""PDF report endpoint (T4).

Reuses the same prepared inputs and inference helpers as `/api/predict`,
then hands off to `app.reports.pdf` for layout. Streamed back to the
client as `application/pdf`.

The enhanced report (Tier 1 + 2D + 3F + 3G) needs every model's output
on the same file for the 8-model comparison page — so we run
`/predict/compare` in-process before invoking the layout step.
"""
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.core.config import settings
from app.inference import predict as inference
from app.inference.registry import MODEL_REGISTRY
from app.reports.pdf import build_pdf
from app.schemas.report import ReportRequest

router = APIRouter()


def _generate_report_sync(file_id: str, model_id: str) -> bytes:
    """All CPU-bound steps run on a worker thread:

    1. Prepare windows / scaler / labels.
    2. Run the *selected* model (drives the main report).
    3. Run every other model on the same prepared inputs for the
       Tier-1C comparison page. Unavailable models degrade gracefully
       (their `unavailable=True` outputs render as "—" in the table).
    4. Render the PDF.
    """
    prepared = inference.prepare_input(file_id)
    threshold = settings.alert_threshold

    selected_output = inference.run_model(model_id, prepared, threshold)

    # 8-model comparison data. We deliberately reuse `prepared` so the
    # heavy `prepare_input` work (CSV parse + windowing + scaler) only
    # happens once.
    compare_results = []
    for mid in MODEL_REGISTRY:
        if mid == model_id:
            compare_results.append(selected_output)
        else:
            compare_results.append(
                inference.run_model(mid, prepared, threshold),
            )

    return build_pdf(prepared, selected_output, threshold, compare_results)


@router.post("/report")
async def generate_report(req: ReportRequest) -> Response:
    """FR-17 / FR-18 / FR-19 — return the multi-page PDF report.

    Sections: Executive Summary, Detection Detail (chart + anomaly
    timeline), Model Comparison (all 8 models), Performance + Confusion
    + ROC/PR (labelled only), Sensor Deep-Dive, Methodology + Appendix.

    PDF rendering and the 8-way inference sweep are CPU-bound; pushed to
    the threadpool so they don't stall any open WebSocket streams.
    """
    pdf_bytes = await run_in_threadpool(
        _generate_report_sync, req.file_id, req.model_id,
    )

    filename = f"pump_detect_report_{req.file_id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
