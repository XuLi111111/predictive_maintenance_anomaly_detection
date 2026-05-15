"""PDF report — multi-page enhanced layout.

Sections (mapped to the design tiers we agreed with the client):

  Page 1 │ Executive Summary
            Tier 1A — verdict card, risk badge, recommended action,
            top-3 anomaly preview, metadata strip.
  Page 2 │ Detection Detail
            Original probability chart, full anomaly timeline table
            (Tier 1B), and the plain-language explanation (FR-18).
  Page 3 │ Model Comparison & Metrics
            8-model comparison (Tier 1C, FR-19), performance metrics
            on this file, plus confusion matrix and ROC / PR curves
            with best-F1 threshold suggestion (Tier 3F+3G — only when
            the file carries ground-truth labels).
  Page 4 │ Sensor Deep-Dive
            Per-channel stats (file-wide vs at-peak-window) and an
            8-panel sparkline grid centred on the peak window. (Tier 2D)
  Page 5 │ Methodology & Appendix
            Preprocessing steps, plain-language methodology, full
            provenance / reproducibility block (Tier 1E).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless — must precede `pyplot` import
import matplotlib.pyplot as plt  # noqa: E402

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from sklearn.metrics import (
        auc,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

from app.core.config import settings
from app.inference.predict import ModelOutput, PreparedInput
from app.inference.registry import MODEL_REGISTRY


# ──────────────────────────────────────────────────────────────────────
# Risk badge thresholds (locked at the 2026-05-12 design review)
# ──────────────────────────────────────────────────────────────────────
# LOW    : no anomaly windows OR peak prob below 0.50
# HIGH   : ≥10% of windows flagged OR peak prob ≥ 0.80
# MEDIUM : otherwise
_RISK_LOW_HEX = "#16a34a"
_RISK_MED_HEX = "#ea580c"
_RISK_HIGH_HEX = "#dc2626"


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────

def build_pdf(
    prepared: PreparedInput,
    output: ModelOutput,
    threshold: float,
    compare_results: list[ModelOutput] | None = None,
) -> bytes:
    """Render the enhanced multi-page report and return the PDF bytes.

    Parameters
    ----------
    prepared
        Shared preprocessing result — gives us detected columns,
        sidecar metadata, sensor matrix, and ground-truth labels.
    output
        The selected model's prediction result (primary subject).
    threshold
        The threshold the user actually ran with.
    compare_results
        Optional list of ``ModelOutput`` for every registered model on
        the same input. Used for the Tier-1C 8-model comparison table.
        When omitted, that section degrades gracefully to a stub.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="pump.detect — Anomaly Report",
        author="pump.detect / Group 14",
    )

    styles = _build_styles()
    story: list = []

    _page_executive_summary(story, styles, prepared, output, threshold)
    story.append(PageBreak())
    _page_detection_detail(story, styles, prepared, output, threshold)
    story.append(PageBreak())
    _page_model_comparison_and_metrics(
        story, styles, prepared, output, threshold, compare_results,
    )
    story.append(PageBreak())
    _page_sensor_deepdive(story, styles, prepared, output)
    story.append(PageBreak())
    _page_methodology_and_appendix(story, styles, prepared, output, threshold)

    doc.build(story)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────
# Page 1: Executive Summary
# ──────────────────────────────────────────────────────────────────────

def _page_executive_summary(story, styles, prepared, output, threshold) -> None:
    level, level_hex = _risk_level(output)

    filename = prepared.sidecar.get("filename", f"{prepared.file_id}.csv")
    model_meta = MODEL_REGISTRY.get(output.model_id)
    model_label = model_meta.name if model_meta else output.model_id
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    story.append(Paragraph("pump.detect — Anomaly Report", styles["Title"]))
    story.append(Paragraph("Executive Summary", styles["H2"]))
    story.append(Spacer(1, 0.3 * cm))

    # Big verdict card
    if output.fault_windows == 0:
        verdict = "Pump operating normally"
        verdict_detail = (
            f"No windows exceeded the alert threshold "
            f"({int(threshold * 100)}%). Highest probability seen was "
            f"{output.peak_prob:.0%}."
        )
    else:
        verdict = f"{output.fault_windows} anomaly window(s) detected"
        verdict_detail = (
            f"Out of {output.total_windows} windows analysed. Peak "
            f"probability {output.peak_prob:.0%} at window #{output.peak_idx}."
        )

    story.append(_verdict_card(verdict, verdict_detail, level, level_hex))
    story.append(Spacer(1, 0.5 * cm))

    # Recommended action
    action = _recommended_action(output, threshold, level)
    story.append(Paragraph("Recommended action", styles["H3"]))
    story.append(Paragraph(action, styles["Body"]))
    story.append(Spacer(1, 0.5 * cm))

    # Top 3 anomaly windows preview
    if output.anomaly_windows:
        story.append(Paragraph("Top anomaly windows (preview)", styles["H3"]))
        timeline_rows = _anomaly_timeline_rows(prepared, output, limit=3)
        story.append(_anomaly_timeline_table(timeline_rows, compact=True))
        story.append(Spacer(1, 0.4 * cm))

    # Metadata strip
    story.append(Paragraph("Report metadata", styles["H3"]))
    info = [
        ["Input file",       filename],
        ["Model used",       f"{model_label} ({output.model_id})"],
        ["Alert threshold",  f"{threshold:.2f} ({int(threshold * 100)}%)"],
        ["Rows analysed",    f"{prepared.df.shape[0]:,}"],
        ["Generated",        generated],
    ]
    story.append(_kv_table(info))


def _verdict_card(verdict: str, detail: str, level: str, level_hex: str) -> Table:
    """The big colorful card at the top of page 1."""
    badge = Paragraph(
        f"<para alignment='center'><b>RISK: {level}</b></para>",
        ParagraphStyle(
            "Badge",
            fontSize=11,
            textColor=colors.white,
            leading=14,
        ),
    )
    cell = Table(
        [
            [Paragraph(
                f"<b>{verdict}</b>",
                ParagraphStyle("V", fontSize=18, leading=22, textColor=colors.white),
            )],
            [Paragraph(detail, ParagraphStyle(
                "VD", fontSize=11, leading=15, textColor=colors.white,
            ))],
            [badge],
        ],
        colWidths=[17 * cm],
    )
    cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(level_hex)),
        ("TEXTCOLOR",  (0, 0), (-1, -1), colors.white),
        ("LEFTPADDING",   (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
        ("TOPPADDING",    (0, 0), (-1, 0),  16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return cell


def _risk_level(output: ModelOutput) -> tuple[str, str]:
    if output.fault_windows == 0 or output.peak_prob < 0.5:
        return "LOW", _RISK_LOW_HEX
    fault_pct = output.fault_windows / max(output.total_windows, 1)
    if fault_pct >= 0.10 or output.peak_prob >= 0.8:
        return "HIGH", _RISK_HIGH_HEX
    return "MEDIUM", _RISK_MED_HEX


def _recommended_action(output: ModelOutput, threshold: float, level: str) -> str:
    if level == "LOW":
        if output.fault_windows == 0:
            return (
                "No action required. The pump showed no behaviour above the "
                f"{int(threshold * 100)}% alert threshold across the entire "
                "file. Continue routine monitoring per the maintenance plan."
            )
        return (
            f"Low concern. {output.fault_windows} short flag(s) were seen "
            f"with peak probability {output.peak_prob:.0%}. Likely transient — "
            "no immediate inspection required, but note the timestamps in the "
            "Anomaly Timeline (Page 2) and re-check at the next scheduled "
            "maintenance window."
        )
    if level == "MEDIUM":
        return (
            f"Inspection recommended. {output.fault_windows} window(s) "
            f"exceeded the {int(threshold * 100)}% threshold with peak "
            f"{output.peak_prob:.0%}. Schedule a visual / vibration check of "
            "the pump during the next maintenance slot, focusing on the time "
            "ranges listed in the Anomaly Timeline."
        )
    return (
        f"<b>Inspect immediately.</b> {output.fault_windows} window(s) "
        f"exceeded the threshold, peak probability {output.peak_prob:.0%}. "
        "Consider pausing the pump and dispatching maintenance to check the "
        "highlighted time ranges (Page 2). The Sensor Deep-Dive (Page 4) "
        "identifies which channel drove the alert."
    )


# ──────────────────────────────────────────────────────────────────────
# Page 2: Detection Detail
# ──────────────────────────────────────────────────────────────────────

def _page_detection_detail(story, styles, prepared, output, threshold) -> None:
    story.append(Paragraph("Detection Detail", styles["H2"]))
    story.append(Paragraph(
        "Per-window anomaly probability over the full uploaded file, "
        "with all flagged windows listed individually.",
        styles["BodyMuted"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    img_buf = _render_probability_chart(output, threshold)
    story.append(Image(img_buf, width=17 * cm, height=7 * cm))
    story.append(Spacer(1, 0.4 * cm))

    if output.anomaly_windows:
        story.append(Paragraph("Anomaly Timeline", styles["H3"]))
        story.append(Paragraph(
            "Each row is a contiguous run of windows whose probability "
            "stayed above the alert threshold. <b>Peak sensor</b> shows "
            "which channel deviated most (in standard deviations) from "
            "the file-wide mean during that run — a quick pointer for "
            "the field engineer.",
            styles["BodyMuted"],
        ))
        story.append(Spacer(1, 0.15 * cm))
        timeline_rows = _anomaly_timeline_rows(prepared, output)
        story.append(_anomaly_timeline_table(timeline_rows, compact=False))
    else:
        story.append(Paragraph(
            "No anomaly windows were flagged in this file.",
            styles["BodyMuted"],
        ))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("What the chart means (plain English)", styles["H3"]))
    story.append(Paragraph(_plain_english(output, threshold), styles["Body"]))


def _render_probability_chart(output: ModelOutput, threshold: float) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    if output.probs:
        ax.plot(output.probs, color="#7c3aed", linewidth=1.6,
                label="P(anomaly in next 10 s)")
        ax.axhline(threshold, color="#b45309", linewidth=1,
                    linestyle="--", label=f"threshold = {threshold:.2f}")
        for run in output.anomaly_windows:
            ax.axvspan(run["start_idx"], run["end_idx"],
                        color="#fde68a", alpha=0.55)
        ax.scatter([output.peak_idx], [output.peak_prob],
                    color="#dc2626", zorder=5, s=30)
    else:
        ax.text(0.5, 0.5, "No windows produced",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=140)
    plt.close(fig)
    img_buf.seek(0)
    return img_buf


# ──────────────────────────────────────────────────────────────────────
# Anomaly Timeline helpers (shared by Pages 1 + 2)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _TimelineRow:
    idx_label: str
    start_clock: str
    end_clock: str
    duration: str
    peak_prob: str
    peak_sensor: str


def _anomaly_timeline_rows(
    prepared: PreparedInput,
    output: ModelOutput,
    limit: int | None = None,
) -> list[_TimelineRow]:
    """Build one row per anomaly run. Always sorted by start_idx; if
    ``limit`` is set, returns the top N by peak_prob."""
    if not output.anomaly_windows:
        return []

    df_sorted = (
        prepared.df.sort_values(settings.timestamp_column).reset_index(drop=True)
    )
    try:
        ts = pd.to_datetime(df_sorted[settings.timestamp_column])
    except (ValueError, TypeError):
        ts = None
    sensor_matrix = df_sorted[settings.sensor_columns].to_numpy(dtype=float)
    file_mean = sensor_matrix.mean(axis=0)
    file_std = sensor_matrix.std(axis=0) + 1e-9
    w = settings.window_size

    runs = output.anomaly_windows
    if limit is not None:
        runs = sorted(runs, key=lambda r: -r["peak_prob"])[:limit]
        runs = sorted(runs, key=lambda r: r["start_idx"])

    rows: list[_TimelineRow] = []
    for i, run in enumerate(runs, 1):
        start_row = w - 1 + int(run["start_idx"])
        end_row = w - 1 + int(run["end_idx"])
        end_row = min(end_row, sensor_matrix.shape[0] - 1)

        window_block = sensor_matrix[start_row : end_row + 1]
        z = (window_block.mean(axis=0) - file_mean) / file_std
        peak_sensor_idx = int(np.argmax(np.abs(z)))
        peak_sensor_name = settings.sensor_columns[peak_sensor_idx]
        peak_z = float(z[peak_sensor_idx])

        if ts is not None and start_row < len(ts):
            t_start = ts.iloc[start_row]
            t_end = ts.iloc[end_row] if end_row < len(ts) else t_start
            duration_s = int((t_end - t_start).total_seconds()) + 1
            start_clock = t_start.strftime("%Y-%m-%d %H:%M:%S")
            end_clock = t_end.strftime("%H:%M:%S")
        else:
            duration_s = end_row - start_row + 1
            start_clock = f"row {start_row}"
            end_clock = f"row {end_row}"

        rows.append(_TimelineRow(
            idx_label=f"#{i}",
            start_clock=start_clock,
            end_clock=end_clock,
            duration=f"{duration_s}s",
            peak_prob=f"{run['peak_prob']:.0%}",
            peak_sensor=f"{peak_sensor_name} ({peak_z:+.1f}σ)",
        ))
    return rows


def _anomaly_timeline_table(rows: list[_TimelineRow], compact: bool) -> Table:
    header = ["#", "Start", "End", "Dur.", "Peak P", "Peak sensor (z vs file)"]
    body = [
        [r.idx_label, r.start_clock, r.end_clock, r.duration,
         r.peak_prob, r.peak_sensor]
        for r in rows
    ]
    col_widths = [0.9 * cm, 4.0 * cm, 2.2 * cm, 1.4 * cm, 1.6 * cm, 7.1 * cm]
    t = Table([header, *body], colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3eafe")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.white, colors.HexColor("#fafaf9")]),
        ("LINEBELOW",  (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ])
    if compact:
        style.add("FONTSIZE", (0, 0), (-1, -1), 8.5)
    t.setStyle(style)
    return t


# ──────────────────────────────────────────────────────────────────────
# Page 3: Model Comparison & Metrics
# ──────────────────────────────────────────────────────────────────────

def _page_model_comparison_and_metrics(
    story, styles, prepared, output, threshold, compare_results,
) -> None:
    story.append(Paragraph("Model Comparison & Performance", styles["H2"]))
    story.append(Paragraph(
        "All 8 pre-trained models run on the same uploaded file. "
        "The selected (★) model drove the main report; the others are "
        "shown for cross-validation — strong agreement means high "
        "confidence, disagreement is worth a second look.",
        styles["BodyMuted"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    if compare_results:
        story.append(_model_comparison_table(compare_results, output.model_id))
        story.append(Spacer(1, 0.2 * cm))
        agreement = _agreement_summary(compare_results, threshold)
        story.append(Paragraph(agreement, styles["BodyMuted"]))
    else:
        story.append(Paragraph(
            "Comparison table unavailable (compare_results not provided).",
            styles["BodyMuted"],
        ))

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "Performance on this file — selected model",
        styles["H3"],
    ))
    if prepared.has_label and output.metrics is not None:
        m = output.metrics
        story.append(_kv_table([
            ["Precision", f"{m['precision']:.3f}"],
            ["Recall",    f"{m['recall']:.3f}"],
            ["F1",        f"{m['f1']:.3f}"],
        ]))
    else:
        story.append(Paragraph(
            "No ground-truth <i>anomaly</i> column was supplied, so "
            "Precision / Recall / F1 cannot be computed for this file "
            "(SRS FR-21). The probability chart and risk verdict above "
            "are unaffected — they don't require labels.",
            styles["BodyMuted"],
        ))

    if prepared.has_label and prepared.labels is not None and _SKLEARN_AVAILABLE:
        probs = np.asarray(output.probs, dtype=float)
        labels = np.asarray(prepared.labels, dtype=int)
        if len(probs) == len(labels) and probs.size > 0 and labels.sum() > 0:
            story.append(Spacer(1, 0.5 * cm))
            _append_confusion_and_curves(story, styles, labels, probs, threshold)


def _model_comparison_table(
    results: Iterable[ModelOutput],
    selected_id: str,
) -> Table:
    header = ["", "Model", "Family",
              "Flags", "Peak P", "F1 (this file)", "F1 (test set)"]
    body: list[list[str]] = []
    for r in results:
        meta = MODEL_REGISTRY.get(r.model_id)
        mark = "★" if r.model_id == selected_id else ""
        name = (meta.name if meta else r.model_id)
        if r.unavailable:
            name += " (unavailable)"
        family = meta.family if meta else "—"
        flags = "—" if r.unavailable else str(r.fault_windows)
        peak = "—" if r.unavailable else f"{r.peak_prob:.2f}"
        file_f1 = "—"
        if not r.unavailable and r.metrics is not None:
            file_f1 = f"{r.metrics['f1']:.2f}"
        test_f1 = "—" if not meta or meta.f1 is None else f"{meta.f1:.2f}"
        body.append([mark, name, family, flags, peak, file_f1, test_f1])

    col_widths = [0.7 * cm, 4.6 * cm, 2.6 * cm,
                  1.4 * cm, 1.6 * cm, 3.2 * cm, 2.9 * cm]
    t = Table([header, *body], colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3eafe")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.white, colors.HexColor("#fafaf9")]),
        ("LINEBELOW",  (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("ALIGN",      (3, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _agreement_summary(results: list[ModelOutput], threshold: float) -> str:
    available = [r for r in results if not r.unavailable]
    if not available:
        return "All comparison models unavailable."
    flagging = [r for r in available if r.fault_windows > 0]
    n = len(available)
    k = len(flagging)
    if k == 0:
        return (
            f"<b>Agreement:</b> all {n} models agree — no windows above "
            f"the {int(threshold * 100)}% threshold on this file."
        )
    if k == n:
        return (
            f"<b>Agreement:</b> all {n} models flag at least one window "
            f"on this file — high confidence in the verdict."
        )
    silent = [
        (MODEL_REGISTRY[r.model_id].name if r.model_id in MODEL_REGISTRY else r.model_id)
        for r in available if r.fault_windows == 0
    ]
    return (
        f"<b>Partial agreement:</b> {k} of {n} models flag at least one "
        f"window. Models that found nothing: {', '.join(silent)}. Cross-"
        f"check the disagreement with the Sensor Deep-Dive on the next page."
    )


def _append_confusion_and_curves(
    story, styles, labels: np.ndarray, probs: np.ndarray, threshold: float,
) -> None:
    """Tier-3F + Tier-3G: confusion matrix at the user's threshold, ROC
    curve + Precision-Recall curve with best-F1 threshold callout."""
    story.append(Paragraph(
        "Diagnostics (labelled data)",
        styles["H3"],
    ))
    pred = (probs >= threshold).astype(int)
    # NB: name the matrix `cmat` — not `cm` — to avoid shadowing the
    # `cm` unit (reportlab.lib.units) that's used as `17 * cm` below.
    cmat = confusion_matrix(labels, pred, labels=[0, 1])
    cm_buf = _render_confusion_matrix(cmat, threshold)
    fp_indices = np.where((pred == 1) & (labels == 0))[0]
    fn_indices = np.where((pred == 0) & (labels == 1))[0]

    table_data = [
        ["", "Predicted normal", "Predicted anomaly"],
        ["Actual normal",   str(cmat[0, 0]),  str(cmat[0, 1])],
        ["Actual anomaly",  str(cmat[1, 0]),  str(cmat[1, 1])],
    ]
    cm_table = Table(table_data, colWidths=[3.6 * cm, 4.2 * cm, 4.2 * cm])
    cm_table.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("BACKGROUND", (1, 0), (-1, 0), colors.HexColor("#f3eafe")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f3eafe")),
        ("FONTNAME",   (1, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN",      (1, 1), (-1, -1), "CENTER"),
        ("LINEBELOW",  (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([
        Paragraph(
            f"<b>Confusion matrix</b> at the configured "
            f"{int(threshold * 100)}% threshold:",
            styles["Body"],
        ),
        Spacer(1, 0.15 * cm),
        cm_table,
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"False positives: {len(fp_indices)} · "
            f"False negatives: {len(fn_indices)}.",
            styles["BodyMuted"],
        ),
    ]))

    # ROC + PR + best-F1 threshold
    story.append(Spacer(1, 0.5 * cm))
    roc_buf, roc_auc, best_threshold, best_f1 = _render_roc_pr(labels, probs)
    story.append(Image(roc_buf, width=17 * cm, height=6.6 * cm))
    story.append(Paragraph(
        f"ROC AUC = <b>{roc_auc:.3f}</b>. Best F1 on this file = "
        f"<b>{best_f1:.3f}</b>, achieved at threshold "
        f"<b>{best_threshold:.2f}</b>. "
        f"Consider trying that threshold from the Upload page if you want "
        f"to optimise precision / recall trade-off for this specific data.",
        styles["BodyMuted"],
    ))


def _render_confusion_matrix(cmat: np.ndarray, threshold: float) -> io.BytesIO:
    # Note: not `cm` — that's the reportlab unit constant at module scope.
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    vmax = max(cmat.max(), 1)
    im = ax.imshow(cmat, cmap="Purples", vmin=0, vmax=vmax)
    for i in range(2):
        for j in range(2):
            txt_color = "white" if cmat[i, j] > vmax * 0.55 else "#1f1147"
            ax.text(j, i, str(cmat[i, j]), ha="center", va="center",
                    fontsize=14, color=txt_color, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"], fontsize=9)
    ax.set_yticklabels(["Normal", "Anomaly"], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual",    fontsize=9)
    ax.set_title(f"@ threshold = {threshold:.2f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_roc_pr(
    labels: np.ndarray, probs: np.ndarray,
) -> tuple[io.BytesIO, float, float, float]:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = float(auc(fpr, tpr))
    prec, rec, pr_thr = precision_recall_curve(labels, probs)
    # F1 across the full PR sweep, then locate the best operating point.
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * prec * rec / np.where(prec + rec == 0, 1, prec + rec)
    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    # pr_thr has len = len(prec) - 1; the last (prec, rec) point doesn't
    # correspond to a threshold.
    if best_idx < len(pr_thr):
        best_threshold = float(pr_thr[best_idx])
    else:
        best_threshold = float(pr_thr[-1]) if len(pr_thr) > 0 else 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(fpr, tpr, color="#7c3aed", lw=2.0)
    ax1.plot([0, 1], [0, 1], color="#9ca3af", lw=0.8, ls="--")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title(f"ROC (AUC = {roc_auc:.3f})", fontsize=10)
    ax1.grid(alpha=0.25)

    ax2.plot(rec, prec, color="#b45309", lw=2.0)
    if best_idx < len(prec):
        ax2.scatter([rec[best_idx]], [prec[best_idx]],
                     color="#dc2626", s=40, zorder=5,
                     label=f"best F1 = {best_f1:.2f}")
        ax2.legend(loc="lower left", fontsize=8, frameon=False)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf, roc_auc, best_threshold, best_f1


# ──────────────────────────────────────────────────────────────────────
# Page 4: Sensor Deep-Dive
# ──────────────────────────────────────────────────────────────────────

def _page_sensor_deepdive(story, styles, prepared, output) -> None:
    story.append(Paragraph("Sensor Deep-Dive", styles["H2"]))

    if not output.probs:
        story.append(Paragraph(
            "No predictions were produced; sensor deep-dive skipped.",
            styles["BodyMuted"],
        ))
        return

    df_sorted = (
        prepared.df.sort_values(settings.timestamp_column).reset_index(drop=True)
    )
    sensor_matrix = df_sorted[settings.sensor_columns].to_numpy(dtype=float)
    w = settings.window_size

    peak_row = w - 1 + output.peak_idx
    peak_row = max(0, min(peak_row, sensor_matrix.shape[0] - 1))
    peak_window_lo = max(0, peak_row - w + 1)
    peak_window_hi = peak_row + 1

    file_mean = sensor_matrix.mean(axis=0)
    file_std = sensor_matrix.std(axis=0) + 1e-9
    win_data = sensor_matrix[peak_window_lo:peak_window_hi]
    win_mean = win_data.mean(axis=0)
    z_scores = (win_mean - file_mean) / file_std

    story.append(Paragraph(
        "How the 8 sensors behaved around the peak window (the moment "
        "the model was most certain something was wrong). The z-score "
        "column tells you how far that channel deviated from its "
        "file-wide average — large absolute values point at the "
        "physical channel that most likely drove the alert.",
        styles["BodyMuted"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # Per-channel stats table
    header = ["Channel", "File mean", "File min",
              "File max", "Peak-window mean", "z-score"]
    body: list[list[str]] = []
    for i, col in enumerate(settings.sensor_columns):
        body.append([
            col,
            f"{file_mean[i]:.3f}",
            f"{sensor_matrix[:, i].min():.3f}",
            f"{sensor_matrix[:, i].max():.3f}",
            f"{win_mean[i]:.3f}",
            f"{z_scores[i]:+.2f}σ",
        ])
    col_widths = [4.3 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm, 3.4 * cm, 2.5 * cm]
    t = Table([header, *body], colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3eafe")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.white, colors.HexColor("#fafaf9")]),
        ("LINEBELOW",  (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
        ("ALIGN",      (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # Auto-narrative naming the most-deviant channel
    peak_sensor_idx = int(np.argmax(np.abs(z_scores)))
    peak_sensor = settings.sensor_columns[peak_sensor_idx]
    peak_z = float(z_scores[peak_sensor_idx])
    direction = "above" if peak_z > 0 else "below"
    narrative = (
        f"<b>Most deviant channel:</b> <i>{peak_sensor}</i> at "
        f"{abs(peak_z):.1f} standard deviations {direction} its "
        f"file-wide mean during the peak window. This is the channel "
        f"the field engineer should inspect first."
    )
    story.append(Paragraph(narrative, styles["Body"]))
    story.append(Spacer(1, 0.4 * cm))

    # Sparkline grid
    story.append(Paragraph(
        "Sensor traces around the peak window (±30 seconds)", styles["H3"],
    ))
    sparkline_buf = _render_sparkline_grid(
        sensor_matrix, peak_row, peak_window_lo, peak_window_hi,
    )
    story.append(Image(sparkline_buf, width=17 * cm, height=8.5 * cm))


def _render_sparkline_grid(
    sensor_matrix: np.ndarray,
    peak_row: int,
    win_lo: int,
    win_hi: int,
) -> io.BytesIO:
    """8-panel grid: each panel is one channel; the peak window is
    highlighted with an amber band and the peak row marked in red."""
    half = 30
    lo = max(0, peak_row - half)
    hi = min(sensor_matrix.shape[0], peak_row + half + 1)

    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True)
    for i, col in enumerate(settings.sensor_columns):
        ax = axes[i // 4, i % 4]
        x = np.arange(lo, hi)
        ax.plot(x, sensor_matrix[lo:hi, i], color="#7c3aed", lw=1.3)
        # Highlight the peak window's span
        ax.axvspan(win_lo, win_hi - 1, color="#fde68a", alpha=0.5)
        ax.axvline(peak_row, color="#dc2626", lw=0.9, ls="--")
        ax.set_title(col, fontsize=8, pad=2)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Amber band = peak anomaly window · red dashed = peak window end",
        fontsize=8, color="#6b7280",
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────────────────────────────
# Page 5: Methodology & Appendix
# ──────────────────────────────────────────────────────────────────────

def _page_methodology_and_appendix(
    story, styles, prepared, output, threshold,
) -> None:
    story.append(Paragraph("Methodology", styles["H2"]))
    story.append(Paragraph("Detected input columns", styles["H3"]))
    detected = prepared.sidecar.get("columns", list(prepared.df.columns))
    story.append(Paragraph(", ".join(detected) if detected else "—", styles["Body"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Preprocessing applied", styles["H3"]))
    bullets = [
        f"Rows sorted chronologically by '{settings.timestamp_column}'.",
        f"Sensor channels standardised using the StandardScaler fit on "
        f"the SKAB training split (no re-fitting on uploaded data).",
        f"Sliding windows of length {settings.window_size} steps with "
        f"stride 1 — same configuration as model training (SRS NFR-05).",
        f"Future horizon for label evaluation: {settings.horizon} steps.",
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles["Body"]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Appendix — Provenance & Reproducibility",
                            styles["H2"]))

    model_meta = MODEL_REGISTRY.get(output.model_id)
    sidecar = prepared.sidecar or {}
    appendix_rows = [
        ["file_id",          prepared.file_id],
        ["Original filename", sidecar.get("filename", "—")],
        ["Rows ingested",    f"{prepared.df.shape[0]:,}"],
        ["Time range",
            f"{sidecar.get('time_range', ['?', '?'])[0]} → "
            f"{sidecar.get('time_range', ['?', '?'])[1]}"],
        ["Has ground-truth label", "yes" if prepared.has_label else "no"],
        ["Original separator", sidecar.get("original_separator", "—")],
        ["Uploaded at",      sidecar.get("uploaded_at", "—")],
        ["Model id",         output.model_id],
        ["Model artifact",   model_meta.artifact if model_meta else "—"],
        ["Model family",     model_meta.family if model_meta else "—"],
        ["Alert threshold",  f"{threshold:.2f}"],
        ["Window size",      str(settings.window_size)],
        ["Forecast horizon", str(settings.horizon)],
        ["Report generated",
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")],
    ]
    story.append(_kv_table(appendix_rows))


# ──────────────────────────────────────────────────────────────────────
# Shared style + helpers
# ──────────────────────────────────────────────────────────────────────

def _plain_english(output: ModelOutput, threshold: float) -> str:
    if output.fault_windows == 0:
        return (
            "The model did not detect any anomaly across the uploaded "
            f"window. The highest probability seen was "
            f"{output.peak_prob:.0%}, which is below the alert "
            f"threshold of {threshold:.0%}. The pump appears to be "
            "operating normally for the period covered by this file."
        )
    pct = output.fault_windows / max(output.total_windows, 1) * 100
    return (
        f"The model flagged {output.fault_windows:,} of "
        f"{output.total_windows:,} windows ({pct:.1f}%) as anomalous, "
        f"meaning the predicted probability of an anomaly within the "
        f"next {settings.horizon} seconds exceeded the alert threshold "
        f"of {threshold:.0%}. The peak probability of "
        f"{output.peak_prob:.0%} occurred at window index "
        f"{output.peak_idx}. Periods that exceed the threshold are "
        f"shaded in the chart above; those are the time ranges most "
        f"worth a human review."
    )


def _build_styles():
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title", parent=base["Title"], fontSize=22,
            textColor=colors.HexColor("#0d0e10"),
            alignment=0, spaceAfter=6, leading=26,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=15,
            textColor=colors.HexColor("#0d0e10"),
            spaceAfter=6, leading=18,
        ),
        "H3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontSize=11,
            textColor=colors.HexColor("#0d0e10"),
            spaceAfter=4, leading=14,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontSize=10, leading=14,
        ),
        "BodyMuted": ParagraphStyle(
            "BodyMuted", parent=base["BodyText"], fontSize=10,
            leading=14, textColor=colors.HexColor("#5b6472"),
        ),
    }


def _kv_table(rows: list[list[str]]) -> Table:
    t = Table(rows, colWidths=[5.5 * cm, 11.5 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR",  (0, 0), (0, -1), colors.HexColor("#5b6472")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW",  (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e7eb")),
    ]))
    return t


__all__ = ["build_pdf", "PageBreak"]
