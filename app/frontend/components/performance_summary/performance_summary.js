/*
  

  This file handles the swap-out of metric numbers when the user
  clicks a different model card on the upload page.

  How it works:
    1. We keep a small lookup table (MODEL_PERFORMANCE) with the real
       test-set scores from my SKAB training runs.
    2. We hook into the existing selectModel() function on the page
       (without modifying it directly) and trigger an update each time
       a model card is clicked.
    3. If the selected model isn't one of mine (LR / KNN / SVM), we
       just show em-dashes so the panel stays consistent.

  Numbers come from:
    - results/lr_parinitha_dataset2.txt
    - results/knn_parinitha_dataset2.txt
    - results/svm_parinitha_dataset2.txt
*/


/* ─── lookup table for the three models I trained ─────────────── */
// keyed exactly by the .mc-name text on the upload page model cards
const MODEL_PERFORMANCE = {
  "Logistic Reg.": { recall: 77.98, precision: 99.51, f1: 87.44 },
  "KNN":           { recall: 54.41, precision: 95.48, f1: 69.32 },
  "SVM":           { recall: 83.39, precision: 98.93, f1: 90.50 }
};


/* ─── helper: format a percentage nicely with the small "%" unit ── */
function formatMetric(value) {
  // null / undefined safety - just in case the lookup misses
  if (value === null || value === undefined) {
    return "—";
  }
  return value.toFixed(2) + '<span class="unit">%</span>';
}


/* ─── main update function ──────────────────────────────────── */
// reads which model card has the .selected class, looks it up,
// and updates the three metric tiles + the active-model badge
function updatePerformancePanel() {
  const selectedCard = document.querySelector(".model-card.selected .mc-name");
  if (!selectedCard) {
    // shouldn't happen but be safe
    return;
  }

  const modelName = selectedCard.textContent.trim();
  const metrics   = MODEL_PERFORMANCE[modelName];

  // always update the badge so the user sees what's currently active
  document.getElementById("perfActiveModel").textContent = modelName;

  if (metrics) {
    // we have real numbers for this model - show them
    document.getElementById("perfRecall").innerHTML    = formatMetric(metrics.recall);
    document.getElementById("perfPrecision").innerHTML = formatMetric(metrics.precision);
    document.getElementById("perfF1").innerHTML        = formatMetric(metrics.f1);
  } else {
    // no classical-ML scores for this model (e.g. Transformer, XGBoost)
    // show em-dashes so the layout stays the same
    document.getElementById("perfRecall").innerHTML    = "—";
    document.getElementById("perfPrecision").innerHTML = "—";
    document.getElementById("perfF1").innerHTML        = "—";
  }
}


/* ─── hook into the existing selectModel function ────────────── */
// the static landing page already has a selectModel() defined.
// instead of editing it (which would create merge conflicts later),
// we wrap it so our update fires after every selection.
const originalSelectModel = window.selectModel;

window.selectModel = function (card) {
  if (typeof originalSelectModel === "function") {
    originalSelectModel(card);
  }
  updatePerformancePanel();
};


/* ─── first paint when the page loads ──────────────────────── */
// run once on page load so the panel shows the default selected model
// (which is SVM in the current static page)
document.addEventListener("DOMContentLoaded", updatePerformancePanel);