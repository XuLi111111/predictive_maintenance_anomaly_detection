



/* severity thresholds - easy to adjust if needed */
const SEVERITY_THRESHOLDS = {
  amber: 0.10   // below this = amber, at or above = red
};

/* status text for each severity level */
const STATUS_TEXT = {
  green: "All clear",
  amber: "Anomalies detected",
  red:   "High anomaly rate"
};


/* work out green / amber / red from the fault ratio */
function severityFor(faultCount, totalWindows) {
  if (totalWindows <= 0) return "green";
  if (faultCount === 0)  return "green";
  const ratio = faultCount / totalWindows;
  return ratio < SEVERITY_THRESHOLDS.amber ? "amber" : "red";
}


/* pull HH:MM:SS out of whatever timestamp format comes back */
function formatFirstFaultTime(timestamp) {
  if (!timestamp) return "—";
  // if it already looks like a time, return as-is
  if (timestamp.length <= 8) return timestamp;
  // otherwise grab the time portion from an ISO or datetime string
  const timePart = timestamp.split("T")[1] || timestamp.split(" ")[1];
  return timePart ? timePart.substring(0, 8) : timestamp;
}


/*
  updateFaultOverview - call this after a prediction run completes.

  summary = {
    faultCount:   number  - total fault windows detected
    totalWindows: number  - total windows in the run
    firstFaultAt: string  - ISO or HH:MM:SS of first fault (optional)
  }
*/
function updateFaultOverview(summary) {
  const panel = document.getElementById("faultOverview");
  if (!panel) return;

  const { faultCount = 0, totalWindows = 0, firstFaultAt = null } = summary || {};
  const severity = severityFor(faultCount, totalWindows);
  const ratio    = totalWindows > 0 ? faultCount / totalWindows : 0;
  const ratioPct = (ratio * 100).toFixed(1);

  // swap colour state - CSS handles the rest via data-severity
  panel.setAttribute("data-severity", severity);

  // update headline number and "out of N" line
  document.getElementById("faultCount").textContent    = faultCount;
  document.getElementById("faultCountSub").textContent =
    `out of ${totalWindows.toLocaleString()} total windows analysed`;

  // update status pill text
  document.querySelector(".fault-status-text").textContent = STATUS_TEXT[severity];

  // show or hide the first-fault timestamp
  const firstTimeEl = document.getElementById("faultFirstTime");
  if (faultCount > 0 && firstFaultAt) {
    firstTimeEl.style.display = "";
    firstTimeEl.innerHTML = `First detected at <strong>${formatFirstFaultTime(firstFaultAt)}</strong>`;
  } else {
    firstTimeEl.style.display = "none";
  }

  // update density bar
  document.getElementById("faultDensityFill").style.width  = `${ratioPct}%`;
  document.getElementById("faultDensityValue").textContent = `${ratioPct}%`;
}


/* expose to global so run-prediction can call it */
window.updateFaultOverview = updateFaultOverview;


/* seed demo data on load so reviewers can see the panel in action */
document.addEventListener("DOMContentLoaded", function () {
  updateFaultOverview({
    faultCount:   12,
    totalWindows: 120,
    firstFaultAt: "00:14:33"
  });
});