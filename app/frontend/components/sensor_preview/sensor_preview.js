


// one colour per sensor line so they're easy to tell apart
const SENSOR_COLORS = [
  "#1d4ed8",  // blue   - Accelerometer1RMS
  "#7c3aed",  // purple - Accelerometer2RMS
  "#b45309",  // amber  - Current
  "#15803d",  // green  - Pressure
  "#dc2626",  // red    - Temperature
  "#0891b2",  // cyan   - Thermocouple
  "#9333ea",  // violet - Voltage
  "#d97706",  // orange - Volume Flow RateRMS
];

// hold a reference to the chart so we can destroy it before rebuilding
let sensorChartInstance = null;


// parse a semicolon-delimited SKAB CSV string into headers and rows
function parseSkabCSV(text) {
  const lines = text.trim().split("\n").filter(l => l.trim());
  if (lines.length < 2) return null;

  // handle both semicolon and comma just in case
  const delimiter  = lines[0].includes(";") ? ";" : ",";
  const headers    = lines[0].split(delimiter).map(h => h.trim());

  // only keep the actual sensor columns, skip metadata columns
  const skipCols   = ["datetime", "anomaly", "changepoint"];
  const sensorCols = headers.filter(h => !skipCols.includes(h));

  const rows = lines.slice(1).map(line => {
    const vals = line.split(delimiter);
    const row  = {};
    headers.forEach((h, i) => { row[h] = vals[i] ? vals[i].trim() : ""; });
    return row;
  });

  return { headers, sensorCols, rows };
}


// build and render the Chart.js chart from the parsed data
function renderSensorChart(parsed) {
  const { sensorCols, rows } = parsed;

  // use the time portion of the datetime column as x-axis labels
  const labels = rows.map((row, i) => {
    if (row["datetime"]) {
      const parts = row["datetime"].split(" ");
      return parts[1] ? parts[1].substring(0, 8) : String(i);
    }
    return String(i);
  });

  // show every Nth label to avoid crowding the x-axis
  const step         = Math.max(1, Math.floor(labels.length / 10));
  const sparseLabels = labels.map((l, i) => i % step === 0 ? l : "");

  // one dataset per sensor column
  const datasets = sensorCols.map((col, idx) => ({
    label:           col,
    data:            rows.map(row => parseFloat(row[col]) || 0),
    borderColor:     SENSOR_COLORS[idx % SENSOR_COLORS.length],
    backgroundColor: "transparent",
    borderWidth:     1.5,
    pointRadius:     0,     // no dots - too many data points to show them
    tension:         0.3,   // slight curve makes the lines easier to read
  }));

  // destroy old chart first so we don't stack them on top of each other
  if (sensorChartInstance) {
    sensorChartInstance.destroy();
    sensorChartInstance = null;
  }

  const ctx = document.getElementById("sensorChart").getContext("2d");

  sensorChartInstance = new Chart(ctx, {
    type: "line",
    data: { labels: sparseLabels, datasets },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 400 },
      plugins: {
        legend: {
          display:  true,
          position: "bottom",
          labels: {
            font:      { family: "'JetBrains Mono', monospace", size: 9 },
            color:     "#6b7280",
            boxWidth:  12,
            padding:   12,
          }
        },
        tooltip: { mode: "index", intersect: false }
      },
      scales: {
        x: {
          ticks: {
            font:        { family: "'JetBrains Mono', monospace", size: 8 },
            color:       "#9ca3af",
            maxRotation: 0,
          },
          grid: { color: "rgba(0,0,0,0.05)" }
        },
        y: {
          ticks: {
            font:  { family: "'JetBrains Mono', monospace", size: 8 },
            color: "#9ca3af",
          },
          grid: { color: "rgba(0,0,0,0.05)" }
        }
      }
    }
  });
}


// read the uploaded file and show the chart
function showSensorPreview(file) {
  const reader = new FileReader();

  reader.onload = function (e) {
    const parsed = parseSkabCSV(e.target.result);

    if (!parsed || parsed.rows.length === 0) {
      // couldn't parse the file - just hide the preview quietly
      document.getElementById("sensorPreviewWrap").style.display = "none";
      return;
    }

    // update the badge with real counts from the file
    document.getElementById("spBadge").textContent =
      `${parsed.sensorCols.length} sensors · ${parsed.rows.length} rows`;

    // show the card and draw the chart
    document.getElementById("sensorPreviewWrap").style.display = "block";
    renderSensorChart(parsed);
  };

  reader.readAsText(file);
}


// wrap the existing handleFile() so our preview fires after every upload
// without touching the original function
const _originalHandleFile = window.handleFile;

window.handleFile = function (input) {
  if (typeof _originalHandleFile === "function") {
    _originalHandleFile(input);
  }
  const file = input.files[0];
  if (file) showSensorPreview(file);
};


// also hide the preview when the file is cleared
const _originalClearFile = window.clearFile;

window.clearFile = function () {
  if (typeof _originalClearFile === "function") {
    _originalClearFile();
  }
  document.getElementById("sensorPreviewWrap").style.display = "none";
  if (sensorChartInstance) {
    sensorChartInstance.destroy();
    sensorChartInstance = null;
  }
};