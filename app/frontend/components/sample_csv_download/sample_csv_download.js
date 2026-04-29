


/*
  downloadSampleCSV - called by the onclick on the download button.
  Fetches the sample file and triggers a browser download.
*/
function downloadSampleCSV() {
  const btn      = document.getElementById("sdcDownloadBtn");
  const filePath = "components/sample_csv_download/sample_skab_format.csv";
  const fileName = "sample_skab_format.csv";

  // give the user some feedback that something is happening
  const originalText = btn.innerHTML;
  btn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <circle cx="12" cy="12" r="10"/>
    </svg>
    Downloading...
  `;
  btn.disabled = true;

  fetch(filePath)
    .then(response => {
      if (!response.ok) {
        throw new Error("File not found");
      }
      return response.blob();
    })
    .then(blob => {
      // create a temporary invisible link and click it to trigger download
      const url  = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href     = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      // restore button after a short delay
      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.disabled  = false;
      }, 1500);
    })
    .catch(() => {
      // if fetch fails (e.g. running locally without a server),
      // fall back to a direct anchor download
      const link    = document.createElement("a");
      link.href     = filePath;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      btn.innerHTML = originalText;
      btn.disabled  = false;
    });
}

// expose to global so the onclick in the HTML can find it
window.downloadSampleCSV = downloadSampleCSV;