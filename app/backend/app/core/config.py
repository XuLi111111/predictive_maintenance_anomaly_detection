import os
from pathlib import Path


class Settings:
    """Runtime configuration. Override via environment variables."""

    artifact_dir: Path = Path(
        os.getenv(
            "ARTIFACT_DIR",
            str(Path(__file__).resolve().parents[2] / "artifacts"),
        )
    )
    upload_tmp_dir: Path = Path(os.getenv("UPLOAD_TMP_DIR", "/tmp/pump_detect_uploads"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))

    # FR-10: must match training-time configuration
    window_size: int = int(os.getenv("WINDOW_SIZE", "20"))
    horizon: int = int(os.getenv("HORIZON", "10"))

    # FR-16: default alert threshold, configurable
    alert_threshold: float = float(os.getenv("ALERT_THRESHOLD", "0.5"))

    # SKAB-trained 8 sensor columns (order matters - matches scaler.pkl)
    sensor_columns: list[str] = [
        "Accelerometer1RMS",
        "Accelerometer2RMS",
        "Current",
        "Pressure",
        "Temperature",
        "Thermocouple",
        "Voltage",
        "Volume Flow RateRMS",
    ]
    timestamp_column: str = "datetime"
    label_column: str = "anomaly"

    cors_origins: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost"
    ).split(",")


settings = Settings()
settings.upload_tmp_dir.mkdir(parents=True, exist_ok=True)
