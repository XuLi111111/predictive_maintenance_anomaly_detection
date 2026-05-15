import uuid
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.upload import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile):
    """
    FR-01/02/03/04/07/21
    Accepts a CSV file, checks it looks like valid SKAB sensor data,
    saves it to a temp folder and returns a file_id for downstream use.
    """

    # only accept csv files, reject anything else straight away
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted. Please upload a .csv file.",
        )

    # read the raw bytes so we can check size and parse it
    contents = await file.read()

    # reject files that are too large to process sensibly
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum allowed size is {settings.max_upload_mb} MB.",
        )

    # try to parse as a dataframe
    # SKAB raw files are semicolon separated, so we try comma first
    # and fall back to semicolon if we only get one column back
    try:
        df = pd.read_csv(BytesIO(contents), sep=",")
        if len(df.columns) == 1:
            # one column usually means the separator is actually a semicolon
            df = pd.read_csv(BytesIO(contents), sep=";")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the file as a CSV. Please check the file format.",
        )

    # no point continuing if the file has no rows
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty. Please upload a file with sensor data.",
        )

    # check all 8 required SKAB sensor columns are present
    # these column names must match exactly what the model was trained on
    missing_cols = [col for col in settings.sensor_columns if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The following required sensor columns are missing: {missing_cols}. "
                f"Expected columns are: {settings.sensor_columns}"
            ),
        )

    # make sure none of the sensor columns have gaps in the data
    # missing readings would cause the model to fail or produce unreliable results
    null_counts = df[settings.sensor_columns].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing values found in these sensor columns: {dict(cols_with_nulls)}. "
                f"All sensor readings must be present."
            ),
        )

    # sensor columns should all be numbers
    # if anything is text or mixed type the scaler will break downstream
    non_numeric_cols = [
        col for col in settings.sensor_columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                f"These sensor columns contain non-numeric values: {non_numeric_cols}. "
                f"All sensor readings must be numeric."
            ),
        )

    # check if the anomaly label column is present
    # this is optional — files without labels can still be used for prediction
    has_label = settings.label_column in df.columns

    # save the file to the temp directory with a unique id
    # downstream endpoints (predict, compare) will use this file_id to load the data
    file_id = str(uuid.uuid4())
    save_path = settings.upload_tmp_dir / f"{file_id}.csv"
    save_path.write_bytes(contents)

    # return the file metadata so the frontend knows the upload worked
    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        rows=len(df),
        columns=list(df.columns),
        has_label=has_label,
        message="File uploaded and validated successfully.",
    )


@router.get("/sample-csv")
def sample_csv():
    """
    FR-05
    Returns a small sample SKAB CSV file so users can see
    the expected format before uploading their own data.
    """

    # the sample file lives in the artifacts folder alongside the trained models
    sample_path = settings.artifact_dir / "sample_skab.csv"

    if not sample_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Sample CSV file not found. Please contact the team.",
        )

    return FileResponse(
        path=str(sample_path),
        filename="sample_skab.csv",
        media_type="text/csv",
    )