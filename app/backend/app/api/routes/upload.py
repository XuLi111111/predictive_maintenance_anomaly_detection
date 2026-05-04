from fastapi import APIRouter, HTTPException, UploadFile

router = APIRouter()


@router.post("/upload")
async def upload_csv(file: UploadFile):
    """FR-01/02/03/04/07/21 — validate SKAB schema, persist to tmp, return file_id.

    TODO (teammate): implement schema validation against settings.sensor_columns,
    detect has_label, persist file under settings.upload_tmp_dir, return metadata.
    """
    raise HTTPException(status_code=501, detail="upload endpoint not implemented")


@router.get("/sample-csv")
def sample_csv():
    """FR-05 — return a downloadable sample SKAB-format CSV.

    TODO (teammate): stream a small valid SKAB CSV from artifacts/.
    """
    raise HTTPException(status_code=501, detail="sample-csv endpoint not implemented")
