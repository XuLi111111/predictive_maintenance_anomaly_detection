from fastapi import APIRouter

from app.inference.registry import MODEL_REGISTRY

router = APIRouter()


@router.get("/models")
def list_models():
    """FR-08 — list the 8 pre-trained models with headline metrics."""
    return [m.public_dict() for m in MODEL_REGISTRY.values()]
