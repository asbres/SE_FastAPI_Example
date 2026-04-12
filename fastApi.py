from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from transformers import pipeline
from PIL import Image
from functools import lru_cache
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@lru_cache()
def get_pipeline():
    return pipeline(
        "image-segmentation",
        model="briaai/RMBG-1.4",
        trust_remote_code=True,
    )


def read_image(upload_file: UploadFile) -> Image.Image:
    if upload_file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только изображения: jpg, jpeg, png, webp",
        )

    image_bytes = upload_file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой или повреждён")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    image = read_image(file)
    pipe = get_pipeline()

    result = pipe(image)
    if isinstance(result, list):
        result = result[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="result.png"'},
    )


@app.post("/get-mask")
async def get_mask(file: UploadFile = File(...)):
    image = read_image(file)
    pipe = get_pipeline()

    mask = pipe(image, return_mask=True)
    if isinstance(mask, list):
        mask = mask[0]

    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="mask.png"'},
    )


@app.get("/health")
async def health_check():
    try:
        pipe = get_pipeline()
        return {"status": "ok", "model_loaded": pipe is not None}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Model not available")