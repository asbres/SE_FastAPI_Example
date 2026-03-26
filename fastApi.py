from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from transformers import pipeline
from PIL import Image
from functools import lru_cache
import io

app = FastAPI()


@lru_cache()
def get_pipeline():
    return pipeline(
        "image-segmentation",
        model="briaai/RMBG-1.4",
        trust_remote_code=True,
    )


def read_image(upload_file: UploadFile) -> Image.Image:

    if upload_file.content_type not in (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только изображения: jpg, jpeg, png, webp",
        )

    image_bytes = upload_file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой или повреждён")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
    return image


@app.post("/remove-background")
def remove_background(file: UploadFile = File(...)):
    image = read_image(file)
    pipe = get_pipeline()

    pillow_image = pipe(image)

    buf = io.BytesIO()
    pillow_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="result.png"'},
    )


@app.post("/get-mask")
def get_mask(file: UploadFile = File(...)):

    image = read_image(file)
    pipe = get_pipeline()

    pillow_mask = pipe(image, return_mask=True)

    buf = io.BytesIO()
    pillow_mask.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="mask.png"'},
    )