from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from .models import YoloDetector


class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)


class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")


class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")


class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")


SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
}


app = FastAPI(title="T-Bank Logo Detection API", version="1.0.0")

# Global model
_detector: YoloDetector | None = None


def get_detector() -> YoloDetector:
    global _detector
    if _detector is None:
        _detector = YoloDetector(weights_path="models/best-yolov8n.pt", device=None, conf_threshold=0.25)
    return _detector


@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)) -> DetectionResponse:
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    try:
        data = await file.read()
        image = Image.open(BytesIO(data))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:  # safety
        raise HTTPException(status_code=500, detail="Failed to read image") from exc

    detector = get_detector()
    boxes = detector.predict(image)

    detections = [Detection(bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)) for x1, y1, x2, y2 in boxes]
    return DetectionResponse(detections=detections)


@app.get("/healthz")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
