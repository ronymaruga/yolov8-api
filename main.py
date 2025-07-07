from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
from pydantic import BaseModel
import os
import urllib.request

app = FastAPI(
    title="YOLOv8 Security Camera API",
    description="API for object detection in security camera images using YOLOv8",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", model_path)
model = YOLO(model_path)

class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

class DetectionResponse(BaseModel):
    detections: List[Detection]
    image_width: int
    image_height: int

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/detect/", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        return DetectionResponse(detections=[], image_width=0, image_height=0)
    
    image_height, image_width = image.shape[:2]

    results = model.predict(image)
    detections = []
    if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            confidence = result.conf[0]
            class_id = result.cls[0]
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=model.names[int(class_id)]
                )
            )

    return DetectionResponse(
        detections=detections,
        image_width=image_width,
        image_height=image_height
    )