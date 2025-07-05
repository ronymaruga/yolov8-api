from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
from pydantic import BaseModel

app = FastAPI(
    title="YOLOv8 Security Camera API",
    description="API for object detection in security camera images using YOLOv8",
    version="0.0.1"
)

# Configure CORS for frontend or external client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Update path if model is stored elsewhere

# Pydantic model for structured JSON response
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
    # Read and process the uploaded image
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_height, image_width = image.shape[:2]

    # Perform object detection
    results = model.predict(image)

    # Extract detection results
    detections = []
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