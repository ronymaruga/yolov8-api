from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
from starlette.responses import Response

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="API for real-time object detection using YOLOv8",
    version="0.0.1"
)

# Configure CORS to allow frontend access (e.g., for React integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read and process the uploaded image
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection
    results = model.predict(image)

    # Process detection results
    detections = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        confidence = result.conf[0]
        class_id = result.cls[0]
        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(confidence),
            "class_id": int(class_id),
            "class_name": model.names[int(class_id)]
        })

    # Optionally, render the image with bounding boxes
    results.render()  # Updates results with boxes and labels
    img = results[0].plot()  # Get annotated image
    img_pil = Image.fromarray(img)
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format="JPEG")

    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")