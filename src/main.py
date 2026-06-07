from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    results = model(img)[0]
    
    detections = []
    for box in results.boxes:
        detections.append({
            "class": "garbage",
            "confidence": round(float(box.conf), 3),
            "bbox": {
                "x1": round(box.xyxy[0][0].item(), 2),
                "y1": round(box.xyxy[0][1].item(), 2),
                "x2": round(box.xyxy[0][2].item(), 2),
                "y2": round(box.xyxy[0][3].item(), 2)
            }
        })
    
    return {
        "detections": detections,
        "count": len(detections)
    }

@app.get("/")
def root():
    return {"message": "Garbage Detection API running"}