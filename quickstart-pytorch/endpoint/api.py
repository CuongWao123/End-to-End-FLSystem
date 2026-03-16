"""FastAPI endpoint for model inference."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import torch
from PIL import Image
import io
import uvicorn

from engine.inference import ModelInference
from prometheus_client import make_asgi_app


app = FastAPI(
    title="Federated Learning Inference API",
    description="API for inference using federated learning models from MinIO",
    version="1.0.0"
)

# Global inference instance
inferencer = None

metrics_app = make_asgi_app()

app.mount("/metrics", metrics_app)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: int
    confidence: float
    all_probabilities: Dict[int, float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global inferencer
    try:
        print("Loading model from MinIO...")
        inferencer = ModelInference()
        inferencer.load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        inferencer = None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Federated Learning Inference API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch-predict (POST)",
            "model_info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if inferencer is not None else "model not loaded",
        model_loaded=inferencer is not None,
        device=str(inferencer.device) if inferencer else "unknown"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict single image.
    
    Args:
        file: Image file (PNG, JPG, etc.)
        
    Returns:
        Prediction result with class and confidence
    """
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        
        # Preprocess and predict
        image_tensor = inferencer.transform(image).unsqueeze(0).to(inferencer.device)
        
        with torch.no_grad():
            outputs = inferencer.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get all class probabilities
        all_probs = {i: float(prob) for i, prob in enumerate(probabilities[0].cpu().numpy())}
        
        return PredictionResponse(
            predicted_class=int(predicted.item()),
            confidence=float(confidence.item()),
            all_probabilities=all_probs
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": inferencer.model_path,
        "device": str(inferencer.device),
        "model_architecture": str(inferencer.model),
        "num_parameters": sum(p.numel() for p in inferencer.model.parameters())
    }


@app.post("/reload-model")
async def reload_model(model_path: str = None):
    """
    Reload model from MinIO.
    
    Args:
        model_path: Optional specific model path, otherwise loads latest
    """
    global inferencer
    
    try:
        inferencer = ModelInference(model_path=model_path)
        inferencer.load_model()
        return {"status": "success", "model_path": inferencer.model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)