from fastapi import FastAPI, UploadFile, HTTPException, File
from trism import TritonModel  # Replace with correct import if `trism` is a typo
import numpy as np
from PIL import Image
import io
import logging
import uvicorn
import time

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Cấu hình Triton Server
TRITON_SERVER_URL = "localhost:8002"  # Correct gRPC port based on logs
MODEL_NAMES = [
    # TensorRT FP16
    "best_b_plan", "last_b_plan", "best_s_plan", "last_s_plan",
    # TensorRT Int8
    "best_b_plan_int8", "last_b_plan_int8", "best_s_plan_int8", "last_s_plan_int8",
    # ONNX FP16
    "best_b_onnx_fp16", "last_b_onnx_fp16", "best_s_onnx_fp16", "last_s_onnx_fp16",
    # ONNX Int8
    "best_b_onnx_int8", "last_b_onnx_int8", "best_s_onnx_int8", "last_s_onnx_int8"
]
MODEL_VERSION = 1

# Retry logic for connecting to Triton
def connect_to_triton(model_name, retries=5, delay=2):
    for attempt in range(retries):
        try:
            model = TritonModel(
                model=model_name,
                version=MODEL_VERSION,
                url=TRITON_SERVER_URL,
                grpc=True
            )
            logger.info(f"Connected to Triton Inference Server at {TRITON_SERVER_URL} for model {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed to connect to Triton: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"Failed to connect to Triton Server after {retries} attempts")
    raise Exception("Unable to connect to Triton Server")

# Initialize first model with retries
try:
    model = connect_to_triton(MODEL_NAMES[0])
except Exception as e:
    logger.error(f"Startup failed: {e}")
    raise SystemExit(1)

# Xử lý ảnh
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((640, 640))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
        return img_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format or processing error.")

@app.get("/")
def index():
    return {"message": "YOLO Triton Model API (TensorRT & ONNX)"}

@app.get("/health")
def health_check():
    try:
        status = {}
        for model_name in MODEL_NAMES:
            try:
                model = connect_to_triton(model_name)
                status[model_name] = {
                    "ready": True,
                    "inputs": [{"name": inp.name, "shape": inp.shape, "dtype": inp.dtype} for inp in model.inputs],
                    "outputs": [{"name": out.name, "shape": out.shape, "dtype": out.dtype} for out in model.outputs]
                }
            except Exception:
                status[model_name] = {"ready": False}
        return {
            "status": "healthy",
            "triton_server": "live",
            "models": status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/predict/{model_name}")
async def predict_yolo(model_name: str, file: UploadFile = File(...)):
    if model_name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available. Choose from {MODEL_NAMES}")
    
    try:
        image_bytes = await file.read()
        logger.info(f"Received prediction request for file: {file.filename} with model: {model_name}")
        img_array = preprocess_image(image_bytes)
        
        # Khởi tạo model với retry
        model = connect_to_triton(model_name)
        
        # Chạy inference
        outputs = model.run(data=[img_array])
        
        # Xử lý outputs (TensorRT có 2 outputs, ONNX có 1)
        if "plan" in model_name:  # TensorRT models
            detections_1 = list(outputs.values())[0]
            detections_2 = list(outputs.values())[1]
            logger.info(f"✅ Inference completed with TensorRT model {model_name}, shapes: {detections_1.shape}, {detections_2.shape}")
            return {
                "model_name": model_name,
                "detections_1": detections_1.tolist(),
                "detections_2": detections_2.tolist()
            }
        else:  # ONNX models
            detections = list(outputs.values())[0]
            logger.info(f"✅ Inference completed with ONNX model {model_name}, detections shape: {detections.shape}")
            return {
                "model_name": model_name,
                "detections": detections.tolist()
            }
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        # Fallback sang ONNX FP16 nếu TensorRT lỗi
        if "plan" in model_name:
            fallback_model = model_name.replace("_plan", "_onnx_fp16")
            if fallback_model in MODEL_NAMES:
                logger.info(f"Attempting fallback to ONNX model: {fallback_model}")
                try:
                    fallback_model_instance = connect_to_triton(fallback_model)
                    outputs = fallback_model_instance.run(data=[img_array])
                    detections = list(outputs.values())[0]
                    logger.info(f"✅ Fallback inference completed with {fallback_model}")
                    return {
                        "model_name": fallback_model,
                        "detections": detections.tolist()
                    }
                except Exception as fallback_e:
                    logger.error(f"❌ Fallback failed: {str(fallback_e)}")
                    raise HTTPException(status_code=500, detail=str(fallback_e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)