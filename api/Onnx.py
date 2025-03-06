from fastapi import FastAPI, UploadFile, HTTPException, File
from trism import TritonModel
import numpy as np
from PIL import Image
import io
import logging
import uvicorn

#logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) #ghi log trong code

app = FastAPI()

#cau hinh triton server
TRITON_SERVER_URL = "localhost:8001" #port GRPC
MODEL_NAMES = ["best_b_onnx","last_b_onnx","best_s_onnx","last_s_onnx"]
MODEL_VERSION = 1

#ket noi voi triton
try:
    model = TritonModel(
        model = MODEL_NAMES[0],
        version = MODEL_VERSION,
        url=TRITON_SERVER_URL,
        grpc = True
    )
    logger.info(f"Connected to Triton Inference Server at {TRITON_SERVER_URL}")
except Exception as e:
    logger.error(f"Failed to connect to Triton Server:{e}")
    raise SystemExit(1)

#xuly anh
def preprocess_image(image_bytes:bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB") #chuyen anh sang RGB
        image = image.resize((640,640)) #resize ve kich thuoc yolo yc
        img_array = np.array(image,dtype=np.float32) / 255.0 # chuyen thanh mang numpy
        img_array = img_array.transpose(2,0,1) # HWC -> CHW
        img_array = np.expand_dims(img_array,axis = 0) #them chieu batch
        return img_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format or processing error.")

@app.get("/")
def index():
    return {"message":"YOLO ONNX Triton Model API"}

@app.get("/health")
def health_check():
    try:
        status = {}
        for model_name in MODEL_NAMES:
            try:
                model = TritonModel(
                    model = model_name,
                    version=MODEL_VERSION,
                    url=TRITON_SERVER_URL,
                    grpc= True
                )
                model_metadata = model.metadata
                status[model_name] = {
                    "ready":True,
                    "inputs": [{"name": inp.name, "shape": inp.shape, "dtype": inp.dtype} for inp in model.inputs],
                    "outputs": [{"name": out.name, "shape": out.shape, "dtype": out.dtype} for out in model.outputs]
                }
            except Exception:
                status[model_name] = {"ready":False}
        return{
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
        model = TritonModel(
            model = model_name,
            version = MODEL_VERSION,
            url=TRITON_SERVER_URL,
            grpc=True
        )
        outputs = model.run(data=[img_array])
        detections = list(outputs.values())[0]
        logger.info(f"✅ Inference completed with model {model_name}, detections shape: {detections.shape}")

        return {
            "model_name":model_name,
            "detections":detections.tolist()
        }
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7000)

##uvicorn Onnx:app --reload --port 7000