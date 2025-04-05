import io
import numpy as np
import cv2
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import StreamingResponse
from trism import TritonModel  
import logging
import uvicorn
import time
import matplotlib.pyplot as plt

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Cấu hình Triton Server
TRITON_SERVER_URL = "localhost:8002"  # Chỉnh sửa cổng nếu cần
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

# Hàm softmax cho vector 1 chiều
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

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

# Initialize first model with retries (dùng cho health check)
try:
    model = connect_to_triton(MODEL_NAMES[0])
except Exception as e:
    logger.error(f"Startup failed: {e}")
    raise SystemExit(1)

# Hàm letterbox_image: resize ảnh sang kích thước cố định (640×640) mà giữ tỷ lệ, thêm padding
def letterbox_image(image, new_shape=(640, 640), color=(114, 114, 114)):
    orig_w, orig_h = image.size
    new_w, new_h = new_shape
    scale = min(new_w / orig_w, new_h / orig_h)
    resize_w = int(orig_w * scale)
    resize_h = int(orig_h * scale)
    image_resized = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)
    # Tạo canvas mới với kích thước cố định và màu nền
    canvas = Image.new("RGB", new_shape, color)
    pad_x = (new_w - resize_w) // 2
    pad_y = (new_h - resize_h) // 2
    canvas.paste(image_resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y

# Hàm tiền xử lý ảnh: sử dụng letterbox để resize ảnh nhưng giữ tỷ lệ ban đầu
def preprocess_image(image_bytes: bytes, model_name: str):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size
        # Resize ảnh theo letterbox: kích thước cố định INPUT_SIZE x INPUT_SIZE
        img_letterboxed, scale, pad_x, pad_y = letterbox_image(image, new_shape=(640, 640))
        # Chuẩn hóa ảnh: chuyển pixel sang khoảng [-1, 1]
        img_np = (np.array(img_letterboxed, dtype=np.float32) / 255.0) * 2 - 1
        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
        return image, img_np, (orig_w, orig_h), (scale, pad_x, pad_y)
    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format or processing error.")

# Hàm đọc ảnh gốc bằng OpenCV để vẽ bounding box
def read_image_cv2(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image

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
        _, img_array, _, _ = preprocess_image(image_bytes, model_name)
        
        model = connect_to_triton(model_name)
        outputs = model.run(data=[img_array])
        
        if "plan" in model_name:  # TensorRT models
            detections_1 = list(outputs.values())[0]
            detections_2 = list(outputs.values())[1]
            logger.info(f"Inference completed with TensorRT model {model_name}, shapes: {detections_1.shape}, {detections_2.shape}")
            return {
                "model_name": model_name,
                "detections_1": detections_1.tolist(),
                "detections_2": detections_2.tolist()
            }
        else:  # ONNX models
            detections = list(outputs.values())[0]
            logger.info(f"Inference completed with ONNX model {model_name}, detections shape: {detections.shape}")
            return {
                "model_name": model_name,
                "detections": detections.tolist()
            }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        if "plan" in model_name:
            fallback_model = model_name.replace("_plan", "_onnx_fp16")
            if fallback_model in MODEL_NAMES:
                logger.info(f"Attempting fallback to ONNX model: {fallback_model}")
                try:
                    fallback_model_instance = connect_to_triton(fallback_model)
                    outputs = fallback_model_instance.run(data=[img_array])
                    detections = list(outputs.values())[0]
                    logger.info(f"Fallback inference completed with {fallback_model}")
                    return {
                        "model_name": fallback_model,
                        "detections": detections.tolist()
                    }
                except Exception as fallback_e:
                    logger.error(f"Fallback failed: {str(fallback_e)}")
                    raise HTTPException(status_code=500, detail=str(fallback_e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_draw/{model_name}")
async def predict_draw(model_name: str, file: UploadFile = File(...)):
    if model_name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available. Choose from {MODEL_NAMES}")
    
    try:
        image_bytes = await file.read()
        logger.info(f"Received predict_draw request for file: {file.filename} with model: {model_name}")
        
        # Tiền xử lý ảnh với letterbox, lấy ảnh gốc, input, kích thước và thông tin letterbox
        orig_image_pil, img_array, (orig_w, orig_h), (scale, pad_x, pad_y) = preprocess_image(image_bytes, model_name)
        # Đọc ảnh gốc bằng OpenCV để vẽ (để giữ màu sắc chuẩn)
        orig_image = read_image_cv2(image_bytes)
        
        model = connect_to_triton(model_name)
        outputs = model.run(data=[img_array])
        
        # Nếu model là TensorRT ("plan"), sử dụng fallback ONNX FP16 cho việc vẽ
        if "plan" in model_name:
            fallback_model = model_name.replace("_plan", "_onnx_fp16")
            logger.info(f"Using fallback ONNX model: {fallback_model} for drawing detections")
            model = connect_to_triton(fallback_model)
            outputs = model.run(data=[img_array])
        
        # Lấy output detections – giả sử shape: (1,7,8400)
        detections = list(outputs.values())[0]
        logger.info(f"Detections shape (raw): {detections.shape}")
        detections = np.squeeze(detections, axis=0)  # từ (1,7,8400) thành (7,8400)
        detections = detections.transpose()           # thành (8400,7)
        logger.info(f"Detections shape (processed): {detections.shape}")
        
        boxes = []
        confidences = []
        class_ids = []
        CONF_THRESHOLD = 0.5
        
        # Các tọa độ (cx, cy, w, h) ở không gian letterboxed (640x640)
        # Để chuyển về tọa độ ảnh gốc:
        # x_orig = (x_letter - pad_x) / scale, y_orig = (y_letter - pad_y) / scale, tương tự cho w và h.
        for detection in detections:
            cx, cy, bw, bh = detection[:4]
            class_logits = detection[4:7]
            probs = softmax(class_logits)
            final_conf = np.max(probs)
            cls_id = int(np.argmax(probs))
            
            if final_conf < CONF_THRESHOLD:
                continue
            
            # Tọa độ trên ảnh letterboxed
            x_letter = cx - bw / 2
            y_letter = cy - bh / 2
            
            # Chuyển sang tọa độ ảnh gốc
            x_orig = int((x_letter - pad_x) / scale)
            y_orig = int((y_letter - pad_y) / scale)
            w_orig = int(bw / scale)
            h_orig = int(bh / scale)
            
            boxes.append([x_orig, y_orig, w_orig, h_orig])
            confidences.append(final_conf)
            class_ids.append(cls_id)
        
        logger.info(f"Number of detections before NMS: {len(boxes)}")
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.45)
        if indices is not None and len(indices) > 0:
            indices = indices.flatten()
        else:
            indices = []
        logger.info(f"Number of final bounding boxes: {len(indices)}")
        
        # Định nghĩa label và màu sắc cho 3 lớp: body, head, face
        labels_list = ["body", "head", "face"]
        colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        for i in indices:
            x, y, w_box, h_box = boxes[i]
            conf_percent = int(confidences[i] * 100)
            lbl = labels_list[class_ids[i]]
            color = colors_list[class_ids[i]]
            text = f"{lbl}: {conf_percent}%"
            cv2.rectangle(orig_image, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.rectangle(orig_image, (x, y - 20), (x + w_box, y), color, -1)
            cv2.putText(orig_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        is_success, buffer = cv2.imencode(".jpg", orig_image)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode output image")
        io_buf = io.BytesIO(buffer.tobytes())
        headers = {"Content-Disposition": f"attachment; filename=result_{model_name}.jpg"}
        return StreamingResponse(io_buf, media_type="image/jpeg", headers=headers)
        
    except Exception as e:
        logger.error(f"Error during predict_draw: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
