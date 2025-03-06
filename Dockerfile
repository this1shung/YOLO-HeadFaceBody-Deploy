FROM hieupth/tritonserver:24.08

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt các công cụ hệ thống
RUN apt-get update && apt-get install -y python3-pip curl && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python (không bao gồm numpy)
RUN pip3 install --no-cache-dir \
    fastapi==0.95.0 \
    uvicorn==0.21.1 \
    pillow==10.2.0 \
    trism \
    tritonclient[all] \
    attrdict \
    python-multipart

# Cài đặt numpy 2.2.3 SAU CÙNG để tránh bị ghi đè
RUN pip3 install --no-cache-dir --force-reinstall numpy==2.2.3

# Copy toàn bộ thư mục models
COPY ./models /models

# Copy code FastAPI
COPY ./api/main.py /app/main.py

# Copy startup script
COPY ./start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports (FastAPI on 8000, Triton HTTP on 8001, gRPC on 8002)
EXPOSE 8000 8001 8002

# Chạy script khởi động
CMD ["/app/start.sh"]
