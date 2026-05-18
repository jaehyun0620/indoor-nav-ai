FROM python:3.11-slim

WORKDIR /app

# Railway 배포 기본값: torch/transformers depth 추론을 피하고 YOLO-only로 실행.
# 로컬 또는 별도 배포에서 Depth Anything을 다시 쓰려면 FAST_MODE=full 로 오버라이드.
ENV FAST_MODE=yolo_only

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 1) numpy 먼저 고정 (다른 패키지가 2.x로 올리기 전에 선점)
RUN pip install --no-cache-dir "numpy==1.26.4"

# 2) torch CPU 전용
RUN pip install --no-cache-dir \
    "torch==2.4.0" "torchvision==0.19.0" \
    --index-url https://download.pytorch.org/whl/cpu

# 3) 나머지 패키지 (numpy 이미 설치됐으므로 재설치 안 됨)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 빌드 시점에 YOLO 모델 미리 다운로드 (런타임 다운로드 타임아웃 방지)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]
