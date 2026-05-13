"""
train.py — YOLOv8n 커스텀 파인튜닝 스크립트
============================================================
실행:
    cd /Users/jaehyun/ai_capstone/capstone
    python backend/training/train.py

학습 완료 후:
    backend/training/runs/detect/indoor_nav/weights/best.pt  ← 최종 모델
"""

from pathlib import Path
from ultralytics import YOLO

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
YAML_PATH  = BASE_DIR / "dataset.yaml"
RUNS_DIR   = BASE_DIR / "runs"

# ── 학습 설정 ─────────────────────────────────────────────────────────────────
CFG = dict(
    model   = "yolov8n.pt",      # 사전학습 가중치 (COCO → transfer learning)
    data    = str(YAML_PATH),
    epochs  = 100,               # 데이터 적으면 150까지 올려도 OK
    imgsz   = 640,
    batch   = 16,                # M5 메모리 여유 있으면 32로 올려도 됨
    device  = "mps",             # Apple Silicon GPU
    project = str(RUNS_DIR / "detect"),
    name    = "indoor_nav",
    exist_ok= True,

    # 오버피팅 방지 (데이터 적을 때 중요)
    dropout = 0.1,
    patience= 30,                # 30 에폭 개선 없으면 조기 종료

    # 데이터 증강 (촬영 데이터 적을 때 효과 큼)
    augment = True,
    flipud  = 0.0,               # 상하 반전 OFF (계단 방향이 달라지면 안 됨)
    fliplr  = 0.5,               # 좌우 반전 ON
    mosaic  = 0.8,               # 모자이크 증강
    hsv_h   = 0.02,              # 색조 변화
    hsv_s   = 0.5,               # 채도 변화
    hsv_v   = 0.4,               # 밝기 변화 (조도 다양성 보완)

    # 저장
    save_period = 10,            # 10 에폭마다 체크포인트 저장
    val     = True,
    plots   = True,              # 학습 곡선 자동 저장
)

# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8n 커스텀 파인튜닝 시작")
    print(f"  데이터셋: {YAML_PATH}")
    print(f"  에폭:     {CFG['epochs']}")
    print(f"  디바이스: {CFG['device']}")
    print("=" * 60)

    model = YOLO(CFG.pop("model"))
    results = model.train(**CFG)

    best = RUNS_DIR / "detect" / "indoor_nav" / "weights" / "best.pt"
    print("\n✅ 학습 완료!")
    print(f"   최적 모델: {best}")
    print("\n다음 단계: .env 파일에서 YOLO_MODEL 경로를 업데이트하세요")
    print(f"   YOLO_MODEL={best}")
