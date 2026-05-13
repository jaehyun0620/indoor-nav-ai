"""
debug_yolo.py
YOLO 탐지 결과를 이미지에 직접 그려서 확인하는 시각화 스크립트.

현재 필터(클래스별 threshold, 최소 bbox 크기)가 적용된 결과와
필터 없는 원본 결과를 나란히 비교할 수 있다.

사용법:
    python debug_yolo.py                         # data/ 전체 이미지 처리
    python debug_yolo.py backend/data/img.jpg    # 특정 이미지 하나
    python debug_yolo.py --conf 0.3              # threshold 조정해서 비교

출력: debug_output/ 폴더에 annotated 이미지 저장
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", default="", help="이미지 경로 (생략 시 data/ 전체)")
parser.add_argument("--conf",      type=float, default=0.3,   help="기본 confidence threshold (기본 0.3)")
parser.add_argument("--conf-person", type=float, default=0.65, help="person 클래스 threshold (기본 0.65)")
parser.add_argument("--min-bbox",  type=float, default=0.015, help="최소 bbox 면적 비율 (기본 0.015)")
parser.add_argument("--no-filter", action="store_true",       help="필터 없이 원본 YOLO 결과만 표시")
parser.add_argument("--resize",    type=int, default=640,      help="YOLO 입력 전 리사이즈 너비 (기본 640, 실제 파이프라인과 동일)")
args = parser.parse_args()

# ── 출력 폴더 ─────────────────────────────────────────────────────────────────
OUT_DIR = Path(SCRIPT_DIR) / "debug_output"
OUT_DIR.mkdir(exist_ok=True)

# ── 이미지 목록 ───────────────────────────────────────────────────────────────
if args.image:
    images = [Path(args.image)]
else:
    data_dir = Path(SCRIPT_DIR) / "backend" / "data"
    images = sorted(data_dir.glob("*.jpg"))[:10]  # 최대 10장

print(f"\n처리할 이미지: {len(images)}장")
print(f"입력 해상도    : {args.resize}px (실제 파이프라인과 동일하게 리사이즈)")
print(f"conf threshold : {args.conf} (person: {args.conf_person})")
print(f"min bbox ratio : {args.min_bbox}")
print(f"필터 적용      : {'OFF' if args.no_filter else 'ON'}\n")

# ── YOLO 로드 ─────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    model = YOLO(os.getenv("YOLO_MODEL", "yolov8n.pt"))
    print("✅ YOLO 모델 로드 완료\n")
except Exception as e:
    print(f"❌ YOLO 로드 실패: {e}")
    sys.exit(1)

# ── 클래스별 색상 ─────────────────────────────────────────────────────────────
# 장애물 클래스 → 빨강 계열 / 정보성 클래스 → 초록 계열 / 기타 → 회색
OBSTACLE_CLASSES = {
    "person", "bicycle", "chair", "bench", "stairs",
    "fire_extinguisher", "trash_can", "table", "backpack",
    "suitcase", "umbrella", "potted plant", "column", "couch",
}
INFO_CLASSES = {"door", "sign", "elevator", "toilet", "classroom"}

def get_color(cls_name: str, filtered_out: bool) -> tuple:
    if filtered_out:
        return (180, 180, 180)       # 필터링된 것 → 회색
    if cls_name in OBSTACLE_CLASSES:
        return (0, 60, 220)          # 장애물 → 빨강 (BGR)
    if cls_name in INFO_CLASSES:
        return (0, 200, 80)          # 정보성 → 초록
    return (200, 140, 0)             # 기타 → 파랑

# ── 한국어 클래스명 ───────────────────────────────────────────────────────────
CLASS_KO = {
    "person": "사람", "bicycle": "자전거", "chair": "의자",
    "bench": "벤치", "stairs": "계단", "fire_extinguisher": "소화기",
    "trash_can": "쓰레기통", "table": "테이블", "backpack": "가방",
    "suitcase": "여행가방", "umbrella": "우산", "potted plant": "화분",
    "column": "기둥", "couch": "소파", "door": "문", "sign": "표지판",
    "elevator": "엘리베이터", "toilet": "화장실", "classroom": "강의실",
}

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
for img_path in images:
    frame_orig = cv2.imread(str(img_path))
    if frame_orig is None:
        print(f"  ⚠️  이미지 로드 실패: {img_path.name}")
        continue

    # 실제 파이프라인과 동일하게 640px로 리사이즈
    oh, ow = frame_orig.shape[:2]
    if ow > args.resize:
        ratio = args.resize / ow
        frame = cv2.resize(frame_orig, (args.resize, int(oh * ratio)), interpolation=cv2.INTER_AREA)
    else:
        frame = frame_orig

    h, w = frame.shape[:2]
    frame_area = h * w
    vis = frame.copy()

    # YOLO 추론 (낮은 threshold로 전체 탐지 — 필터는 우리가 직접 적용)
    results = model(frame, conf=args.conf, verbose=False)

    passed = []    # 필터 통과한 탐지
    filtered = []  # 필터에 걸린 탐지

    for result in results:
        for box in result.boxes:
            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            bbox_area = (x2 - x1) * (y2 - y1)

            det = dict(cls=cls_name, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2,
                       area_ratio=bbox_area / frame_area)

            if args.no_filter:
                passed.append(det)
                continue

            # 필터 1: 클래스별 threshold
            class_threshold = args.conf_person if cls_name == "person" else args.conf
            if conf < class_threshold:
                det["filter_reason"] = f"conf {conf:.2f} < {class_threshold}"
                filtered.append(det)
                continue

            # 필터 2: 최소 bbox 크기
            if bbox_area / frame_area < args.min_bbox:
                det["filter_reason"] = f"bbox {bbox_area/frame_area*100:.1f}% < {args.min_bbox*100:.1f}%"
                filtered.append(det)
                continue

            VALID_CLASSES = {
                "person", "chair", "bicycle", "bench",
                "backpack", "suitcase", "umbrella",
                "potted plant", "table", "couch",
            }
            if cls_name not in VALID_CLASSES:
                det["filter_reason"] = f"복도 외 클래스 ({cls_name})"
                filtered.append(det)
                continue

            passed.append(det)

    # ── 필터링된 것 먼저 그리기 (회색 점선) ──────────────────────────────────
    for det in filtered:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)
        label = f"[제외] {CLASS_KO.get(det['cls'], det['cls'])} {det['conf']:.2f}"
        reason = det.get("filter_reason", "")
        cv2.putText(vis, label,  (x1, max(y1-18, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(vis, reason, (x1, max(y1- 4, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    # ── 통과한 것 그리기 (두꺼운 컬러 박스) ──────────────────────────────────
    for det in passed:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cls_name = det["cls"]
        conf = det["conf"]
        color = get_color(cls_name, filtered_out=False)
        area_pct = det["area_ratio"] * 100

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        ko_name = CLASS_KO.get(cls_name, cls_name)
        label = f"{ko_name} {conf:.2f} ({area_pct:.1f}%)"
        label_y = max(y1 - 8, 14)

        # 라벨 배경
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, label_y - lh - 2), (x1 + lw + 4, label_y + 2), color, -1)
        cv2.putText(vis, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── 범례 + 요약 ──────────────────────────────────────────────────────────
    summary = f"통과: {len(passed)}  제외: {len(filtered)}  | {img_path.name}"
    cv2.rectangle(vis, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(vis, summary, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / f"debug_{img_path.stem}.jpg"
    cv2.imwrite(str(out_path), vis)

    # 터미널 출력
    print(f"📸 {img_path.name}  원본({ow}×{oh}) → YOLO입력({w}×{h})")
    if passed:
        for det in passed:
            ko = CLASS_KO.get(det["cls"], det["cls"])
            tag = "🔴 장애물" if det["cls"] in OBSTACLE_CLASSES else "🟢 정보"
            print(f"   {tag}  {ko:8s}  conf={det['conf']:.2f}  bbox={det['area_ratio']*100:.1f}%")
    else:
        print("   (탐지 없음)")

    if filtered:
        print(f"   ── 필터로 제외된 것 ({len(filtered)}개) ──")
        for det in filtered:
            ko = CLASS_KO.get(det["cls"], det["cls"])
            print(f"   ⚪ {ko:8s}  conf={det['conf']:.2f}  이유: {det.get('filter_reason','')}")
    print()

print(f"✅ 완료 — debug_output/ 폴더에서 결과 이미지 확인\n")
