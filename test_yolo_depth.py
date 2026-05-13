"""
test_yolo_depth.py
yolo_midas.py (YOLOv8 + Depth Anything V2) 단독 테스트 스크립트.

사용법:
    python test_yolo_depth.py <이미지_경로>

예시:
    python test_yolo_depth.py test.jpg
"""

import sys
import cv2
import numpy as np

# ── 경로 설정 ────────────────────────────────────────────────────────────────
# capstone/ 폴더에서 실행해야 한다 (python test_yolo_depth.py)
import os
sys.path.insert(0, os.path.dirname(__file__))

from backend.models.yolo_midas import YOLOMiDaSWrapper


def draw_results(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """탐지 결과를 이미지에 시각화한다."""
    vis = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls   = det["class"]
        dist  = det["distance_m"]
        conf  = det["conf"]
        label = f"{cls} {dist:.1f}m (conf={conf:.2f})"

        # bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 텍스트 배경
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        # 텍스트
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_yolo_depth.py <이미지_경로>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    # ── 이미지 로드 ──────────────────────────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[오류] 이미지 로딩 실패: {image_path}")
        sys.exit(1)

    H, W = frame.shape[:2]
    print(f"\n[이미지] {image_path}  크기={W}×{H}")

    # ── 모델 초기화 ──────────────────────────────────────────────────────────
    # 고해상도 이미지를 실제 앱과 비슷한 해상도로 리사이즈
    # (실제 WebSocket 스트리밍은 보통 1280×720 이하로 들어옴)
    MAX_WIDTH = 1280
    if W > MAX_WIDTH:
        scale = MAX_WIDTH / W
        frame = cv2.resize(frame, (MAX_WIDTH, int(H * scale)))
        H, W = frame.shape[:2]
        print(f"[리사이즈] 실제 앱 해상도로 축소: {W}×{H}")

    print("\n[초기화] YOLOv8 + Depth Anything V2 로딩 중... (최초 1회만)")
    wrapper = YOLOMiDaSWrapper(
        yolo_model="yolov8n.pt",
        midas_model="large",   # small / base / large  ← 정확도 최우선
        conf_threshold=0.4,
        depth_interval=1,      # 테스트용: 매 프레임 depth 실행
    )
    print("[초기화] 완료!\n")

    # ── 추론 ─────────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 1: YOLO 탐지 결과")
    print("=" * 55)

    detections, fast_result = wrapper.run(frame)

    if not detections:
        print("  탐지된 장애물 없음 (허용 클래스: person, chair, stair)")
    else:
        for i, det in enumerate(detections):
            print(f"  [{i+1}] 클래스  : {det['class']}")
            print(f"       거리     : {det['distance_m']:.2f} m")
            print(f"       신뢰도   : {det['conf']:.3f}")
            x1,y1,x2,y2 = det['bbox']
            print(f"       bbox     : ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})")
            print()

    print("=" * 55)
    print("  STEP 2: fast_result 요약 (우선순위 모듈에 전달되는 값)")
    print("=" * 55)
    print(f"  has_obstacle  : {fast_result['has_obstacle']}")
    print(f"  closest_class : {fast_result['class']}")
    print(f"  distance_m    : {fast_result['distance_m']} m")
    print(f"  direction     : {fast_result['direction']}")
    print(f"  conf          : {fast_result['conf']:.3f}")
    print()

    # ── 시각화 저장 ──────────────────────────────────────────────────────────
    if detections:
        vis = draw_results(frame, detections)
        out_path = image_path.replace(".", "_result.")
        if out_path == image_path:
            out_path = image_path + "_result.jpg"
        cv2.imwrite(out_path, vis)
        print(f"[시각화] 결과 이미지 저장됨: {out_path}")
    else:
        print("[시각화] 탐지 결과 없음 → 시각화 파일 생성 생략")

    print("\n완료!")


if __name__ == "__main__":
    main()
