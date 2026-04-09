"""
Depth Anything V2 Metric 테스트 스크립트
- 복도 이미지로 거리 추정 결과 확인
- YOLO 탐지 결과와 조합 테스트
"""

import sys
import os
import time
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "backend", "data")
TEST_IMAGE = os.path.join(DATA_DIR, "KakaoTalk_20260330_184714834.jpg")

# ── 1) 이미지 로드 ────────────────────────────────────────────────────────────
print("=" * 55)
print("  Depth Anything V2 Metric 거리 추정 테스트")
print("=" * 55)

img_bgr = cv2.imread(TEST_IMAGE)
if img_bgr is None:
    print(f"[ERROR] 이미지 로드 실패: {TEST_IMAGE}")
    sys.exit(1)

# 실제 카메라 해상도(640x480)로 리사이즈해서 테스트 (속도 현실적으로 측정)
img_bgr = cv2.resize(img_bgr, (640, 480))
h, w = img_bgr.shape[:2]
print(f"\n[이미지] {os.path.basename(TEST_IMAGE)}  → {w}x{h}px 으로 리사이즈 (실제 카메라 해상도)\n")

# ── 2) Depth Anything V2 로드 및 추론 ────────────────────────────────────────
print("[1단계] Depth Anything V2 모델 로드 중...")
t0 = time.time()

from transformers import pipeline as hf_pipeline
from PIL import Image

# 디바이스 선택
import torch
if torch.backends.mps.is_available():
    device = "mps"
    pipe_device = "mps"
    print("       디바이스: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = "cuda"
    pipe_device = 0
    print("       디바이스: CUDA (GPU)")
else:
    device = "cpu"
    pipe_device = -1
    print("       디바이스: CPU")

pipe = hf_pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    device=pipe_device,
)
load_time = time.time() - t0
print(f"       모델 로드 완료 ({load_time:.1f}초)\n")

# ── 3) 깊이 추정 ─────────────────────────────────────────────────────────────
print("[2단계] 깊이맵 생성 중...")
t1 = time.time()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

result = pipe(pil_img)

# predicted_depth = 실제 미터 단위 텐서 (이게 진짜 거리값)
# result["depth"] 는 시각화용 PIL 이미지 (0~255 픽셀값이라 거리 아님)
depth_tensor = result["predicted_depth"]
depth_np = depth_tensor.squeeze().cpu().numpy().astype(np.float32)

# 카메라 해상도로 리사이즈
if depth_np.shape[:2] != img_bgr.shape[:2]:
    depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

infer_time = time.time() - t1
print(f"       추론 완료 ({infer_time:.2f}초)\n")

# ── 4) 결과 분석 ─────────────────────────────────────────────────────────────
print("[3단계] 거리 추정 결과 분석")
print("-" * 55)

# 이미지를 5x3 격자로 나눠서 각 구역 거리 측정
rows, cols = 3, 5
cell_h, cell_w = h // rows, w // cols

print(f"  이미지를 {cols}x{rows} 격자로 나눠 거리 측정 (단위: m)\n")
print("  위치         거리(m)")
print("  " + "-" * 30)

positions = {
    (0, 0): "왼쪽-상단",
    (0, 2): "중앙-상단",
    (0, 4): "오른쪽-상단",
    (1, 0): "왼쪽-중앙",
    (1, 2): "정중앙  ",
    (1, 4): "오른쪽-중앙",
    (2, 0): "왼쪽-하단",
    (2, 2): "중앙-하단",
    (2, 4): "오른쪽-하단",
}

for (r, c), label in positions.items():
    cy = int((r + 0.5) * cell_h)
    cx = int((c + 0.5) * cell_w)
    half = 15
    patch = depth_np[
        max(0, cy-half):min(h, cy+half),
        max(0, cx-half):min(w, cx+half)
    ]
    dist = float(np.median(patch))
    print(f"  {label}  →  {dist:.2f}m")

print()
print(f"  전체 깊이 범위: {depth_np.min():.2f}m ~ {depth_np.max():.2f}m")
print(f"  전체 평균 거리: {depth_np.mean():.2f}m")

# ── 5) 핵심 지점 요약 ─────────────────────────────────────────────────────────
print()
print("[4단계] 핵심 판단 지점 (복도 정면 기준)")
print("-" * 55)

# 정면 중앙 (사용자가 걷는 방향)
front_patch = depth_np[
    h//3 : 2*h//3,
    w//3 : 2*w//3
]
front_dist = float(np.median(front_patch))
print(f"  정면 중앙 거리:   {front_dist:.2f}m")

if front_dist < 1.0:
    status = "🔴 즉시 경고 구간 (1m 미만)"
elif front_dist < 2.0:
    status = "🟡 주의 구간 (1~2m)"
else:
    status = "🟢 안전 구간 (2m 초과)"
print(f"  판단 결과:        {status}")

# ── 6) 추론 속도 요약 ─────────────────────────────────────────────────────────
print()
print("[5단계] 성능 요약")
print("-" * 55)
print(f"  모델 로드:  {load_time:.1f}초  (최초 1회만)")
print(f"  추론 속도:  {infer_time:.2f}초 / 프레임")
print(f"  예상 FPS:   {1/infer_time:.1f} fps")
print()
print("=" * 55)
print("  테스트 완료")
print("=" * 55)
