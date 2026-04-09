"""
test_vlm.py
VLM 단독 테스트 스크립트 — 이미지를 직접 넣고 응답 확인

사용법:
    python test_vlm.py                            # data/ 폴더 첫 번째 이미지 사용
    python test_vlm.py backend/data/내이미지.jpg  # 이미지 직접 지정
    python test_vlm.py backend/data/내이미지.jpg 엘리베이터  # 목표물 지정

capstone/ 디렉토리에서 실행
"""

import sys
import os
import asyncio
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(SCRIPT_DIR, "backend", ".env"))

from backend.modules.prompt_designer import build_prompt, parse_vlm_response
from backend.modules.context_builder import build_context
from backend.channels.slow_channel import VLMClient


# ── 인자 처리 ─────────────────────────────────────────────────────────────────

# 이미지 경로
if len(sys.argv) >= 2:
    IMAGE_PATH = sys.argv[1]
else:
    # data/ 폴더 첫 번째 이미지 자동 선택
    data_dir = os.path.join(SCRIPT_DIR, "backend", "data")
    jpgs = sorted(f for f in os.listdir(data_dir) if f.endswith(".jpg"))
    if not jpgs:
        print("❌ backend/data/ 에 .jpg 이미지가 없습니다.")
        sys.exit(1)
    IMAGE_PATH = os.path.join(data_dir, jpgs[0])

# 목표물
TARGET = sys.argv[2] if len(sys.argv) >= 3 else "화장실"

# VLM 이미지 크기
VLM_SIZE = int(os.getenv("VLM_IMAGE_SIZE", "480"))


# ── 이미지 로드 + 리사이즈 ────────────────────────────────────────────────────

import cv2
import numpy as np

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    print(f"❌ 이미지 로드 실패: {IMAGE_PATH}")
    sys.exit(1)

h, w = img_bgr.shape[:2]
if w > VLM_SIZE:
    ratio   = VLM_SIZE / w
    img_bgr = cv2.resize(img_bgr, (VLM_SIZE, int(h * ratio)))

_, enc    = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
img_bytes = enc.tobytes()

print(f"\n{'='*60}")
print(f"  VLM 단독 테스트")
print(f"{'='*60}")
print(f"  이미지  : {os.path.basename(IMAGE_PATH)}  ({w}×{h} → {img_bgr.shape[1]}×{img_bgr.shape[0]})")
print(f"  목표물  : {TARGET}")
print(f"  전송크기: {len(img_bytes)/1024:.1f} KB")
print(f"  모델    : {os.getenv('VLM_PROVIDER', 'gemini').upper()}  /  {os.getenv('GEMINI_MODEL') or os.getenv('OPENAI_MODEL')}")


# ── 프롬프트 생성 (YOLO 없이 — 순수 VLM 판단력 확인용) ───────────────────────

yolo_context = "탐지 정보 없음 (YOLO 미실행)"   # YOLO 없이 순수 VLM 테스트
prompt       = build_prompt(yolo_context, TARGET, condition="proposed")

print(f"\n── 전송 프롬프트 ──────────────────────────────────────────")
print(prompt)


# ── VLM 호출 ─────────────────────────────────────────────────────────────────

async def run():
    client = VLMClient()

    print(f"\n── VLM 호출 중... ────────────────────────────────────────")
    t0 = time.time()
    try:
        raw_text = await client.call(prompt, img_bytes)
    except Exception as e:
        print(f"❌ API 오류: {e}")
        return
    elapsed = time.time() - t0

    print(f"  응답 시간: {elapsed:.2f}초")
    print(f"\n── 원본 응답 ─────────────────────────────────────────────")
    print(raw_text)

    parsed = parse_vlm_response(raw_text)

    print(f"\n── 파싱 결과 ─────────────────────────────────────────────")
    print(f"  goal_visible    : {parsed['goal_visible']}")
    print(f"  goal_direction  : {parsed['goal_direction']}")
    print(f"  goal_distance   : {parsed['goal_distance']}")
    print(f"  confidence      : {parsed['confidence']:.2f}")
    print(f"  reasoning       : {parsed['reasoning']}")
    print(f"  tts_message     : {parsed.get('tts_message', '(없음)')}")
    print(f"{'='*60}\n")


asyncio.run(run())
