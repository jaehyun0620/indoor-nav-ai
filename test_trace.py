"""
test_trace.py
이미지 한 장을 파이프라인 전체에 통과시키며 각 단계 output을 출력한다.

실행 예시:
  python test_trace.py                                          # 기본 이미지, 목적지=화장실
  python test_trace.py backend/data/obstacle/chair.JPG 강의실  # 이미지·목적지 지정
"""

import sys, os, time, json, asyncio
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "backend/data/KakaoTalk_20260330_184714834.jpg"
TARGET   = sys.argv[2] if len(sys.argv) > 2 else "화장실"

IMG_PATH = os.path.join(ROOT, IMG_PATH) if not os.path.isabs(IMG_PATH) else IMG_PATH

# ── 색상 출력 헬퍼 ─────────────────────────────────────────────────────────────
def hdr(title):
    print(f"\n\033[1;36m{'━'*60}\033[0m")
    print(f"\033[1;36m  {title}\033[0m")
    print(f"\033[1;36m{'━'*60}\033[0m")

def kv(key, val):
    print(f"  \033[33m{key:<22}\033[0m {val}")

def ok(msg):  print(f"  \033[32m✔  {msg}\033[0m")
def err(msg): print(f"  \033[31m✘  {msg}\033[0m")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\033[1;35m{'═'*60}")
print(f"  파이프라인 트레이서")
print(f"  이미지 : {os.path.basename(IMG_PATH)}")
print(f"  목적지 : {TARGET}")
print(f"{'═'*60}\033[0m")

# ── 이미지 로드 ───────────────────────────────────────────────────────────────
hdr("STEP 0  이미지 로드")
frame_bgr = cv2.imread(IMG_PATH)
if frame_bgr is None:
    err(f"이미지를 읽을 수 없음: {IMG_PATH}")
    sys.exit(1)

frame_bgr = cv2.resize(frame_bgr, (640, 480))
H, W = frame_bgr.shape[:2]
ok(f"로드 완료")
kv("경로", IMG_PATH)
kv("해상도", f"{W}×{H}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: YOLOv8 raw 탐지
# ══════════════════════════════════════════════════════════════════════════════
hdr("STEP 1  YOLOv8 원시 탐지")

from backend.models.yolo_midas import YOLOMiDaSWrapper

wrapper = YOLOMiDaSWrapper(
    yolo_model=os.path.join(ROOT, "yolov8n.pt"),
    conf_threshold=0.4,
    depth_interval=1,   # 매 프레임 depth 계산 (테스트용)
)

t0 = time.time()
raw_detections, fast_result_raw = wrapper.run(frame_bgr)
elapsed = time.time() - t0

kv("처리 시간", f"{elapsed:.2f}s")
kv("탐지 객체 수", f"{len(raw_detections)}개")

if raw_detections:
    print()
    print(f"  {'클래스':<15} {'거리':>6}  {'신뢰도':>6}  bbox")
    print(f"  {'─'*55}")
    for d in raw_detections:
        bbox_str = "[{:.0f},{:.0f},{:.0f},{:.0f}]".format(*d["bbox"])
        print(f"  {d['class']:<15} {d['distance_m']:>5.2f}m  {d['conf']:>6.2f}  {bbox_str}")
else:
    print("  (탐지 없음)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Depth Anything V2 깊이맵
# ══════════════════════════════════════════════════════════════════════════════
hdr("STEP 2  Depth Anything V2 깊이맵")

from backend.models.yolo_midas import estimate_depth_map

t0 = time.time()
depth_map = estimate_depth_map(frame_bgr)
elapsed = time.time() - t0

kv("처리 시간", f"{elapsed:.2f}s")
kv("깊이맵 해상도", str(depth_map.shape))
kv("최솟값 (가까운 것)", f"{depth_map.min():.2f}m")
kv("최댓값 (먼 것)",     f"{depth_map.max():.2f}m")
kv("중앙 정면 거리",     f"{float(np.median(depth_map[160:320, 213:427])):.2f}m")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: FastChannel 필터링 + Temporal 안정화
# ══════════════════════════════════════════════════════════════════════════════
hdr("STEP 3  FastChannel (필터링 + Temporal 안정화)")

from backend.channels.fast_channel import FastChannel

fc = FastChannel(
    yolo_model=os.path.join(ROOT, "yolov8n.pt"),
    conf_threshold=0.4,
    depth_interval=1,
)

# 첫 프레임 (prev_detections 비어있음)
t0 = time.time()
fc_out1 = fc.process_frame(frame_bgr)
elapsed = time.time() - t0

print(f"\n  [1번째 프레임]  처리 시간: {elapsed:.2f}s")
kv("필터 후 탐지 수", f"{len(fc_out1['detections'])}개  ← temporal 필터 (첫 프레임은 항상 0)")
print(f"  prev_detections: {len(fc.prev_detections)}개 저장됨")

# 두 번째 프레임 (같은 이미지, prev가 있음)
fc_out2 = fc.process_frame(frame_bgr)
print(f"\n  [2번째 프레임]  (같은 이미지 재처리)")
kv("필터 후 탐지 수", f"{len(fc_out2['detections'])}개")

if fc_out2["detections"]:
    print()
    print(f"  {'클래스':<15} {'거리':>6}  {'신뢰도':>6}  방향")
    print(f"  {'─'*45}")
    for d in fc_out2["detections"]:
        print(f"  {d['class']:<15} {d['distance_m']:>5.2f}m  {d['conf']:>6.2f}")
else:
    print("  (안정화 후 탐지 없음 — 거리 0.5m 이내 연속 탐지 필요)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Context Builder → VLM 주입 텍스트
# ══════════════════════════════════════════════════════════════════════════════
hdr("STEP 4  Context Builder → VLM 주입 텍스트")

from backend.modules.context_builder import build_context, build_obstacle_summary

yolo_context = fc_out2["yolo_context"]
obstacle_summary = build_obstacle_summary(fc_out2["detections"])

kv("yolo_context", repr(yolo_context) if yolo_context else "(빈 문자열)")
print()
kv("obstacle_summary.closest_class",    obstacle_summary.get("closest_class", "—"))
kv("obstacle_summary.closest_distance", f"{obstacle_summary.get('closest_distance', 0):.2f}m")
kv("obstacle_summary.direction",        obstacle_summary.get("direction", "—"))
kv("obstacle_summary.has_obstacle",     str(obstacle_summary.get("has_obstacle")))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Prompt Designer → 실제 프롬프트
# ══════════════════════════════════════════════════════════════════════════════
hdr("STEP 5  Prompt Designer → VLM에 보내는 프롬프트")

from backend.modules.prompt_designer import build_prompt

prompt = build_prompt(yolo_context or "탐지된 장애물 없음", TARGET, condition="proposed")
lines = prompt.strip().split("\n")
print(f"\n  총 {len(lines)}줄 / {len(prompt)}자")
print(f"\n  ── 프롬프트 전문 ──────────────────────────────────")
for line in lines:
    print(f"  {line}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: VLM API 호출 → raw 응답
# ══════════════════════════════════════════════════════════════════════════════
hdr("STEP 6  VLM API 호출 (GPT-4o)")

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, "backend", ".env"))

from backend.channels.slow_channel import SlowChannel

sc = SlowChannel()

# 이미지를 512px로 축소 (main.py와 동일하게)
def resize_for_vlm(frame, max_px=512):
    h, w = frame.shape[:2]
    scale = max_px / max(h, w)
    if scale < 1.0:
        return cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return frame

small_frame = resize_for_vlm(frame_bgr)
kv("VLM 입력 해상도", f"{small_frame.shape[1]}×{small_frame.shape[0]}")

async def call_vlm():
    t0 = time.time()
    result = await sc.process_instant(small_frame, yolo_context or "탐지된 장애물 없음", TARGET)
    elapsed = time.time() - t0
    return result, elapsed

vlm_result, vlm_elapsed = asyncio.run(call_vlm())

kv("처리 시간", f"{vlm_elapsed:.2f}s")

raw = vlm_result.get("raw", {})
print()
kv("goal_visible",    str(raw.get("goal_visible")))
kv("goal_direction",  raw.get("goal_direction", "—"))
kv("goal_distance",   raw.get("goal_distance", "—"))
kv("confidence",      str(raw.get("confidence")))
kv("reasoning",       raw.get("reasoning", "—"))
kv("tts_message",     raw.get("tts_message", "—"))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Consistency Filter
# ══════════════════════════════════════════════════════════════════════════════
hdr("STEP 7  Consistency Filter (단일 프레임 시뮬레이션)")

from backend.modules.consistency_filter import ConsistencyFilter

cf = ConsistencyFilter()

direction = raw.get("goal_direction", "unknown")
confidence = raw.get("confidence", 0.0)

print(f"\n  VLM 결과: direction={direction}  conf={confidence}")
print()

# 1회 추가
cf.add(direction, confidence)
d1, t1 = cf.get_guidance()
kv("1회 후 확정 방향", f"{d1}  →  \"{t1}\"")

# 2회 추가 (같은 값)
cf.add(direction, confidence)
d2, t2 = cf.get_guidance()
kv("2회 후 확정 방향", f"{d2}  →  \"{t2}\"")

# 3회 추가
cf.add(direction, confidence)
d3, t3 = cf.get_guidance()
kv("3회 후 확정 방향", f"{d3}  →  \"{t3}\"")

confirmed_direction = d3
confirmed_tts = t3

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Priority Module → 최종 판단
# ══════════════════════════════════════════════════════════════════════════════
hdr("STEP 8  Priority Module → 최종 출력")

from backend.modules.priority_module import PriorityModule

pm = PriorityModule()

fast_summary = obstacle_summary
slow_summary = {
    "confirmed_direction": confirmed_direction,
    "tts_text": confirmed_tts,
}

decision = pm.decide(fast_summary, slow_summary)

print()
kv("message_type",     decision["message_type"])
kv("tts_text",         decision["tts_text"])
kv("priority",         str(decision["priority"]))
kv("suppress_guidance",str(decision["suppress_guidance"]))

# ══════════════════════════════════════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\033[1;35m{'═'*60}")
print(f"  최종 요약")
print(f"{'═'*60}\033[0m")
print(f"  이미지  : {os.path.basename(IMG_PATH)}")
print(f"  목적지  : {TARGET}")
print(f"  YOLO    : {len(raw_detections)}개 탐지 → 필터 후 {len(fc_out2['detections'])}개")
print(f"  VLM     : {direction} (신뢰도 {confidence})")
print(f"  필터    : {confirmed_direction}")
print(f"  최종 TTS: \033[1;32m{decision['tts_text']}\033[0m")
print(f"  타입    : {decision['message_type']}  priority={decision['priority']}")
print()
