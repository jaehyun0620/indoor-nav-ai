"""
전체 파이프라인 통합 테스트 스크립트
capstone/ 루트에서 실행: python test_pipeline.py

테스트 순서:
  1. context_builder   - YOLO 결과 텍스트 변환
  2. prompt_designer   - 프롬프트 생성 + VLM 응답 파싱
  3. consistency_filter - 다수결 방향 확정
  4. priority_module   - 경로 A/B 분기 판단
  5. scene_memory      - 장면 기억 버퍼
  6. DA2 깊이 추정     - 복도 이미지로 실제 거리 측정
  7. YOLO + DA2 통합   - FastChannel 전체 실행
"""

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  {status}  {name}" + (f"  →  {detail}" if detail else ""))
    return condition

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ─────────────────────────────────────────────────────────
# 1. context_builder
# ─────────────────────────────────────────────────────────
section("1. context_builder")

from backend.modules.context_builder import build_context, build_obstacle_summary

mock_detections = [
    {"class": "person", "bbox": [280, 100, 360, 400], "distance_m": 1.8, "conf": 0.91},  # 2m 이내
    {"class": "chair",  "bbox": [450, 150, 550, 380], "distance_m": 3.5, "conf": 0.77},
    {"class": "stairs", "bbox": [10,  100, 100, 400], "distance_m": 4.0, "conf": 0.22},  # 낮은 신뢰도
]

ctx = build_context(mock_detections, frame_width=640)
check("텍스트 생성됨",           len(ctx) > 0,                ctx[:50])
check("가까운 순서 정렬",         ctx.index("사람") < ctx.index("의자"), "사람(2.1m) → 의자(3.5m)")
check("낮은 신뢰도 필터링",       "계단" not in ctx,           "conf 0.22 제외됨")
check("방향 텍스트 포함",         "정면" in ctx or "왼쪽" in ctx or "오른쪽" in ctx)

summary = build_obstacle_summary(mock_detections)
check("가장 가까운 장애물",        summary["closest_class"] == "사람", summary["closest_class"])
check("2m 이내 장애물 감지",       summary["has_obstacle"] == True,    f"거리 {summary['closest_distance']}m")

# ─────────────────────────────────────────────────────────
# 2. prompt_designer
# ─────────────────────────────────────────────────────────
section("2. prompt_designer")

from backend.modules.prompt_designer import build_prompt, parse_vlm_response

# 프롬프트 생성
p_proposed   = build_prompt(ctx, "화장실", condition="proposed")
p_baseline   = build_prompt(ctx, "화장실", condition="baseline")
p_structured = build_prompt(ctx, "화장실", condition="structured")

check("Proposed 프롬프트 — YOLO 컨텍스트 포함",   "사람" in p_proposed)
check("Proposed 프롬프트 — JSON 형식 강제",        "goal_direction" in p_proposed)
check("Baseline 프롬프트 — 단순 질문",             "goal_direction" not in p_baseline)
check("Structured 프롬프트 — JSON만 강제",         "goal_direction" in p_structured and "사람" not in p_structured)
check("목표물 설명 확장",                          "남자화장실" in p_proposed)

# VLM 응답 파싱
good_json = '{"goal_visible": true, "goal_direction": "right", "goal_distance": "약 5m", "confidence": 0.85, "reasoning": "오른쪽에 화장실 표지판이 보입니다", "tts_message": "오른쪽으로 이동하세요"}'
low_conf  = '{"goal_visible": true, "goal_direction": "left",  "goal_distance": "unknown", "confidence": 0.2, "reasoning": "불확실합니다", "tts_message": ""}'
bad_json  = "죄송합니다, 화장실을 찾을 수 없습니다."
markdown  = '```json\n{"goal_visible": false, "goal_direction": "straight", "goal_distance": "약 3m", "confidence": 0.75, "reasoning": "직진", "tts_message": "앞으로 이동하세요"}\n```'

r1 = parse_vlm_response(good_json)
r2 = parse_vlm_response(low_conf)
r3 = parse_vlm_response(bad_json)
r4 = parse_vlm_response(markdown)

check("정상 JSON 파싱",              r1["goal_direction"] == "right",    f"direction={r1['goal_direction']}")
check("confidence 낮으면 unknown",   r2["goal_direction"] == "unknown",  f"conf=0.2 → {r2['goal_direction']}")
check("JSON 없으면 기본값 반환",      r3["goal_direction"] == "unknown")
check("마크다운 코드블록 파싱",       r4["goal_direction"] == "straight", f"direction={r4['goal_direction']}")
check("tts_message 파싱",            r1.get("tts_message") == "오른쪽으로 이동하세요")

# ─────────────────────────────────────────────────────────
# 3. consistency_filter
# ─────────────────────────────────────────────────────────
section("3. consistency_filter")

from backend.modules.consistency_filter import ConsistencyFilter

f = ConsistencyFilter()

# 버퍼 비어있을 때
d0, t0 = f.get_guidance()
check("버퍼 비어있으면 unknown",     d0 == "unknown", t0)

# 1회만 추가
f.add("right", 0.85)
d1, t1 = f.get_guidance()
check("1회만 추가 → 아직 분석 중",  d1 == "unknown", t1)

# 2회 일치 → 확정
f.add("right", 0.79)
d2, t2 = f.get_guidance()
check("2/2 일치 → 방향 확정",       d2 == "right",   t2)

# 노이즈 1개 섞여도 유지
f.add("left", 0.60)
d3, t3 = f.get_guidance()
check("3개 중 2개 일치 → 여전히 right", d3 == "right", f"버퍼: right×2, left×1")

# confidence 낮으면 unknown 처리
f2 = ConsistencyFilter()
f2.add("right", 0.3)   # conf < 0.4 → unknown 처리됨
f2.add("right", 0.3)
d4, _ = f2.get_guidance()
check("low conf → unknown 강제",    d4 == "unknown")

# unknown 연속 메시지 단계
f3 = ConsistencyFilter()
f3.add("unknown", 0.9)
f3.add("unknown", 0.9)
_, m1 = f3.get_guidance()
_, m2 = f3.get_guidance()
_, m3 = f3.get_guidance()
check("unknown 1회 메시지",         "기다려" in m1,       m1)
check("unknown 2회 메시지",         "카메라" in m2,       m2)
check("unknown 3회 메시지",         "둘러보" in m3,       m3)

# 리셋
f.reset()
check("reset 후 버퍼 비워짐",       len(f.buffer) == 0)

# ─────────────────────────────────────────────────────────
# 4. priority_module
# ─────────────────────────────────────────────────────────
section("4. priority_module")

from backend.modules.priority_module import PriorityModule

pm = PriorityModule()

# 경로 A: 즉각 경고 (1m 미만)
r_warn = pm.decide(
    {"class": "계단", "distance_m": 0.7, "has_obstacle": True},
    {"confirmed_direction": "right", "tts_text": "오른쪽"},
)
check("1m 미만 → warning",          r_warn["message_type"] == "warning",   r_warn["tts_text"])
check("warning → suppress_guidance", r_warn["suppress_guidance"] == True)
check("warning → priority 1",        r_warn["priority"] == 1)

# 경로 A+: 주의 (1~2m)
r_cau = pm.decide(
    {"class": "사람", "distance_m": 1.5, "has_obstacle": True},
    {"confirmed_direction": "right", "tts_text": "오른쪽"},
)
check("1~2m → caution",             r_cau["message_type"] == "caution",   r_cau["tts_text"])
check("caution → 방향 안내 병합",    "오른쪽" in r_cau["tts_text"],        r_cau["tts_text"])

# 경로 B: 방향 안내
r_gui = pm.decide(
    {"class": "", "distance_m": 5.0, "has_obstacle": False},
    {"confirmed_direction": "left", "tts_text": "왼쪽으로 이동하세요"},
)
check("안전 + 방향 확정 → guidance", r_gui["message_type"] == "guidance",   r_gui["tts_text"])
check("guidance → priority 2",       r_gui["priority"] == 2)

# 방향 미확정
r_unk = pm.decide(
    {"class": "", "distance_m": 5.0, "has_obstacle": False},
    {"confirmed_direction": "unknown", "tts_text": "분석 중"},
)
check("방향 미확정 → unknown",       r_unk["message_type"] == "unknown",   r_unk["tts_text"])
check("unknown → priority 3",        r_unk["priority"] == 3)

# ─────────────────────────────────────────────────────────
# 5. scene_memory
# ─────────────────────────────────────────────────────────
section("5. scene_memory")

from backend.modules.scene_memory import SceneMemory

sm = SceneMemory(maxlen=5, ttl=10.0)

sm.update(mock_detections, {"goal_direction": "right", "confidence": 0.85})
sm.update(mock_detections, {"goal_direction": "right", "confidence": 0.80})

check("recent 항목 조회",       len(sm.get_recent(3)) == 2)
check("마지막 방향 반환",        sm.get_last_direction() == "right")

hint = sm.get_context_for_prompt()
check("프롬프트 힌트 생성",      "오른쪽" in hint,   hint)

summary_txt = sm.get_context_summary()
check("컨텍스트 요약 생성",      "사람" in summary_txt, summary_txt)

sm.reset()
check("reset 후 비워짐",         sm.get_last_direction() is None)

# ─────────────────────────────────────────────────────────
# 6. Depth Anything V2 (실제 모델 추론)
# ─────────────────────────────────────────────────────────
section("6. Depth Anything V2 — 복도 이미지 거리 추정")

import cv2
import numpy as np

IMG_PATH = os.path.join(SCRIPT_DIR, "backend", "data", "KakaoTalk_20260330_184714834.jpg")
img_bgr = cv2.imread(IMG_PATH)
img_bgr = cv2.resize(img_bgr, (640, 480))

print(f"  이미지 로드: {os.path.basename(IMG_PATH)} → 640x480")

try:
    from backend.models.yolo_midas import estimate_depth_map, bbox_center_depth

    t_start = time.time()
    depth_map = estimate_depth_map(img_bgr)
    elapsed = time.time() - t_start

    check("깊이맵 생성 성공",        depth_map is not None)
    check("깊이맵 해상도 일치",       depth_map.shape == (480, 640),    str(depth_map.shape))
    # MPS 첫 번째 추론은 워밍업으로 느릴 수 있음 → 5초로 완화
    check("추론 시간 5초 이내",       elapsed < 5.0,                    f"{elapsed:.2f}초")
    check("거리 범위 현실적 (0~30m)", depth_map.max() < 30.0,          f"최대 {depth_map.max():.1f}m")
    check("최솟값 0 이상",            depth_map.min() >= 0.0,           f"최솟값 {depth_map.min():.2f}m")

    # 복도 정면 중앙 거리
    front = float(np.median(depth_map[160:320, 213:427]))
    check("정면 중앙 거리 현실적 (2~20m)", 2.0 <= front <= 20.0, f"{front:.2f}m")

    # bbox 거리 추출 테스트
    mock_bbox = [260, 100, 380, 400]   # 화면 중앙 물체
    dist = bbox_center_depth(depth_map, mock_bbox)
    check("bbox 중심 거리 추출",      dist > 0,   f"{dist:.2f}m")

except Exception as e:
    check("DA2 추론", False, str(e))

# ─────────────────────────────────────────────────────────
# 7. FastChannel 통합 (YOLO + DA2)
# ─────────────────────────────────────────────────────────
section("7. FastChannel 통합 — YOLO + DA2")

try:
    from backend.channels.fast_channel import FastChannel

    fc = FastChannel(
        yolo_model="yolov8n.pt",
        midas_model="small",
        conf_threshold=0.4,
        depth_interval=5,
    )

    with open(IMG_PATH, "rb") as f_img:
        img_bytes = f_img.read()

    t2 = time.time()
    out = fc.process_bytes(img_bytes)
    elapsed2 = time.time() - t2

    check("FastChannel 실행 성공",    out is not None)
    check("detections 키 존재",       "detections" in out)
    check("fast_result 키 존재",      "fast_result" in out)
    check("yolo_context 키 존재",     "yolo_context" in out)

    fr = out["fast_result"]
    check("distance_m 포함",          "distance_m" in fr,  f"{fr.get('distance_m')}m")
    check("has_obstacle 포함",        "has_obstacle" in fr, str(fr.get("has_obstacle")))

    ctx2 = out["yolo_context"]
    check("YOLO 컨텍스트 생성",       len(ctx2) > 0, ctx2[:60])

    print(f"\n  탐지 객체 수:  {len(out['detections'])}개")
    print(f"  처리 시간:     {elapsed2:.2f}초")
    for det in out["detections"][:3]:
        print(f"    - {det['class']:15s}  {det['distance_m']:.2f}m  conf={det['conf']:.2f}")

except Exception as e:
    check("FastChannel 통합", False, str(e))

# ─────────────────────────────────────────────────────────
# 최종 결과
# ─────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  최종 결과")
print(f"{'='*55}")

total  = len(results)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = total - passed

print(f"\n  통과: {passed} / {total}")
if failed:
    print(f"\n  실패 항목:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"    {FAIL}  {name}  {detail}")
else:
    print("\n  모든 테스트 통과 🎉")
print()
