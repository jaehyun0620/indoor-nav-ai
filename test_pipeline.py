#!/usr/bin/env python3
"""
test_pipeline.py
전체 파이프라인 추적 테스트 스크립트.
이미지를 입력하면 각 모듈의 입력/출력을 순서대로 시각화한다.

사용법:
    cd /Users/jaehyun/ai_capstone/capstone
    python test_pipeline.py <이미지경로> [목표물]

예시:
    python test_pipeline.py test_image.jpg 화장실
    python test_pipeline.py test_image.jpg 강의실
    python test_pipeline.py test_image.jpg 엘리베이터
"""

import sys
import os
import asyncio
import time

# ── 프로젝트 루트를 sys.path에 추가 ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "backend", ".env"))

# ── ANSI 색상 ────────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
WHITE   = "\033[37m"

MAX_WIDTH = 1280  # 입력 이미지 최대 너비 (리사이즈 기준)


# ── 터미널 출력 헬퍼 ─────────────────────────────────────────────────────────

def header(step: str, title: str, color: str = CYAN) -> None:
    width = 62
    bar = "═" * width
    print(f"\n{color}{BOLD}╔{bar}╗")
    print(f"║  {step}  {title:<{width - len(step) - 3}}║")
    print(f"╚{bar}╝{RESET}")


def item(key: str, val: str, color: str = WHITE) -> None:
    print(f"  {DIM}│{RESET}  {BOLD}{key:<20}{RESET}{color}{val}{RESET}")


def label(name: str, color: str = BOLD) -> None:
    print(f"  {color}{name}{RESET}")


def blank() -> None:
    print()


def ruler() -> None:
    print(f"  {DIM}{'─' * 60}{RESET}")


# ── 이미지 유틸 ──────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)))
    return frame


def save_yolo_vis(frame: np.ndarray, detections: list, path: str) -> None:
    """YOLO bbox + 클래스/신뢰도/거리 레이블을 그려 저장."""
    vis = frame.copy()
    palette = {
        "person": (0,   0, 255),
        "chair":  (0, 165, 255),
        "stair":  (0, 255, 255),
    }
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls   = det["class"]
        conf  = det["conf"]
        dist  = det["distance_m"]
        color = palette.get(cls, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_txt = f"{cls}  conf={conf:.2f}  {dist:.1f}m"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label_txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(path, vis)


def save_depth_map(depth_map: np.ndarray, path: str) -> None:
    """
    Depth Anything V2 깊이맵을 INFERNO 컬러맵으로 시각화.
    값이 클수록 멀다 (어두움=가까움, 밝음=멀다).
    하단에 0m ~ 10m 눈금 범례를 추가한다.
    """
    clipped    = np.clip(depth_map, 0, 10.0)
    normalized = ((clipped / 10.0) * 255).astype(np.uint8)
    colored    = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

    h, w = colored.shape[:2]
    legend_h = 44
    legend = np.zeros((legend_h, w, 3), dtype=np.uint8)

    # 그라데이션 바
    for i in range(w):
        v = int((i / w) * 255)
        c = cv2.applyColorMap(np.array([[v]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
        legend[4:legend_h - 4, i] = c

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(legend, "0m (NEAR)",  (4,  legend_h - 6), font, 0.42, (255, 255, 255), 1)
    cv2.putText(legend, "10m (FAR)",  (w - 82, legend_h - 6), font, 0.42, (255, 255, 255), 1)
    cv2.putText(legend, "Depth Anything V2 Metric Indoor",
                (w // 2 - 120, legend_h - 6), font, 0.42, (200, 200, 200), 1)

    cv2.imwrite(path, np.vstack([colored, legend]))


def save_ocr_vis(frame: np.ndarray, ocr_results: list, path: str) -> None:
    """
    EasyOCR 검출 영역을 원본 이미지 위에 그려 저장.
    EasyOCR bbox 형식: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4개 꼭짓점)
    """
    vis = frame.copy()

    if not ocr_results:
        # 검출 없음 표시
        cv2.putText(vis, "No OCR results", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(path, vis)
        return

    for res in ocr_results:
        pts   = res["bbox"]   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        text  = res["text"]
        conf  = res["conf"]

        # conf에 따라 색상 변경 (높을수록 초록, 낮을수록 노랑)
        color = (0, 255, 0) if conf >= 0.7 else (0, 200, 200)

        # 4-코너 폴리곤 그리기
        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)

        # 텍스트 레이블 (좌상단 기준)
        tx = int(min(p[0] for p in pts))
        ty = int(min(p[1] for p in pts)) - 6
        if ty < 14:
            ty = int(max(p[1] for p in pts)) + 18

        label_txt = f"{text}  ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (tx, ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
        cv2.putText(vis, label_txt, (tx + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite(path, vis)


def save_depth_overlay(frame: np.ndarray, depth_map: np.ndarray,
                       detections: list, path: str) -> None:
    """원본 + 깊이맵 반투명 오버레이 + bbox 중심 거리 표기."""
    clipped    = np.clip(depth_map, 0, 10.0)
    normalized = ((clipped / 10.0) * 255).astype(np.uint8)
    colored    = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    overlay    = cv2.addWeighted(frame, 0.45, colored, 0.55, 0)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        dist = det["distance_m"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), -1)
        cv2.putText(overlay, f"{dist:.1f}m", (x1 + 4, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(path, overlay)


# ── 메인 파이프라인 ──────────────────────────────────────────────────────────

async def run_pipeline(image_path: str, target: str) -> None:
    from backend.models.yolo_midas import YOLOMiDaSWrapper, bbox_center_depth
    from backend.modules.context_builder import build_context, build_obstacle_summary
    from backend.modules.prompt_designer import build_prompt
    from backend.modules.priority_module import PriorityModule
    from backend.channels.slow_channel import SlowChannel

    out_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "pipeline_output")
    os.makedirs(out_dir, exist_ok=True)
    base    = os.path.splitext(os.path.basename(image_path))[0]
    t_start = time.time()

    # ─────────────────────────────────────────────────────
    # STEP 0 · 이미지 로드
    # ─────────────────────────────────────────────────────
    header("STEP 0", "이미지 로드", BLUE)
    label("▶ 입력")
    item("경로", image_path)

    frame = load_image(image_path)
    H, W  = frame.shape[:2]
    orig_path = os.path.join(out_dir, f"{base}_00_original.jpg")
    cv2.imwrite(orig_path, frame)

    blank()
    label("◀ 출력", GREEN)
    item("해상도", f"{W} × {H} px", GREEN)
    item("저장",   orig_path, DIM)

    # ─────────────────────────────────────────────────────
    # STEP 1 · YOLOv8 탐지
    # ─────────────────────────────────────────────────────
    header("STEP 1", "YOLOv8 탐지", CYAN)
    label("▶ 입력")
    item("프레임",    f"{W}×{H} BGR")
    item("모델",      "yolov8n.pt  (사전학습, 재학습 없음)")
    item("conf_thr",  "0.40")
    item("valid_cls", "person / chair / stair")

    t1 = time.time()
    wrapper = YOLOMiDaSWrapper(
        yolo_model     = "yolov8n.pt",
        midas_model    = "small",
        conf_threshold = 0.40,
        depth_interval = 1,   # 테스트: 매 프레임 DA2 실행
    )
    detections, fast_result = wrapper.run(frame)
    elapsed_yolo = time.time() - t1

    blank()
    label("◀ 출력 · detections 리스트", GREEN)
    if detections:
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            print(f"  {DIM}│{RESET}  [{i+1}] "
                  f"{YELLOW}{d['class']:<10}{RESET}"
                  f"conf={GREEN}{d['conf']:.3f}{RESET}  "
                  f"dist={CYAN}{d['distance_m']:.2f}m{RESET}  "
                  f"bbox=[{x1},{y1},{x2},{y2}]")
    else:
        item("결과", "(탐지된 객체 없음)", DIM)

    blank()
    label("◀ 출력 · fast_result (우선순위 모듈용 요약)", GREEN)
    item("class",        fast_result["class"] or "(없음)",   YELLOW)
    item("distance_m",   f"{fast_result['distance_m']:.2f} m", CYAN)
    item("direction",    fast_result["direction"],             WHITE)
    item("has_obstacle", str(fast_result["has_obstacle"]),
         RED if fast_result["has_obstacle"] else GREEN)
    item("conf",         f"{fast_result['conf']:.3f}",         WHITE)
    item("처리 시간",    f"{elapsed_yolo*1000:.0f} ms",        DIM)

    yolo_vis = os.path.join(out_dir, f"{base}_01_yolo_bbox.jpg")
    save_yolo_vis(frame, detections, yolo_vis)
    print(f"\n  {GREEN}✔ YOLO 시각화 저장 → {yolo_vis}{RESET}")

    # ─────────────────────────────────────────────────────
    # STEP 2 · Depth Anything V2 깊이맵
    # ─────────────────────────────────────────────────────
    header("STEP 2", "Depth Anything V2 Metric Indoor (깊이맵)", MAGENTA)
    label("▶ 입력")
    item("프레임",   f"{W}×{H} BGR")
    item("모델",     "Depth-Anything-V2-Metric-Indoor-Small-hf")
    item("출력단위", "미터 (m)  — 값이 클수록 멀다")

    depth_map = wrapper._depth_cache   # STEP 1에서 이미 계산된 캐시 사용
    d_min  = float(depth_map.min())
    d_max  = float(depth_map.max())
    d_mean = float(depth_map.mean())

    blank()
    label("◀ 출력 · 깊이맵 통계", GREEN)
    item("shape",  f"{depth_map.shape[0]} × {depth_map.shape[1]}",   WHITE)
    item("최솟값", f"{d_min:.2f} m  ← 가장 가까운 픽셀",             CYAN)
    item("최댓값", f"{d_max:.2f} m  ← 가장 먼 픽셀",                 WHITE)
    item("평균값", f"{d_mean:.2f} m",                                  WHITE)

    if detections:
        blank()
        label("◀ bbox 중심 거리 (depth_map 직접 조회)", YELLOW)
        for d in detections:
            cd = bbox_center_depth(depth_map, d["bbox"])
            item(d["class"], f"bbox 중심 → {cd:.2f} m", YELLOW)

    depth_path   = os.path.join(out_dir, f"{base}_02_depth_map.jpg")
    overlay_path = os.path.join(out_dir, f"{base}_03_depth_overlay.jpg")
    save_depth_map(depth_map, depth_path)
    save_depth_overlay(frame, depth_map, detections, overlay_path)
    print(f"\n  {GREEN}✔ 깊이맵 저장    → {depth_path}{RESET}")
    print(f"  {GREEN}✔ 오버레이 저장  → {overlay_path}{RESET}")

    # ─────────────────────────────────────────────────────
    # STEP 3 · OCR 파이프라인
    # ─────────────────────────────────────────────────────
    header("STEP 3", "OCR 파이프라인  (크롭 → 4배 확대 → CLAHE → 이진화 → EasyOCR)", CYAN)
    label("▶ 입력")
    item("프레임",       f"{W}×{H} BGR  (전체 이미지 대상)")
    item("언어",         "ko + en")
    item("conf_thr",     "0.40")
    item("전처리 전략",  "원본 시도 → avg_conf < 0.5 이면 CLAHE+이진화 fallback")

    ocr_path = os.path.join(out_dir, f"{base}_04_ocr.jpg")

    try:
        from backend.modules.ocr_pipeline import read_text, find_target_sign
        _easyocr_ok = True
    except ImportError:
        _easyocr_ok = False

    if not _easyocr_ok:
        blank()
        print(f"  {YELLOW}⚠  easyocr 미설치 — OCR 건너뜀  (pip install easyocr){RESET}")
        ocr_results = []
    else:
        try:
            t_ocr = time.time()
            ocr_results = read_text(frame, bbox=None, conf_threshold=0.40)
            elapsed_ocr = time.time() - t_ocr

            blank()
            label("◀ 출력 · OCR 검출 결과", GREEN)
            if ocr_results:
                for i, r in enumerate(ocr_results):
                    pts = r["bbox"]
                    # bbox 좌상단 좌표만 표시 (EasyOCR 4-코너 포맷)
                    x = int(pts[0][0]) if pts else 0
                    y = int(pts[0][1]) if pts else 0
                    print(f"  {DIM}│{RESET}  [{i+1}] "
                          f"{YELLOW}\"{r['text']}\"{RESET}  "
                          f"conf={GREEN}{r['conf']:.3f}{RESET}  "
                          f"위치=({x},{y})")

                # 목표물 키워드 탐색
                matched = find_target_sign(ocr_results, target)
                blank()
                label("◀ 목표물 키워드 탐색", GREEN)
                if matched:
                    item("매칭 결과", f"\"{matched['text']}\"  conf={matched['conf']:.3f}", GREEN)
                else:
                    item("매칭 결과", f"'{target}' 포함 텍스트 없음", DIM)
            else:
                item("결과", "(인식된 텍스트 없음)", DIM)

            item("처리 시간", f"{elapsed_ocr*1000:.0f} ms", DIM)

        except Exception as e:
            blank()
            print(f"  {RED}OCR 오류: {e}{RESET}")
            ocr_results = []

    save_ocr_vis(frame, ocr_results, ocr_path)
    print(f"\n  {GREEN}✔ OCR 시각화 저장 → {ocr_path}{RESET}")

    # ─────────────────────────────────────────────────────
    # STEP 4 · Context Builder
    # ─────────────────────────────────────────────────────
    header("STEP 4", "Context Builder  (YOLO 결과 → VLM 프롬프트 텍스트 변환)", CYAN)
    label("▶ 입력")
    item("detections",  f"{len(detections)}개 (STEP 1 출력)")
    item("frame_width", str(W))
    item("conf_thr",    "0.40")

    yolo_context     = build_context(detections, frame_width=W, conf_threshold=0.40)
    obstacle_summary = build_obstacle_summary(detections, frame_width=W)

    blank()
    label("◀ 출력 · yolo_context  (VLM 프롬프트에 삽입되는 텍스트)", GREEN)
    for line in yolo_context.splitlines():
        print(f"  {DIM}│{RESET}  {YELLOW}{line}{RESET}")

    blank()
    label("◀ 출력 · obstacle_summary", GREEN)
    item("closest_class",    obstacle_summary["closest_class"] or "(없음)", YELLOW)
    item("closest_distance", f"{obstacle_summary['closest_distance']:.2f} m", CYAN)
    item("direction",        obstacle_summary["direction"],                   WHITE)
    item("has_obstacle",     str(obstacle_summary["has_obstacle"]),
         RED if obstacle_summary["has_obstacle"] else GREEN)

    # ─────────────────────────────────────────────────────
    # STEP 4 · Prompt Designer
    # ─────────────────────────────────────────────────────
    header("STEP 4", "Prompt Designer  (구조화 프롬프트 생성)", CYAN)
    label("▶ 입력")
    item("yolo_context", "↑ (STEP 3 출력)")
    item("target",       target)
    item("condition",    "proposed  (YOLO 주입 + JSON 응답 강제)")

    prompt = build_prompt(yolo_context, target, condition="proposed")
    if len(prompt) > 600:
        preview = (prompt[:300]
                   + f"\n  {DIM}... (중략 {len(prompt)-500}자) ...{RESET}\n"
                   + prompt[-200:])
    else:
        preview = prompt

    blank()
    label("◀ 출력 · prompt 미리보기", GREEN)
    ruler()
    for line in preview.splitlines():
        print(f"  {DIM}│  {line}{RESET}")
    ruler()
    item("전체 길이", f"{len(prompt)} 문자", DIM)

    # ─────────────────────────────────────────────────────
    # STEP 5 · VLM API
    # ─────────────────────────────────────────────────────
    header("STEP 5", "Slow Channel  ─  VLM API (GPT-4o / Gemini)", RED)

    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
    api_ok     = has_openai or has_gemini
    provider   = "openai" if has_openai else ("gemini" if has_gemini else "없음")

    label("▶ 입력")
    item("image",        f"{W}×{H}  JPEG bytes")
    item("yolo_context", "↑ (STEP 3 출력)")
    item("target",       target)
    item("method",       "process_instant()  (일관성 필터 우회, 즉시 반환)")
    item("provider",     provider, GREEN if api_ok else RED)

    if not api_ok:
        blank()
        print(f"  {YELLOW}⚠  API 키 미설정 — VLM 호출을 건너뜁니다.{RESET}")
        print(f"  {DIM}   backend/.env 에 OPENAI_API_KEY 또는 GEMINI_API_KEY 를 추가하세요.{RESET}")
        slow_result = {
            "confirmed_direction": "unknown",
            "tts_text": "API 키 없음 (테스트 건너뜀)",
            "unknown_streak": 0,
            "raw": {},
        }
    else:
        _, img_enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = img_enc.tobytes()

        slow_ch = SlowChannel(condition="proposed")
        t5 = time.time()
        slow_result = await slow_ch.process_instant(image_bytes, yolo_context, target)
        elapsed_vlm = time.time() - t5

        raw = slow_result.get("raw", {})
        blank()
        label("◀ 출력 · VLM 파싱 결과 (raw JSON)", GREEN)
        item("goal_visible",   str(raw.get("goal_visible")),   YELLOW)
        item("goal_direction", str(raw.get("goal_direction")), CYAN)
        item("goal_distance",  str(raw.get("goal_distance")),  WHITE)
        item("confidence",     str(raw.get("confidence")),     WHITE)

        reasoning = raw.get("reasoning", "")
        item("reasoning",
             (reasoning[:100] + "…") if len(reasoning) > 100 else reasoning, DIM)

        tts_msg = raw.get("tts_message", "")
        item("tts_message",
             (tts_msg[:80] + "…") if len(tts_msg) > 80 else tts_msg, YELLOW)

        blank()
        label("◀ 출력 · 일관성 필터 통과 후", GREEN)
        item("confirmed_direction", slow_result["confirmed_direction"], CYAN)
        item("tts_text",            slow_result["tts_text"],            YELLOW)
        item("unknown_streak",      str(slow_result["unknown_streak"]), WHITE)
        item("처리 시간",           f"{elapsed_vlm*1000:.0f} ms",       DIM)

    # ─────────────────────────────────────────────────────
    # STEP 6 · Priority Module
    # ─────────────────────────────────────────────────────
    header("STEP 6", "Priority Module  ─  경로 A / B 분기 (최종 판단)", CYAN)
    label("▶ 입력 · fast_result")
    item("class",        fast_result["class"] or "(없음)", YELLOW)
    item("distance_m",   f"{fast_result['distance_m']:.2f} m", CYAN)
    item("has_obstacle", str(fast_result["has_obstacle"]),
         RED if fast_result["has_obstacle"] else GREEN)

    blank()
    label("▶ 입력 · slow_result")
    item("confirmed_direction", slow_result["confirmed_direction"], CYAN)
    item("tts_text",            slow_result["tts_text"],            WHITE)

    final = PriorityModule().decide(fast_result, slow_result)

    route_label = {
        "warning":  "경로 A  — 즉각 경고 (2m 미만)",
        "caution":  "경로 A  — 주의 경고 (2~4m)",
        "guidance": "경로 B  — VLM 방향 안내",
        "unknown":  "경로 B  — 방향 미확정 (unknown)",
    }.get(final["message_type"], "경로 B")

    route_color = RED if final["message_type"] in ("warning", "caution") else GREEN

    blank()
    label("◀ 출력 · 최종 결정", GREEN)
    ruler()
    item("분기",              route_label,                  route_color)
    item("message_type",      final["message_type"],        route_color)
    item("priority",          str(final["priority"]),       WHITE)
    item("suppress_guidance", str(final["suppress_guidance"]), WHITE)
    ruler()
    print(f"\n  {BOLD}🔊  TTS 출력: {route_color}{final['tts_text']}{RESET}\n")

    # ─────────────────────────────────────────────────────
    # 최종 요약
    # ─────────────────────────────────────────────────────
    total = time.time() - t_start
    header("RESULT", "전체 파이프라인 결과 요약", GREEN)

    print(f"""
  {BOLD}{'─'*60}{RESET}
  입력 이미지   : {os.path.basename(image_path)}  ({W}×{H})
  목표물        : {target}
  탐지 객체 수  : {len(detections)}개
  has_obstacle  : {RED if fast_result['has_obstacle'] else GREEN}{fast_result['has_obstacle']}{RESET}  (임계값 4.0m)
  가장 가까운 것: {YELLOW}{fast_result['class'] or '없음'}{RESET}  {CYAN}{fast_result['distance_m']:.1f}m{RESET}
  VLM 방향      : {CYAN}{slow_result['confirmed_direction']}{RESET}
  {BOLD}{'─'*60}
  🔊  최종 TTS  : {route_color}{final['tts_text']}{RESET}
  {BOLD}{'─'*60}{RESET}
  총 처리 시간  : {total*1000:.0f} ms

  {DIM}저장된 시각화 파일:{RESET}
    {DIM}00  원본          →{RESET} {orig_path}
    {DIM}01  YOLO bbox     →{RESET} {yolo_vis}
    {DIM}02  깊이맵        →{RESET} {depth_path}
    {DIM}03  깊이+오버레이 →{RESET} {overlay_path}
    {DIM}04  OCR 검출 영역 →{RESET} {ocr_path}
""")


# ── 진입점 ───────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(f"""
{BOLD}사용법:{RESET}  python test_pipeline.py <이미지경로> [목표물]

{BOLD}예시:{RESET}
  python test_pipeline.py test_image.jpg 화장실
  python test_pipeline.py test_image.jpg 강의실
  python test_pipeline.py test_image.jpg 엘리베이터

{DIM}목표물을 생략하면 기본값 '화장실'이 사용됩니다.{RESET}
        """)
        sys.exit(1)

    image_path = sys.argv[1]
    target     = sys.argv[2] if len(sys.argv) >= 3 else "화장실"

    if not os.path.exists(image_path):
        print(f"{RED}파일을 찾을 수 없습니다: {image_path}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}{'═' * 64}")
    print(f"  🚀  캡스톤 파이프라인 추적 테스트")
    print(f"{'═' * 64}{RESET}")
    print(f"  이미지  : {image_path}")
    print(f"  목표물  : {target}\n")

    asyncio.run(run_pipeline(image_path, target))


if __name__ == "__main__":
    main()
