"""
test_video.py
동영상 파일을 프레임 단위로 분해해서 전체 파이프라인을 오프라인으로 테스트한다.

실시간 촬영 없이도 복도 영상 한 장으로 반복 테스트 및 로그 분석이 가능하다.

사용법:
    python test_video.py 영상경로.mp4
    python test_video.py 영상경로.mp4 --target 엘리베이터
    python test_video.py 영상경로.mp4 --target 화장실 --fps 2 --no-vlm
    python test_video.py 영상경로.mp4 --target 화장실 --condition baseline

옵션:
    --target    목표물 (기본: 화장실)
    --fps       처리할 초당 프레임 수 (기본: 1, VLM 비용 절감)
    --no-vlm    VLM 호출 없이 YOLO+DA2 결과만 확인
    --condition baseline / structured / proposed (기본: proposed)
    --out       CSV 저장 경로 (기본: logs/result_<타임스탬프>.csv)
    --max-sec   최대 처리 시간(초), 0이면 전체 (기본: 0)

출력:
    - 터미널 실시간 로그
    - logs/result_<타임스탬프>.csv  (프레임별 결과 전체)
    - logs/result_<타임스탬프>_summary.txt  (집계 요약)
"""

import argparse
import asyncio
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(SCRIPT_DIR, "backend", ".env"))

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="동영상 오프라인 파이프라인 테스트")
parser.add_argument("video", help="테스트할 동영상 파일 경로")
parser.add_argument("--target",    default="화장실",   help="목표물 (기본: 화장실)")
parser.add_argument("--fps",       type=float, default=1.0, help="처리 FPS (기본: 1)")
parser.add_argument("--no-vlm",    action="store_true",     help="VLM 호출 없이 YOLO+DA2만 실행")
parser.add_argument("--condition", default="proposed",       help="실험 조건: baseline/structured/proposed")
parser.add_argument("--out",       default="",               help="CSV 출력 경로 (기본: 자동 생성)")
parser.add_argument("--max-sec",   type=float, default=0,    help="최대 처리 초 (0=전체)")
args = parser.parse_args()

# ── 출력 경로 설정 ────────────────────────────────────────────────────────────

LOG_DIR = Path(SCRIPT_DIR) / "logs"
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path  = args.out if args.out else str(LOG_DIR / f"result_{timestamp}.csv")
summary_path = csv_path.replace(".csv", "_summary.txt")

# ── 모듈 로드 ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  동영상 오프라인 파이프라인 테스트")
print(f"{'='*60}")
print(f"  영상    : {args.video}")
print(f"  목표물  : {args.target}")
print(f"  조건    : {args.condition}")
print(f"  처리FPS : {args.fps}")
print(f"  VLM     : {'OFF (YOLO+DA2만)' if args.no_vlm else 'ON'}")
print(f"  출력    : {csv_path}")
print()

print("  모듈 로딩 중...")

from backend.channels.fast_channel import FastChannel
from backend.modules.priority_module import PriorityModule
from backend.modules.consistency_filter import ConsistencyFilter

fast_channel = FastChannel(
    yolo_model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
    midas_model=os.getenv("DA2_MODEL_SIZE", "small"),
    conf_threshold=float(os.getenv("YOLO_CONF", "0.4")),
    depth_interval=int(os.getenv("DA2_DEPTH_INTERVAL", "5")),
)
priority_module = PriorityModule()
print("  ✅ FastChannel 로드 완료")

if not args.no_vlm:
    from backend.channels.slow_channel import SlowChannel
    slow_channel = SlowChannel(
        provider=os.getenv("VLM_PROVIDER", "openai"),
        condition=args.condition,
    )
    SLOW_INTERVAL = float(os.getenv("SLOW_CHANNEL_INTERVAL", "2.5"))
    print(f"  ✅ SlowChannel 로드 완료 (VLM 호출 간격: {SLOW_INTERVAL}초)")
else:
    slow_channel = None
    SLOW_INTERVAL = 9999

# ── VLM 이미지 리사이즈 헬퍼 ──────────────────────────────────────────────────

def resize_for_vlm(frame_bgr: np.ndarray) -> bytes:
    size    = int(os.getenv("VLM_IMAGE_SIZE", "512"))
    quality = int(os.getenv("VLM_JPEG_QUALITY", "82"))
    h, w = frame_bgr.shape[:2]
    if w > size:
        ratio = size / w
        frame_bgr = cv2.resize(frame_bgr, (size, int(h * ratio)), interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return enc.tobytes()

# ── CSV 헤더 ─────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "frame_idx", "time_sec",
    # YOLO + DA2
    "has_obstacle", "closest_class", "closest_dist_m", "detection_count", "yolo_context",
    # VLM (no-vlm 모드면 빈 값)
    "vlm_direction", "vlm_confidence", "vlm_reasoning", "vlm_latency_sec",
    # 필터 결과
    "confirmed_direction", "tts_text",
    # 최종 판단
    "message_type", "priority", "final_tts",
]

# ── 메인 루프 ─────────────────────────────────────────────────────────────────

async def run():
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {args.video}")
        sys.exit(1)

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    frame_step   = max(1, int(video_fps / args.fps))   # 처리할 프레임 간격

    max_frames = int(args.max_sec * video_fps) if args.max_sec > 0 else total_frames

    print(f"  영상 정보: {total_frames}프레임 / {video_fps:.1f}fps / {duration_sec:.1f}초")
    print(f"  처리 간격: {frame_step}프레임마다 1회 ({args.fps}fps 기준)")
    print(f"  예상 처리: {min(total_frames, max_frames) // frame_step}프레임\n")

    rows = []
    last_slow_time = -SLOW_INTERVAL   # 첫 프레임에서 즉시 VLM 호출되도록
    frame_idx = 0
    processed = 0

    # ConsistencyFilter는 slow_channel 내부에 있지만,
    # no-vlm 모드에서도 dummy 결과를 만들기 위해 별도 유지
    dummy_slow = {"confirmed_direction": "unknown", "tts_text": "VLM 비활성화", "raw": {}}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= max_frames:
                break

            # 지정 간격이 아니면 스킵
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            time_sec = frame_idx / video_fps
            processed += 1

            # ── 빠른 채널 (YOLO + DA2) ────────────────────────────────────
            t_fast = time.time()
            fast_out    = await asyncio.to_thread(fast_channel.process_frame, frame)
            fast_result = fast_out["fast_result"]
            yolo_context = fast_out["yolo_context"]
            fast_ms = (time.time() - t_fast) * 1000

            # ── 느린 채널 (VLM) ───────────────────────────────────────────
            vlm_dir = vlm_conf = vlm_reason = ""
            vlm_latency = 0.0
            slow_result = dummy_slow

            now = time.time()
            if not args.no_vlm and (now - last_slow_time) >= SLOW_INTERVAL:
                img_bytes = resize_for_vlm(frame)
                t_vlm = time.time()
                try:
                    slow_result  = await slow_channel.process_instant(img_bytes, yolo_context, args.target)
                    vlm_latency  = time.time() - t_vlm
                    raw          = slow_result.get("raw", {})
                    vlm_dir      = raw.get("goal_direction", "")
                    vlm_conf     = raw.get("confidence", "")
                    vlm_reason   = raw.get("reasoning", "")
                    last_slow_time = now
                except Exception as e:
                    vlm_dir = f"ERROR:{str(e)[:40]}"

            # ── 우선순위 판단 ─────────────────────────────────────────────
            decision = priority_module.decide(fast_result, slow_result)

            # ── 터미널 출력 ────────────────────────────────────────────────
            obs_str  = f"⚠️  {fast_result['class']} {fast_result['distance_m']:.1f}m" \
                       if fast_result["has_obstacle"] else "✅ 장애물 없음"
            dir_str  = slow_result.get("confirmed_direction", "unknown")
            msg_type = decision["message_type"]
            color    = {"warning": "🔴", "caution": "🟡", "guidance": "🟢", "unknown": "⚪"}.get(msg_type, "⚫")

            print(
                f"  [{time_sec:6.1f}s | f{frame_idx:05d}]  "
                f"{obs_str:<22}  "
                f"VLM={vlm_dir or '-':<10}  "
                f"confirmed={dir_str:<10}  "
                f"{color} {decision['final_tts'] if 'final_tts' in decision else decision['tts_text']}"
            )

            # ── CSV 저장 ──────────────────────────────────────────────────
            row = {
                "frame_idx":         frame_idx,
                "time_sec":          round(time_sec, 2),
                "has_obstacle":      fast_result["has_obstacle"],
                "closest_class":     fast_result["class"],
                "closest_dist_m":    round(fast_result["distance_m"], 2),
                "detection_count":   len(fast_out["detections"]),
                "yolo_context":      yolo_context.replace("\n", " | "),
                "vlm_direction":     vlm_dir,
                "vlm_confidence":    vlm_conf,
                "vlm_reasoning":     vlm_reason,
                "vlm_latency_sec":   round(vlm_latency, 2),
                "confirmed_direction": slow_result.get("confirmed_direction", ""),
                "tts_text":          slow_result.get("tts_text", ""),
                "message_type":      decision["message_type"],
                "priority":          decision["priority"],
                "final_tts":         decision["tts_text"],
            }
            writer.writerow(row)
            rows.append(row)

            frame_idx += 1

    cap.release()

    # ── 집계 요약 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  처리 완료: {processed}프레임")
    print(f"{'='*60}")

    if not rows:
        print("  처리된 프레임 없음")
        return

    total = len(rows)
    obs_count   = sum(1 for r in rows if r["has_obstacle"])
    vlm_rows    = [r for r in rows if r["vlm_direction"]]
    unknown_cnt = sum(1 for r in rows if r["confirmed_direction"] == "unknown")
    dir_counts  = {}
    for r in rows:
        d = r["confirmed_direction"]
        dir_counts[d] = dir_counts.get(d, 0) + 1

    msg_counts = {}
    for r in rows:
        m = r["message_type"]
        msg_counts[m] = msg_counts.get(m, 0) + 1

    avg_vlm_latency = (
        sum(float(r["vlm_latency_sec"]) for r in vlm_rows) / len(vlm_rows)
        if vlm_rows else 0
    )

    summary_lines = [
        f"동영상 파이프라인 테스트 요약",
        f"{'='*50}",
        f"영상      : {args.video}",
        f"목표물    : {args.target}",
        f"실험조건  : {args.condition}",
        f"처리일시  : {timestamp}",
        f"",
        f"[프레임 통계]",
        f"  총 처리 프레임  : {total}",
        f"  장애물 감지     : {obs_count}회 ({obs_count/total*100:.1f}%)",
        f"  VLM 호출        : {len(vlm_rows)}회",
        f"  VLM 평균 응답   : {avg_vlm_latency:.2f}초",
        f"",
        f"[방향 확정 결과]",
    ]
    for d, cnt in sorted(dir_counts.items(), key=lambda x: -x[1]):
        label = {"left": "왼쪽", "right": "오른쪽", "straight": "직진", "unknown": "미확정", "": "(없음)"}.get(d, d)
        summary_lines.append(f"  {label:<10}: {cnt}회 ({cnt/total*100:.1f}%)")

    summary_lines += [
        f"",
        f"[최종 메시지 타입]",
    ]
    for m, cnt in sorted(msg_counts.items(), key=lambda x: -x[1]):
        summary_lines.append(f"  {m:<12}: {cnt}회 ({cnt/total*100:.1f}%)")

    summary_lines += [
        f"",
        f"[출력 파일]",
        f"  CSV     : {csv_path}",
        f"  요약    : {summary_path}",
    ]

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\n  📄 CSV   저장 완료: {csv_path}")
    print(f"  📋 요약  저장 완료: {summary_path}\n")


asyncio.run(run())
