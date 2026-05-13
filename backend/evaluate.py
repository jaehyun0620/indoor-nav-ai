"""
evaluate.py
테스트 비디오에서 프레임을 추출하여 FastChannel + VLM 3조건 비교 평가를 실행한다.

사용법:
    cd /Users/jaehyun/ai_capstone/capstone
    python -m backend.evaluate                          # 전체 실행
    python -m backend.evaluate --skip-vlm               # FastChannel만
    python -m backend.evaluate --target 엘리베이터       # 목표 변경
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# capstone/ 를 sys.path에 추가해 from backend.xxx 임포트를 허용
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


# ── 프레임 추출 ───────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    interval_sec: float = 1.0,
) -> Tuple[List[Tuple[int, float, np.ndarray]], float, float]:
    """비디오에서 interval_sec 간격으로 프레임을 추출한다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오를 열 수 없음: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    step         = max(1, int(fps * interval_sec))

    frames = []
    idx = 0
    while idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((idx, idx / fps, frame))
        idx += step

    cap.release()
    return frames, fps, duration


def preprocess_frame(frame_bgr: np.ndarray, target_w=640, target_h=480) -> np.ndarray:
    """세로 영상 회전 + 리사이즈."""
    h, w = frame_bgr.shape[:2]
    if h > w:  # 세로 촬영 → 가로로 회전
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.resize(frame_bgr, (target_w, target_h))


# ── FastChannel 테스트 ────────────────────────────────────────────────────────

async def run_fast_channel(
    frames: List[Tuple[int, float, np.ndarray]],
) -> List[Dict]:
    from backend.channels.fast_channel import FastChannel

    fast = FastChannel(
        yolo_model     = os.getenv("YOLO_MODEL", "yolov8n.pt"),
        midas_model    = os.getenv("DA2_MODEL_SIZE", "small"),
        conf_threshold = float(os.getenv("YOLO_CONF", "0.4")),
        depth_interval = int(os.getenv("DA2_DEPTH_INTERVAL", "2")),
    )

    results = []
    for frame_idx, timestamp, frame_bgr in frames:
        frame_bgr = preprocess_frame(frame_bgr)
        t0 = time.time()
        try:
            out          = await asyncio.to_thread(fast.process_frame, frame_bgr)
            latency_ms   = (time.time() - t0) * 1000
            fr           = out["fast_result"]

            results.append({
                "frame_idx":          frame_idx,
                "timestamp_sec":      round(timestamp, 2),
                "has_obstacle":       fr["has_obstacle"],
                "obstacle_class":     fr["class"],
                "obstacle_dist_m":    round(fr["distance_m"], 2),
                "obstacle_direction": fr["direction"],
                "detection_count":    len(out["detections"]),
                "yolo_context":       out["yolo_context"],
                "fast_latency_ms":    round(latency_ms, 1),
                "status":             "ok",
            })
        except Exception as e:
            results.append({
                "frame_idx":          frame_idx,
                "timestamp_sec":      round(timestamp, 2),
                "has_obstacle":       False,
                "obstacle_class":     "",
                "obstacle_dist_m":    999.0,
                "obstacle_direction": "정면",
                "detection_count":    0,
                "yolo_context":       "탐지된 객체 없음",
                "fast_latency_ms":    0.0,
                "status":             f"error:{e}",
            })
        print(
            f"  [{frame_idx:4d}] T={timestamp:5.1f}s  "
            f"장애물={'있음' if results[-1]['has_obstacle'] else '없음':2}  "
            f"cls={results[-1]['obstacle_class'] or '-':8}  "
            f"dist={results[-1]['obstacle_dist_m']:5.1f}m  "
            f"{results[-1]['fast_latency_ms']:.0f}ms"
        )

    return results


# ── VLM 3조건 비교 ────────────────────────────────────────────────────────────

async def run_vlm_comparison(
    fast_results: List[Dict],
    frames_dict:  Dict[int, np.ndarray],
    target:       str,
    vlm_sample:   int = 3,
) -> List[Dict]:
    from backend.channels.slow_channel import SlowChannel

    # N프레임 간격으로 샘플링
    sampled = [r for i, r in enumerate(fast_results) if i % vlm_sample == 0]
    conditions = ["baseline", "structured", "proposed"]
    all_results = []

    for condition in conditions:
        print(f"\n  [조건: {condition}] {len(sampled)}프레임 테스트 중...")
        slow = SlowChannel(condition=condition)

        for r in sampled:
            frame_bgr = frames_dict.get(r["frame_idx"])
            if frame_bgr is None:
                continue

            _, enc    = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_bytes = enc.tobytes()

            t0 = time.time()
            try:
                res        = await slow.process_instant(img_bytes, r["yolo_context"], target)
                vlm_ms     = (time.time() - t0) * 1000
                raw        = res.get("raw", {})
                direction  = res["confirmed_direction"]
                confidence = float(raw.get("confidence", 0.0))

                # JSON 파싱 성공 여부: raw 딕셔너리가 비어있지 않으면 성공
                json_ok = bool(raw and raw.get("goal_direction"))

                all_results.append({
                    "condition":      condition,
                    "frame_idx":      r["frame_idx"],
                    "timestamp_sec":  r["timestamp_sec"],
                    "direction":      direction,
                    "confidence":     round(confidence, 3),
                    "goal_visible":   bool(raw.get("goal_visible", False)),
                    "goal_distance":  str(raw.get("goal_distance", "unknown")),
                    "tts_text":       res.get("tts_text", ""),
                    "reasoning":      str(raw.get("reasoning", ""))[:120],
                    "vlm_latency_ms": round(vlm_ms, 1),
                    "json_parse_ok":  json_ok,
                    "status":         "ok",
                })
                print(
                    f"    T={r['timestamp_sec']:5.1f}s  "
                    f"dir={direction:8}  conf={confidence:.2f}  "
                    f"json={'OK' if json_ok else 'NG'}  {vlm_ms:.0f}ms"
                )

            except Exception as e:
                all_results.append({
                    "condition":      condition,
                    "frame_idx":      r["frame_idx"],
                    "timestamp_sec":  r["timestamp_sec"],
                    "direction":      "error",
                    "confidence":     0.0,
                    "goal_visible":   False,
                    "goal_distance":  "unknown",
                    "tts_text":       "",
                    "reasoning":      "",
                    "vlm_latency_ms": 0.0,
                    "json_parse_ok":  False,
                    "status":         f"error:{e}",
                })
                print(f"    T={r['timestamp_sec']:5.1f}s  ERROR: {e}")

            await asyncio.sleep(0.3)   # API 속도 제한 방지

        await asyncio.sleep(1.5)       # 조건 간 휴식

    return all_results


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

def compute_metrics(fast_results: List[Dict], vlm_results: List[Dict]) -> Dict:
    metrics = {}

    # FastChannel
    n = len(fast_results)
    obstacle_frames = [r for r in fast_results if r["has_obstacle"]]
    ok_frames       = [r for r in fast_results if r["status"] == "ok"]
    latencies       = [r["fast_latency_ms"] for r in ok_frames]

    metrics["fast_channel"] = {
        "total_frames":       n,
        "ok_frames":          len(ok_frames),
        "obstacle_frames":    len(obstacle_frames),
        "obstacle_rate_pct":  round(len(obstacle_frames) / n * 100, 1) if n else 0,
        "avg_latency_ms":     round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p95_latency_ms":     round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,
        "error_frames":       n - len(ok_frames),
    }

    # 장애물 클래스 빈도
    cls_freq: Dict[str, int] = {}
    for r in obstacle_frames:
        c = r["obstacle_class"]
        cls_freq[c] = cls_freq.get(c, 0) + 1
    metrics["fast_channel"]["obstacle_class_freq"] = cls_freq

    # VLM 조건별
    for cond in ["baseline", "structured", "proposed"]:
        cond_r = [r for r in vlm_results if r["condition"] == cond]
        if not cond_r:
            continue
        n_c       = len(cond_r)
        known     = [r for r in cond_r if r["direction"] not in ("unknown", "error")]
        high_conf = [r for r in cond_r if r["confidence"] >= 0.6]
        json_ok   = [r for r in cond_r if r["json_parse_ok"]]
        lats      = [r["vlm_latency_ms"] for r in cond_r if r["vlm_latency_ms"] > 0]
        visible   = [r for r in cond_r if r["goal_visible"]]

        # 방향 분포
        dir_dist: Dict[str, int] = {}
        for r in cond_r:
            d = r["direction"]
            dir_dist[d] = dir_dist.get(d, 0) + 1

        metrics[cond] = {
            "total_samples":            n_c,
            "direction_known_rate_pct": round(len(known) / n_c * 100, 1) if n_c else 0,
            "unknown_rate_pct":         round((n_c - len(known)) / n_c * 100, 1) if n_c else 0,
            "goal_visible_rate_pct":    round(len(visible) / n_c * 100, 1) if n_c else 0,
            "high_confidence_rate_pct": round(len(high_conf) / n_c * 100, 1) if n_c else 0,
            "json_parse_success_pct":   round(len(json_ok) / n_c * 100, 1) if n_c else 0,
            "avg_vlm_latency_ms":       round(sum(lats) / len(lats), 1) if lats else 0,
            "p95_vlm_latency_ms":       round(sorted(lats)[int(len(lats) * 0.95)], 1) if len(lats) >= 2 else (lats[0] if lats else 0),
            "direction_distribution":   dir_dist,
        }

    # 할루시네이션 필터 효과 (structured vs proposed unknown_rate 비교)
    if "structured" in metrics and "proposed" in metrics:
        unk_struct   = metrics["structured"]["unknown_rate_pct"]
        unk_proposed = metrics["proposed"]["unknown_rate_pct"]
        # unknown rate 감소 = 필터가 잘못된 방향을 걸러낸 효과
        metrics["filter_effect"] = {
            "structured_unknown_rate_pct": unk_struct,
            "proposed_unknown_rate_pct":   unk_proposed,
            "note": (
                "proposed의 unknown_rate가 structured보다 높으면 "
                "일관성 필터가 불확실한 응답을 추가로 차단함"
            ),
        }

    return metrics


# ── CSV / JSON 저장 ───────────────────────────────────────────────────────────

def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── 메인 ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="실내 길 안내 시스템 평가")
    parser.add_argument("--video",      default="data/video/test_video.MOV")
    parser.add_argument("--target",     default="화장실")
    parser.add_argument("--interval",   type=float, default=1.0,
                        help="프레임 샘플링 간격 (초, 기본 1.0)")
    parser.add_argument("--vlm-sample", type=int,   default=3,
                        help="VLM 호출 샘플 비율 — fast 결과 N개 중 1개 (기본 3)")
    parser.add_argument("--output",     default="logs/eval",
                        help="결과 저장 디렉터리")
    parser.add_argument("--skip-vlm",   action="store_true",
                        help="VLM 호출 생략 (FastChannel만 테스트)")
    args = parser.parse_args()

    LINE = "=" * 62
    print(f"\n{LINE}")
    print(f"  실내 길 안내 시스템 — 평가 실행")
    print(f"  비디오 : {args.video}")
    print(f"  목표물 : {args.target}")
    print(f"  VLM    : {'생략' if args.skip_vlm else '3조건 비교'}")
    print(f"{LINE}\n")

    # ── 프레임 추출 ─────────────────────────────────────────────────────
    print(f"[1/4] 비디오 프레임 추출 (간격 {args.interval}초)")
    frames, fps, duration = extract_frames(args.video, args.interval)
    print(f"      {len(frames)}프레임 추출 완료 (총 {duration:.1f}초, {fps:.0f}fps)\n")

    # 전처리 및 frames_dict 구성
    processed: List[Tuple[int, float, np.ndarray]] = []
    frames_dict: Dict[int, np.ndarray] = {}
    for fid, ts, frm in frames:
        frm = preprocess_frame(frm)
        processed.append((fid, ts, frm))
        frames_dict[fid] = frm

    # ── FastChannel ─────────────────────────────────────────────────────
    print(f"[2/4] FastChannel (YOLO + Depth Anything V2) 테스트")
    fast_results = await run_fast_channel(processed)

    obs_n  = sum(1 for r in fast_results if r["has_obstacle"])
    avg_ms = sum(r["fast_latency_ms"] for r in fast_results) / max(len(fast_results), 1)
    print(f"\n      완료: {len(fast_results)}프레임 | 장애물 {obs_n}건 | 평균 {avg_ms:.0f}ms\n")

    # ── VLM 비교 ────────────────────────────────────────────────────────
    vlm_results: List[Dict] = []
    if not args.skip_vlm:
        sample_n = max(1, len(fast_results) // args.vlm_sample)
        print(f"[3/4] VLM 3조건 비교 (~{sample_n}프레임 × 3조건 = ~{sample_n*3}회 API 호출)")
        vlm_results = await run_vlm_comparison(
            fast_results, frames_dict, args.target, args.vlm_sample
        )
        print(f"\n      완료: {len(vlm_results)}건\n")
    else:
        print(f"[3/4] VLM 생략 (--skip-vlm)\n")

    # ── 저장 및 지표 계산 ────────────────────────────────────────────────
    print(f"[4/4] 결과 저장 및 지표 계산")
    out = args.output
    fast_csv    = f"{out}/fast_channel.csv"
    vlm_csv     = f"{out}/vlm_comparison.csv"
    metrics_json= f"{out}/metrics_summary.json"

    save_csv(fast_results, fast_csv)
    if vlm_results:
        save_csv(vlm_results, vlm_csv)

    metrics = compute_metrics(fast_results, vlm_results)
    save_json(metrics, metrics_json)

    # ── 콘솔 요약 ────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print(f"  평가 결과 요약")
    print(f"{LINE}")

    fm = metrics.get("fast_channel", {})
    print(f"\n  [FastChannel]")
    print(f"    분석 프레임   : {fm.get('total_frames',0)}개")
    print(f"    장애물 감지   : {fm.get('obstacle_frames',0)}건 ({fm.get('obstacle_rate_pct',0)}%)")
    print(f"    평균 처리속도 : {fm.get('avg_latency_ms',0)}ms  "
          f"(P95: {fm.get('p95_latency_ms',0)}ms)")
    freq = fm.get("obstacle_class_freq", {})
    if freq:
        print(f"    감지 클래스   : {dict(sorted(freq.items(), key=lambda x:-x[1]))}")

    if vlm_results:
        print(f"\n  [VLM 3조건 비교]")
        hdr = f"  {'조건':<12} {'방향확정':>7} {'목표보임':>7} {'고신뢰도':>7} {'JSON성공':>7} {'평균지연':>9}"
        print(f"\n{hdr}")
        print(f"  {'-'*56}")
        for cond in ["baseline", "structured", "proposed"]:
            m = metrics.get(cond, {})
            if not m:
                continue
            print(
                f"  {cond:<12} "
                f"{m['direction_known_rate_pct']:>6}%  "
                f"{m['goal_visible_rate_pct']:>6}%  "
                f"{m['high_confidence_rate_pct']:>6}%  "
                f"{m['json_parse_success_pct']:>6}%  "
                f"{m['avg_vlm_latency_ms']:>7}ms"
            )

    print(f"\n  저장 경로: {out}/")
    print(f"    {fast_csv}")
    if vlm_results:
        print(f"    {vlm_csv}")
    print(f"    {metrics_json}")
    print(f"\n{LINE}\n")

    return metrics


if __name__ == "__main__":
    asyncio.run(main())
