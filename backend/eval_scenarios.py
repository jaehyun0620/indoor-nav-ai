"""
eval_scenarios.py — 25개 시나리오 기반 3조건 비교 평가
========================================================
영상 파일 + 정답 CSV를 입력받아 3가지 조건으로 VLM을 호출하고
방향 정확도 / 장애물 경고 누락률 / 할루시네이션 발생률 등 지표를 산출한다.

조건:
  baseline   : 이미지만 → VLM 자유 응답
  structured : 이미지 → VLM JSON 강제 (YOLO 없음)
  proposed   : 이미지 + YOLO 컨텍스트 → VLM JSON 강제 + 장애물 우선순위

준비:
  1. backend/evaluation/scenarios.csv 작성 (--create-template 으로 생성)
  2. backend/evaluation/videos/ 에 영상 파일 넣기
  3. scenarios.csv의 video_path, gt_direction 수정

실행:
  cd /Users/jaehyun/ai_capstone/capstone

  # 템플릿 생성 (처음 한 번만)
  python backend/eval_scenarios.py --create-template

  # 전체 평가
  python backend/eval_scenarios.py

  # 특정 조건만
  python backend/eval_scenarios.py --condition proposed

  # 앞 5개만 테스트
  python backend/eval_scenarios.py --limit 5
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "backend" / ".env")

from backend.modules.context_builder import build_context, build_obstacle_summary
from backend.modules.prompt_designer import build_prompt, parse_vlm_response
from backend.channels.slow_channel import VLMClient

# ── 경로 ──────────────────────────────────────────────────────────────────────
EVAL_DIR    = ROOT / "backend" / "evaluation"
VIDEOS_DIR  = EVAL_DIR / "videos"
RESULTS_DIR = EVAL_DIR / "results"
SCENARIOS_CSV = EVAL_DIR / "scenarios.csv"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 방향 정규화 ────────────────────────────────────────────────────────────────
DIR_ALIAS = {
    "왼쪽": "left", "좌측": "left", "왼편": "left",
    "오른쪽": "right", "우측": "right", "오른편": "right",
    "직진": "straight", "앞": "straight", "정면": "straight",
    "도착": "arrived", "arrived": "arrived",
    "모름": "unknown", "불명": "unknown",
}

def norm_dir(raw: str) -> str:
    return DIR_ALIAS.get(raw.strip().lower(), raw.strip().lower())


# ── YOLO 초기화 ────────────────────────────────────────────────────────────────
class YOLODetector:
    def __init__(self):
        model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")
        conf = float(os.getenv("YOLO_CONF", "0.4"))
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.conf = conf
            self.ok = True
            print(f"  [YOLO] {Path(model_path).name}")
        except Exception as e:
            print(f"  [YOLO] 로드 실패: {e}")
            self.ok = False

    def detect(self, frame: np.ndarray) -> List[Dict]:
        if not self.ok:
            return []
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        h, w = frame.shape[:2]
        dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox_h = max(y2 - y1, 1)
            dets.append({
                "class":      self.model.names[cls_id],
                "bbox":       [x1, y1, x2, y2],
                "distance_m": round(max(0.5, 300 / bbox_h), 1),
                "conf":       float(box.conf[0]),
            })
        return dets


# ── 이미지/영상 → 대표 프레임 ──────────────────────────────────────────────────
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif'}

def extract_frame(media_path: str, timestamp: float = -1) -> Optional[np.ndarray]:
    """
    이미지 파일 또는 영상 파일에서 프레임을 반환한다.
    - 이미지(.jpg/.png 등): 그대로 읽어서 반환
    - 영상(.mp4/.mov 등): timestamp 위치 프레임 추출 (기본: 중간)
    세로 방향 파일은 자동으로 가로로 회전한다.
    """
    ext = Path(media_path).suffix.lower()

    # ── 이미지 파일 ──────────────────────────────────────────────────────────
    if ext in IMAGE_EXTS:
        frame = cv2.imread(media_path)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if h > w:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.resize(frame, (640, 480))

    # ── 영상 파일 ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target = int(timestamp * fps) if timestamp >= 0 else total // 2
    target = min(max(target, 0), total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.resize(frame, (640, 480))


def frame_to_bytes(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ── Baseline 자유 응답 파싱 ────────────────────────────────────────────────────
def parse_baseline(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["도착", "바로 앞", "바로앞", "arrived"]):
        return "arrived"
    scores = {
        "left":     sum(1 for k in ["왼쪽","left","좌측","왼편"] if k in t),
        "right":    sum(1 for k in ["오른쪽","right","우측","오른편"] if k in t),
        "straight": sum(1 for k in ["직진","straight","정면","앞으로","앞"] if k in t),
        "unknown":  sum(1 for k in ["모르","알 수 없","unknown","보이지 않","없"] if k in t),
    }
    best = max(scores, key=lambda x: scores[x])
    return best if scores[best] > 0 else "unknown"


# ── 단일 시나리오 평가 ────────────────────────────────────────────────────────
async def eval_one(
    sc: Dict,
    client: VLMClient,
    yolo: YOLODetector,
    conditions: List[str],
) -> Dict:
    sid     = sc["id"]
    vpath   = sc["video_path"]
    target  = sc["target"]
    gt_dir  = norm_dir(sc["gt_direction"])
    has_obs = sc.get("has_obstacle", "false").strip().lower() == "true"
    tgt_vis = sc.get("target_visible", "true").strip().lower() == "true"
    ts      = float(sc.get("timestamp_sec", -1))

    print(f"\n  ── 시나리오 {sid} ─ {sc.get('memo','')} ──")
    print(f"     목표:{target}  정답:{gt_dir}  장애물:{has_obs}  목표보임:{tgt_vis}")

    # 프레임 추출
    frame = extract_frame(vpath, ts)
    if frame is None:
        print(f"     ⚠️  영상 열기 실패: {vpath}")
        return {"id": sid, "error": "no_frame"}

    img_bytes = frame_to_bytes(frame)

    # YOLO 탐지
    dets     = yolo.detect(frame) if yolo.ok else []
    ctx      = build_context(dets, frame_width=640)
    obs_info = build_obstacle_summary(dets, frame_width=640)

    row = {
        "id": sid, "memo": sc.get("memo",""),
        "target": target, "gt_direction": gt_dir,
        "has_obstacle": has_obs, "target_visible": tgt_vis,
        "yolo_detections": len(dets),
    }

    for cond in conditions:
        print(f"     [{cond:10s}] ", end="", flush=True)
        t0 = time.time()
        try:
            # ── proposed: 장애물 우선 경고 (VLM 우회) ────────────────────────
            if cond == "proposed" and obs_info["has_obstacle"]:
                dist = obs_info["closest_distance"]
                cls  = obs_info["closest_class"]
                warn = f"즉시 멈추세요. {cls}이(가) 있습니다." if dist < 1.0 \
                       else f"주의: {cls} {dist:.1f}m 앞"
                lat = time.time() - t0
                row[f"{cond}_pred_dir"]      = "obstacle_warning"
                row[f"{cond}_latency_s"]     = round(lat, 2)
                row[f"{cond}_correct"]       = False   # 방향 정확도에 포함 안 함
                row[f"{cond}_obs_warned"]    = True
                row[f"{cond}_hallucination"] = False
                row[f"{cond}_raw"]           = warn
                print(f"장애물 경고 → '{warn[:50]}'")
                continue

            # ── VLM 호출 ─────────────────────────────────────────────────────
            prompt = build_prompt(ctx if cond == "proposed" else "탐지된 객체 없음",
                                  target, condition=cond)
            raw_text = await client.call(prompt, img_bytes)
            lat = time.time() - t0

            # ── 방향 파싱 ─────────────────────────────────────────────────────
            if cond == "baseline":
                pred_dir = parse_baseline(raw_text)
                conf     = None
                vis      = None
            else:
                parsed   = parse_vlm_response(raw_text)
                pred_dir = parsed.get("goal_direction", "unknown")
                conf     = parsed.get("confidence")
                vis      = parsed.get("goal_visible")

            # ── 지표 계산 ─────────────────────────────────────────────────────
            is_correct = (pred_dir == gt_dir)

            # 할루시네이션: 목표물 안 보이는데 방향 확정 응답
            is_halluc = (not tgt_vis) and (pred_dir not in ("unknown","arrived","error"))

            # 장애물 경고 누락: 장애물 있는데 경고 못 함 (baseline/structured)
            obs_warned = False  # baseline/structured는 YOLO bypass 없음

            row[f"{cond}_pred_dir"]      = pred_dir
            row[f"{cond}_latency_s"]     = round(lat, 2)
            row[f"{cond}_confidence"]    = round(conf, 3) if conf is not None else ""
            row[f"{cond}_goal_visible"]  = vis
            row[f"{cond}_correct"]       = is_correct
            row[f"{cond}_obs_warned"]    = obs_warned
            row[f"{cond}_hallucination"] = is_halluc
            row[f"{cond}_raw"]           = raw_text[:100].replace("\n"," ")

            mark = "✅" if is_correct else "❌"
            hall = " ⚠️할루" if is_halluc else ""
            print(f"pred={pred_dir:8s} {mark} ({lat:.1f}s){hall}")

        except Exception as e:
            lat = time.time() - t0
            row[f"{cond}_pred_dir"]      = "error"
            row[f"{cond}_latency_s"]     = round(lat, 2)
            row[f"{cond}_correct"]       = False
            row[f"{cond}_obs_warned"]    = False
            row[f"{cond}_hallucination"] = False
            row[f"{cond}_raw"]           = str(e)[:80]
            print(f"ERROR: {e}")

        await asyncio.sleep(0.4)  # API 속도 제한

    return row


# ── 지표 집계 ─────────────────────────────────────────────────────────────────
def compute_metrics(rows: List[Dict], conditions: List[str]) -> Dict:
    metrics = {}
    for cond in conditions:
        total       = 0
        correct     = 0
        obs_total   = 0
        obs_miss    = 0
        hall_total  = 0
        hall_count  = 0
        unk_total   = 0   # target_visible=false 시나리오
        unk_correct = 0   # unknown/arrived 로 정확히 처리
        latencies   = []

        for r in rows:
            if "error" in r or f"{cond}_pred_dir" not in r:
                continue

            pred    = r.get(f"{cond}_pred_dir", "error")
            gt      = r.get("gt_direction")
            has_obs = r.get("has_obstacle", False)
            tgt_vis = r.get("target_visible", True)
            warned  = r.get(f"{cond}_obs_warned", False)
            is_hall = r.get(f"{cond}_hallucination", False)
            lat     = r.get(f"{cond}_latency_s", -1)

            # 장애물 경고(우회)인 경우 방향 정확도 집계 제외
            if pred == "obstacle_warning":
                if has_obs:
                    obs_total += 1
                    # 우회는 경고한 것으로 처리
                continue

            total += 1

            if r.get(f"{cond}_correct", False):
                correct += 1

            if has_obs:
                obs_total += 1
                if not warned:
                    obs_miss += 1  # baseline/structured는 경고 안 함

            if not tgt_vis:
                hall_total += 1
                unk_total  += 1
                if is_hall:
                    hall_count += 1
                else:
                    unk_correct += 1

            if lat >= 0:
                latencies.append(lat)

        metrics[cond] = {
            "방향_정확도(%)":        round(correct / total * 100, 1) if total else 0,
            "장애물_경고_누락률(%)":  round(obs_miss / obs_total * 100, 1) if obs_total else 0,
            "할루시네이션_발생률(%)": round(hall_count / hall_total * 100, 1) if hall_total else 0,
            "unknown_정확처리율(%)":  round(unk_correct / unk_total * 100, 1) if unk_total else 0,
            "평균_응답지연(s)":       round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "최대_응답지연(s)":       round(max(latencies), 2) if latencies else 0,
            "방향_정답수":            correct,
            "총_방향_시나리오":       total,
            "장애물_시나리오":        obs_total,
            "장애물_누락수":          obs_miss,
            "할루시네이션_발생수":    hall_count,
            "불확실_시나리오":        unk_total,
        }
    return metrics


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_summary(metrics: Dict, conditions: List[str]):
    print("\n" + "=" * 72)
    print("📊 평가 결과 요약")
    print("=" * 72)

    rows_def = [
        ("방향_정확도(%)",        "방향 정확도",          "%",  "↑ 높을수록 좋음  (목표 ≥70%)"),
        ("장애물_경고_누락률(%)",  "장애물 경고 누락률",   "%",  "↓ 낮을수록 좋음  (목표 0%)"),
        ("할루시네이션_발생률(%)", "할루시네이션 발생률",  "%",  "↓ 낮을수록 좋음"),
        ("unknown_정확처리율(%)",  "Unknown 정확 처리율",  "%",  "↑ 높을수록 좋음"),
        ("평균_응답지연(s)",       "평균 응답 지연",       "s",  "↓ 낮을수록 좋음  (목표 ≤3s)"),
    ]

    header = f"{'지표':<24}" + "".join(f"{c:>14}" for c in conditions)
    print(header)
    print("-" * 72)
    for key, label, unit, note in rows_def:
        line = f"{label:<24}"
        for c in conditions:
            val = metrics.get(c, {}).get(key, "-")
            line += f"{str(val)+unit:>14}"
        print(f"{line}   {note}")
    print("=" * 72)

    print("\n[상세]")
    for c in conditions:
        m = metrics.get(c, {})
        print(f"  {c}: 방향정답 {m.get('방향_정답수',0)}/{m.get('총_방향_시나리오',0)}  "
              f"| 장애물누락 {m.get('장애물_누락수',0)}/{m.get('장애물_시나리오',0)}  "
              f"| 할루 {m.get('할루시네이션_발생수',0)}/{m.get('불확실_시나리오',0)}")


# ── CSV 저장 ──────────────────────────────────────────────────────────────────
def save_results(rows: List[Dict], metrics: Dict, conditions: List[str]):
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 상세 결과
    detail = RESULTS_DIR / f"detail_{ts}.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(detail, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # 지표 요약
    summary = RESULTS_DIR / f"metrics_{ts}.csv"
    with open(summary, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["지표"] + conditions)
        all_keys = list(next(iter(metrics.values())).keys())
        for key in all_keys:
            w.writerow([key] + [metrics[c].get(key, "") for c in conditions])

    # JSON도 저장
    json_path = RESULTS_DIR / f"metrics_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n  💾 상세 결과: {detail.name}")
    print(f"  💾 지표 요약: {summary.name}")
    print(f"  💾 JSON:     {json_path.name}")
    print(f"  📂 저장 위치: {RESULTS_DIR}")


# ── scenarios.csv 템플릿 생성 ─────────────────────────────────────────────────
def create_template():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    template = [
        # id, video_path, target, gt_direction, has_obstacle, target_visible, timestamp_sec, memo
        # 화장실 8개
        (1,  "backend/evaluation/videos/toilet_01.mp4", "화장실", "straight", "false", "true",  -1, "화장실 표지판 정면"),
        (2,  "backend/evaluation/videos/toilet_02.mp4", "화장실", "left",     "false", "true",  -1, "화장실 표지판 왼쪽"),
        (3,  "backend/evaluation/videos/toilet_03.mp4", "화장실", "right",    "false", "true",  -1, "화장실 표지판 오른쪽"),
        (4,  "backend/evaluation/videos/toilet_04.mp4", "화장실", "arrived",  "false", "true",  -1, "화장실 문 바로 앞"),
        (5,  "backend/evaluation/videos/toilet_05.mp4", "화장실", "unknown",  "false", "false", -1, "화장실 안 보이는 복도"),
        (6,  "backend/evaluation/videos/toilet_06.mp4", "화장실", "straight", "true",  "true",  -1, "화장실 방향+사람 장애물"),
        (7,  "backend/evaluation/videos/toilet_07.mp4", "화장실", "straight", "true",  "true",  -1, "화장실 방향+의자 장애물"),
        (8,  "backend/evaluation/videos/toilet_08.mp4", "화장실", "straight", "false", "true",  -1, "멀리서 화장실 표지판"),
        # 강의실 8개
        (9,  "backend/evaluation/videos/class_01.mp4",  "강의실", "straight", "false", "true",  -1, "강의실 번호판 정면"),
        (10, "backend/evaluation/videos/class_02.mp4",  "강의실", "left",     "false", "true",  -1, "강의실 번호판 왼쪽"),
        (11, "backend/evaluation/videos/class_03.mp4",  "강의실", "right",    "false", "true",  -1, "강의실 번호판 오른쪽"),
        (12, "backend/evaluation/videos/class_04.mp4",  "강의실", "arrived",  "false", "true",  -1, "강의실 문 바로 앞"),
        (13, "backend/evaluation/videos/class_05.mp4",  "강의실", "unknown",  "false", "false", -1, "강의실 안 보이는 복도"),
        (14, "backend/evaluation/videos/class_06.mp4",  "강의실", "straight", "true",  "true",  -1, "강의실 방향+사람 장애물"),
        (15, "backend/evaluation/videos/class_07.mp4",  "강의실", "straight", "true",  "true",  -1, "강의실 방향+계단 장애물"),
        (16, "backend/evaluation/videos/class_08.mp4",  "강의실", "left",     "false", "true",  -1, "여러 강의실 문 복도"),
        # 엘리베이터 9개
        (17, "backend/evaluation/videos/elev_01.mp4",   "엘리베이터", "straight", "false", "true",  -1, "엘리베이터 정면"),
        (18, "backend/evaluation/videos/elev_02.mp4",   "엘리베이터", "left",     "false", "true",  -1, "엘리베이터 왼쪽"),
        (19, "backend/evaluation/videos/elev_03.mp4",   "엘리베이터", "right",    "false", "true",  -1, "엘리베이터 오른쪽"),
        (20, "backend/evaluation/videos/elev_04.mp4",   "엘리베이터", "arrived",  "false", "true",  -1, "엘리베이터 바로 앞"),
        (21, "backend/evaluation/videos/elev_05.mp4",   "엘리베이터", "unknown",  "false", "false", -1, "엘리베이터 안 보이는 복도"),
        (22, "backend/evaluation/videos/elev_06.mp4",   "엘리베이터", "straight", "true",  "true",  -1, "엘리베이터+사람 장애물"),
        (23, "backend/evaluation/videos/elev_07.mp4",   "엘리베이터", "straight", "true",  "true",  -1, "엘리베이터+계단 근처"),
        (24, "backend/evaluation/videos/elev_08.mp4",   "엘리베이터", "arrived",  "false", "true",  -1, "엘리베이터 문 열린 상태"),
        (25, "backend/evaluation/videos/elev_09.mp4",   "엘리베이터", "straight", "false", "true",  -1, "어두운 복도 엘리베이터"),
    ]

    fields = ["id","video_path","target","gt_direction",
              "has_obstacle","target_visible","timestamp_sec","memo"]
    with open(SCENARIOS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(template)

    print(f"✅ 템플릿 생성: {SCENARIOS_CSV}")
    print(f"✅ 영상 폴더:   {VIDEOS_DIR}")
    print()
    print("다음 단계:")
    print("  1. backend/evaluation/videos/ 에 영상 파일 넣기")
    print("     파일명은 scenarios.csv 의 video_path 와 일치해야 함")
    print("     예: toilet_01.mp4, class_01.mp4, elev_01.mp4 ...")
    print("  2. scenarios.csv 열어서 gt_direction, has_obstacle 등 정답 확인")
    print("  3. python backend/eval_scenarios.py 실행")


# ── 메인 ──────────────────────────────────────────────────────────────────────
async def main(scenarios_path: Path, conditions: List[str], limit: Optional[int]):
    print("=" * 60)
    print("📹 25개 시나리오 3조건 비교 평가")
    print(f"   조건: {conditions}")
    print("=" * 60)

    client = VLMClient()
    print(f"  [VLM] {client.provider} / {client.model}")

    yolo = YOLODetector() if "proposed" in conditions else \
           type("NoYOLO", (), {"ok": False, "detect": lambda s, f: []})()

    # 시나리오 로드
    with open(scenarios_path, encoding="utf-8-sig") as f:
        scenarios = list(csv.DictReader(f))
    if limit:
        scenarios = scenarios[:limit]

    # 절대경로 보정 + 이미지 확장자 자동 탐색
    # video_path가 .mp4여도 같은 이름의 이미지가 있으면 이미지 사용
    IMAGE_FALLBACK = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.heic', '.HEIF']
    VIDEO_EXTS     = ['.mp4', '.mov', '.MOV', '.MP4', '.avi']

    def resolve_path(vp: str):
        """CSV의 video_path에서 실제 파일 경로를 찾는다. 이미지/영상 모두 지원."""
        base_stem = Path(vp).stem
        base_dir  = Path(vp).parent

        # 1. 그대로 존재하면 사용
        for root_prefix in [Path('.'), ROOT]:
            full = root_prefix / vp
            if full.exists():
                return str(full)

        # 2. 같은 이름으로 이미지 확장자 탐색 (jpg, jpeg, png ...)
        for root_prefix in [Path('.'), ROOT]:
            for ext in IMAGE_FALLBACK + VIDEO_EXTS:
                candidate = root_prefix / base_dir / f"{base_stem}{ext}"
                if candidate.exists():
                    return str(candidate)

        return None  # 못 찾음

    valid, skipped = [], []
    for sc in scenarios:
        resolved = resolve_path(sc["video_path"])
        if resolved:
            sc["video_path"] = resolved
            valid.append(sc)
        else:
            skipped.append(sc)

    if skipped:
        print(f"\n  ⚠️  파일 없는 시나리오 {len(skipped)}개 건너뜀:")
        for s in skipped[:3]:
            print(f"     {s['id']}: {s['video_path']}")
        if len(skipped) > 3:
            print(f"     ... 외 {len(skipped)-3}개")

    if not valid:
        print("\n평가할 파일이 없습니다.")
        print("아래 폴더에 사진 또는 영상을 넣어주세요:")
        print(f"  {VIDEOS_DIR}")
        print("파일명 예시: toilet_01.jpg / class_01.png / elev_01.mp4")
        return

    print(f"\n  실행: {len(valid)}/{len(scenarios)}개 시나리오\n")

    # 평가 실행
    all_rows = []
    for i, sc in enumerate(valid):
        print(f"[{i+1:02d}/{len(valid):02d}]", end="")
        row = await eval_one(sc, client, yolo, conditions)
        all_rows.append(row)

    # 지표 계산 및 출력
    metrics = compute_metrics(all_rows, conditions)
    print_summary(metrics, conditions)
    save_results(all_rows, metrics, conditions)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="25 시나리오 3조건 평가")
    ap.add_argument("--scenarios", default=str(SCENARIOS_CSV))
    ap.add_argument("--condition", nargs="+",
                    default=["baseline","structured","proposed"],
                    choices=["baseline","structured","proposed"])
    ap.add_argument("--limit", type=int, default=None, help="앞 N개만 실행 (테스트용)")
    ap.add_argument("--create-template", action="store_true",
                    help="scenarios.csv 템플릿 생성 후 종료")
    args = ap.parse_args()

    if args.create_template:
        create_template()
        sys.exit(0)

    sc_path = Path(args.scenarios)
    if not sc_path.exists():
        print(f"시나리오 파일 없음 → 템플릿 자동 생성")
        create_template()

    asyncio.run(main(sc_path, args.condition, args.limit))
