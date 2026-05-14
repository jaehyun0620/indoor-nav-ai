"""
yolo_midas.py
YOLOv8 + Depth Anything V2 Metric 래퍼.

- YOLOv8     : ultralytics (재학습 없이 추론만)
- Depth Anything V2 Metric Indoor : HuggingFace transformers
  - 출력이 실제 미터 단위 (scale_factor 보정 불필요)
  - 실내 환경 특화 학습 모델
  - MiDaS 대비 정확도 대폭 향상
"""

import os
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
# torch 는 _load_depth_anything_v2() 안에서 지연 import
# → 서버 시작 시 torch 미설치 환경에서도 정상 기동 가능

# ── YOLOv8 ──────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ── Depth Anything V2 ────────────────────────────────────────────────────────

_DA2_PIPE = None
_DA2_DEVICE = None

# 실내 복도 환경 최적화 모델
# Small  → 빠름, 정확도 MiDaS 대비 대폭 향상
# Base   → 균형
# Large  → 가장 정확, 느림
DA2_MODEL_MAP = {
    "small":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base":   "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


def _load_depth_anything_v2(model_size: str = "small") -> None:
    """Depth Anything V2 Metric 모델을 처음 한 번만 로드한다."""
    global _DA2_PIPE, _DA2_DEVICE

    if _DA2_PIPE is not None:
        return

    import torch  # 지연 import — 서버 기동 시 불필요

    # 디바이스 선택: M-시리즈 Mac → MPS, NVIDIA → CUDA, 그 외 → CPU
    if torch.backends.mps.is_available():
        _DA2_DEVICE = "mps"
        pipe_device = "mps"
    elif torch.cuda.is_available():
        _DA2_DEVICE = "cuda"
        pipe_device = 0
    else:
        _DA2_DEVICE = "cpu"
        pipe_device = -1

    from transformers import pipeline as hf_pipeline

    model_name = DA2_MODEL_MAP.get(model_size, DA2_MODEL_MAP["small"])
    _DA2_PIPE = hf_pipeline(
        task="depth-estimation",
        model=model_name,
        device=pipe_device,
    )


def estimate_depth_map(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Depth Anything V2 Metric으로 깊이맵을 반환한다.

    MiDaS와 달리 출력값이 실제 미터 단위다.
    값이 클수록 카메라에서 멀리 있는 픽셀이다. (MiDaS와 반대)

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR 형식의 카메라 프레임

    Returns
    -------
    np.ndarray
        float32 깊이맵 (단위: 미터, frame_bgr과 동일한 H×W)
    """
    _load_depth_anything_v2(
        model_size=os.getenv("DA2_MODEL_SIZE", "small")
    )

    from PIL import Image

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    result = _DA2_PIPE(pil_image)

    # predicted_depth = 실제 미터 단위 텐서
    # result["depth"] 는 시각화용 PIL 이미지 (0~255 픽셀값, 거리 아님)
    depth_np = result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)

    # 원본 해상도로 리사이즈 (모델 내부 해상도와 다를 수 있음)
    if depth_np.shape[:2] != frame_bgr.shape[:2]:
        depth_np = cv2.resize(
            depth_np,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return depth_np


def bbox_center_depth(
    depth_map: np.ndarray,
    bbox: List[float],
) -> float:
    """
    bbox 내부 중심 영역(20×20px)의 중앙값 깊이를 반환한다.
    Depth Anything V2 사용 시 반환값은 미터 단위 거리다.

    Parameters
    ----------
    depth_map : np.ndarray
        estimate_depth_map() 의 반환값 (미터 단위)
    bbox : List[float]
        [x1, y1, x2, y2] 픽셀 좌표

    Returns
    -------
    float
        중앙값 깊이 (미터)
    """
    h, w = depth_map.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]

    cx = max(0, min((x1 + x2) // 2, w - 1))
    cy = max(0, min((y1 + y2) // 2, h - 1))

    half = 10
    patch = depth_map[
        max(0, cy - half): min(h, cy + half),
        max(0, cx - half): min(w, cx + half),
    ]

    if patch.size == 0:
        return float(depth_map[cy, cx])

    return float(np.median(patch))


def inverse_depth_to_meters(
    depth_m: float,
    depth_map: np.ndarray = None,
    scale_factor: float = 1.0,
) -> float:
    """
    Depth Anything V2 Metric 사용 시 이미 미터 단위이므로
    그대로 반환한다. (scale_factor, depth_map 인자는 하위 호환용)
    """
    if depth_m <= 0:
        return 999.0
    return round(float(depth_m), 2)


# ── YOLOMiDaSWrapper ────────────────────────────────────────────────────────

class YOLOMiDaSWrapper:
    """
    YOLOv8 탐지 + Depth Anything V2 Metric 거리 추정 래퍼 클래스.
    (클래스명은 하위 호환성을 위해 유지)

    최적화 전략:
    - YOLO  : 매 프레임 실행 (장애물 위치 파악)
    - DA2   : depth_interval 프레임마다 1회 실행 후 캐시 재사용
              연속 프레임 사이 깊이맵 변화가 작으므로 재사용해도 안전
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        midas_model: str = "small",       # DA2 모델 크기: small / base / large
        conf_threshold: float = 0.4,
        scale_factor: float = 1.0,        # DA2는 미터 단위 직접 출력, 사용 안 함
        depth_interval: int = 2,          # DA2를 몇 프레임마다 실행할지
    ):
        """
        Parameters
        ----------
        yolo_model : str
            ultralytics YOLO 모델 경로 또는 공식 모델명
        midas_model : str
            Depth Anything V2 모델 크기 ("small" | "base" | "large")
        conf_threshold : float
            YOLO confidence 임계값
        scale_factor : float
            하위 호환용 (Depth Anything V2에서는 사용 안 함)
        depth_interval : int
            DA2 깊이 추정 실행 주기 (기본 5 → 5프레임에 1회 실행)
            환경변수 DA2_DEPTH_INTERVAL 로 재정의 가능
        """
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics 패키지가 설치되지 않았습니다. pip install ultralytics")

        # ── 커스텀 모델: elevator / stairs 전용 ──────────────────────────────
        self.model = _YOLO(yolo_model)

        # ── 기본 모델: person / chair 등 COCO 일반 장애물 전용 ───────────────
        # 커스텀 파인튜닝으로 사람 감지 능력이 없어진 것을 보완
        base_model_path = os.getenv("YOLO_BASE_MODEL", "yolov8n.pt")
        try:
            self.base_model = _YOLO(base_model_path)
        except Exception:
            self.base_model = None

        self.conf_threshold = conf_threshold
        self.scale_factor = scale_factor
        self.model_size = midas_model
        self.depth_interval = int(os.getenv("DA2_DEPTH_INTERVAL", str(depth_interval)))

        # 커스텀 모델이 탐지할 클래스 (파인튜닝한 것)
        self.custom_classes = {"elevator", "stairs", "stair"}
        # 기본 모델이 탐지할 클래스 (COCO 중 복도 장애물)
        self.base_classes = {"person", "chair", "table", "suitcase", "backpack",
                             "stairs", "stair"}
        # 전체 허용 클래스
        self.valid_classes = self.custom_classes | self.base_classes

        # ── 클래스별 confidence 임계값 오버라이드 ────────────────────────────
        # 복도 환경에서 오탐이 많은 클래스는 전역 threshold보다 높게 설정한다.
        # 환경변수로도 조정 가능: YOLO_CONF_PERSON=0.65
        self.class_conf_overrides: dict = {
            "person": float(os.getenv("YOLO_CONF_PERSON", "0.65")),
            # 필요 시 추가: "chair": 0.55, "bicycle": 0.60
        }

        # ── 최소 bbox 크기 필터 ──────────────────────────────────────────────
        # 프레임 전체 면적 대비 bbox 면적이 이 비율 미만이면 noise로 제거.
        # 너무 작은 bbox는 멀리 있거나 배경 패턴으로 인한 오탐일 가능성이 높다.
        # 환경변수: YOLO_MIN_BBOX_RATIO=0.015
        self.min_bbox_area_ratio: float = float(os.getenv("YOLO_MIN_BBOX_RATIO", "0.015"))

        # 깊이맵 캐시
        self._depth_cache: Optional[np.ndarray] = None
        self._frame_count: int = 0

        _load_depth_anything_v2(midas_model)

    def run(self, frame_bgr: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        프레임 한 장에 대해 YOLO 탐지 + Depth Anything V2 거리 추정을 실행한다.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR 카메라 프레임 (640×480 권장)

        Returns
        -------
        Tuple[List[Dict], Dict]
            detections : 탐지 결과 리스트
                [{"class": str, "bbox": [...], "distance_m": float, "conf": float}, ...]
            fast_result : 우선순위 모듈용 요약 dict
        """
        from backend.modules.context_builder import build_obstacle_summary

        # 1) DA2 깊이맵 — depth_interval 프레임마다 갱신, 나머지는 캐시 재사용
        # 카운터를 증가 전에 체크해야 depth_interval=2 시 프레임 1,3,5... 에서 실행됨
        # (증가 후 체크하면 첫 프레임(count=1)과 두 번째 프레임(count=2)이 연속 실행되는 off-by-one 발생)
        if self._depth_cache is None or self._frame_count % self.depth_interval == 0:
            self._depth_cache = estimate_depth_map(frame_bgr)
        self._frame_count += 1

        depth_map = self._depth_cache

        # 2) YOLO 탐지 (매 프레임 실행)
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        detections: List[Dict] = []

        def _extract(results, allowed_classes):
            """YOLO 결과에서 허용 클래스만 필터링해 반환"""
            found = []
            for result in results:
                for box in result.boxes:
                    cls_id  = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf    = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox    = [x1, y1, x2, y2]

                    if cls_name not in allowed_classes:
                        continue

                    class_threshold = self.class_conf_overrides.get(cls_name, self.conf_threshold)
                    if conf < class_threshold:
                        continue

                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area / frame_area < self.min_bbox_area_ratio:
                        continue

                    distance_m = bbox_center_depth(depth_map, bbox)
                    found.append({"class": cls_name, "bbox": bbox,
                                  "distance_m": distance_m, "conf": conf})
            return found

        # 메인 모델 실행 (yolov8n.pt 단독 또는 커스텀 모델)
        main_results = self.model(frame_bgr, conf=self.conf_threshold, verbose=False)
        detections += _extract(main_results, self.valid_classes)

        # 기본 모델이 별도로 있으면 추가 탐지 (커스텀+기본 듀얼 모드)
        if self.base_model is not None:
            base_results = self.base_model(frame_bgr, conf=self.conf_threshold, verbose=False)
            detections += _extract(base_results, self.base_classes)

        # 4) 가장 가까운 장애물 요약
        summary = build_obstacle_summary(detections, frame_width=frame_bgr.shape[1])
        fast_result = {
            "class": summary["closest_class"],
            "distance_m": summary["closest_distance"],
            "direction": summary["direction"],
            "has_obstacle": summary["has_obstacle"],
            "conf": detections[0]["conf"] if detections else 0.0,
            "bbox": detections[0]["bbox"] if detections else [0, 0, 0, 0],
            "detections": detections,
        }

        return detections, fast_result
