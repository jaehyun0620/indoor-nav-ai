"""
yolo_midas.py
YOLOv8 + MiDaS 래퍼.
- YOLOv8: ultralytics (재학습 없이 추론만)
- MiDaS: torch.hub (DPT_Large 또는 MiDaS_small)
"""

import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch

# ── YOLOv8 ──────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO as _YOLO

    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ── MiDaS ────────────────────────────────────────────────────────────────────

_MIDAS_MODEL = None
_MIDAS_TRANSFORM = None
_MIDAS_DEVICE = None


def _load_midas(model_type: str = "MiDaS_small") -> None:
    """MiDaS 모델을 처음 한 번만 로드한다."""
    global _MIDAS_MODEL, _MIDAS_TRANSFORM, _MIDAS_DEVICE

    if _MIDAS_MODEL is not None:
        return

    _MIDAS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    _MIDAS_MODEL.to(_MIDAS_DEVICE)
    _MIDAS_MODEL.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        _MIDAS_TRANSFORM = transforms.dpt_transform
    else:
        _MIDAS_TRANSFORM = transforms.small_transform


def estimate_depth_map(frame_bgr: np.ndarray) -> np.ndarray:
    """
    MiDaS로 프레임 전체의 역깊이 맵(inverse depth map)을 반환한다.
    반환값이 클수록 카메라에 가까운 픽셀이다.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR 형식의 카메라 프레임

    Returns
    -------
    np.ndarray
        float32 역깊이 맵 (frame_bgr과 동일한 H×W)
    """
    _load_midas()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = _MIDAS_TRANSFORM(frame_rgb).to(_MIDAS_DEVICE)

    with torch.no_grad():
        prediction = _MIDAS_MODEL(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy().astype(np.float32)
    return depth_map


def inverse_depth_to_meters(
    inv_depth: float,
    depth_map: np.ndarray,
    scale_factor: float = 5.0,
) -> float:
    """
    역깊이 값을 미터 단위 거리로 변환한다.
    MiDaS는 절대 거리를 직접 출력하지 않으므로
    depth_map의 통계를 이용해 정규화 후 scale_factor를 곱한다.

    scale_factor는 현장 보정값 (기본 5.0m, 실내 복도 기준).
    """
    d_max = float(depth_map.max()) if depth_map.max() > 0 else 1.0
    normalized = inv_depth / d_max          # 0.0 ~ 1.0 (클수록 가까움)
    if normalized <= 0:
        return 999.0
    distance_m = scale_factor / normalized  # 가까울수록 normalized 크고 distance 작음
    return round(float(distance_m), 2)


def bbox_center_depth(
    depth_map: np.ndarray,
    bbox: List[float],
) -> float:
    """
    bbox 내부 중심 영역(20×20px)의 중앙값 역깊이를 반환한다.

    Parameters
    ----------
    depth_map : np.ndarray
        estimate_depth_map() 의 반환값
    bbox : List[float]
        [x1, y1, x2, y2] 픽셀 좌표

    Returns
    -------
    float
        중앙값 역깊이 값
    """
    h, w = depth_map.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # bbox 중심 좌표
    cx = max(0, min((x1 + x2) // 2, w - 1))
    cy = max(0, min((y1 + y2) // 2, h - 1))

    # 중심 주변 20×20 영역
    half = 10
    patch = depth_map[
        max(0, cy - half): min(h, cy + half),
        max(0, cx - half): min(w, cx + half),
    ]

    if patch.size == 0:
        return float(depth_map[cy, cx])

    return float(np.median(patch))


# ── YOLOMiDaSWrapper ────────────────────────────────────────────────────────

class YOLOMiDaSWrapper:
    """
    YOLOv8 탐지 + MiDaS 거리 추정을 함께 수행하는 래퍼 클래스.
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        midas_model: str = "MiDaS_small",
        conf_threshold: float = 0.4,
        scale_factor: float = 5.0,
    ):
        """
        Parameters
        ----------
        yolo_model : str
            ultralytics YOLO 모델 경로 또는 공식 모델명 (예: "yolov8n.pt")
        midas_model : str
            MiDaS 모델 타입 ("MiDaS_small" | "DPT_Large" | "DPT_Hybrid")
        conf_threshold : float
            YOLO confidence 임계값
        scale_factor : float
            MiDaS 역깊이 → 미터 변환 보정값 (현장 캘리브레이션 필요)
        """
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics 패키지가 설치되지 않았습니다. pip install ultralytics")

        self.model = _YOLO(yolo_model)
        self.conf_threshold = conf_threshold
        self.scale_factor = scale_factor
        self.midas_model_type = midas_model
        _load_midas(midas_model)

    def run(self, frame_bgr: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        프레임 한 장에 대해 YOLO 탐지 + MiDaS 거리 추정을 실행한다.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR 카메라 프레임 (640×480 권장)

        Returns
        -------
        Tuple[List[Dict], Dict]
            detections : 탐지 결과 리스트
                [
                    {
                        "class": str,
                        "bbox": [x1, y1, x2, y2],
                        "distance_m": float,
                        "conf": float
                    },
                    ...
                ]
            fast_result : 우선순위 모듈용 요약 dict
                {
                    "class": str,          # 가장 가까운 객체 (한국어)
                    "distance_m": float,
                    "bbox": list,
                    "conf": float,
                    "detections": list     # 전체 탐지 목록
                }
        """
        from backend.modules.context_builder import CLASS_KO, build_obstacle_summary

        # 1) MiDaS 깊이 맵
        depth_map = estimate_depth_map(frame_bgr)

        # 2) YOLO 탐지
        results = self.model(frame_bgr, conf=self.conf_threshold, verbose=False)
        detections: List[Dict] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [x1, y1, x2, y2]

                # 3) MiDaS 거리 추정
                inv_d = bbox_center_depth(depth_map, bbox)
                distance_m = inverse_depth_to_meters(inv_d, depth_map, self.scale_factor)

                detections.append(
                    {
                        "class": cls_name,
                        "bbox": bbox,
                        "distance_m": distance_m,
                        "conf": conf,
                    }
                )

        # 4) 요약 (가장 가까운 장애물)
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
