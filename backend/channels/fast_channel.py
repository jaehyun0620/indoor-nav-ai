"""
fast_channel.py
빠른 채널: YOLOv8 + Depth 기반 거리 추정 + 안정화 필터 (30fps 목표)
"""

import base64
from typing import Dict

import cv2
import numpy as np

from backend.models.yolo_midas import YOLOMiDaSWrapper
from backend.modules.context_builder import build_context


class FastChannel:
    """
    빠른 채널 실행 클래스.

    주요 기능:
    - YOLOv8 객체 탐지
    - Depth 기반 거리 추정
    - 오탐 제거 (confidence / bbox / distance 필터)
    - temporal 안정화 (이전 프레임 비교)
    - VLM용 context 생성
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        midas_model: str = "small",
        conf_threshold: float = 0.6,   # 🔥 강화
        scale_factor: float = 1.0,
        depth_interval: int = 2,
    ):
        self.wrapper = YOLOMiDaSWrapper(
            yolo_model=yolo_model,
            midas_model=midas_model,
            conf_threshold=conf_threshold,
            scale_factor=scale_factor,
            depth_interval=depth_interval,
        )

        # 🔥 이전 프레임 저장 (temporal filter)
        self.prev_detections = []

    # ─────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        프레임 입력 → 필터링 → 안정화 → context 생성
        """

        detections, fast_result = self.wrapper.run(frame_bgr)

        H, W = frame_bgr.shape[:2]

        # ─────────────────────────────
        # 1. 1차 필터링
        # ─────────────────────────────
        filtered = []

        for det in detections:
            conf = det.get("conf", 0)
            dist = det.get("distance_m", 999)
            bbox = det.get("bbox", [0, 0, 0, 0])
            cls = det.get("class", "")

            x1, y1, x2, y2 = bbox
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))

            # 🔥 필터 1: confidence
            if conf < 0.6:
                continue

            # 🔥 필터 2: bbox 크기 (너무 작으면 제거)
            if area < 0.01 * (W * H):
                continue

            # 🔥 필터 3: 거리 제한
            if dist > 5.0:
                continue

            # 🔥 (선택) 사람만 쓰고 싶으면 활성화
            # if cls != "person":
            #     continue

            filtered.append(det)

        # ─────────────────────────────
        # 2. temporal 안정화
        # ─────────────────────────────
        # 안전 우선 임계 거리: 이 거리 이내 장애물은 temporal 필터 없이 즉시 통과
        SAFETY_DIST = 3.0  # m  (필터3 상한인 5.0m 와 구분된 값)

        if not self.prev_detections:
            # 첫 프레임 또는 reset 직후 — 비교 대상 없으므로 필터 없이 통과
            stable = filtered
        else:
            stable = []
            for det in filtered:
                # SAFETY_DIST 이내: 안전 우선으로 즉시 통과 (1프레임 지연 없음)
                if det["distance_m"] <= SAFETY_DIST:
                    stable.append(det)
                    continue
                # SAFETY_DIST 초과 (3~5m): 이전 프레임에도 비슷한 거리에 같은 클래스가
                # 있었던 경우에만 stable에 포함 → 노이즈성 오탐 제거
                for prev in self.prev_detections:
                    if (
                        prev.get("class") == det.get("class")
                        and abs(det["distance_m"] - prev["distance_m"]) < 0.8
                    ):
                        stable.append(det)
                        break

        # 다음 프레임을 위해 저장
        self.prev_detections = filtered

        # ─────────────────────────────
        # 3. context 생성
        # ─────────────────────────────
        yolo_context = build_context(stable, frame_width=W)

        return {
            "detections": stable,
            "fast_result": fast_result,
            "yolo_context": yolo_context,
        }

    # ─────────────────────────────────────────────

    def process_bytes(self, image_bytes: bytes) -> Dict:
        """
        JPEG/PNG 바이트 입력 처리
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("이미지 디코딩 실패")

        return self.process_frame(frame)

    # ─────────────────────────────────────────────

    def process_base64(self, b64_string: str) -> Dict:
        """
        Base64 이미지 입력 처리 (WebSocket용)
        """
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(b64_string)
        return self.process_bytes(image_bytes)