"""
fast_channel.py
빠른 채널: YOLOv8 + MiDaS로 장애물을 탐지하고 거리를 추정한다. (30fps 목표)
"""

import base64
from typing import Dict, List, Optional

import cv2
import numpy as np

from backend.models.yolo_midas import YOLOMiDaSWrapper
from backend.modules.context_builder import build_context, build_obstacle_summary


class FastChannel:
    """
    빠른 채널 실행 클래스.
    YOLOv8 + MiDaS 추론 결과를 반환하며,
    context_builder를 통해 VLM 프롬프트용 텍스트도 함께 제공한다.
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        midas_model: str = "small",
        conf_threshold: float = 0.4,
        scale_factor: float = 1.0,
        depth_interval: int = 5,
    ):
        self.wrapper = YOLOMiDaSWrapper(
            yolo_model=yolo_model,
            midas_model=midas_model,
            conf_threshold=conf_threshold,
            scale_factor=scale_factor,
            depth_interval=depth_interval,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        BGR 프레임을 받아 탐지 결과를 반환한다.

        Parameters
        ----------
        frame_bgr : np.ndarray
            카메라에서 읽은 BGR 프레임

        Returns
        -------
        dict
            {
                "detections": List[Dict],   # 전체 탐지 목록
                "fast_result": Dict,        # 우선순위 모듈용 요약
                "yolo_context": str         # VLM 프롬프트용 텍스트
            }
        """
        detections, fast_result = self.wrapper.run(frame_bgr)
        yolo_context = build_context(detections, frame_width=frame_bgr.shape[1])

        return {
            "detections": detections,
            "fast_result": fast_result,
            "yolo_context": yolo_context,
        }

    def process_bytes(self, image_bytes: bytes) -> Dict:
        """
        JPEG/PNG 바이트 스트림을 받아 탐지 결과를 반환한다.
        (FastAPI multipart 업로드 대응)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("이미지 디코딩 실패")
        return self.process_frame(frame)

    def process_base64(self, b64_string: str) -> Dict:
        """
        Base64 인코딩된 이미지 문자열을 받아 탐지 결과를 반환한다.
        (WebSocket 프레임 스트리밍 대응)
        """
        # data:image/jpeg;base64,... 헤더 제거
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_string)
        return self.process_bytes(image_bytes)
