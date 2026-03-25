"""
priority_module.py
빠른 채널(장애물)과 느린 채널(방향 안내)의 결과를 합류시켜
최종 TTS 메시지와 우선순위를 결정하는 모듈.

경로 A: 장애물 감지 → VLM 완전 우회, 즉각 경고 출력
경로 B: 안전 확인 → VLM 방향 안내 출력
"""

from typing import Dict


class PriorityModule:
    # 즉각 경고 거리 임계값 (m)
    CRITICAL_DISTANCE = 1.0
    CAUTION_DISTANCE = 2.0

    def decide(self, fast_result: Dict, slow_result: Dict) -> Dict:
        """
        빠른 채널 결과와 느린 채널 결과를 받아 최종 출력 메시지를 결정한다.

        Parameters
        ----------
        fast_result : dict
            빠른 채널(YOLOv8 + MiDaS) 출력:
            {
                "class": str,           # 가장 가까운 장애물 클래스 (한국어)
                "distance_m": float,    # 추정 거리 (m)
                "bbox": list,           # [x1, y1, x2, y2]
                "conf": float,          # 신뢰도
                "detections": list      # 전체 탐지 목록
            }
        slow_result : dict
            느린 채널(VLM + ConsistencyFilter) 출력:
            {
                "confirmed_direction": str,  # left / right / straight / unknown
                "tts_text": str,             # 한국어 안내 문장
                "unknown_streak": int
            }

        Returns
        -------
        dict
            {
                "message_type": str,       # "warning" / "caution" / "guidance" / "unknown"
                "tts_text": str,           # 최종 TTS 출력 문장
                "priority": int,           # 1(긴급) ~ 3(낮음)
                "suppress_guidance": bool  # True면 VLM 방향 안내 억제
            }
        """
        distance = fast_result.get("distance_m", 999.0)
        obj_class = fast_result.get("class", "장애물")
        direction = slow_result.get("confirmed_direction", "unknown")
        tts_text = slow_result.get("tts_text", "현재 위치를 파악 중입니다")

        # ── 경로 A: 즉각 경고 ──────────────────────────────────────────────
        if distance < self.CRITICAL_DISTANCE:
            return {
                "message_type": "warning",
                "tts_text": f"즉시 멈추세요. {obj_class}이 있습니다.",
                "priority": 1,
                "suppress_guidance": True,
            }

        if self.CRITICAL_DISTANCE <= distance < self.CAUTION_DISTANCE:
            caution = f"주의: {obj_class} {distance:.1f}m 앞"
            if direction and direction != "unknown":
                combined = f"{caution}, {tts_text}"
            else:
                combined = caution
            return {
                "message_type": "caution",
                "tts_text": combined,
                "priority": 1,
                "suppress_guidance": False,
            }

        # ── 경로 B: VLM 방향 안내 ────────────────────────────────────────
        if direction and direction != "unknown":
            return {
                "message_type": "guidance",
                "tts_text": tts_text,
                "priority": 2,
                "suppress_guidance": False,
            }

        # ── 방향 미확정 ───────────────────────────────────────────────────
        return {
            "message_type": "unknown",
            "tts_text": tts_text,
            "priority": 3,
            "suppress_guidance": False,
        }
