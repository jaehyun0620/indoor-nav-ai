"""
scene_memory.py
최근 N프레임의 탐지 결과와 VLM 응답을 버퍼링하여
장면 맥락(Scene Context)을 유지하는 모듈.
"""

import time
from collections import deque
from typing import Dict, List, Optional


class SceneMemory:
    """
    최근 장면 정보를 짧은 시간 동안 기억해두는 버퍼.

    주요 용도:
    - 이전 프레임의 탐지 결과를 참조해 현재 프롬프트를 보강
    - 방향 안내 이력을 유지해 급격한 방향 변경을 억제
    - 디버깅/로깅용 히스토리 제공
    """

    def __init__(self, maxlen: int = 10, ttl: float = 5.0):
        """
        Parameters
        ----------
        maxlen : int
            기억할 최대 항목 수 (기본 10)
        ttl : float
            항목 유효 시간 (초, 기본 5.0)
        """
        self.maxlen = maxlen
        self.ttl = ttl
        self._buffer: deque = deque(maxlen=maxlen)

    def update(self, detections: List[Dict], vlm_result: Dict) -> None:
        """
        새 프레임의 탐지 결과와 VLM 파싱 결과를 버퍼에 추가한다.

        Parameters
        ----------
        detections : List[Dict]
            fast_channel 탐지 결과 목록
        vlm_result : Dict
            prompt_designer.parse_vlm_response() 파싱 결과
        """
        self._buffer.append(
            {
                "timestamp": time.time(),
                "detections": detections,
                "vlm": vlm_result,
            }
        )

    def get_recent(self, n: int = 3) -> List[Dict]:
        """
        TTL 이내의 최근 n개 항목을 반환한다 (오래된 것부터).

        Parameters
        ----------
        n : int
            가져올 최대 항목 수

        Returns
        -------
        List[Dict]
            유효한 최근 항목 리스트
        """
        now = time.time()
        valid = [e for e in self._buffer if now - e["timestamp"] < self.ttl]
        return valid[-n:]

    def get_last_direction(self) -> Optional[str]:
        """
        버퍼에서 가장 최근의 확정 방향을 반환한다.
        unknown이 아닌 마지막 방향을 우선한다.

        Returns
        -------
        str or None
            "left" / "right" / "straight" / None
        """
        for entry in reversed(list(self._buffer)):
            direction = entry.get("vlm", {}).get("goal_direction")
            if direction and direction != "unknown":
                return direction
        return None

    def get_context_summary(self) -> str:
        """
        최근 탐지된 객체 빈도를 텍스트로 요약한다.
        (프롬프트 보강용)

        Returns
        -------
        str
            예: "최근 탐지: 사람(3회), 의자(2회)"
        """
        from collections import Counter
        from backend.modules.context_builder import CLASS_KO

        recent = self.get_recent(n=5)
        if not recent:
            return "최근 탐지 없음"

        counter: Counter = Counter()
        for entry in recent:
            for det in entry.get("detections", []):
                cls_en = det.get("class", "")
                cls_ko = CLASS_KO.get(cls_en, cls_en)
                counter[cls_ko] += 1

        if not counter:
            return "최근 탐지 없음"

        summary = ", ".join(f"{cls}({cnt}회)" for cls, cnt in counter.most_common(5))
        return f"최근 탐지: {summary}"

    def get_context_for_prompt(self) -> str:
        """
        최근 확정 방향 이력을 프롬프트 보강용 텍스트로 반환한다.
        느린 채널 프롬프트에 주입해 VLM이 방향 일관성을 유지하도록 돕는다.

        Returns
        -------
        str
            예: "이전 분석에서 목적지는 왼쪽으로 확인됨. 현재도 왼쪽인지 재확인해줘."
            이력 없으면 빈 문자열.
        """
        last = self.get_last_direction()
        if not last:
            return ""
        label = {"left": "왼쪽", "right": "오른쪽", "straight": "직진"}.get(last, last)
        return f"이전 분석에서 목적지는 {label}으로 확인됨. 현재도 {label}인지 재확인해줘."

    def reset(self) -> None:
        """버퍼를 초기화한다."""
        self._buffer.clear()
