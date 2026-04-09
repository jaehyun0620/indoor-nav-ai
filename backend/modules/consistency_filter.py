"""
consistency_filter.py
VLM 응답의 일관성을 검증하는 필터.
deque(maxlen=3) 버퍼 + TTL + 다수결(2/3) 방향 확정.

TTL 설계 기준:
  - VLM 호출 간격(SLOW_CHANNEL_INTERVAL, 기본 2.5초) * 버퍼크기(3) = 7.5초
  - 여유 margin 4.5초 추가 → 기본 TTL = 12초
  - TTL이 호출 간격보다 짧으면 버퍼가 항상 비어 방향 확정 불가
  - TTL이 너무 길면(30초) 사용자가 방향을 틀어도 이전 결과가 유효로 남아
    틀린 방향을 계속 안내하는 문제 발생 → 12초로 단축
"""

from collections import deque, Counter
import time
from typing import Tuple


class ConsistencyFilter:
    def __init__(
        self,
        buffer_size: int = 3,
        agree_threshold: int = 2,
        conf_min: float = 0.6,   # 0.4 → 0.6 복원 (설계 원칙: confidence 0.6 미만은 unknown 처리)
        ttl: float = 12.0,       # 30.0 → 12.0 (2.5초 간격 * 3 + 여유 4.5초)
    ):
        """
        Parameters
        ----------
        buffer_size : int
            최근 응답을 저장할 버퍼 크기 (기본 3)
        agree_threshold : int
            방향 확정에 필요한 최소 일치 횟수 (기본 2, 즉 3회 중 2회)
        conf_min : float
            이 값 미만의 confidence는 direction을 unknown으로 처리 (기본 0.6)
            낮은 신뢰도 결과가 버퍼에 쌓이면 틀린 방향이 합의를 통과할 수 있음
        ttl : float
            버퍼 항목 유효 시간 (초, 기본 12.0).
            초과 항목은 유효하지 않으므로 방향 전환 후 빠르게 새 방향으로 전환됨
        """
        self.buffer: deque = deque(maxlen=buffer_size)
        self.agree_threshold = agree_threshold
        self.conf_min = conf_min
        self.ttl = ttl
        self.unknown_streak: int = 0

    def add(self, direction: str, confidence: float) -> None:
        """
        VLM 응답 하나를 버퍼에 추가한다.

        Parameters
        ----------
        direction : str
            VLM이 반환한 goal_direction ("left" / "right" / "straight" / "unknown")
        confidence : float
            VLM이 반환한 confidence (0.0 ~ 1.0)
        """
        if confidence < self.conf_min:
            direction = "unknown"

        self.buffer.append(
            {
                "direction": direction,
                "confidence": confidence,
                "timestamp": time.time(),
            }
        )

    def get_guidance(self) -> Tuple[str, str]:
        """
        버퍼 내 유효 응답을 기반으로 확정 방향과 TTS 텍스트를 반환한다.

        Returns
        -------
        Tuple[str, str]
            (confirmed_direction, tts_text)
            confirmed_direction: "left" / "right" / "straight" / "unknown"
            tts_text: 사용자에게 읽어줄 한국어 문장
        """
        now = time.time()
        valid = [r for r in self.buffer if now - r["timestamp"] < self.ttl]

        # 유효 응답 부족 (버퍼 전체 기준 — TTL 안에 있는 것만 카운트)
        if len(valid) < self.agree_threshold:
            return "unknown", "아직 분석 중입니다"

        directions = [r["direction"] for r in valid]
        counter = Counter(directions)
        top_dir, top_count = counter.most_common(1)[0]

        if top_count >= self.agree_threshold:
            if top_dir == "unknown":
                return self._handle_unknown()
            self.unknown_streak = 0
            return top_dir, self._to_korean(top_dir)

        return self._handle_unknown()

    def _handle_unknown(self) -> Tuple[str, str]:
        """unknown 연속 횟수에 따라 단계별 유도 메시지를 반환한다."""
        self.unknown_streak += 1
        if self.unknown_streak == 1:
            return "unknown", "잠시 기다려주세요"
        elif self.unknown_streak == 2:
            return "unknown", "카메라를 천천히 움직여주세요"
        else:
            self.unknown_streak = 0
            return "unknown", "주변을 천천히 둘러보세요"

    def _to_korean(self, direction: str) -> str:
        """영문 방향을 한국어 안내 문장으로 변환한다."""
        mapping = {
            "left": "왼쪽으로 이동하세요",
            "right": "오른쪽으로 이동하세요",
            "straight": "앞으로 직진하세요",
        }
        return mapping.get(direction, "방향을 파악 중입니다")

    def reset(self) -> None:
        """버퍼와 unknown_streak을 초기화한다."""
        self.buffer.clear()
        self.unknown_streak = 0
