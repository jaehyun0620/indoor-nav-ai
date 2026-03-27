"""
navigation_session.py
목적지 도착까지 지속 안내를 관리하는 세션 모듈.

상태머신:
  idle → navigating → arrived
"""

import re
import time
from collections import deque


class NavigationSession:
    ARRIVAL_DISTANCE = 1.5  # 목적지 도착 판정 거리 (m)
    ARRIVAL_CONFIRM = 2     # 연속 도착 조건 충족 횟수 (할루시네이션 방지)

    def __init__(self):
        self.state = "idle"          # idle / navigating / arrived
        self.target: str = ""
        self.start_time: float = 0.0
        self._direction_history: deque = deque(maxlen=5)
        self._arrival_count: int = 0

    # ── 세션 제어 ────────────────────────────────────────────────────────────

    def start(self, target: str) -> None:
        """네비게이션 세션 시작."""
        self.state = "navigating"
        self.target = target
        self.start_time = time.time()
        self._direction_history.clear()
        self._arrival_count = 0

    def stop(self) -> None:
        """세션 강제 종료."""
        self.state = "idle"
        self._arrival_count = 0

    @property
    def is_navigating(self) -> bool:
        return self.state == "navigating"

    @property
    def elapsed(self) -> float:
        """세션 시작 후 경과 시간 (초)."""
        return time.time() - self.start_time if self.start_time else 0.0

    # ── 방향 추적 ────────────────────────────────────────────────────────────

    def update_direction(self, direction: str) -> None:
        """확정된 방향을 이력에 추가한다. unknown은 저장하지 않는다."""
        if direction and direction != "unknown":
            self._direction_history.append(direction)

    def get_progress_feedback(self, current_direction: str) -> str:
        """
        이전 방향과 현재 방향을 비교해 진행 피드백 문장을 반환한다.

        - 최근 3회 방향이 동일하면: "○○ 방향으로 잘 가고 있습니다"
        - 방향이 바뀌었으면: 빈 문자열 (방향 안내 자체가 이미 제공됨)
        - 이력 부족 or unknown: 빈 문자열

        Returns
        -------
        str
            피드백 문장 or ""
        """
        if current_direction == "unknown" or len(self._direction_history) < 3:
            return ""

        recent = list(self._direction_history)[-3:]
        if all(d == current_direction for d in recent):
            label = {"left": "왼쪽", "right": "오른쪽", "straight": "직진"}.get(current_direction, "")
            return f"{label} 방향으로 잘 가고 있습니다"

        return ""

    # ── 도착 판정 ────────────────────────────────────────────────────────────

    def check_arrival(
        self,
        goal_visible: bool,
        goal_distance_str: str,
        confidence: float,
    ) -> bool:
        """
        VLM 결과를 기반으로 목적지 도착 여부를 판정한다.
        ARRIVAL_CONFIRM회 연속으로 조건을 충족해야 True를 반환해
        할루시네이션에 의한 오도착 판정을 방지한다.

        Parameters
        ----------
        goal_visible : bool
            VLM이 목적지를 화면에서 식별했는지 여부
        goal_distance_str : str
            VLM이 반환한 거리 문자열 (예: "약 1.2m")
        confidence : float
            VLM 신뢰도 (0.0 ~ 1.0)

        Returns
        -------
        bool
            True면 도착 확정, 세션 상태를 "arrived"로 변경
        """
        distance_m = self._parse_distance(goal_distance_str)
        condition_met = (
            goal_visible
            and confidence >= 0.75
            and distance_m <= self.ARRIVAL_DISTANCE
        )

        if condition_met:
            self._arrival_count += 1
        else:
            # 조건 미충족 시 카운트를 하나 감소 (일시적 오인식 허용)
            self._arrival_count = max(0, self._arrival_count - 1)

        if self._arrival_count >= self.ARRIVAL_CONFIRM:
            self.state = "arrived"
            return True
        return False

    def arrival_message(self) -> str:
        """도착 TTS 메시지를 반환한다."""
        elapsed = int(self.elapsed)
        return f"{self.target}에 도착했습니다. {elapsed}초 만에 안내를 완료했습니다."

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_distance(distance_str: str) -> float:
        """'약 1.5m' 형식에서 숫자를 추출한다. 파싱 실패 시 999 반환."""
        m = re.search(r"(\d+\.?\d*)", str(distance_str))
        return float(m.group(1)) if m else 999.0
