"""
context_builder.py
YOLO + MiDaS 탐지 결과를 VLM 프롬프트용 텍스트로 변환하는 모듈.

클래스 분류 설계:
- OBSTACLE_CLASSES : 복도에서 실제로 충돌 가능한 물리적 장애물
- INFO_CLASSES     : 길 안내에 참고할 정보성 객체 (장애물 판정 제외)
- CLASS_KO         : 위 두 집합의 합집합 (한국어 매핑)

OBSTACLE_CLASSES에 없는 클래스가 탐지되어도 장애물 경고는 발생하지 않는다.
"""

from typing import List, Dict, Set


# ── 물리적 장애물 클래스 (복도 내 충돌 위험 있는 것만) ──────────────────────
OBSTACLE_CLASSES: Set[str] = {
    "person",           # 사람
    "bicycle",          # 자전거 (복도 주차 종종 있음)
    "chair",            # 의자
    "bench",            # 벤치
    "stairs",           # 계단 (위험도 높음)
    "fire_extinguisher",# 소화기 (벽 돌출)
    "trash_can",        # 쓰레기통
    "table",            # 테이블
    "backpack",         # 바닥에 놓인 가방
    "suitcase",         # 여행가방
    "umbrella",         # 우산 (바닥에 세워진 경우)
    "potted plant",     # 화분
    "column",           # 기둥
    "couch",            # 소파 (일부 건물 복도)
}

# ── 정보성 클래스 (VLM 컨텍스트 제공용, 장애물 판정 제외) ───────────────────
INFO_CLASSES: Set[str] = {
    "door",             # 문 — 목표 방향 추론에 유용
    "sign",             # 표지판 — 목적지 위치 파악
    "elevator",         # 엘리베이터 — 목표물
    "toilet",           # 화장실 — 목표물
    "classroom",        # 강의실 — 목표물
}

# ── 한국어 클래스명 매핑 (OBSTACLE + INFO 합집합) ────────────────────────────
CLASS_KO = {
    # 장애물
    "person":            "사람",
    "bicycle":           "자전거",
    "chair":             "의자",
    "bench":             "벤치",
    "stairs":            "계단",
    "fire_extinguisher": "소화기",
    "trash_can":         "쓰레기통",
    "table":             "테이블",
    "backpack":          "가방",
    "suitcase":          "여행가방",
    "umbrella":          "우산",
    "potted plant":      "화분",
    "column":            "기둥",
    "couch":             "소파",
    # 정보성
    "door":              "문",
    "sign":              "표지판",
    "elevator":          "엘리베이터",
    "toilet":            "화장실",
    "classroom":         "강의실",
}

# bbox 수평 위치 → 방향 텍스트 (프레임 폭 기준)
def _bbox_to_direction(bbox: List[float], frame_width: int = 640) -> str:
    """bbox 중심 x좌표를 기준으로 좌/중앙/우 방향 반환."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    ratio = cx / frame_width

    if ratio < 0.35:
        return "왼쪽"
    elif ratio > 0.65:
        return "오른쪽"
    else:
        return "정면"


def _distance_label(distance_m: float) -> str:
    """거리 수치를 사람이 읽기 쉬운 텍스트로 변환."""
    if distance_m <= 0:
        return "거리 불명"
    return f"{distance_m:.1f}m"


def build_context(
    detections: List[Dict],
    frame_width: int = 640,
    conf_threshold: float = 0.4,
) -> str:
    """
    YOLO + MiDaS 탐지 결과 리스트를 VLM 프롬프트용 텍스트로 변환한다.

    Parameters
    ----------
    detections : List[Dict]
        각 항목은 fast_channel이 출력하는 dict:
        {
            "class": str,           # YOLO 클래스명 (영문)
            "bbox": [x1,y1,x2,y2],  # 픽셀 좌표
            "distance_m": float,    # MiDaS 추정 거리 (m)
            "conf": float           # YOLO 신뢰도
        }
    frame_width : int
        카메라 프레임 가로 해상도 (기본 640)
    conf_threshold : float
        이 값 미만의 탐지는 컨텍스트에서 제외

    Returns
    -------
    str
        VLM 프롬프트에 삽입할 텍스트 블록.
        예:
            사람: 정면 2.1m (신뢰도 0.91)
            계단: 오른쪽 3.5m (신뢰도 0.85)
            의자: 왼쪽 1.2m (신뢰도 0.77)
    """
    if not detections:
        return "탐지된 객체 없음"

    # 신뢰도 필터링 후 거리 기준 오름차순 정렬 (가까운 것 우선)
    filtered = [d for d in detections if d.get("conf", 0) >= conf_threshold]
    filtered.sort(key=lambda d: d.get("distance_m", 999))

    if not filtered:
        return "탐지된 객체 없음"

    lines = []
    for det in filtered:
        cls_en = det.get("class", "unknown")
        cls_ko = CLASS_KO.get(cls_en, cls_en)
        bbox = det.get("bbox", [0, 0, 0, 0])
        distance_m = det.get("distance_m", -1)
        conf = det.get("conf", 0.0)

        direction = _bbox_to_direction(bbox, frame_width)
        dist_str = _distance_label(distance_m)

        lines.append(f"{cls_ko}: {direction} {dist_str} (신뢰도 {conf:.2f})")

    return "\n".join(lines)


def build_obstacle_summary(detections: List[Dict], frame_width: int = 640) -> Dict:
    """
    우선순위 판단 모듈이 사용할 가장 가까운 장애물 요약 정보를 반환한다.

    OBSTACLE_CLASSES 에 속한 탐지 결과만 장애물로 판정한다.
    문·표지판·목표물 등 정보성 객체는 장애물 판정에서 제외된다.

    Returns
    -------
    dict
        {
            "closest_class": str,      # 가장 가까운 장애물 클래스 (한국어)
            "closest_distance": float,
            "direction": str,          # 왼쪽 / 정면 / 오른쪽
            "has_obstacle": bool       # 2m 이내 실제 장애물 존재 여부
        }
    """
    # 정보성 객체 제외 — OBSTACLE_CLASSES 에 속한 것만 필터링
    obstacle_dets = [
        d for d in detections
        if d.get("class") in OBSTACLE_CLASSES
    ]

    if not obstacle_dets:
        return {
            "closest_class": "",
            "closest_distance": 999.0,
            "direction": "정면",
            "has_obstacle": False,
        }

    nearest = min(obstacle_dets, key=lambda d: d.get("distance_m", 999))
    cls_en = nearest.get("class", "unknown")
    distance = nearest.get("distance_m", 999.0)
    bbox = nearest.get("bbox", [0, 0, 0, 0])

    return {
        "closest_class": CLASS_KO.get(cls_en, cls_en),
        "closest_distance": distance,
        "direction": _bbox_to_direction(bbox, frame_width),
        "has_obstacle": distance < 2.0,
    }
