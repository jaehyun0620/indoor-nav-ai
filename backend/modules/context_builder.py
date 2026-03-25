"""
context_builder.py
YOLO + MiDaS 탐지 결과를 VLM 프롬프트용 텍스트로 변환하는 모듈.
"""

from typing import List, Dict


# 한국어 클래스명 매핑
CLASS_KO = {
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "chair": "의자",
    "bench": "벤치",
    "door": "문",
    "stairs": "계단",
    "elevator": "엘리베이터",
    "toilet": "화장실",
    "classroom": "강의실",
    "fire_extinguisher": "소화기",
    "trash_can": "쓰레기통",
    "table": "테이블",
    "backpack": "가방",
    "suitcase": "여행가방",
    "umbrella": "우산",
    "couch": "소파",
    "potted plant": "화분",
    "tv": "TV",
    "laptop": "노트북",
    "cell phone": "휴대폰",
    "bottle": "병",
    "cup": "컵",
    "clock": "시계",
    "book": "책",
    "keyboard": "키보드",
    "mouse": "마우스",
    "remote": "리모컨",
    "scissors": "가위",
    "dog": "강아지",
    "cat": "고양이",
    "bird": "새",
    "sports ball": "공",
    "dining table": "식탁",
    "sink": "세면대",
    "refrigerator": "냉장고",
    "microwave": "전자레인지",
    "bed": "침대",
    "mirror": "거울",
    "window": "창문",
    "column": "기둥",
    "sign": "표지판",
    "traffic light": "신호등",
    "stop sign": "정지 표지판",
    "fire hydrant": "소화전",
    "bus": "버스",
    "truck": "트럭",
    "motorcycle": "오토바이",
    "bicycle": "자전거",
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

    Returns
    -------
    dict
        {
            "closest_class": str,      # 가장 가까운 객체 클래스 (한국어)
            "closest_distance": float,
            "direction": str,          # 왼쪽 / 정면 / 오른쪽
            "has_obstacle": bool       # 2m 이내 장애물 존재 여부
        }
    """
    if not detections:
        return {
            "closest_class": "",
            "closest_distance": 999.0,
            "direction": "정면",
            "has_obstacle": False,
        }

    nearest = min(detections, key=lambda d: d.get("distance_m", 999))
    cls_en = nearest.get("class", "unknown")
    distance = nearest.get("distance_m", 999.0)
    bbox = nearest.get("bbox", [0, 0, 0, 0])

    return {
        "closest_class": CLASS_KO.get(cls_en, cls_en),
        "closest_distance": distance,
        "direction": _bbox_to_direction(bbox, frame_width),
        "has_obstacle": distance < 2.0,
    }
