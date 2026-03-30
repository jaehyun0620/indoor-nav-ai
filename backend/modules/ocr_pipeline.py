"""
ocr_pipeline.py
강의실 번호 및 표지판 텍스트 인식 파이프라인.
크롭 → 4배 확대 → CLAHE → 이진화 → EasyOCR
"""

import re
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False


# ── EasyOCR 싱글톤 ───────────────────────────────────────────────────────────

_reader: Optional[object] = None


def _get_reader(languages: List[str] = ("ko", "en")) -> object:
    """EasyOCR Reader를 처음 한 번만 초기화한다."""
    global _reader
    if _reader is None:
        if not _EASYOCR_AVAILABLE:
            raise ImportError("easyocr 패키지가 설치되지 않았습니다. pip install easyocr")
        _reader = easyocr.Reader(list(languages), gpu=False)
    return _reader


# ── 전처리 파이프라인 ────────────────────────────────────────────────────────

def preprocess(image: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    OCR 전처리: 확대 → 그레이스케일 → CLAHE → 적응형 이진화.

    Parameters
    ----------
    image : np.ndarray
        BGR 또는 그레이스케일 크롭 이미지
    scale : int
        확대 배율 (기본 4)

    Returns
    -------
    np.ndarray
        이진화된 그레이스케일 이미지
    """
    # 4배 확대
    h, w = image.shape[:2]
    enlarged = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 그레이스케일
    if len(enlarged.shape) == 3:
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    else:
        gray = enlarged.copy()

    # CLAHE (대비 향상)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 적응형 이진화
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    return binary


def crop_bbox(image: np.ndarray, bbox: List[float], padding: int = 5) -> np.ndarray:
    """
    bbox 좌표로 이미지를 크롭한다.

    Parameters
    ----------
    image : np.ndarray
        원본 BGR 프레임
    bbox : List[float]
        [x1, y1, x2, y2]
    padding : int
        상하좌우 여백 픽셀

    Returns
    -------
    np.ndarray
        크롭된 이미지
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image[y1:y2, x1:x2]


# ── 메인 OCR 함수 ─────────────────────────────────────────────────────────────

def read_text(
    image: np.ndarray,
    bbox: Optional[List[float]] = None,
    languages: List[str] = ("ko", "en"),
    conf_threshold: float = 0.4,
    preprocess_fallback_threshold: float = 0.5,
) -> List[Dict]:
    """
    이미지 (또는 bbox 크롭 영역)에서 텍스트를 인식한다.

    전략: 원본 이미지로 먼저 시도 → 평균 conf가 낮으면 전처리 후 재시도.
    적응형 이진화가 오히려 깨끗한 이미지를 망가뜨릴 수 있으므로
    전처리는 fallback으로만 사용한다.

    Parameters
    ----------
    image : np.ndarray
        BGR 원본 프레임
    bbox : List[float], optional
        크롭할 bbox. None이면 전체 이미지 사용
    languages : List[str]
        EasyOCR 언어 목록
    conf_threshold : float
        최종 결과 신뢰도 임계값 (이하는 제외)
    preprocess_fallback_threshold : float
        이 값 미만이면 전처리 후 재시도 (기본 0.5)

    Returns
    -------
    List[Dict]
        [{"text": str, "conf": float, "bbox": ...}, ...]
    """
    reader = _get_reader(list(languages))

    roi = crop_bbox(image, bbox) if bbox is not None else image

    def _run(img: np.ndarray) -> List[Dict]:
        results = reader.readtext(img)
        return [
            {"text": t.strip(), "conf": round(c, 3), "bbox": coords}
            for coords, t, c in results
            if c >= conf_threshold and t.strip()
        ]

    # 1차: 원본 이미지 그대로 시도
    output = _run(roi)
    avg_conf = sum(r["conf"] for r in output) / len(output) if output else 0.0

    # 2차: conf 낮으면 전처리 후 재시도
    if avg_conf < preprocess_fallback_threshold:
        processed = preprocess(roi)
        fallback = _run(processed)
        if fallback:
            fb_avg = sum(r["conf"] for r in fallback) / len(fallback)
            if fb_avg > avg_conf:
                output = fallback

    return output


def extract_room_number(text: str) -> Optional[str]:
    """
    OCR 텍스트에서 강의실 번호를 추출한다.
    예: "101호", "B203", "공학관 301" → "101", "B203", "301"

    Returns
    -------
    str or None
        추출된 강의실 번호. 없으면 None.
    """
    # 숫자 앞에 영문자가 붙는 패턴 (B101, S302 등)
    match = re.search(r"[A-Za-z]?\d{3,4}", text)
    if match:
        return match.group().upper()
    return None


def find_target_sign(
    ocr_results: List[Dict],
    target: str,
) -> Optional[Dict]:
    """
    OCR 결과 중 목표물 표지판 텍스트가 포함된 항목을 찾는다.

    Parameters
    ----------
    ocr_results : List[Dict]
        read_text() 반환값
    target : str
        찾을 키워드 (예: "화장실", "엘리베이터")

    Returns
    -------
    dict or None
        매칭된 OCR 결과 항목
    """
    for item in ocr_results:
        if target in item["text"]:
            return item
    return None
