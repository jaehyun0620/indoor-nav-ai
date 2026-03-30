"""
prompt_designer.py
VLM에 전달할 구조화 프롬프트를 생성하는 모듈.
JSON 응답 형식 강제 + confidence 기반 unknown 처리 규칙 포함.
"""

from typing import List, Dict


# 목표물 한국어 → 영문 매핑 (VLM 프롬프트용)
TARGET_KO_TO_DESC = {
    "강의실": "강의실 (문 번호가 있는 교실)",
    "화장실": "화장실 (남자화장실 / 여자화장실 표시)",
    "엘리베이터": "엘리베이터 (승강기, 위아래 화살표 버튼)",
}

PROMPT_TEMPLATE = """\
당신은 시각장애인을 돕는 실내 안내 시스템입니다.

[탐지된 객체 정보 - 신뢰도 높음]
{yolo_context}

위 탐지 정보와 이미지를 함께 분석하여 반드시 아래 JSON 형식으로만 응답하세요.
다른 텍스트는 절대 포함하지 마세요.

목표물: {target}

{{
  "goal_visible": true 또는 false,
  "goal_direction": "left" 또는 "right" 또는 "straight" 또는 "unknown",
  "goal_distance": "약 Xm" 또는 "unknown",
  "confidence": 0.0에서 1.0 사이 숫자,
  "reasoning": "판단 근거 한 문장"
}}

규칙:
1. 목표물이 직접 보이지 않더라도, 복도 구조·표지판·화살표 등을 근거로 방향을 추론하세요.
2. confidence가 0.4 미만일 때만 goal_direction을 "unknown"으로 설정하세요.
3. 탐지된 객체 정보와 모순되는 응답은 금지합니다.
4. 장애물이 1~2m 이내에 있을 경우 confidence 기준을 0.6으로 높여서 판단하세요.
5. reasoning은 반드시 한국어 한 문장으로 작성하세요.\
"""

# 비교 실험 Baseline용 프롬프트 (구조화 없음)
BASELINE_PROMPT_TEMPLATE = """\
이미지를 보고 {target}이(가) 어느 방향에 있는지 알려주세요.\
"""

# 비교 실험 +구조화 조건 (JSON만 강제, YOLO 주입 없음)
STRUCTURED_ONLY_PROMPT_TEMPLATE = """\
당신은 시각장애인을 돕는 실내 안내 시스템입니다.

이미지를 분석하여 반드시 아래 JSON 형식으로만 응답하세요.
다른 텍스트는 절대 포함하지 마세요.

목표물: {target}

{{
  "goal_visible": true 또는 false,
  "goal_direction": "left" 또는 "right" 또는 "straight" 또는 "unknown",
  "goal_distance": "약 Xm" 또는 "unknown",
  "confidence": 0.0에서 1.0 사이 숫자,
  "reasoning": "판단 근거 한 문장"
}}

규칙:
1. confidence가 0.6 미만이면 goal_direction은 반드시 "unknown"으로 설정하세요.
4. goal_visible이 false이면 goal_direction은 "unknown"으로 설정하세요.\
"""


def build_prompt(
    yolo_context: str,
    target: str,
    condition: str = "proposed",
) -> str:
    """
    VLM에 전달할 프롬프트 문자열을 생성한다.

    Parameters
    ----------
    yolo_context : str
        context_builder.build_context() 가 반환한 텍스트
    target : str
        사용자가 요청한 목표물 (예: "화장실", "강의실", "엘리베이터")
    condition : str
        실험 조건 선택:
        - "baseline"   : 자유 응답 (Baseline)
        - "structured" : JSON 강제만 (+ 구조화)
        - "proposed"   : YOLO 주입 + JSON 강제 (Proposed, 기본값)

    Returns
    -------
    str
        완성된 프롬프트 문자열
    """
    # 매핑에 없으면 사용자가 말한 그대로 사용
    target_desc = TARGET_KO_TO_DESC.get(target, target)

    if condition == "baseline":
        return BASELINE_PROMPT_TEMPLATE.format(target=target_desc)

    if condition == "structured":
        return STRUCTURED_ONLY_PROMPT_TEMPLATE.format(target=target_desc)

    # proposed (기본)
    return PROMPT_TEMPLATE.format(
        yolo_context=yolo_context,
        target=target_desc,
    )


def parse_vlm_response(response_text: str) -> Dict:
    """
    VLM 응답 텍스트에서 JSON을 파싱한다.
    파싱 실패 시 안전한 기본값(unknown)을 반환한다.

    Parameters
    ----------
    response_text : str
        VLM이 반환한 원본 텍스트

    Returns
    -------
    dict
        {
            "goal_visible": bool,
            "goal_direction": str,   # left / right / straight / unknown
            "goal_distance": str,
            "confidence": float,
            "reasoning": str
        }
    """
    import json
    import re

    default = {
        "goal_visible": False,
        "goal_direction": "unknown",
        "goal_distance": "unknown",
        "confidence": 0.0,
        "reasoning": "응답 파싱 실패",
    }

    if not response_text:
        return default

    # 응답에서 JSON 블록 추출 (마크다운 코드블록 포함 대응)
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if not json_match:
        return default

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return default

    # confidence < 0.4 이면 direction 강제 unknown
    confidence = float(parsed.get("confidence", 0.0))
    direction = parsed.get("goal_direction", "unknown")
    if confidence < 0.4:
        direction = "unknown"

    return {
        "goal_visible": bool(parsed.get("goal_visible", False)),
        "goal_direction": direction,
        "goal_distance": str(parsed.get("goal_distance", "unknown")),
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning", "")),
    }
