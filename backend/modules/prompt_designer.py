"""
prompt_designer.py
VLM에 전달할 구조화 프롬프트를 생성하는 모듈.
JSON 응답 형식 강제 + confidence 기반 unknown 처리 규칙 포함.
"""

from typing import List, Dict


# ── 목표물 설명: VLM이 어떤 시각적 단서를 찾아야 하는지 구체적으로 안내 ────────
TARGET_KO_TO_DESC = {
    "화장실": (
        "화장실 (남녀 픽토그램, '화장실'·'TOILET'·'WC' 표지판, "
        "파란색/초록색 안내판, 문에 붙은 남녀 그림)"
    ),
    "강의실": (
        "강의실 (문 옆 번호판·호실 번호, '○○○호' 표기, "
        "나무문·철문에 붙은 번호 스티커, 복도 천장의 호실 방향 화살표)"
    ),
    "엘리베이터": (
        "엘리베이터 (승강기, 금속 문, '↑↓' 버튼 패널, "
        "'ELEVATOR'·'승강기' 표지판, 천장의 방향 화살표)"
    ),
}

PROMPT_TEMPLATE = """\
당신은 시각장애인의 눈이 되어주는 실내 길 안내 도우미입니다.
지금 사용자가 스마트폰으로 촬영한 건물 복도 이미지를 보고 있습니다.
이미지를 꼼꼼히 분석해서 목적지까지 구체적이고 자연스러운 안내를 해주세요.

━━━ 센서 보조 정보 (참고용 — 이미지 판단 우선) ━━━
{yolo_context}
※ 센서 오차가 있을 수 있으므로, 이미지에서 직접 확인한 내용을 최우선으로 판단하세요.

━━━ 이미지 분석 체크리스트 ━━━
다음 항목들을 순서대로 확인하세요:
  ① 벽면·천장의 표지판, 방향 화살표, 픽토그램
  ② 문의 위치·색상·번호판
  ③ 복도의 구조 (직선, 꺾임, 갈림길)
  ④ 바닥 유도 블록, 점자 블록
  ⑤ 기타 랜드마크 (계단, 소화기, 게시판, 창문)

━━━ 목표물 ━━━
{target}

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

━━━ 중요: 필드를 순서대로 채우세요 ━━━
reasoning을 먼저 작성하고, tts_message는 그 reasoning을 구어체로 변환하세요.

{{
  "goal_visible": true 또는 false,
  "goal_direction": "left" 또는 "right" 또는 "straight" 또는 "unknown",
  "goal_distance": "약 Xm" 또는 "unknown",
  "confidence": 0.0에서 1.0 사이 숫자,
  "reasoning": "이미지에서 직접 확인한 내용을 구체적으로 서술 (표지판 위치·색상, 문 번호, 복도 구조 등)",
  "tts_message": "위 reasoning의 핵심을 친근한 구어체 두 문장으로 변환 (아래 규칙 참조)"
}}

━━━ reasoning 작성 기준 ━━━
이미지에서 실제로 확인한 시각 요소를 구체적으로 적으세요.
  좋은 예: "왼쪽 상단 벽면에 파란색 화장실 픽토그램 표지판이 있고, 복도가 왼쪽으로 꺾임"
  좋은 예: "정면 복도 끝에 금속 엘리베이터 문과 버튼 패널이 보임, 약 8m 거리 추정"
  나쁜 예: "화장실이 왼쪽에 있음" (시각 근거 없음 ❌)

━━━ tts_message 변환 규칙 ━━━

[구조] reasoning을 바탕으로 두 문장:
  첫 문장 = reasoning에서 확인한 시각 요소를 자연스럽게 전달
  둘째 문장 = 지금 해야 할 행동

[어투] 친근한 구어체, 70자 이내
  ✅ "왼쪽 벽에 파란 화장실 표지판이 보여요. 왼쪽으로 꺾어서 약 5미터 이동하세요."
  ✅ "복도 오른쪽 끝에 엘리베이터 문이 보여요. 오른쪽으로 쭉 걸어가세요."
  ✅ "천장에 화장실 방향 화살표가 있어요. 화살표 방향인 왼쪽으로 이동하세요."
  ✅ "바로 앞에 사람이 있어요. 잠깐 멈추고 오른쪽으로 돌아서 가세요."
  ✅ "아직 표지판은 안 보이지만 복도가 오른쪽으로 이어져 있어요. 오른쪽으로 계속 가세요."
  ❌ "목적지는 왼쪽 방향입니다." (로봇 말투 금지)
  ❌ "왼쪽으로 이동하시기 바랍니다." (공식 말투 금지)

━━━ 판단 규칙 ━━━
1. confidence < 0.4 → goal_direction 반드시 "unknown"
2. 장애물 1~2m 이내 → confidence 기준 0.6으로 상향
3. YOLO 참고 정보와 이미지가 다르면 → 이미지를 따름\
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
        "tts_message": str(parsed.get("tts_message", "")),
    }
