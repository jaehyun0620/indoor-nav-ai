"""
slow_channel.py
느린 채널: VLM API 호출 + 일관성 필터 → 방향 안내 (2~3초 주기)

설계 원칙:
- VLM HTTP 호출은 httpx.AsyncClient 를 사용해 FastAPI 이벤트 루프를 블로킹하지 않는다.
- 호출 메서드는 모두 async def 로 선언한다.
- httpx.AsyncClient 는 VLMClient 인스턴스당 하나를 생성해 재사용한다.
  (매 호출마다 새 클라이언트를 만들면 TCP 연결을 매번 새로 맺어 지연 발생)
"""

import base64
import json
import os
from typing import Dict, Optional

import httpx

from backend.modules.prompt_designer import build_prompt, build_orientation_prompt, parse_vlm_response
from backend.modules.consistency_filter import ConsistencyFilter


# ── VLM 클라이언트 ───────────────────────────────────────────────────────────

class VLMClient:
    """
    GPT-4o 또는 Gemini 1.5 Flash API를 호출하는 클라이언트.
    환경변수로 사용할 모델을 선택한다.

    httpx.AsyncClient를 인스턴스 레벨에서 재사용해 TCP 연결 비용을 줄인다.
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Parameters
        ----------
        provider : str, optional
            "openai" 또는 "gemini". 미지정 시 환경변수 VLM_PROVIDER 참조.
            기본값은 "openai".
        """
        self.provider = (
            provider
            or os.getenv("VLM_PROVIDER", "openai")
        ).lower()
        self.mock_enabled = os.getenv("MOCK_VLM", "false").lower() in ("1", "true", "yes", "on")

        if self.mock_enabled:
            self.api_key = ""
            self.model = "mock-vlm"
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY", "")
            self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        else:
            raise ValueError(f"지원하지 않는 VLM provider: {self.provider}")

        # 연결 재사용을 위해 persistent client 생성 (매 호출마다 새로 만들지 않음)
        self._client = httpx.AsyncClient(timeout=20.0)

    async def call(self, prompt: str, image_bytes: bytes) -> str:
        """
        VLM API를 비동기로 호출하여 응답 텍스트를 반환한다.

        Parameters
        ----------
        prompt : str
            prompt_designer.build_prompt() 가 생성한 프롬프트
        image_bytes : bytes
            JPEG/PNG 이미지 바이트

        Returns
        -------
        str
            VLM 응답 텍스트 (JSON 문자열 기대)
        """
        if self.mock_enabled:
            return self._call_mock(prompt)

        if self.provider == "openai":
            return await self._call_openai(prompt, image_bytes)
        return await self._call_gemini(prompt, image_bytes)

    def _call_mock(self, prompt: str) -> str:
        """
        집/오프라인 테스트용 VLM 대체 응답.
        실제 이미지를 분석하지 않고 환경변수 값으로 JSON을 만들어
        WebSocket, 필터, TTS, UI 흐름을 비용 없이 검증한다.
        """
        direction = os.getenv("MOCK_VLM_DIRECTION", "straight").lower()
        if direction not in ("left", "right", "straight", "unknown"):
            direction = "unknown"

        confidence = float(os.getenv("MOCK_VLM_CONFIDENCE", "0.85"))
        visible = os.getenv("MOCK_VLM_VISIBLE", "true").lower() in ("1", "true", "yes", "on")
        distance = os.getenv("MOCK_VLM_DISTANCE", "약 5m")

        target = "목적지"
        if "━━━ 목표물 ━━━" in prompt:
            target = prompt.split("━━━ 목표물 ━━━", 1)[1].strip().splitlines()[0].strip() or target
        elif "목표물:" in prompt:
            target = prompt.split("목표물:", 1)[1].splitlines()[0].strip() or target
        target = target.split(" ", 1)[0]

        label = {
            "left": "왼쪽",
            "right": "오른쪽",
            "straight": "정면",
            "unknown": "알 수 없는 방향",
        }[direction]

        tts_message = os.getenv(
            "MOCK_VLM_TTS",
            f"테스트 모드입니다. {target} 안내는 {label} 방향입니다.",
        )

        return json.dumps(
            {
                "goal_visible": visible,
                "goal_direction": direction,
                "goal_distance": distance if visible else "unknown",
                "confidence": confidence,
                "reasoning": "MOCK_VLM 테스트 모드에서 생성한 응답",
                "tts_message": tts_message,
            },
            ensure_ascii=False,
        )

    async def _call_openai(self, prompt: str, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "max_tokens": 400,          # tts_message 2문장 허용으로 증가
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        }
        resp = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def _call_gemini(self, prompt: str, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"maxOutputTokens": 400},  # tts_message 2문장 허용으로 증가
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()
        # safety filter 등으로 candidates가 없는 경우 처리
        candidates = body.get("candidates", [])
        if not candidates:
            raise ValueError(f"Gemini candidates 없음: {body.get('promptFeedback', '')}")
        return candidates[0]["content"]["parts"][0]["text"]


# ── SlowChannel ──────────────────────────────────────────────────────────────

class SlowChannel:
    """
    느린 채널 실행 클래스.
    VLM API 호출 → JSON 파싱 → ConsistencyFilter → 방향 확정.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        condition: str = "proposed",
    ):
        """
        Parameters
        ----------
        provider : str, optional
            "openai" 또는 "gemini"
        condition : str
            실험 조건 ("baseline" / "structured" / "proposed")
        """
        self.vlm = VLMClient(provider=provider)
        self.filter = ConsistencyFilter()
        self.condition = condition

    async def process(
        self,
        image_bytes: bytes,
        yolo_context: str,
        target: str,
    ) -> Dict:
        """
        이미지와 YOLO 컨텍스트를 받아 방향 안내를 반환한다.

        Parameters
        ----------
        image_bytes : bytes
            카메라 프레임 JPEG 바이트
        yolo_context : str
            context_builder.build_context() 결과
        target : str
            목표물 (예: "화장실")

        Returns
        -------
        dict
            {
                "confirmed_direction": str,  # left / right / straight / unknown
                "tts_text": str,
                "unknown_streak": int,
                "raw": dict                  # VLM 파싱 결과 (디버깅용)
            }
        """
        prompt = build_prompt(yolo_context, target, condition=self.condition)

        try:
            raw_text = await self.vlm.call(prompt, image_bytes)
        except Exception as e:
            # API 오류 시 unknown 처리
            return {
                "confirmed_direction": "unknown",
                "tts_text": f"VLM 오류: {str(e)[:60]}",
                "unknown_streak": self.filter.unknown_streak,
                "raw": {},
            }

        parsed = parse_vlm_response(raw_text)
        self.filter.add(parsed["goal_direction"], parsed["confidence"])
        confirmed_dir, fallback_tts = self.filter.get_guidance()

        # tts_text 결정 규칙:
        # - confirmed_dir이 unknown이면 필터의 fallback_tts를 사용한다.
        #   (VLM tts_message가 "왼쪽으로 가세요"여도 필터가 아직 방향을 확정하지 않았으면
        #    그 메시지를 내보내면 안 됨 → confirmed_dir와 tts_text 불일치 버그 방지)
        # - confirmed_dir이 실제 방향이면 VLM tts_message를 우선 사용하되,
        #   없으면 fallback_tts를 사용한다.
        if confirmed_dir == "unknown":
            tts_text = fallback_tts
        else:
            tts_message = parsed.get("tts_message", "").strip()
            tts_text = tts_message if tts_message else fallback_tts

        return {
            "confirmed_direction": confirmed_dir,
            "tts_text": tts_text,
            "unknown_streak": self.filter.unknown_streak,
            "raw": parsed,
        }

    async def process_instant(
        self,
        image_bytes: bytes,
        yolo_context: str,
        target: str,
    ) -> Dict:
        """
        VLM을 1회 호출하고 ConsistencyFilter를 거치지 않고 즉시 결과를 반환한다.
        사용자가 명시적으로 방향 조회를 요청할 때 사용.
        필터 버퍼에는 결과를 추가해 이후 연속성 유지.

        Returns
        -------
        dict
            {
                "confirmed_direction": str,
                "tts_text": str,
                "unknown_streak": int,
                "raw": dict
            }
        """
        prompt = build_prompt(yolo_context, target, condition=self.condition)

        try:
            raw_text = await self.vlm.call(prompt, image_bytes)
        except Exception as e:
            return {
                "confirmed_direction": "unknown",
                "tts_text": f"VLM 오류: {str(e)[:60]}",
                "unknown_streak": self.filter.unknown_streak,
                "raw": {},
            }

        parsed = parse_vlm_response(raw_text)
        # 필터 버퍼에는 추가하되, 결과는 즉시 반환
        self.filter.add(parsed["goal_direction"], parsed["confidence"])

        direction = parsed["goal_direction"]
        mapping = {"left": "왼쪽", "right": "오른쪽", "straight": "직진"}

        # VLM이 생성한 자연어 안내문 우선 사용, 없으면 fallback
        tts_message = parsed.get("tts_message", "").strip()
        if tts_message:
            tts_text = tts_message
        elif direction != "unknown":
            tts_text = f"목적지는 {mapping.get(direction, direction)} 방향입니다"
        else:
            tts_text = "방향을 파악하지 못했습니다"

        return {
            "confirmed_direction": direction,
            "tts_text": tts_text,
            "unknown_streak": self.filter.unknown_streak,
            "raw": parsed,
        }

    async def run_orientation(
        self,
        image_bytes: bytes,
        yolo_context: str,
        target: str,
    ) -> Dict:
        """
        네비게이션 시작 직후 첫 프레임에 대해 초기 장면 파악(orientation)을 수행한다.
        ConsistencyFilter를 거치지 않고 VLM 1회 응답을 즉시 반환한다.

        Parameters
        ----------
        image_bytes : bytes
            첫 번째 카메라 프레임 JPEG 바이트
        yolo_context : str
            context_builder.build_context() 결과
        target : str
            목표물 (예: "화장실")

        Returns
        -------
        dict
            {
                "goal_visible": bool,
                "goal_direction": str,
                "scene_type": str,
                "tts_text": str,
                "raw": dict
            }
        """
        prompt = build_orientation_prompt(yolo_context, target)

        try:
            raw_text = await self.vlm.call(prompt, image_bytes)
        except Exception as e:
            return {
                "goal_visible": False,
                "goal_direction": "unknown",
                "scene_type": "unknown",
                "tts_text": f"{target} 안내를 시작합니다. 카메라를 복도 방향으로 향해주세요.",
                "raw": {"error": str(e)},
            }

        parsed = parse_vlm_response(raw_text)

        # orientation 전용 필드 추출
        scene_type = ""
        try:
            import json, re as _re
            cleaned = _re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()
            jm = _re.search(r"\{[\s\S]*\}", cleaned)
            if jm:
                full = json.loads(jm.group())
                scene_type = str(full.get("scene_type", ""))
        except Exception:
            pass

        # ── 할루시네이션 후처리 가드 ────────────────────────────────────
        # reasoning에 추정·불확실 표현이 있으면 goal_visible을 False로 강제.
        # 프롬프트 규칙만으로는 LLM이 무시하는 경우가 있으므로 코드 레벨에서 이중 보호.
        GUESS_KEYWORDS = [
            "예상", "추정", "것으로 보임", "있을 것",
            "위치할 것", "보일 것", "될 것으로", "일 것이다",
            "있을 수도", "가능성", "것 같습니다", "듯합니다",
        ]
        reasoning_text = parsed.get("reasoning", "")
        if parsed.get("goal_visible") and any(kw in reasoning_text for kw in GUESS_KEYWORDS):
            parsed["goal_visible"] = False
            parsed["goal_direction"] = "unknown"

        # goal_visible이 후처리로 바뀐 경우 tts_message도 재생성
        goal_visible  = parsed.get("goal_visible", False)
        goal_direction = parsed.get("goal_direction", "unknown")

        tts_message = parsed.get("tts_message", "").strip()
        # goal_visible이 False로 바뀐 경우 기존 tts_message는 잘못된 내용일 수 있으므로 무효화
        if not goal_visible and tts_message and any(
            word in tts_message for word in ["보여요", "있어요", "확인됩니다"]
        ):
            tts_message = ""

        if not tts_message:
            # fallback: 간단한 안내문 생성
            if goal_visible:
                dir_map = {"left": "왼쪽", "right": "오른쪽", "straight": "앞쪽"}
                d = dir_map.get(goal_direction, "")
                tts_message = (
                    f"{target}이 보입니다. {d}으로 이동하세요." if d
                    else f"{target}이 가까이 있습니다."
                )
            else:
                sc = f" {scene_type}에" if scene_type and scene_type != "unknown" else ""
                tts_message = (
                    f"현재{sc} 계신 것 같아요. "
                    f"{target}을 찾고 있어요. 앞쪽으로 이동해보세요."
                )

        return {
            "goal_visible":   goal_visible,
            "goal_direction": goal_direction,
            "scene_type":     scene_type,
            "tts_text":       tts_message,
            "raw":            parsed,
        }

    def reset(self) -> None:
        """목표가 바뀌거나 세션이 재시작될 때 필터를 초기화한다."""
        self.filter.reset()
