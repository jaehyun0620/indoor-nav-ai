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
import os
from typing import Dict, Optional

import httpx

from backend.modules.prompt_designer import build_prompt, parse_vlm_response
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

        if self.provider == "openai":
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
        if self.provider == "openai":
            return await self._call_openai(prompt, image_bytes)
        return await self._call_gemini(prompt, image_bytes)

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

    def reset(self) -> None:
        """목표가 바뀌거나 세션이 재시작될 때 필터를 초기화한다."""
        self.filter.reset()
