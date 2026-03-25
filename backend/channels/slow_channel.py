"""
slow_channel.py
느린 채널: VLM API 호출 + 일관성 필터 → 방향 안내 (2~3초 주기)
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

    def call(self, prompt: str, image_bytes: bytes) -> str:
        """
        VLM API를 호출하여 응답 텍스트를 반환한다.

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
            return self._call_openai(prompt, image_bytes)
        return self._call_gemini(prompt, image_bytes)

    def _call_openai(self, prompt: str, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "max_tokens": 300,
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
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str, image_bytes: bytes) -> str:
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
            "generationConfig": {"maxOutputTokens": 300},
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


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

    def process(
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
            raw_text = self.vlm.call(prompt, image_bytes)
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
        confirmed_dir, tts_text = self.filter.get_guidance()

        return {
            "confirmed_direction": confirmed_dir,
            "tts_text": tts_text,
            "unknown_streak": self.filter.unknown_streak,
            "raw": parsed,
        }

    def reset(self) -> None:
        """목표가 바뀌거나 세션이 재시작될 때 필터를 초기화한다."""
        self.filter.reset()
