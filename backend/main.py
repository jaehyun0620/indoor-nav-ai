"""
main.py
FastAPI 서버 진입점.
빠른 채널(YOLOv8+MiDaS)과 느린 채널(VLM)을 통합하고
우선순위 판단 모듈을 거쳐 최종 TTS 메시지를 반환한다.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from backend.channels.fast_channel import FastChannel
from backend.channels.slow_channel import SlowChannel
from backend.modules.priority_module import PriorityModule
from backend.modules.scene_memory import SceneMemory

# ── 싱글톤 인스턴스 ──────────────────────────────────────────────────────────

fast_channel: Optional[FastChannel] = None
slow_channel: Optional[SlowChannel] = None
priority_module = PriorityModule()
scene_memory = SceneMemory()

# 느린 채널 쿨다운 제어 (2~3초 주기)
_last_slow_time: float = 0.0
SLOW_CHANNEL_INTERVAL: float = float(os.getenv("SLOW_CHANNEL_INTERVAL", "2.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델을 로드한다."""
    global fast_channel, slow_channel
    fast_channel = FastChannel(
        yolo_model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        midas_model=os.getenv("MIDAS_MODEL", "MiDaS_small"),
        conf_threshold=float(os.getenv("YOLO_CONF", "0.4")),
        scale_factor=float(os.getenv("MIDAS_SCALE", "5.0")),
    )
    slow_channel = SlowChannel(
        provider=os.getenv("VLM_PROVIDER", "openai"),
        condition=os.getenv("EXPERIMENT_CONDITION", "proposed"),
    )
    yield


app = FastAPI(title="Indoor Navigation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST 엔드포인트 ─────────────────────────────────────────────────────────

class NavigationResponse(BaseModel):
    message_type: str       # warning / caution / guidance / unknown
    tts_text: str
    priority: int
    suppress_guidance: bool
    detections: list
    yolo_context: str
    slow_raw: dict


@app.post("/navigate", response_model=NavigationResponse)
async def navigate(
    frame: UploadFile = File(...),
    target: str = Form("화장실"),
):
    """
    카메라 프레임 1장을 받아 TTS 안내 메시지를 반환한다.

    - frame: JPEG/PNG 이미지
    - target: 목표물 ("강의실" / "화장실" / "엘리베이터")
    """
    import time

    global _last_slow_time

    image_bytes = await frame.read()

    # ── 빠른 채널 (매 요청마다 실행) ───────────────────────────────────────
    fast_output = fast_channel.process_bytes(image_bytes)
    fast_result = fast_output["fast_result"]
    yolo_context = fast_output["yolo_context"]
    detections = fast_output["detections"]

    # ── 느린 채널 (쿨다운 적용) ────────────────────────────────────────────
    now = time.time()
    if now - _last_slow_time >= SLOW_CHANNEL_INTERVAL:
        _last_slow_time = now
        slow_result = slow_channel.process(image_bytes, yolo_context, target)
    else:
        # 쿨다운 중: 마지막 필터 상태 재사용
        confirmed_dir, tts_text = slow_channel.filter.get_guidance()
        slow_result = {
            "confirmed_direction": confirmed_dir,
            "tts_text": tts_text,
            "unknown_streak": slow_channel.filter.unknown_streak,
            "raw": {},
        }

    # SceneMemory 업데이트
    scene_memory.update(detections, slow_result.get("raw", {}))

    # ── 우선순위 판단 ──────────────────────────────────────────────────────
    decision = priority_module.decide(fast_result, slow_result)

    return NavigationResponse(
        message_type=decision["message_type"],
        tts_text=decision["tts_text"],
        priority=decision["priority"],
        suppress_guidance=decision["suppress_guidance"],
        detections=detections,
        yolo_context=yolo_context,
        slow_raw=slow_result.get("raw", {}),
    )


@app.post("/reset")
async def reset_session():
    """세션(일관성 필터 + SceneMemory)을 초기화한다."""
    slow_channel.reset()
    scene_memory.reset()
    return {"status": "reset"}


# ── WebSocket 엔드포인트 ─────────────────────────────────────────────────────

@app.websocket("/ws/navigate")
async def ws_navigate(websocket: WebSocket):
    """
    WebSocket으로 Base64 인코딩된 프레임을 받아 실시간 안내를 반환한다.

    메시지 형식 (클라이언트 → 서버):
        JSON: { "frame": "<base64>", "target": "화장실" }

    메시지 형식 (서버 → 클라이언트):
        JSON: NavigationResponse 형태
    """
    import json
    import time

    global _last_slow_time

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            b64_frame = data.get("frame", "")
            target = data.get("target", "화장실")

            # 빠른 채널
            fast_output = fast_channel.process_base64(b64_frame)
            fast_result = fast_output["fast_result"]
            yolo_context = fast_output["yolo_context"]
            detections = fast_output["detections"]

            # 느린 채널 (쿨다운)
            now = time.time()
            if now - _last_slow_time >= SLOW_CHANNEL_INTERVAL:
                _last_slow_time = now
                import base64 as _b64
                raw_b64 = b64_frame.split(",", 1)[-1] if "," in b64_frame else b64_frame
                image_bytes = _b64.b64decode(raw_b64)
                slow_result = slow_channel.process(image_bytes, yolo_context, target)
            else:
                confirmed_dir, tts_text = slow_channel.filter.get_guidance()
                slow_result = {
                    "confirmed_direction": confirmed_dir,
                    "tts_text": tts_text,
                    "unknown_streak": slow_channel.filter.unknown_streak,
                    "raw": {},
                }

            scene_memory.update(detections, slow_result.get("raw", {}))
            decision = priority_module.decide(fast_result, slow_result)

            await websocket.send_json({
                "message_type": decision["message_type"],
                "tts_text": decision["tts_text"],
                "priority": decision["priority"],
                "suppress_guidance": decision["suppress_guidance"],
                "yolo_context": yolo_context,
            })

    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health():
    return {"status": "ok"}
