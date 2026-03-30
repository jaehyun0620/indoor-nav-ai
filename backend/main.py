"""
main.py
FastAPI 서버 진입점.

빠른 채널(YOLOv8+MiDaS)과 느린 채널(VLM)을 통합하고
우선순위 판단 모듈을 거쳐 최종 TTS 메시지를 반환한다.

WebSocket /ws/navigate 가 핵심 엔드포인트:
  - 클라이언트가 접속하면 네비게이션 세션이 시작된다.
  - 1초마다 프레임을 전송하면 안내 메시지를 돌려받는다.
  - 도착 판정 시 "arrived" 메시지를 보내고 세션을 종료한다.
"""

import asyncio
import base64
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from backend.channels.fast_channel import FastChannel
from backend.channels.slow_channel import SlowChannel
from backend.modules.navigation_session import NavigationSession
from backend.modules.priority_module import PriorityModule
from backend.modules.scene_memory import SceneMemory

# ── 싱글톤 인스턴스 ──────────────────────────────────────────────────────────

fast_channel: Optional[FastChannel] = None
slow_channel: Optional[SlowChannel] = None
priority_module = PriorityModule()

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


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _resize_for_vlm(image_bytes: bytes) -> bytes:
    """VLM 전송용으로 이미지를 축소한다. 비용 절감 목적."""
    import cv2
    import numpy as np
    size = int(os.getenv("VLM_IMAGE_SIZE", "320"))
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    if w <= size:
        return image_bytes
    ratio = size / w
    resized = cv2.resize(img, (size, int(h * ratio)))
    _, enc = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return enc.tobytes()


def _process_frame(image_bytes: bytes, target: str, last_slow_time: float, scene_memory: SceneMemory):
    """
    프레임 1장을 빠른/느린 채널에 통과시켜 결과를 반환한다.

    Returns
    -------
    dict: fast_result, slow_result, yolo_context, detections, raw_vlm, new_slow_time
    """
    # 빠른 채널
    fast_output = fast_channel.process_bytes(image_bytes)
    fast_result = fast_output["fast_result"]
    yolo_context = fast_output["yolo_context"]
    detections = fast_output["detections"]

    # scene_memory 컨텍스트를 프롬프트에 추가
    memory_hint = scene_memory.get_context_for_prompt()
    enriched_context = f"{yolo_context}\n{memory_hint}".strip() if memory_hint else yolo_context

    # 느린 채널 (쿨다운 주기만 체크, 장면 변화 조건 제거)
    raw_vlm = {}
    now = time.time()

    if now - last_slow_time >= SLOW_CHANNEL_INTERVAL:
        small_image = _resize_for_vlm(image_bytes)
        slow_result = slow_channel.process(small_image, enriched_context, target)
        raw_vlm = slow_result.get("raw", {})
        new_slow_time = now
    else:
        confirmed_dir, tts_text = slow_channel.filter.get_guidance()
        slow_result = {
            "confirmed_direction": confirmed_dir,
            "tts_text": tts_text,
            "unknown_streak": slow_channel.filter.unknown_streak,
            "raw": {},
        }
        new_slow_time = last_slow_time

    scene_memory.update(detections, raw_vlm)

    return {
        "fast_result": fast_result,
        "slow_result": slow_result,
        "yolo_context": yolo_context,
        "detections": detections,
        "raw_vlm": raw_vlm,
        "new_slow_time": new_slow_time,
    }


# ── REST 엔드포인트 (단발성 테스트용) ───────────────────────────────────────

class NavigationResponse(BaseModel):
    message_type: str
    tts_text: str
    priority: int
    suppress_guidance: bool
    arrived: bool = False
    detections: list
    yolo_context: str
    slow_raw: dict


_rest_slow_time: float = 0.0
_rest_scene_memory = SceneMemory()


@app.post("/navigate", response_model=NavigationResponse)
async def navigate(
    frame: UploadFile = File(...),
    target: str = Form("화장실"),
):
    """
    단발성 테스트용 REST 엔드포인트.
    실제 서비스는 WebSocket /ws/navigate 를 사용한다.
    """
    global _rest_slow_time

    image_bytes = await frame.read()
    result = _process_frame(image_bytes, target, _rest_slow_time, _rest_scene_memory)
    _rest_slow_time = result["new_slow_time"]

    decision = priority_module.decide(result["fast_result"], result["slow_result"])

    return NavigationResponse(
        message_type=decision["message_type"],
        tts_text=decision["tts_text"],
        priority=decision["priority"],
        suppress_guidance=decision["suppress_guidance"],
        detections=result["detections"],
        yolo_context=result["yolo_context"],
        slow_raw=result["raw_vlm"],
    )


@app.post("/reset")
async def reset_session():
    """REST 세션 초기화."""
    global _rest_slow_time
    slow_channel.reset()
    _rest_scene_memory.reset()
    _rest_slow_time = 0.0
    return {"status": "reset"}


# ── WebSocket 엔드포인트 (지속 네비게이션) ───────────────────────────────────

@app.websocket("/ws/navigate")
async def ws_navigate(websocket: WebSocket):
    """
    지속 네비게이션 WebSocket 엔드포인트.

    ── 클라이언트 → 서버 메시지 형식 ──
    시작:  { "action": "start",  "target": "화장실" }
    프레임: { "action": "frame",  "frame": "<base64>", "target": "화장실" }
    중지:  { "action": "stop" }

    ── 서버 → 클라이언트 메시지 형식 ──
    일반:  { "message_type": "guidance"|"caution"|"warning"|"unknown",
             "tts_text": "...", "priority": 1~3,
             "arrived": false, "progress": "..." }
    도착:  { "message_type": "arrived", "tts_text": "화장실에 도착했습니다.",
             "arrived": true }
    시작확인: { "message_type": "started", "tts_text": "화장실 안내를 시작합니다" }
    중지확인: { "message_type": "stopped", "tts_text": "안내를 중지했습니다" }
    """
    await websocket.accept()

    session = NavigationSession()
    scene_memory = SceneMemory()
    last_slow_time: float = 0.0

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "frame")

            # ── 세션 시작 ───────────────────────────────────────────────────
            if action == "start":
                target = data.get("target", "화장실")
                session.start(target)
                slow_channel.reset()
                scene_memory.reset()
                last_slow_time = 0.0
                await websocket.send_json({
                    "message_type": "started",
                    "tts_text": f"{target} 안내를 시작합니다",
                    "arrived": False,
                    "progress": "",
                })
                continue

            # ── 세션 중지 ───────────────────────────────────────────────────
            if action == "stop":
                session.stop()
                await websocket.send_json({
                    "message_type": "stopped",
                    "tts_text": "안내를 중지했습니다",
                    "arrived": False,
                    "progress": "",
                })
                break

            # ── 프레임 처리 ─────────────────────────────────────────────────
            if not session.is_navigating:
                continue

            b64_frame = data.get("frame", "")
            target = data.get("target", session.target)

            # base64 → bytes
            raw_b64 = b64_frame.split(",", 1)[-1] if "," in b64_frame else b64_frame
            image_bytes = base64.b64decode(raw_b64)

            # 채널 처리
            result = _process_frame(image_bytes, target, last_slow_time, scene_memory)
            last_slow_time = result["new_slow_time"]

            confirmed_dir = result["slow_result"].get("confirmed_direction", "unknown")
            session.update_direction(confirmed_dir)

            # 우선순위 판단
            decision = priority_module.decide(result["fast_result"], result["slow_result"])

            # 도착 판정 (VLM이 실행된 경우에만)
            raw_vlm = result["raw_vlm"]
            if raw_vlm:
                arrived = session.check_arrival(
                    goal_visible=raw_vlm.get("goal_visible", False),
                    goal_distance_str=str(raw_vlm.get("goal_distance", "unknown")),
                    confidence=float(raw_vlm.get("confidence", 0.0)),
                )
                if arrived:
                    await websocket.send_json({
                        "message_type": "arrived",
                        "tts_text": session.arrival_message(),
                        "priority": 1,
                        "suppress_guidance": True,
                        "arrived": True,
                        "progress": "",
                        "yolo_context": result["yolo_context"],
                    })
                    session.stop()
                    break

            # 진행 피드백 (방향 안내 중일 때만 대체)
            progress = session.get_progress_feedback(confirmed_dir)
            tts_out = decision["tts_text"]
            if progress and decision["message_type"] == "guidance":
                tts_out = progress

            await websocket.send_json({
                "message_type": decision["message_type"],
                "tts_text": tts_out,
                "priority": decision["priority"],
                "suppress_guidance": decision["suppress_guidance"],
                "arrived": False,
                "progress": progress,
                "yolo_context": result["yolo_context"],
            })

    except WebSocketDisconnect:
        session.stop()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": fast_channel is not None}
