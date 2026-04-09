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
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("nav")

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

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
    """서버 시작 시 모델을 로드한다. 모델 미설치 환경에서는 경고만 출력하고 기동."""
    global fast_channel, slow_channel
    try:
        fast_channel = FastChannel(
            yolo_model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
            midas_model=os.getenv("DA2_MODEL_SIZE", "small"),
            conf_threshold=float(os.getenv("YOLO_CONF", "0.4")),
            scale_factor=float(os.getenv("MIDAS_SCALE", "1.0")),
            depth_interval=int(os.getenv("DA2_DEPTH_INTERVAL", "5")),
        )
        log.info("✅ FastChannel (YOLO + DA2) 로드 완료")
    except Exception as e:
        log.warning(f"⚠️  FastChannel 로드 실패 (모델 미설치?): {e}")
        fast_channel = None

    try:
        slow_channel = SlowChannel(
            provider=os.getenv("VLM_PROVIDER", "openai"),
            condition=os.getenv("EXPERIMENT_CONDITION", "proposed"),
        )
        log.info("✅ SlowChannel (VLM) 로드 완료")
    except Exception as e:
        log.warning(f"⚠️  SlowChannel 로드 실패: {e}")
        slow_channel = None

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
    """VLM 전송용으로 이미지를 리사이즈한다.

    표지판 글씨 인식을 위해 512px 기준으로 유지.
    VLM_IMAGE_SIZE 환경변수로 조정 가능 (기본 512).
    JPEG 품질은 82로 설정 — 70은 표지판 텍스트가 뭉개짐.
    """
    import cv2
    import numpy as np
    size = int(os.getenv("VLM_IMAGE_SIZE", "512"))   # 320 → 512 (표지판 인식 개선)
    quality = int(os.getenv("VLM_JPEG_QUALITY", "82"))  # 70 → 82 (텍스트 선명도 개선)
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    if w <= size:
        # 이미 작은 이미지도 품질 재인코딩은 적용
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return enc.tobytes()
    ratio = size / w
    resized = cv2.resize(img, (size, int(h * ratio)), interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return enc.tobytes()


async def _process_frame(image_bytes: bytes, target: str, last_slow_time: float, scene_memory: SceneMemory):
    """
    프레임 1장을 빠른/느린 채널에 통과시켜 결과를 반환한다.
    slow_channel.process() 가 async 이므로 이 함수도 async 여야 한다.

    Returns
    -------
    dict: fast_result, slow_result, yolo_context, detections, raw_vlm, new_slow_time
    """
    # 빠른 채널 (CPU 바운드 → to_thread 로 이벤트 루프 보호)
    fast_output = await asyncio.to_thread(fast_channel.process_bytes, image_bytes)
    fast_result = fast_output["fast_result"]
    yolo_context = fast_output["yolo_context"]
    detections = fast_output["detections"]

    # scene_memory 컨텍스트를 프롬프트에 추가
    # 주의: get_context_for_prompt()는 "이전 방향으로 재확인해줘" 형태의 힌트를 주는데,
    # 사용자가 방향을 바꾼 상황에서는 오히려 VLM을 이전 방향으로 편향(bias)시킬 수 있다.
    # 환경변수 SCENE_MEMORY_HINT=0 으로 비활성화 가능 (기본: 비활성화).
    use_memory_hint = os.getenv("SCENE_MEMORY_HINT", "0") == "1"
    if use_memory_hint:
        memory_hint = scene_memory.get_context_for_prompt()
        enriched_context = f"{yolo_context}\n{memory_hint}".strip() if memory_hint else yolo_context
    else:
        enriched_context = yolo_context

    # 느린 채널 (쿨다운 주기만 체크, 장면 변화 조건 제거)
    raw_vlm = {}
    now = time.time()

    if now - last_slow_time >= SLOW_CHANNEL_INTERVAL:
        log.info(f"[SLOW] VLM 호출 시작 | target={target} | context={enriched_context!r}")
        small_image = _resize_for_vlm(image_bytes)
        slow_result = await slow_channel.process(small_image, enriched_context, target)
        raw_vlm = slow_result.get("raw", {})
        new_slow_time = now
        log.info(f"[SLOW] VLM 응답 | dir={raw_vlm.get('goal_direction')} conf={raw_vlm.get('confidence')} | confirmed={slow_result['confirmed_direction']} | tts={slow_result['tts_text']!r}")
    else:
        confirmed_dir, tts_text = slow_channel.filter.get_guidance()
        slow_result = {
            "confirmed_direction": confirmed_dir,
            "tts_text": tts_text,
            "unknown_streak": slow_channel.filter.unknown_streak,
            "raw": {},
        }
        new_slow_time = last_slow_time
        wait_sec = SLOW_CHANNEL_INTERVAL - (now - last_slow_time)
        log.info(f"[SLOW] 쿨다운 중 ({wait_sec:.1f}초 후 호출) | 현재={confirmed_dir}")

    log.info(f"[YOLO] {yolo_context!r} | obstacle={fast_result.get('has_obstacle')} dist={fast_result.get('distance_m')}m")
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
    result = await _process_frame(image_bytes, target, _rest_slow_time, _rest_scene_memory)
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
    last_slow_time: float = 0.0      # frame 액션 자동 VLM 호출 쿨다운 추적
    warmup_done: bool = False         # True가 되기 전까지 VLM 안내를 먼저 실행
    obstacle_warn_count: int = 0      # 같은 장애물 연속 경고 횟수
    last_obstacle_class: str = ""     # 이전 경고에서의 장애물 클래스
    last_guidance_tts: str = ""       # 마지막 VLM 안내 텍스트 캐시 (쿨다운 재전송용)
    last_guidance_type: str = "unknown"  # 마지막 메시지 타입 캐시
    last_resend_time: float = 0.0     # 쿨다운 중 캐시 재전송 마지막 시간

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
                # 세션 상태 초기화
                last_slow_time = 0.0
                warmup_done = False
                obstacle_warn_count = 0
                last_obstacle_class = ""
                last_guidance_tts = ""
                last_guidance_type = "unknown"
                last_resend_time = 0.0
                await websocket.send_json({
                    "message_type": "started",
                    "tts_text": f"{target} 안내를 시작합니다. 잠시만 기다려 주세요.",
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

            if not session.is_navigating:
                continue

            # ── base64 → bytes (frame / query 공통) ──────────────────────
            b64_frame = data.get("frame", "")
            target = data.get("target", session.target)
            raw_b64 = b64_frame.split(",", 1)[-1] if "," in b64_frame else b64_frame
            image_bytes = base64.b64decode(raw_b64)

            # ── 프레임 처리 (자동 1초 주기) ─────────────────────────────────
            if action == "frame":
                fast_output = await asyncio.to_thread(fast_channel.process_bytes, image_bytes)
                fast_result = fast_output["fast_result"]
                yolo_context = fast_output["yolo_context"]
                detections = fast_output["detections"]

                log.info(f"[YOLO] {yolo_context!r} | obstacle={fast_result.get('has_obstacle')} dist={fast_result.get('distance_m')}m")

                # ── warmup 전: 장애물 체크 무시, VLM 첫 안내 우선 실행 ───────
                if not warmup_done:
                    now = time.time()
                    if now - last_slow_time >= SLOW_CHANNEL_INTERVAL:
                        memory_hint = scene_memory.get_context_for_prompt()
                        enriched_context = f"{yolo_context}\n{memory_hint}".strip() if memory_hint else yolo_context
                        log.info(f"[WARMUP] VLM 첫 안내 호출 | target={target}")
                        small_image = _resize_for_vlm(image_bytes)
                        # process_instant(): 필터 게이트 우회, VLM 결과 즉시 반환
                        slow_result = await slow_channel.process_instant(small_image, enriched_context, target)
                        raw_vlm = slow_result.get("raw", {})
                        last_slow_time = now
                        warmup_done = True
                        scene_memory.update(detections, raw_vlm)
                        confirmed_dir = slow_result.get("confirmed_direction", "unknown")
                        session.update_direction(confirmed_dir)
                        decision = priority_module.decide(fast_result, slow_result)
                        log.info(f"[WARMUP] 완료 | dir={confirmed_dir} tts={slow_result['tts_text']!r}")
                        # 캐시 업데이트: 쿨다운 중 재전송에 사용
                        last_guidance_tts = slow_result["tts_text"]
                        last_guidance_type = decision["message_type"]
                        last_resend_time = now
                        await websocket.send_json({
                            "message_type": decision["message_type"],
                            "tts_text": slow_result["tts_text"],
                            "priority": decision["priority"],
                            "suppress_guidance": False,
                            "arrived": False,
                            "progress": "",
                            "yolo_context": yolo_context,
                            "debug": {
                                "vlm_called": True,
                                "vlm_direction": raw_vlm.get("goal_direction", "-"),
                                "vlm_confidence": raw_vlm.get("confidence", 0),
                                "vlm_reasoning": raw_vlm.get("reasoning", ""),
                                "vlm_goal_distance": raw_vlm.get("goal_distance", "unknown"),
                                "confirmed_direction": confirmed_dir,
                                "filter_buffer_size": len(slow_channel.filter.buffer),
                                "unknown_streak": slow_result.get("unknown_streak", 0),
                                "obstacle_dist": 999,
                            },
                        })
                    continue

                # ── warmup 완료 후 정상 분기 ────────────────────────────────
                if fast_result.get("has_obstacle"):
                    obj_class = fast_result.get("class", "장애물")

                    # 연속 경고 횟수 누적
                    if obj_class == last_obstacle_class:
                        obstacle_warn_count += 1
                    else:
                        obstacle_warn_count = 1
                        last_obstacle_class = obj_class

                    if obstacle_warn_count <= 3:
                        # ── 경로 A-1: 3회 이하 → 즉각 경고 (VLM 우회) ───────
                        cached_dir, cached_tts = slow_channel.filter.get_guidance()
                        cached_slow = {"confirmed_direction": cached_dir, "tts_text": cached_tts}
                        decision = priority_module.decide(fast_result, cached_slow)
                        scene_memory.update(detections, {})
                        log.info(f"[OBSTACLE] 경고 {obstacle_warn_count}회 | {obj_class} {fast_result.get('distance_m')}m")
                        await websocket.send_json({
                            "message_type": decision["message_type"],
                            "tts_text": decision["tts_text"],
                            "priority": decision["priority"],
                            "suppress_guidance": decision["suppress_guidance"],
                            "arrived": False,
                            "progress": "",
                            "yolo_context": yolo_context,
                            "debug": {
                                "vlm_called": False,
                                "vlm_direction": cached_dir,
                                "vlm_confidence": 0,
                                "vlm_reasoning": "",
                                "vlm_goal_distance": "unknown",
                                "confirmed_direction": cached_dir,
                                "filter_buffer_size": len(slow_channel.filter.buffer),
                                "unknown_streak": slow_channel.filter.unknown_streak,
                                "obstacle_dist": fast_result.get("distance_m", 999),
                            },
                        })
                    else:
                        # ── 경로 A-2: 4회 이상 → VLM에 장애물 상황 넘겨 우회 안내 ─
                        now = time.time()
                        if now - last_slow_time >= SLOW_CHANNEL_INTERVAL:

                            memory_hint = scene_memory.get_context_for_prompt()
                            enriched_context = f"{yolo_context}\n{memory_hint}".strip() if memory_hint else yolo_context
                            log.info(f"[OBSTACLE→VLM] 반복 장애물 → VLM 우회 안내 | {obj_class}")
                            small_image = _resize_for_vlm(image_bytes)
                            # process_instant(): 장애물 상황에서 즉각 우회 방향 필요
                            slow_result = await slow_channel.process_instant(small_image, enriched_context, target)
                            raw_vlm = slow_result.get("raw", {})
                            last_slow_time = now
                            scene_memory.update(detections, raw_vlm)
                            # "앞에 사람이 있습니다. {VLM 방향 안내}" 형식으로 결합
                            dist = fast_result.get("distance_m", 0)
                            obstacle_prefix = f"앞에 {obj_class}이 있습니다. "
                            vlm_tts = slow_result.get("tts_text", "")
                            combined_tts = obstacle_prefix + vlm_tts if vlm_tts else f"{obstacle_prefix}천천히 이동하세요."
                            log.info(f"[OBSTACLE→VLM] 결합 안내: {combined_tts!r}")
                            # 캐시 업데이트
                            last_guidance_tts = combined_tts
                            last_guidance_type = "caution"
                            last_resend_time = now
                            await websocket.send_json({
                                "message_type": "caution",
                                "tts_text": combined_tts,
                                "priority": 1,
                                "suppress_guidance": False,
                                "arrived": False,
                                "progress": "",
                                "yolo_context": yolo_context,
                                "debug": {
                                    "vlm_called": True,
                                    "vlm_direction": raw_vlm.get("goal_direction", "-"),
                                    "vlm_confidence": raw_vlm.get("confidence", 0),
                                    "vlm_reasoning": raw_vlm.get("reasoning", ""),
                                    "vlm_goal_distance": raw_vlm.get("goal_distance", "unknown"),
                                    "confirmed_direction": raw_vlm.get("goal_direction", "unknown"),
                                    "filter_buffer_size": len(slow_channel.filter.buffer),
                                    "unknown_streak": slow_result.get("unknown_streak", 0),
                                    "obstacle_dist": fast_result.get("distance_m", 999),
                                },
                            })
                        else:
                            # ── A-2 쿨다운 중: 마지막 장애물+VLM 안내 재전송 ─────
                            if last_guidance_tts and (now - last_resend_time >= SLOW_CHANNEL_INTERVAL / 2):
                                last_resend_time = now
                                log.info(f"[A-2 RESEND] 쿨다운 중 캐시 재전송 | tts={last_guidance_tts!r}")
                                await websocket.send_json({
                                    "message_type": "caution",
                                    "tts_text": last_guidance_tts,
                                    "priority": 1,
                                    "suppress_guidance": False,
                                    "arrived": False,
                                    "progress": "",
                                    "yolo_context": yolo_context,
                                    "cached": True,
                                })
                else:
                    # ── 경로 B: 장애물 없음 → 자동 VLM 안내 (쿨다운 적용) ──
                    obstacle_warn_count = 0
                    last_obstacle_class = ""
                    now = time.time()
                    if now - last_slow_time >= SLOW_CHANNEL_INTERVAL:
                        memory_hint = scene_memory.get_context_for_prompt()
                        enriched_context = f"{yolo_context}\n{memory_hint}".strip() if memory_hint else yolo_context
                        log.info(f"[AUTO] VLM 자동 호출 | target={target}")
                        small_image = _resize_for_vlm(image_bytes)
                        # process_instant(): 쿨다운 간격(3s)이 이미 빈도를 제어하므로 필터 게이트 불필요
                        slow_result = await slow_channel.process_instant(small_image, enriched_context, target)
                        raw_vlm = slow_result.get("raw", {})
                        last_slow_time = now
                        scene_memory.update(detections, raw_vlm)

                        confirmed_dir = slow_result.get("confirmed_direction", "unknown")
                        session.update_direction(confirmed_dir)
                        decision = priority_module.decide(fast_result, slow_result)

                        # 도착 판정
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
                                    "yolo_context": yolo_context,
                                })
                                session.stop()
                                break

                        progress = session.get_progress_feedback(confirmed_dir)
                        tts_out = progress if (progress and decision["message_type"] == "guidance") else decision["tts_text"]
                        log.info(f"[AUTO] VLM 응답 | dir={confirmed_dir} tts={tts_out!r}")
                        # 캐시 업데이트: 쿨다운 중 재전송에 사용
                        last_guidance_tts = tts_out
                        last_guidance_type = decision["message_type"]
                        last_resend_time = now
                        await websocket.send_json({
                            "message_type": decision["message_type"],
                            "tts_text": tts_out,
                            "priority": decision["priority"],
                            "suppress_guidance": decision["suppress_guidance"],
                            "arrived": False,
                            "progress": progress,
                            "yolo_context": yolo_context,
                            "debug": {
                                "vlm_called": True,
                                "vlm_direction": raw_vlm.get("goal_direction", "-"),
                                "vlm_confidence": raw_vlm.get("confidence", 0),
                                "vlm_reasoning": raw_vlm.get("reasoning", ""),
                                "vlm_goal_distance": raw_vlm.get("goal_distance", "unknown"),
                                "confirmed_direction": confirmed_dir,
                                "filter_buffer_size": len(slow_channel.filter.buffer),
                                "unknown_streak": slow_result.get("unknown_streak", 0),
                                "obstacle_dist": 999,
                            },
                        })
                    else:
                        # 쿨다운 중 — 마지막 안내 재전송 (SLOW_CHANNEL_INTERVAL/2 주기)
                        scene_memory.update(detections, {})
                        if last_guidance_tts and (now - last_resend_time >= SLOW_CHANNEL_INTERVAL / 2):
                            last_resend_time = now
                            log.info(f"[RESEND] 쿨다운 중 캐시 재전송 | tts={last_guidance_tts!r}")
                            await websocket.send_json({
                                "message_type": last_guidance_type,
                                "tts_text": last_guidance_tts,
                                "priority": 2,
                                "suppress_guidance": False,
                                "arrived": False,
                                "progress": "",
                                "yolo_context": yolo_context,
                                "cached": True,
                            })
                continue

            # ── VLM 방향 조회 (사용자 요청 시) ─────────────────────────────
            if action == "query":
                try:
                    fast_output = await asyncio.to_thread(fast_channel.process_bytes, image_bytes)
                    fast_result = fast_output["fast_result"]
                    yolo_context = fast_output["yolo_context"]
                    detections = fast_output["detections"]

                    memory_hint = scene_memory.get_context_for_prompt()
                    enriched_context = f"{yolo_context}\n{memory_hint}".strip() if memory_hint else yolo_context

                    log.info(f"[QUERY] VLM 즉시 호출 | target={target} | context={enriched_context!r}")
                    small_image = _resize_for_vlm(image_bytes)
                    slow_result = await slow_channel.process_instant(small_image, enriched_context, target)
                    raw_vlm = slow_result.get("raw", {})
                    log.info(f"[QUERY] VLM 응답 | dir={raw_vlm.get('goal_direction')} conf={raw_vlm.get('confidence')} | tts={slow_result['tts_text']!r}")

                    scene_memory.update(detections, raw_vlm)

                    confirmed_dir = slow_result.get("confirmed_direction", "unknown")
                    session.update_direction(confirmed_dir)

                    decision = priority_module.decide(fast_result, slow_result)

                    # 도착 판정
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
                                "yolo_context": yolo_context,
                                "query_response": True,
                            })
                            session.stop()
                            break

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
                        "yolo_context": yolo_context,
                        "query_response": True,
                        "debug": {
                            "vlm_direction": raw_vlm.get("goal_direction", "-"),
                            "vlm_confidence": raw_vlm.get("confidence", 0),
                            "vlm_reasoning": raw_vlm.get("reasoning", ""),
                            "vlm_goal_distance": raw_vlm.get("goal_distance", "unknown"),
                            "vlm_called": True,
                            "confirmed_direction": confirmed_dir,
                            "filter_buffer_size": len(slow_channel.filter.buffer),
                            "unknown_streak": slow_result.get("unknown_streak", 0),
                            "obstacle_dist": fast_result.get("distance_m", 999),
                        },
                    })
                except Exception as e:
                    log.error(f"[QUERY] 처리 오류: {e}", exc_info=True)
                    await websocket.send_json({
                        "message_type": "unknown",
                        "tts_text": "분석 중 오류가 발생했습니다",
                        "priority": 3,
                        "suppress_guidance": False,
                        "arrived": False,
                        "progress": "",
                        "yolo_context": "",
                        "query_response": True,
                        "debug": {"error": str(e)},
                    })
                continue

    except WebSocketDisconnect:
        session.stop()


@app.post("/tts")
async def tts_proxy(text: str = Form(...)):
    """
    Naver Clova Voice TTS 프록시 엔드포인트.
    프론트엔드에서 직접 Naver API를 호출하면 CORS 문제가 발생하므로
    백엔드가 대신 호출하고 MP3 바이트를 반환한다.

    환경변수:
        NAVER_TTS_CLIENT_ID     : Naver Cloud Platform Application Client ID
        NAVER_TTS_CLIENT_SECRET : Naver Cloud Platform Application Client Secret
        NAVER_TTS_SPEAKER       : 목소리 (기본: vara — 차분한 남성 내레이션)
    """
    client_id     = os.getenv("NAVER_TTS_CLIENT_ID", "")
    client_secret = os.getenv("NAVER_TTS_CLIENT_SECRET", "")
    speaker       = os.getenv("NAVER_TTS_SPEAKER", "vara")

    if not client_id or not client_secret:
        raise HTTPException(status_code=503, detail="Naver TTS API 키가 설정되지 않았습니다.")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts",
                headers={
                    "X-NCP-APIGW-API-KEY-ID": client_id,
                    "X-NCP-APIGW-API-KEY":    client_secret,
                    "Content-Type":           "application/x-www-form-urlencoded",
                },
                content=f"speaker={speaker}&volume=0&speed=0&pitch=0&format=mp3&text={text}".encode(),
            )
    except Exception as e:
        log.error(f"[TTS] Naver API 호출 실패: {e}")
        raise HTTPException(status_code=502, detail=f"Naver TTS 연결 오류: {e}")

    if resp.status_code != 200:
        log.warning(f"[TTS] Naver API 오류 {resp.status_code}: {resp.text[:100]}")
        raise HTTPException(status_code=resp.status_code, detail="Naver TTS API 오류")

    log.info(f"[TTS] 합성 완료 | speaker={speaker} | {len(resp.content):,}bytes | text={text[:30]!r}")
    return Response(content=resp.content, media_type="audio/mpeg")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": fast_channel is not None}
