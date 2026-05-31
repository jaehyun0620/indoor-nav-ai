"""
main.py  ―  Indoor Navigation FastAPI Server
============================================================
동작 모드: 실시간 안전 감시 + 사용자 요청 기반 방향 안내

  [빠른 채널] 매 프레임 YOLO + DA2 → 장애물 즉각 경고
  [요청 기반] 사용자가 query 액션을 보낼 때만 VLM 호출

흐름:
  - 시작 후 YOLO는 계속 실행해 장애물을 감시한다.
  - VLM(느린 채널)은 자동으로 반복 호출하지 않는다.
  - 사용자가 query 를 보내면 현재 프레임을 VLM에 보내 방향을 안내한다.
  - query 중에도 장애물이 가까우면 VLM 결과보다 장애물 경고를 우선한다.

로그 태그:
  [FAST]      빠른 채널 YOLO 처리 결과
  [OBSTACLE]  장애물 경고 발송
  [QUERY]     사용자 요청 기반 VLM 호출
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nav")

from backend.channels.fast_channel import FastChannel
from backend.channels.slow_channel import SlowChannel
from backend.modules.navigation_session import NavigationSession

# ── 환경변수 ──────────────────────────────────────────────────────────────────
WARN_COOLDOWN = float(os.getenv("WARN_COOLDOWN", "3.0"))
# OCR(강의실 번호 인식)은 EasyOCR이 무겁고 CPU에서 느려 기본 비활성.
# 데모/로컬에서 ENABLE_OCR=true 로 켤 수 있다.
ENABLE_OCR = os.getenv("ENABLE_OCR", "false").lower() in ("1", "true", "yes", "on")

# ── 싱글톤 ───────────────────────────────────────────────────────────────────
fast_channel: Optional[FastChannel] = None
slow_channel: Optional[SlowChannel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fast_channel, slow_channel
    try:
        fast_channel = FastChannel(
            yolo_model     = os.getenv("YOLO_MODEL", "yolov8n.pt"),
            midas_model    = os.getenv("DA2_MODEL_SIZE", "small"),
            conf_threshold = float(os.getenv("YOLO_CONF", "0.4")),
            depth_interval = int(os.getenv("DA2_DEPTH_INTERVAL", "2")),
        )
        log.info("✅ FastChannel (YOLO + DA2) 로드 완료")
    except Exception as e:
        log.warning(f"⚠️  FastChannel 로드 실패: {e}")

    try:
        slow_channel = SlowChannel(
            provider  = os.getenv("VLM_PROVIDER", "openai"),
            condition = os.getenv("EXPERIMENT_CONDITION", "proposed"),
        )
        log.info("✅ SlowChannel (VLM) 로드 완료")
    except Exception as e:
        log.warning(f"⚠️  SlowChannel 로드 실패: {e}")

    yield


app = FastAPI(title="Indoor Navigation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _josa(word: str, jong: str, no_jong: str) -> str:
    if not word:
        return no_jong
    last = word[-1]
    if 0xAC00 <= ord(last) <= 0xD7A3:
        return jong if (ord(last) - 0xAC00) % 28 != 0 else no_jong
    return no_jong


def _resize_for_vlm(image_bytes: bytes) -> bytes:
    import cv2, numpy as np
    size    = int(os.getenv("VLM_IMAGE_SIZE", "768"))
    quality = int(os.getenv("VLM_JPEG_QUALITY", "85"))
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    if w > size:
        img = cv2.resize(img, (size, int(h * size / w)), interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return enc.tobytes()


def _ocr_room_number(image_bytes: bytes) -> str:
    """
    프레임에서 강의실 번호를 OCR로 인식해 '○○○호' 문자열을 반환한다.
    ENABLE_OCR=true 일 때만 호출되며, 실패/미인식 시 빈 문자열을 반환한다.
    (EasyOCR은 무거우므로 asyncio.to_thread로 호출할 것)
    """
    try:
        import cv2, numpy as np
        from backend.modules.ocr_pipeline import read_text, extract_room_number

        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        results = read_text(img)  # 전체 프레임 OCR
        for item in sorted(results, key=lambda r: -r["conf"]):
            room = extract_room_number(item["text"])
            if room:
                return room
    except Exception as e:
        log.warning(f"[OCR] 실패: {e}")
    return ""


def _empty_fast_output(error: str = "") -> dict:
    """FastChannel 추론 실패 시 VLM query를 계속 진행하기 위한 안전 fallback."""
    return {
        "fast_result": {
            "class": "",
            "distance_m": 999.0,
            "direction": "unknown",
            "has_obstacle": False,
            "conf": 0.0,
            "bbox": [0, 0, 0, 0],
            "detections": [],
        },
        "yolo_context": "탐지된 객체 없음",
        "detections": [],
        "fast_error": error,
    }


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/navigate")
async def ws_navigate(websocket: WebSocket):
    """
    프로토콜:
      바이너리 프레임  → 순수 JPEG bytes (YOLO 감시용)
      텍스트 JSON      → { action: start | stop | query, target?, frame? }

    동작:
      start  → YOLO 감시 시작, 안내 메시지 발송
      stop   → 감시 중단
      query  → 현재 프레임 VLM 분석 (1회), 장애물 있으면 장애물 경고 우선
      frame  → YOLO 처리, 장애물 감지 시 경고 (자동 VLM 없음)
    """
    await websocket.accept()

    target            : str   = "화장실"
    is_running        : bool  = False
    last_warn_time    : float = 0.0
    last_obstacle_cls : str   = ""
    orientation_done  : bool  = False   # 세션당 최초 1회만 실행
    session = NavigationSession()

    # WebSocket 동시 쓰기 방지 락
    send_lock = asyncio.Lock()

    async def safe_send(msg: dict) -> None:
        try:
            async with send_lock:
                await websocket.send_json(msg)
        except Exception as e:
            log.warning(f"[SEND] 전송 실패: {e}")

    # ── 메인 루프 ─────────────────────────────────────────────────────────
    try:
        while True:
            msg = await websocket.receive()
            msg_type = msg.get("type")

            if msg_type == "websocket.disconnect":
                raise WebSocketDisconnect(code=msg.get("code", 1000))

            # ── 바이너리 프레임: JPEG 직송 ───────────────────────────────
            if msg_type == "websocket.receive" and msg.get("bytes"):
                if not is_running:
                    continue
                image_bytes = msg["bytes"]
                action = "frame"
                data = {}
            else:
                raw_text = msg.get("text", "{}")
                try:
                    data = json.loads(raw_text)
                except Exception:
                    continue
                action = data.get("action", "frame")
                image_bytes = None

            # ── START ───────────────────────────────────────────────────
            if action == "start":
                target = data.get("target", "화장실")
                slow_channel.reset()
                session.start(target)
                last_warn_time    = 0.0
                last_obstacle_cls = ""
                orientation_done  = False
                is_running        = True

                log.info(f"[SESSION] 시작 | target={target}")
                await safe_send({
                    "message_type": "monitoring",
                    "tts_text": f"{target} 안내를 시작합니다. 주변 장면을 파악하고 있어요.",
                    "direction": "unknown",
                })
                continue

            # ── STOP ────────────────────────────────────────────────────
            if action == "stop":
                session.stop()
                is_running = False
                log.info("[SESSION] 중지")
                await safe_send({
                    "message_type": "stopped",
                    "tts_text":     "안내를 중지했습니다.",
                    "direction":    "unknown",
                })
                break

            if not is_running:
                continue

            # ── 텍스트 JSON frame / query: base64 디코딩 ────────────────
            if image_bytes is None:
                b64 = data.get("frame", "")
                raw_b64 = b64.split(",", 1)[-1] if "," in b64 else b64
                if not raw_b64:
                    continue
                image_bytes = base64.b64decode(raw_b64)
                target = data.get("target", target)

            # ── QUERY: 사용자 명시 방향 조회 ────────────────────────────
            if action == "query":
                log.info(f"[QUERY] target={target!r} 방향 조회 시작")
                try:
                    try:
                        if fast_channel is None:
                            raise RuntimeError("FastChannel not loaded")
                        fast_out = await asyncio.to_thread(fast_channel.process_bytes, image_bytes)
                    except Exception as e:
                        log.error(f"[QUERY→FAST] 빠른 채널 실패, VLM만 진행: {e}", exc_info=True)
                        fast_out = _empty_fast_output(str(e))

                    fast_result = fast_out["fast_result"]
                    yolo_ctx    = fast_out["yolo_context"]
                    fast_error  = fast_out.get("fast_error", "")

                    # 쿼리 중에도 가까운 장애물이 있으면 장애물 경고 우선
                    if fast_result["has_obstacle"]:
                        dist_q = fast_result["distance_m"]
                        cls_q  = fast_result["class"]
                        ig = _josa(cls_q, "이", "가")
                        if dist_q < 2.0:
                            tts_q  = f"즉시 멈추세요. {cls_q}{ig} 바로 앞에 있습니다."
                            mtype_q = "warning"
                        else:
                            tts_q  = f"주의하세요. {cls_q}{ig} {dist_q:.1f}m 앞에 있습니다."
                            mtype_q = "caution"
                        log.info(f"[QUERY→OBSTACLE] {mtype_q} | {tts_q!r}")
                        await safe_send({
                            "message_type":  mtype_q,
                            "tts_text":      tts_q,
                            "direction":     "unknown",
                            "query_response": True,
                            "debug": {
                                "stage": "query_obstacle",
                                "yolo_context": yolo_ctx,
                                "fast_error": fast_error,
                                "obstacle": {
                                    "class": cls_q,
                                    "distance_m": dist_q,
                                    "direction": fast_result.get("direction", "unknown"),
                                },
                            },
                        })
                        continue

                    slow_res  = await slow_channel.process_instant(
                        _resize_for_vlm(image_bytes), yolo_ctx, target
                    )
                    direction = slow_res.get("confirmed_direction", "unknown")
                    raw       = slow_res.get("raw", {})
                    tts_out   = slow_res["tts_text"]
                    session.update_direction(direction)
                    mtype     = "guidance" if raw.get("goal_visible") else "searching"

                    # OCR: 강의실 목표이고 목표가 보이면 번호판을 읽어 안내에 덧붙임
                    if ENABLE_OCR and raw.get("goal_visible") and "강의실" in target:
                        room = await asyncio.to_thread(_ocr_room_number, image_bytes)
                        if room:
                            tts_out = f"{tts_out} {room}호가 보입니다."
                            log.info(f"[QUERY→OCR] 강의실 번호 인식: {room}")

                    log.info(
                        f"[QUERY] visible={raw.get('goal_visible')} dir={direction} "
                        f"conf={raw.get('confidence')} dist={raw.get('goal_distance')} | "
                        f"{tts_out!r}"
                    )
                    await safe_send({
                        "message_type":   mtype,
                        "tts_text":       tts_out,
                        "direction":      direction,
                        "query_response": True,
                        "debug": {
                            "stage": "query_vlm",
                            "yolo_context": yolo_ctx,
                            "fast_error": fast_error,
                            "goal_visible": raw.get("goal_visible", False),
                            "goal_direction": raw.get("goal_direction", "unknown"),
                            "clock_position": raw.get("clock_position", ""),
                            "goal_distance": raw.get("goal_distance", "unknown"),
                            "confidence": raw.get("confidence", 0.0),
                            "reasoning": raw.get("reasoning", ""),
                            "tts_message": raw.get("tts_message", ""),
                        },
                    })

                except Exception as e:
                    log.error(f"[QUERY] 오류: {e}", exc_info=True)
                    await safe_send({
                        "message_type":   "searching",
                        "tts_text":       "방향을 파악하지 못했습니다.",
                        "direction":      "unknown",
                        "query_response": True,
                        "debug": {
                            "stage": "query_error",
                            "error": str(e),
                        },
                    })
                continue

            if action != "frame":
                continue

            # ════════════════════════════════════════════════════════════
            # YOLO + DA2 빠른 채널 — 장애물 감시
            # ════════════════════════════════════════════════════════════
            try:
                if fast_channel is None:
                    raise RuntimeError("FastChannel not loaded")
                fast_out    = await asyncio.to_thread(fast_channel.process_bytes, image_bytes)
                fast_result = fast_out["fast_result"]
                yolo_ctx    = fast_out["yolo_context"]
            except Exception as e:
                log.error(f"[FAST] 오류: {e}", exc_info=True)
                continue

            # ── 첫 프레임 초기 장면 파악 (orientation) ──────────────────
            if not orientation_done:
                orientation_done = True
                log.info(f"[ORIENTATION] 첫 프레임 장면 파악 시작 | target={target!r}")

                async def _run_orientation(img: bytes, ctx: str, tgt: str) -> None:
                    try:
                        ori = await slow_channel.run_orientation(
                            _resize_for_vlm(img), ctx, tgt
                        )
                        log.info(
                            f"[ORIENTATION] visible={ori['goal_visible']} "
                            f"dir={ori['goal_direction']} scene={ori['scene_type']!r} | "
                            f"{ori['tts_text']!r}"
                        )
                        await safe_send({
                            "message_type": "orientation",
                            "tts_text":     ori["tts_text"],
                            "direction":    ori["goal_direction"],
                            "scene_type":   ori["scene_type"],
                            "goal_visible": ori["goal_visible"],
                            "debug": {
                                "stage": "orientation",
                                "yolo_context": ctx,
                                "reasoning": ori["raw"].get("reasoning", ""),
                                "confidence": ori["raw"].get("confidence", 0.0),
                            },
                        })
                    except Exception as e:
                        log.error(f"[ORIENTATION] 오류: {e}", exc_info=True)

                asyncio.create_task(_run_orientation(image_bytes, yolo_ctx, target))

            now  = time.time()
            dist = fast_result["distance_m"]
            cls  = fast_result["class"]

            log.debug(
                f"[FAST] obstacle={fast_result['has_obstacle']}  "
                f"dist={dist:.1f}m  cls={cls!r}"
            )

            # ── 장애물 감지 → 즉각 경고 ─────────────────────────────────
            if fast_result["has_obstacle"]:

                # 경고 쿨다운 (같은 장애물 반복 경고 방지)
                if now - last_warn_time < WARN_COOLDOWN:
                    continue

                last_warn_time    = now
                last_obstacle_cls = cls
                ig = _josa(cls, "이", "가")

                if dist < 2.0:
                    tts   = f"즉시 멈추세요. {cls}{ig} 바로 앞에 있습니다."
                    mtype = "warning"
                else:
                    tts   = f"주의하세요. {cls}{ig} {dist:.1f}m 앞에 있습니다."
                    mtype = "caution"

                log.info(f"[OBSTACLE] {mtype} | dist={dist:.1f}m | {tts!r}")
                await safe_send({"message_type": mtype, "tts_text": tts, "direction": "unknown"})
                continue

            # ── 장애물 없음 → 조용히 감시 (자동 VLM 없음) ──────────────
            last_obstacle_cls = ""
            log.debug(f"[FAST] 장애물 없음 — 감시 중")

    except WebSocketDisconnect:
        session.stop()
        log.info("[WS] 연결 해제")

    except Exception as e:
        log.error(f"[WS] 예외: {e}", exc_info=True)


# ── TTS 프록시 ───────────────────────────────────────────────────────────────

@app.post("/tts")
async def tts_proxy(text: str = Form(...)):
    client_id     = os.getenv("NAVER_TTS_CLIENT_ID", "")
    client_secret = os.getenv("NAVER_TTS_CLIENT_SECRET", "")
    speaker       = os.getenv("NAVER_TTS_SPEAKER", "vara")

    if not client_id or not client_secret:
        return Response(status_code=204)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts",
                headers={
                    "X-NCP-APIGW-API-KEY-ID": client_id,
                    "X-NCP-APIGW-API-KEY":    client_secret,
                },
                data={
                    "speaker": speaker,
                    "volume": "0",
                    "speed": "0",
                    "pitch": "0",
                    "format": "mp3",
                    "text": text,
                },
            )
    except Exception as e:
        log.error(f"[TTS] 오류: {e}")
        return Response(status_code=204)

    if resp.status_code != 200:
        return Response(status_code=204)

    log.info(f"[TTS] {len(resp.content):,}bytes | {text[:30]!r}")
    return Response(content=resp.content, media_type="audio/mpeg")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "fast_channel": fast_channel is not None,
        "slow_channel": slow_channel is not None,
        "fast_mode": os.getenv("FAST_MODE", "full"),
    }
