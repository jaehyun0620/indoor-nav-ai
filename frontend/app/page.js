"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSTT } from "./hooks/useSTT";
import { useTTS } from "./hooks/useTTS";

const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL || "https://indoor-nav-ai-production.up.railway.app"
).replace(/\/+$/, "");
const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws/navigate";

// 모바일(iOS/Android) 감지 — LTE 환경에서 프레임 크기·주기 자동 조정
const IS_MOBILE = typeof navigator !== "undefined" &&
  /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

const CAPTURE_INTERVAL_MS   = IS_MOBILE ? 800  : 500;   // 모바일: 800ms, PC: 500ms
const CAPTURE_JPEG_QUALITY  = IS_MOBILE ? 0.6  : 0.8;   // 모바일: 낮은 화질로 전송량 감소
const CAPTURE_WIDTH         = IS_MOBILE ? 480  : 640;   // 모바일: 480px
const MAX_WS_BUFFERED_BYTES = 512 * 1024;

const PRESET_TARGETS = ["화장실", "강의실", "엘리베이터"];

const DIRECTION = {
  left: { label: "왼쪽", mark: "←" },
  right: { label: "오른쪽", mark: "→" },
  straight: { label: "직진", mark: "↑" },
  unknown: { label: "확인 중", mark: "·" },
};

const MESSAGE = {
  warning: { label: "위험", color: "#ef4444", bg: "#2a0c0c" },
  caution: { label: "주의", color: "#f59e0b", bg: "#261a07" },
  guidance: { label: "안내", color: "#22c55e", bg: "#071f12" },
  searching: { label: "탐색", color: "#3b82f6", bg: "#071629" },
  orientation: { label: "파악 중", color: "#38bdf8", bg: "#071a26" },
  monitoring: { label: "감시", color: "#94a3b8", bg: "#0b1220" },
  arrived: { label: "도착", color: "#a855f7", bg: "#160d24" },
  stopped: { label: "중지", color: "#94a3b8", bg: "#0b1220" },
  ready: { label: "준비", color: "#94a3b8", bg: "#0b1220" },
};

const styles = {
  page: {
    minHeight: "100vh",
    background: "#080b10",
    color: "#f8fafc",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
    display: "flex",
    justifyContent: "center",
  },
  shell: {
    width: "100%",
    maxWidth: 460,
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    padding: "18px 16px 20px",
    gap: 14,
    boxSizing: "border-box",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  title: {
    margin: 0,
    fontSize: 21,
    lineHeight: 1.2,
    fontWeight: 800,
    letterSpacing: 0,
  },
  subtitle: {
    margin: "4px 0 0",
    fontSize: 13,
    lineHeight: 1.35,
    color: "#94a3b8",
  },
  pill: {
    minWidth: 72,
    height: 34,
    padding: "0 11px",
    borderRadius: 999,
    border: "1px solid #243244",
    background: "#0d1420",
    color: "#cbd5e1",
    fontSize: 12,
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 7,
    whiteSpace: "nowrap",
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#475569",
    flex: "0 0 auto",
  },
  targetPanel: {
    display: "grid",
    gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
    gap: 8,
  },
  targetButton: {
    height: 48,
    borderRadius: 8,
    border: "1px solid #243244",
    background: "#101722",
    color: "#cbd5e1",
    fontSize: 14,
    fontWeight: 750,
    cursor: "pointer",
  },
  inputRow: {
    display: "grid",
    gridTemplateColumns: "1fr 50px",
    gap: 8,
  },
  input: {
    width: "100%",
    height: 46,
    padding: "0 13px",
    borderRadius: 8,
    border: "1px solid #243244",
    background: "#0d1420",
    color: "#f8fafc",
    fontSize: 15,
    outline: "none",
    boxSizing: "border-box",
  },
  iconButton: {
    width: 50,
    height: 46,
    borderRadius: 8,
    border: "1px solid #243244",
    background: "#101722",
    color: "#e2e8f0",
    fontSize: 19,
    fontWeight: 800,
    cursor: "pointer",
  },
  cameraBox: {
    width: "100%",
    aspectRatio: "4 / 3",
    borderRadius: 8,
    overflow: "hidden",
    background: "#111827",
    border: "1px solid #243244",
    position: "relative",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block",
    background: "#111827",
  },
  cameraEmpty: {
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#64748b",
    fontSize: 15,
    fontWeight: 650,
  },
  overlayTop: {
    position: "absolute",
    top: 10,
    left: 10,
    right: 10,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    pointerEvents: "none",
  },
  overlayBadge: {
    height: 30,
    padding: "0 10px",
    borderRadius: 999,
    background: "rgba(8, 11, 16, 0.76)",
    border: "1px solid rgba(226, 232, 240, 0.14)",
    display: "flex",
    alignItems: "center",
    gap: 7,
    fontSize: 12,
    fontWeight: 800,
    color: "#e2e8f0",
  },
  directionOverlay: {
    position: "absolute",
    left: "50%",
    bottom: 12,
    transform: "translateX(-50%)",
    minWidth: 150,
    height: 48,
    borderRadius: 999,
    background: "rgba(8, 11, 16, 0.84)",
    border: "1px solid rgba(34, 197, 94, 0.45)",
    color: "#bbf7d0",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    fontSize: 15,
    fontWeight: 850,
    pointerEvents: "none",
  },
  directionMark: {
    width: 28,
    height: 28,
    borderRadius: 999,
    background: "#12351f",
    color: "#86efac",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 18,
    lineHeight: 1,
  },
  status: {
    minHeight: 88,
    borderRadius: 8,
    border: "1px solid #243244",
    background: "#0d1420",
    padding: 14,
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    gap: 8,
  },
  statusHead: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  statusLabel: {
    fontSize: 12,
    fontWeight: 850,
    letterSpacing: 0,
  },
  statusTarget: {
    fontSize: 12,
    color: "#94a3b8",
    whiteSpace: "nowrap",
  },
  statusText: {
    margin: 0,
    color: "#e2e8f0",
    fontSize: 16,
    lineHeight: 1.45,
    fontWeight: 720,
  },
  actions: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 10,
  },
  primaryAction: {
    height: 58,
    borderRadius: 8,
    border: "1px solid #2563eb",
    background: "#1d4ed8",
    color: "#fff",
    fontSize: 16,
    fontWeight: 850,
    cursor: "pointer",
  },
  secondaryAction: {
    height: 58,
    borderRadius: 8,
    border: "1px solid #334155",
    background: "#111827",
    color: "#e2e8f0",
    fontSize: 16,
    fontWeight: 850,
    cursor: "pointer",
  },
  startAction: {
    height: 58,
    borderRadius: 8,
    border: "1px solid #16a34a",
    background: "#15803d",
    color: "#fff",
    fontSize: 16,
    fontWeight: 850,
    cursor: "pointer",
  },
  error: {
    borderRadius: 8,
    border: "1px solid #7f1d1d",
    background: "#2a0c0c",
    color: "#fecaca",
    padding: "11px 12px",
    fontSize: 13,
    lineHeight: 1.35,
  },
};

export default function HomePage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const frameReadyRef = useRef(false);
  const rvcHandleRef = useRef(null);
  const lastSpokenRef = useRef({ text: "", type: "", at: 0 });
  const tapCountRef = useRef(0);
  const tapTimerRef = useRef(null);

  const [target, setTarget] = useState("");
  const [navState, setNavState] = useState("idle");
  const [decision, setDecision] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [debugInfo, setDebugInfo] = useState("초기 상태");
  const [started, setStarted] = useState(false);   // 온보딩 시작 화면 통과 여부

  const heartbeatRef         = useRef(null);  // 주행 중 주기적 안심 멘트 타이머
  const intentionalCloseRef  = useRef(false); // 사용자가 의도적으로 중지했는지
  const reconnectTimerRef    = useRef(null);  // 재연결 예약 타이머
  const reconnectAttemptsRef = useRef(0);     // 재연결 시도 횟수
  const connectWSRef         = useRef(null);  // 최신 connectWS 참조 (onclose에서 사용)
  const navTargetRef         = useRef("");    // 재연결 시 사용할 목적지

  const { transcript, isListening, start: startSTT, stop: stopSTT, reset: resetSTT } = useSTT();
  const { speak, stop: stopTTS, clearPending, unlockAudio, isSpeaking } = useTTS();
  const isSpeakingRef = useRef(false);  // TTS 재생 상태 ref (STT 시작 타이밍 판단용)
  const sttTimerRef   = useRef(null);   // STT 대기 타이머

  const isRunning = navState === "navigating";
  const direction = DIRECTION[decision?.direction] || DIRECTION.unknown;

  // isSpeaking → ref 미러 (setTimeout 클로저에서 최신 값 참조용)
  useEffect(() => { isSpeakingRef.current = isSpeaking; }, [isSpeaking]);

  // ── 주행 중 주기적 안심 멘트 (heartbeat) ────────────────────────────────
  // 시각장애인 사용자는 화면을 볼 수 없으므로, 한동안 아무 안내가 없으면
  // 시스템이 멈춘 건지 알 수 없다. 18초간 발화가 없으면 작동 중임을 알린다.
  useEffect(() => {
    if (!isRunning) {
      if (heartbeatRef.current) clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
      return;
    }
    heartbeatRef.current = setInterval(() => {
      const idle = Date.now() - lastSpokenRef.current.at;
      if (idle > 18000 && !isSpeakingRef.current && !isQuerying) {
        speak("계속 안내 중입니다. 방향이 궁금하면 화면을 한 번 탭하세요.", false);
        lastSpokenRef.current = { text: "heartbeat", type: "heartbeat", at: Date.now() };
      }
    }, 3000);
    return () => {
      if (heartbeatRef.current) clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    };
  }, [isRunning, isQuerying, speak]);

  const messageType = decision?.message_type || (isRunning ? "monitoring" : "ready");
  const message = MESSAGE[messageType] || MESSAGE.monitoring;

  const startCamera = useCallback(async () => {
    // 이미 카메라 스트림이 활성화돼 있으면 중복 실행 방지
    if (videoRef.current?.srcObject) return true;
    setCameraError(null);
    setCameraReady(false);
    setDebugInfo("카메라 권한 요청 중");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width:  { ideal: CAPTURE_WIDTH },
          height: { ideal: IS_MOBILE ? 360 : 480 },
        },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraReady(true);
        setDebugInfo("카메라 준비 완료");
      }
      return true;
    } catch (e) {
      const message = `카메라 오류: ${e.message}`;
      setCameraError(message);
      setDebugInfo(message);
      // 시각장애인 사용자를 위해 카메라 실패도 음성으로 안내
      speak("카메라를 열 수 없습니다. 카메라 권한을 허용했는지 확인해 주세요.", true);
      return false;
    }
  }, [speak]);

  const stopCamera = useCallback(() => {
    videoRef.current?.srcObject?.getTracks().forEach((track) => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraReady(false);
  }, []);

  const startFrameCache = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const capture = () => {
      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        frameReadyRef.current = true;
      }
    };

    if (video.requestVideoFrameCallback) {
      const loop = () => {
        capture();
        rvcHandleRef.current = video.requestVideoFrameCallback(loop);
      };
      rvcHandleRef.current = video.requestVideoFrameCallback(loop);
    } else {
      const loop = () => {
        capture();
        rvcHandleRef.current = requestAnimationFrame(loop);
      };
      rvcHandleRef.current = requestAnimationFrame(loop);
    }
  }, []);

  const captureCurrentFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return false;
    if (video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) return false;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    frameReadyRef.current = true;
    return true;
  }, []);

  const stopFrameCache = useCallback(() => {
    const video = videoRef.current;
    if (rvcHandleRef.current != null) {
      video?.requestVideoFrameCallback
        ? video.cancelVideoFrameCallback(rvcHandleRef.current)
        : cancelAnimationFrame(rvcHandleRef.current);
      rvcHandleRef.current = null;
    }
    frameReadyRef.current = false;
  }, []);

  const handleMessage = useCallback((data) => {
    const { message_type, tts_text, query_response } = data;
    setDecision(data);
    if (data.debug) {
      const parts = [
        `서버 응답: ${message_type}`,
        `stage=${data.debug.stage || "-"}`,
        `visible=${String(data.debug.goal_visible ?? "-")}`,
        `dir=${data.debug.goal_direction || data.direction || "-"}`,
        `conf=${data.debug.confidence ?? "-"}`,
      ];
      if (data.debug.error) parts.push(`error=${data.debug.error}`);
      if (data.debug.reasoning) parts.push(`reason=${String(data.debug.reasoning).slice(0, 120)}`);
      setDebugInfo(parts.join(" | "));
    } else {
      setDebugInfo(`서버 응답: ${message_type || "unknown"}`);
    }

    if (message_type === "arrived") {
      intentionalCloseRef.current = true;   // 도착 → 자동 재연결 막기
      stopTTS();
      speak(tts_text, true);
      setNavState("arrived");
      clearInterval(intervalRef.current);
      stopFrameCache();
      stopCamera();
      if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.close();
      wsRef.current = null;
      setWsConnected(false);
      setTimeout(() => {
        setNavState("idle");
        setDecision(null);
        setTarget("");
      }, 6000);
      return;
    }

    if (message_type === "stopped") return;

    if (query_response) {
      setIsQuerying(false);
      speak(tts_text, true);
      lastSpokenRef.current = { text: tts_text, type: message_type, at: Date.now() };
      return;
    }

    const now = Date.now();
    const prev = lastSpokenRef.current;
    const elapsed = now - prev.at;

    if (message_type === "warning") {
      // 같은 경고 텍스트는 5초 쿨다운 — 매 프레임 interrupt로 음성이 잘리는 현상 방지
      if (tts_text !== prev.text || elapsed > 5000) {
        // 촉각 채널 — 시끄러운 환경/소리 못 듣는 경우 대비 강한 진동
        if (typeof navigator !== "undefined" && navigator.vibrate) {
          navigator.vibrate([120, 60, 120]);
        }
        speak(tts_text, true);
        lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      }
      return;
    }

    if (message_type === "caution") {
      // warning 직후 5초 이내에는 caution 억제 — warning↔caution 진동으로 음성 끊김 방지
      if (prev.type === "warning" && elapsed < 5000) return;
      if (tts_text !== prev.text || elapsed > 4000) {
        speak(tts_text, false);
        lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      }
      return;
    }

    if (message_type === "monitoring") {
      speak(tts_text, false);
      lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      return;
    }

    // 초기 장면 파악 — 세션 시작 직후 한 번 발화
    if (message_type === "orientation") {
      speak(tts_text, false);
      lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      return;
    }

    if (message_type === "guidance" || message_type === "searching") {
      if (tts_text !== prev.text || elapsed > 8000) {
        speak(tts_text, false);
        lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      }
    }
  }, [speak, stopTTS, stopFrameCache, stopCamera]);

  // WebSocket 연결 수립 (최초 연결 + 재연결 공용)
  const connectWS = useCallback((dest, isReconnect = false) => {
    setDebugInfo(isReconnect ? "재연결 시도 중" : `WebSocket 연결 시도: ${WS_URL}`);
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0;
      setDebugInfo(isReconnect ? "재연결됨" : "WebSocket 연결됨");
      ws.send(JSON.stringify({ action: "start", target: dest }));
      setNavState("navigating");
      setWsConnected(true);
      setIsQuerying(false);
      if (!isReconnect) {
        setDecision(null);
        lastSpokenRef.current = { text: "", type: "", at: 0 };
      } else {
        // 재연결 성공을 사용자에게 음성으로 알림
        speak("다시 연결되었습니다. 계속 안내합니다.", true);
        lastSpokenRef.current = { text: "재연결", type: "monitoring", at: Date.now() };
      }
      startFrameCache();
      intervalRef.current = setInterval(() => {
        if (
          ws.readyState === WebSocket.OPEN &&
          frameReadyRef.current &&
          ws.bufferedAmount < MAX_WS_BUFFERED_BYTES
        ) {
          canvasRef.current?.toBlob((blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
          }, "image/jpeg", CAPTURE_JPEG_QUALITY);
        }
      }, CAPTURE_INTERVAL_MS);
    };

    ws.onmessage = (e) => {
      try {
        const parsed = JSON.parse(e.data);
        handleMessage(parsed);
      } catch {
        setDebugInfo("서버 응답 파싱 실패");
      }
    };

    ws.onerror = () => {
      console.error(`[WS] 연결 오류: ${WS_URL}`);
      setDebugInfo(`WebSocket 오류: ${WS_URL}`);
      setWsConnected(false);
      // onclose가 이어서 호출되므로 재연결은 onclose에서 처리
    };

    ws.onclose = (event) => {
      setDebugInfo(`WebSocket 종료: code=${event.code}`);
      clearInterval(intervalRef.current);
      stopFrameCache();
      setIsQuerying(false);
      setWsConnected(false);

      // 사용자가 의도적으로 중지한 경우 → 재연결 안 함
      if (intentionalCloseRef.current) {
        intentionalCloseRef.current = false;
        setNavState("idle");
        return;
      }

      // 예기치 않은 끊김 → 자동 재연결 (최대 5회, 지수 백오프)
      if (reconnectAttemptsRef.current < 5) {
        reconnectAttemptsRef.current += 1;
        const attempt = reconnectAttemptsRef.current;
        const delay = Math.min(1000 * attempt, 4000);
        if (attempt === 1) {
          speak("연결이 끊겼습니다. 다시 연결하고 있습니다. 잠시만 기다려 주세요.", true);
        }
        setDebugInfo(`재연결 예약 (${attempt}/5, ${delay}ms)`);
        reconnectTimerRef.current = setTimeout(() => {
          connectWSRef.current?.(dest, true);
        }, delay);
      } else {
        // 재연결 한계 초과 → 안내 종료
        setNavState("idle");
        setCameraError("서버에 연결할 수 없습니다.");
        speak("서버에 다시 연결하지 못했습니다. 안내를 종료합니다.", true);
      }
    };
  }, [startFrameCache, stopFrameCache, handleMessage, speak]);

  // onclose 콜백에서 최신 connectWS를 참조하기 위한 ref 동기화
  useEffect(() => { connectWSRef.current = connectWS; }, [connectWS]);

  const startNavigation = useCallback(async (navTarget) => {
    const dest = (navTarget || target).trim();
    if (!dest) return;
    navTargetRef.current = dest;
    intentionalCloseRef.current = false;
    reconnectAttemptsRef.current = 0;
    const ok = await startCamera();
    if (!ok) return;
    connectWS(dest, false);
  }, [target, startCamera, connectWS]);

  // STT 인식 완료 → 목적지 설정 + TTS 재생 완료 후 안내 시작
  // (startNavigation 정의 이후에 위치해야 const TDZ 오류가 발생하지 않음)
  useEffect(() => {
    if (!transcript || navState !== "idle") return;
    const cleaned = transcript
      .replace(/찾아줘|가고\s?싶어|어디야|알려줘|데려다줘|가자|가줘|보여줘|어디에|어디로/g, "")
      .trim();
    if (cleaned.length >= 1) {
      setTarget(cleaned);
      speak(`${cleaned}으로 안내를 시작합니다`);
      resetSTT();
      // TTS "○○ 안내를 시작합니다" 재생이 끝난 뒤 startNavigation 호출
      // → 폴링으로 isSpeaking이 false가 되면 실행 (최대 4초 대기)
      let waited = 0;
      const poll = setInterval(() => {
        waited += 100;
        if (!isSpeakingRef.current || waited >= 4000) {
          clearInterval(poll);
          startNavigation(cleaned);
        }
      }, 100);
    }
  }, [transcript, navState, speak, resetSTT, startNavigation]);

  // ── STT 인식 실패/빈 결과 재안내 ────────────────────────────────────────
  // 음성 인식이 끝났는데 아무것도 못 알아들으면, 시각장애인 사용자가 멈춰버린
  // 줄 알 수 있으므로 다시 말하도록 음성으로 유도한다.
  const prevListeningRef = useRef(false);
  useEffect(() => {
    const ended = prevListeningRef.current && !isListening;
    prevListeningRef.current = isListening;
    if (ended && navState === "idle" && !transcript.trim()) {
      speak("잘 못 들었어요. 화면을 한 번 탭하고 다시 말씀해 주세요.", true);
    }
  }, [isListening, transcript, navState, speak]);

  const stopNavigation = useCallback(() => {
    // 의도적 중지 → onclose의 자동 재연결을 막는다
    intentionalCloseRef.current = true;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    reconnectAttemptsRef.current = 0;
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stop" }));
      ws.close();
    }
    clearInterval(intervalRef.current);
    stopFrameCache();
    stopCamera();
    stopTTS();
    clearPending();
    setNavState("idle");
    setWsConnected(false);
    setDecision(null);
    setIsQuerying(false);
    setDebugInfo("안내 중지됨");
    speak("안내를 중지했습니다");
  }, [stopCamera, stopFrameCache, stopTTS, clearPending, speak]);

  const queryDirection = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      const message = "서버 연결이 아직 준비되지 않았습니다.";
      setCameraError(message);
      setDebugInfo(`query 실패: WebSocket state=${ws?.readyState ?? "none"}`);
      speak(message, false);
      return;
    }
    if (!captureCurrentFrame() && !frameReadyRef.current) {
      const message = "카메라 화면이 아직 준비되지 않았습니다. 잠시 후 다시 눌러주세요.";
      setCameraError(message);
      setDebugInfo("query 실패: frameReady=false");
      speak(message, false);
      return;
    }
    if (isQuerying) return;
    const b64 = canvasRef.current?.toDataURL("image/jpeg", 0.8);
    if (!b64) {
      const message = "카메라 이미지를 캡처하지 못했습니다.";
      setCameraError(message);
      setDebugInfo("query 실패: canvas b64 없음");
      speak(message, false);
      return;
    }
    ws.send(JSON.stringify({ action: "query", frame: b64, target }));
    setIsQuerying(true);
    setCameraError(null);
    setDebugInfo("query 전송 완료");
    speak("분석 중입니다", false);
  }, [target, speak, isQuerying, captureCurrentFrame]);

  useEffect(() => () => {
    clearInterval(intervalRef.current);
    if (tapTimerRef.current) clearTimeout(tapTimerRef.current);
    if (sttTimerRef.current) clearInterval(sttTimerRef.current);
    if (heartbeatRef.current) clearInterval(heartbeatRef.current);
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    intentionalCloseRef.current = true;  // 언마운트 시 재연결 방지
    wsRef.current?.close();
    stopCamera();
  }, [stopCamera]);

  const statusText = isRunning
    ? decision?.tts_text || "화면을 한 번 터치하면 방향을 안내해 드립니다."
    : target
      ? `${target} 안내를 시작할 수 있습니다.`
      : "목적지를 선택하세요.";

  // ── 화면 탭 접근성 (시각장애인용) ──────────────────────────────────────────
  // 버튼/입력 영역을 제외한 화면 아무 곳이나 탭
  //   한 번  : 대기 중 → 목적지 음성 입력 / 안내 중 → 방향 즉시 조회
  //   두 번  : 대기 중(목적지 설정됨) → 안내 시작 / 안내 중 → 안내 중지
  const handleScreenTap = useCallback(
    (e) => {
      const tag = e.target.tagName;
      if (tag === "BUTTON" || tag === "INPUT") return;

      // iOS/Android 오디오 컨텍스트 unlock — 모든 탭에서 항상 호출 (멱등)
      unlockAudio();

      tapCountRef.current += 1;
      if (tapTimerRef.current) clearTimeout(tapTimerRef.current);

      tapTimerRef.current = setTimeout(() => {
        const count = tapCountRef.current;
        tapCountRef.current = 0;

        if (count === 1) {
          if (navState === "idle") {
            // iOS: recognition.start()는 제스처로부터 1초 이내 호출해야 함.
            // TTS 완료를 기다리면 제스처 윈도우를 벗어나 STT가 차단됨.
            // → STT를 먼저 즉시 시작, TTS는 짧은 지연 후 재생 (피드백 최소화).
            if (sttTimerRef.current) clearInterval(sttTimerRef.current);
            startSTT();               // 제스처 직후 즉시 — iOS 제스처 윈도우 준수
            setTimeout(() => {
              speak("말씀하세요");    // 짧은 안내음 (300ms 후) — STT와 겹치는 시간 최소화
            }, 300);
          } else if (navState === "navigating") {
            queryDirection();
          }
        } else if (count >= 2) {
          if (navState === "idle" && target.trim()) {
            speak("안내를 시작합니다");
            setTimeout(() => startNavigation(), 500);
          } else if (navState === "navigating") {
            stopNavigation();
          } else if (navState === "idle" && !target.trim()) {
            speak("목적지를 먼저 설정해 주세요. 화면을 한 번 탭하면 음성으로 입력할 수 있습니다");
          }
        }
      }, 300);
    },
    [navState, target, speak, startSTT, queryDirection, startNavigation, stopNavigation]
  );

  // ── 온보딩 시작 화면 진입 ───────────────────────────────────────────────
  // 첫 탭에서 ① iOS 오디오 unlock ② 사용법 음성 안내를 함께 처리한다.
  const handleStart = useCallback(() => {
    unlockAudio();
    setStarted(true);
    speak(
      "시각장애인 실내 길 안내입니다. " +
      "화장실, 강의실, 엘리베이터로 안내할 수 있어요. " +
      "목적지를 말하려면 화면을 한 번, " +
      "안내를 시작하거나 멈추려면 화면을 두 번 탭하세요."
    );
  }, [unlockAudio, speak]);

  // ── 도움말 재안내 (길게 누르기) ─────────────────────────────────────────
  const speakHelp = useCallback(() => {
    if (navState === "navigating") {
      speak("안내 중입니다. 방향이 궁금하면 화면을 한 번, 안내를 멈추려면 두 번 탭하세요.", true);
    } else {
      speak("목적지를 말하려면 화면을 한 번, 안내를 시작하려면 두 번 탭하세요.", true);
    }
  }, [navState, speak]);

  // ── 온보딩 시작 화면 ───────────────────────────────────────────────────
  if (!started) {
    return (
      <main
        style={{
          minHeight: "100vh",
          background: "#080b10",
          color: "#f8fafc",
          fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          padding: "32px 24px",
          cursor: "pointer",
        }}
        onClick={handleStart}
        role="button"
        aria-label="화면을 탭하여 시작하고 사용법을 들으세요"
      >
        <div style={{ fontSize: 56, marginBottom: 24 }} aria-hidden="true">🧭</div>
        <h1 style={{ fontSize: 26, fontWeight: 850, margin: "0 0 12px" }}>
          실내 보조 안내
        </h1>
        <p style={{ fontSize: 16, color: "#94a3b8", lineHeight: 1.6, margin: "0 0 36px", maxWidth: 320 }}>
          시각장애인을 위한 실내 길 안내입니다.<br />
          화장실 · 강의실 · 엘리베이터를 찾아드려요.
        </p>
        <div
          style={{
            border: "2px solid #2563eb",
            background: "#10254a",
            color: "#bfdbfe",
            borderRadius: 16,
            padding: "22px 28px",
            fontSize: 20,
            fontWeight: 800,
            boxShadow: "0 0 32px rgba(37,99,235,0.35)",
          }}
        >
          화면을 탭하여 시작
        </div>
        <p style={{ fontSize: 13, color: "#64748b", marginTop: 28, lineHeight: 1.6, maxWidth: 300 }}>
          탭하면 음성으로 사용법을 안내합니다.<br />
          소리가 나오도록 무음 모드를 해제해 주세요.
        </p>
      </main>
    );
  }

  return (
    <main
      style={styles.page}
      onClick={handleScreenTap}
      onContextMenu={(e) => { e.preventDefault(); speakHelp(); }}
    >
      <div style={styles.shell}>
        <header style={styles.header}>
          <div>
            <h1 style={styles.title}>실내 보조 안내</h1>
            <p style={styles.subtitle}>목적지: {target || "미선택"}</p>
          </div>
          <div style={{
            ...styles.pill,
            borderColor: wsConnected ? "#14532d" : "#243244",
            color: wsConnected ? "#86efac" : "#94a3b8",
          }}>
            <span style={{ ...styles.dot, background: wsConnected ? "#22c55e" : "#475569" }} />
            {wsConnected ? "연결됨" : "대기"}
            {isSpeaking ? " · 음성" : ""}
          </div>
        </header>

        {!isRunning && navState !== "arrived" && (
          <>
            <div style={styles.targetPanel}>
              {PRESET_TARGETS.map((item) => {
                const selected = target === item;
                return (
                  <button
                    key={item}
                    type="button"
                    onClick={() => {
                      unlockAudio();
                      setTarget(item);
                      speak(`${item} 안내를 시작합니다`);
                      setTimeout(() => startNavigation(item), 1000);
                    }}
                    style={{
                      ...styles.targetButton,
                      borderColor: selected ? "#2563eb" : "#243244",
                      background: selected ? "#10254a" : "#101722",
                      color: selected ? "#bfdbfe" : "#cbd5e1",
                    }}
                  >
                    {item}
                  </button>
                );
              })}
            </div>

            <div style={styles.inputRow}>
              <input
                type="text"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && target.trim()) startNavigation();
                }}
                placeholder="직접 입력 후 Enter"
                style={styles.input}
              />
              <button
                type="button"
                onPointerDown={startSTT}
                onPointerUp={stopSTT}
                style={{
                  ...styles.iconButton,
                  borderColor: isListening ? "#2563eb" : "#243244",
                  background: isListening ? "#10254a" : "#101722",
                }}
                aria-label="음성 입력"
              >
                {isListening ? "..." : "음성"}
              </button>
            </div>
          </>
        )}

        <section style={styles.cameraBox}>
          <video ref={videoRef} style={styles.video} playsInline muted />
          <canvas ref={canvasRef} style={{ display: "none" }} />

          {!isRunning && navState !== "arrived" && (
            <div style={styles.cameraEmpty}>카메라 대기</div>
          )}

          {isRunning && (
            <>
              <div style={styles.overlayTop}>
                <div style={styles.overlayBadge}>
                  <span style={{ ...styles.dot, background: "#ef4444" }} />
                  감시 중
                </div>
                <div style={styles.overlayBadge}>{target}</div>
              </div>

              {decision?.message_type === "warning" && (
                <div style={{
                  position: "absolute",
                  inset: 0,
                  border: "4px solid #ef4444",
                  borderRadius: 8,
                  pointerEvents: "none",
                }} />
              )}

              {isQuerying && (
                <div style={styles.directionOverlay}>
                  <span style={styles.directionMark}>···</span>
                  분석 중
                </div>
              )}

              {/* 방향 화살표: query 응답뿐 아니라 orientation/guidance에서도 표시 */}
              {!isQuerying && decision?.direction && decision.direction !== "unknown" && (
                <div style={styles.directionOverlay}>
                  <span style={styles.directionMark}>{direction.mark}</span>
                  {direction.label}
                </div>
              )}

              {/* 청중용 자막: 현재 안내 멘트를 화면 하단에 크게 표시 */}
              {decision?.tts_text && (
                <div style={{
                  position: "absolute",
                  left: 10, right: 10, bottom: 10,
                  background: "rgba(8,11,16,0.78)",
                  border: `1px solid ${message.color}`,
                  borderRadius: 10,
                  padding: "10px 12px",
                  fontSize: 15,
                  fontWeight: 700,
                  lineHeight: 1.4,
                  color: "#f1f5f9",
                  pointerEvents: "none",
                }}>
                  {decision.tts_text}
                </div>
              )}
            </>
          )}

          {navState === "arrived" && (
            <div style={{
              position: "absolute",
              inset: 0,
              background: "rgba(8, 11, 16, 0.86)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#e9d5ff",
              fontSize: 20,
              fontWeight: 850,
            }}>
              도착했습니다
            </div>
          )}
        </section>

        <section style={{
          ...styles.status,
          background: message.bg,
          borderColor: message.color,
        }}>
          <div style={styles.statusHead}>
            <span style={{ ...styles.statusLabel, color: message.color }}>{message.label}</span>
            <span style={styles.statusTarget}>
              {isRunning ? (cameraReady ? "카메라 준비" : "카메라 준비 중") : isListening ? "듣는 중" : "준비"}
            </span>
          </div>
          <p style={styles.statusText}>{statusText}</p>
        </section>

        {cameraError && <div style={styles.error}>{cameraError}</div>}

        <details style={{
          border: "1px solid #1f2937",
          borderRadius: 8,
          background: "#0b1220",
          color: "#94a3b8",
          padding: "10px 12px",
          fontSize: 12,
          lineHeight: 1.5,
        }}>
          <summary style={{ cursor: "pointer", color: "#cbd5e1", fontWeight: 750 }}>
            연결 진단
          </summary>
          <div style={{ marginTop: 8, wordBreak: "break-all" }}>
            <div>API: {API_BASE}</div>
            <div>WS: {WS_URL}</div>
            <div>카메라: {cameraReady ? "ready" : "not ready"}</div>
            <div>WebSocket: {wsConnected ? "connected" : "disconnected"}</div>
            <div>마지막 상태: {debugInfo}</div>
          </div>
        </details>

        {isRunning ? (
          <div style={styles.actions}>
            <button
              type="button"
              onClick={queryDirection}
              disabled={isQuerying}
              style={{
                ...styles.primaryAction,
                opacity: isQuerying ? 0.62 : 1,
                cursor: isQuerying ? "not-allowed" : "pointer",
              }}
            >
              {isQuerying ? "확인 중" : "현재 위치 확인"}
            </button>
            <button type="button" onClick={() => { unlockAudio(); stopNavigation(); }} style={styles.secondaryAction}>
              중지
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => { unlockAudio(); startNavigation(); }}
            disabled={navState === "arrived" || !target.trim()}
            style={{
              ...styles.startAction,
              opacity: target.trim() && navState !== "arrived" ? 1 : 0.45,
              cursor: target.trim() && navState !== "arrived" ? "pointer" : "not-allowed",
            }}
          >
            보조 안내 시작
          </button>
        )}
      </div>
    </main>
  );
}
