"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSTT } from "./hooks/useSTT";
import { useTTS } from "./hooks/useTTS";
import VoiceButton from "./components/VoiceButton";

const WS_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
  .replace(/^http/, "ws") + "/ws/navigate";

const CAPTURE_INTERVAL_MS = 1000;

const TARGETS = [
  { id: "화장실", label: "화장실", icon: "🚻" },
  { id: "강의실", label: "강의실", icon: "📚" },
  { id: "엘리베이터", label: "엘리베이터", icon: "🛗" },
];

// ── 방향 화살표 SVG ─────────────────────────────────────────────────────────
function DirectionArrow({ direction }) {
  const config = {
    straight: {
      rotate: "rotate-0",
      color: "text-green-400",
      glow: "drop-shadow(0 0 20px #4ade80)",
      label: "직진",
    },
    left: {
      rotate: "-rotate-90",
      color: "text-blue-400",
      glow: "drop-shadow(0 0 20px #60a5fa)",
      label: "좌회전",
    },
    right: {
      rotate: "rotate-90",
      color: "text-blue-400",
      glow: "drop-shadow(0 0 20px #60a5fa)",
      label: "우회전",
    },
    unknown: {
      rotate: "rotate-0",
      color: "text-gray-500",
      glow: "none",
      label: "분석 중",
    },
  }[direction] ?? {
    rotate: "rotate-0",
    color: "text-gray-500",
    glow: "none",
    label: "—",
  };

  if (direction === "unknown" || !direction) {
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="w-20 h-20 rounded-full border-4 border-gray-700 flex items-center justify-center">
          <span className="text-gray-600 text-3xl font-bold">?</span>
        </div>
        <span className="text-gray-500 text-sm font-medium">{config.label}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={`${config.rotate} transition-transform duration-500 ${config.color}`}
        style={{ filter: config.glow }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 80 80"
          fill="currentColor"
          className="w-20 h-20 animate-bounce-subtle"
          aria-hidden="true"
        >
          <path d="M40 6 L68 62 L40 50 L12 62 Z" />
        </svg>
      </div>
      <span className={`text-sm font-semibold ${config.color}`}>{config.label}</span>
    </div>
  );
}

// ── 상태 카드 ────────────────────────────────────────────────────────────────
function StatusCard({ decision }) {
  if (!decision) return null;

  const typeConfig = {
    warning: {
      bg: "bg-red-950/80 border-red-500/60",
      text: "text-red-300",
      icon: "⚠️",
      badge: "bg-red-500",
      badgeText: "위험",
    },
    caution: {
      bg: "bg-amber-950/80 border-amber-500/60",
      text: "text-amber-300",
      icon: "⚡",
      badge: "bg-amber-500",
      badgeText: "주의",
    },
    guidance: {
      bg: "bg-green-950/80 border-green-500/40",
      text: "text-green-300",
      icon: "🧭",
      badge: "bg-green-600",
      badgeText: "안내",
    },
    arrived: {
      bg: "bg-blue-950/80 border-blue-400/60",
      text: "text-blue-300",
      icon: "✅",
      badge: "bg-blue-500",
      badgeText: "도착",
    },
    unknown: {
      bg: "bg-gray-900/80 border-gray-600/40",
      text: "text-gray-400",
      icon: "🔍",
      badge: "bg-gray-600",
      badgeText: "탐색",
    },
  }[decision.message_type] ?? {
    bg: "bg-gray-900/80 border-gray-600/40",
    text: "text-gray-400",
    icon: "💬",
    badge: "bg-gray-600",
    badgeText: "정보",
  };

  return (
    <div
      className={`w-full max-w-lg rounded-2xl border px-5 py-4 ${typeConfig.bg} animate-slide-up`}
      role="status"
      aria-live="assertive"
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl mt-0.5 shrink-0" aria-hidden="true">
          {typeConfig.icon}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className={`text-xs font-bold px-2 py-0.5 rounded-full text-white ${typeConfig.badge}`}
            >
              {typeConfig.badgeText}
            </span>
            {decision.goal_distance && decision.goal_distance !== "unknown" && (
              <span className="text-xs text-gray-400">{decision.goal_distance}</span>
            )}
          </div>
          <p className={`text-lg font-semibold leading-snug ${typeConfig.text}`}>
            {decision.tts_text}
          </p>
        </div>
      </div>
    </div>
  );
}

// ── 신뢰도 바 ────────────────────────────────────────────────────────────────
function ConfidenceBar({ confidence }) {
  if (confidence == null) return null;
  const pct = Math.round(confidence * 100);
  const color =
    pct >= 75 ? "bg-green-500" : pct >= 60 ? "bg-amber-500" : "bg-gray-600";
  return (
    <div className="w-full max-w-lg">
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>신뢰도</span>
        <span>{pct}%</span>
      </div>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ── 경과 시간 훅 ─────────────────────────────────────────────────────────────
function useElapsedTime(isRunning) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(null);

  useEffect(() => {
    if (isRunning) {
      startRef.current = Date.now() - elapsed * 1000;
      const id = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
      }, 1000);
      return () => clearInterval(id);
    } else {
      setElapsed(0);
    }
  }, [isRunning]); // eslint-disable-line

  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

// ── 메인 페이지 ──────────────────────────────────────────────────────────────
export default function HomePage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  const [target, setTarget] = useState("화장실");
  const [status, setStatus] = useState("목적지를 선택하고 시작하세요");
  const [isRunning, setIsRunning] = useState(false);
  const [lastDecision, setLastDecision] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [navState, setNavState] = useState("idle");
  const [wsConnected, setWsConnected] = useState(false);

  const elapsedTime = useElapsedTime(isRunning);

  const { transcript, isListening, start: startSTT, stop: stopSTT, reset: resetSTT, error: sttError } =
    useSTT();
  const { speak, isSpeaking } = useTTS();

  // STT 결과로 목적지 변경
  useEffect(() => {
    if (!transcript) return;
    const found = TARGETS.find((t) => transcript.includes(t.id));
    if (found) {
      setTarget(found.id);
      speak(`목적지를 ${found.label}(으)로 설정했습니다`);
      resetSTT();
    }
  }, [transcript, speak, resetSTT]);

  // 카메라 시작
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (e) {
      setCameraError(`카메라 오류: ${e.message}`);
    }
  }, []);

  // 카메라 종료
  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  // 프레임 캡처 → WebSocket 전송
  const captureAndSend = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    const b64 = canvas.toDataURL("image/jpeg", 0.8);
    ws.send(JSON.stringify({ action: "frame", frame: b64, target }));
  }, [target]);

  // ── TTS 쿨다운 관리 ─────────────────────────────────────────────────────────
  const lastSpokenTypeRef = useRef("");
  const lastSpokenTextRef = useRef("");
  const lastSpokenAtRef   = useRef(0);

  const speakIfNeeded = useCallback((tts_text, message_type) => {
    const now = Date.now();
    const elapsed = now - lastSpokenAtRef.current;

    // 안전 경고는 무조건 즉시 발화
    if (message_type === "warning" || message_type === "caution" || message_type === "arrived") {
      speak(tts_text);
      lastSpokenAtRef.current = now;
      lastSpokenTextRef.current = tts_text;
      lastSpokenTypeRef.current = message_type;
      return;
    }

    // guidance: 방향이 바뀌거나 8초 지나면 발화
    if (message_type === "guidance") {
      if (tts_text !== lastSpokenTextRef.current || elapsed > 8000) {
        speak(tts_text);
        lastSpokenAtRef.current = now;
        lastSpokenTextRef.current = tts_text;
        lastSpokenTypeRef.current = message_type;
      }
      return;
    }

    // unknown: 15초에 한 번만 발화
    if (message_type === "unknown") {
      if (elapsed > 15000) {
        speak(tts_text);
        lastSpokenAtRef.current = now;
        lastSpokenTextRef.current = tts_text;
        lastSpokenTypeRef.current = message_type;
      }
      return;
    }
  }, [speak]);

  // WebSocket 메시지 처리
  const handleWsMessage = useCallback(
    (data) => {
      setLastDecision(data);
      setStatus(data.tts_text);
      speakIfNeeded(data.tts_text, data.message_type);

      if (data.message_type === "arrived") {
        clearInterval(intervalRef.current);
        stopCamera();
        setIsRunning(false);
        setNavState("arrived");
        setWsConnected(false);
      }
    },
    [speakIfNeeded, stopCamera]
  );

  // 안내 시작
  const startNavigation = useCallback(async () => {
    await startCamera();

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ action: "start", target }));
      setIsRunning(true);
      setNavState("navigating");
      setWsConnected(true);
      setLastDecision(null);
      intervalRef.current = setInterval(captureAndSend, CAPTURE_INTERVAL_MS);
    };

    ws.onmessage = (e) => {
      try {
        handleWsMessage(JSON.parse(e.data));
      } catch {}
    };

    ws.onerror = () => {
      setStatus("서버 연결 오류");
      setIsRunning(false);
      setNavState("idle");
      setWsConnected(false);
    };

    ws.onclose = () => {
      clearInterval(intervalRef.current);
      setIsRunning(false);
      setWsConnected(false);
    };
  }, [startCamera, captureAndSend, handleWsMessage, target]);

  // 안내 중지
  const stopNavigation = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stop" }));
      ws.close();
    }
    clearInterval(intervalRef.current);
    stopCamera();
    setIsRunning(false);
    setNavState("idle");
    setWsConnected(false);
    setLastDecision(null);
    setStatus("안내를 중지했습니다");
    speak("안내를 중지했습니다");
  }, [stopCamera, speak]);

  // 언마운트 정리
  useEffect(() => {
    return () => {
      clearInterval(intervalRef.current);
      wsRef.current?.close();
      stopCamera();
    };
  }, [stopCamera]);

  const currentDirection = lastDecision?.goal_direction ?? null;
  const targetInfo = TARGETS.find((t) => t.id === target);

  return (
    <main className="min-h-screen bg-gray-950 flex flex-col items-center pb-8 select-none">

      {/* ── 헤더 ── */}
      <header className="w-full max-w-lg px-4 pt-5 pb-3 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">실내 길 안내</h1>
          <p className="text-xs text-gray-500 mt-0.5">Indoor Navigation AI</p>
        </div>
        {/* 연결 상태 */}
        <div className="flex items-center gap-2">
          {isSpeaking && (
            <span className="text-xs text-purple-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse-fast inline-block" />
              음성 출력
            </span>
          )}
          <div
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${
              wsConnected
                ? "border-green-700 text-green-400 bg-green-950/50"
                : "border-gray-700 text-gray-500 bg-gray-900/50"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                wsConnected ? "bg-green-400 animate-ping-slow" : "bg-gray-600"
              }`}
            />
            {wsConnected ? "연결됨" : "대기"}
          </div>
        </div>
      </header>

      <div className="w-full max-w-lg px-4 flex flex-col gap-4">

        {/* ── 목적지 선택 ── */}
        <section aria-label="목적지 선택">
          <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide">목적지</p>
          <div className="grid grid-cols-3 gap-2">
            {TARGETS.map((t) => (
              <button
                key={t.id}
                onClick={() => !isRunning && setTarget(t.id)}
                aria-pressed={target === t.id}
                disabled={isRunning}
                className={[
                  "flex flex-col items-center gap-1.5 py-3 rounded-2xl border-2 font-semibold transition-all duration-150",
                  target === t.id
                    ? "border-blue-500 bg-blue-950/60 text-blue-300"
                    : "border-gray-700/60 bg-gray-900/40 text-gray-400 hover:border-gray-600",
                  isRunning ? "opacity-40 cursor-not-allowed" : "cursor-pointer active:scale-95",
                ].join(" ")}
              >
                <span className="text-2xl" aria-hidden="true">{t.icon}</span>
                <span className="text-sm">{t.label}</span>
              </button>
            ))}
          </div>
        </section>

        {/* ── 카메라 뷰 ── */}
        <section aria-label="카메라 화면" className="relative">
          <div className="relative w-full aspect-video bg-gray-900 rounded-2xl overflow-hidden border border-gray-800">
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              playsInline
              muted
              aria-label="카메라 화면"
            />
            <canvas ref={canvasRef} className="hidden" />

            {/* 꺼짐 상태 */}
            {!isRunning && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-gray-600">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-10 h-10 opacity-40">
                  <path d="M2 6.75A2.75 2.75 0 014.75 4h9.5A2.75 2.75 0 0117 6.75v.5l3.726-2.483A.75.75 0 0122 5.5v13a.75.75 0 01-1.274.533L17 16.75v.5A2.75 2.75 0 0114.25 20h-9.5A2.75 2.75 0 012 17.25V6.75z" />
                </svg>
                <span className="text-sm opacity-60">카메라 꺼짐</span>
              </div>
            )}

            {/* 안내 중 오버레이 */}
            {isRunning && (
              <>
                {/* 코너 가이드 */}
                <div className="absolute top-3 left-3 w-6 h-6 border-t-2 border-l-2 border-white/30 rounded-tl-sm" />
                <div className="absolute top-3 right-3 w-6 h-6 border-t-2 border-r-2 border-white/30 rounded-tr-sm" />
                <div className="absolute bottom-3 left-3 w-6 h-6 border-b-2 border-l-2 border-white/30 rounded-bl-sm" />
                <div className="absolute bottom-3 right-3 w-6 h-6 border-b-2 border-r-2 border-white/30 rounded-br-sm" />

                {/* 상단 정보 바 */}
                <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-3 pt-2 pb-1.5 bg-gradient-to-b from-black/60 to-transparent">
                  <span className="flex items-center gap-1.5 text-xs text-red-400 font-semibold">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-ping-slow" />
                    REC
                  </span>
                  <span className="text-xs text-white/70 font-mono">{elapsedTime}</span>
                  <span className="text-xs text-white/70">{targetInfo?.icon} {target}</span>
                </div>

                {/* 경고 시 카메라 테두리 강조 */}
                {lastDecision?.message_type === "warning" && (
                  <div className="absolute inset-0 border-4 border-red-500/70 rounded-2xl animate-pulse pointer-events-none" />
                )}
              </>
            )}
          </div>
        </section>

        {/* ── 방향 패널 ── */}
        {isRunning && (
          <section
            aria-label="방향 안내"
            className="glass-card rounded-2xl px-6 py-5 flex items-center justify-between animate-fade-in"
          >
            <div className="flex flex-col gap-1">
              <span className="text-xs text-gray-500 uppercase tracking-wide">방향</span>
              <span className="text-2xl font-bold text-white">
                {currentDirection === "straight" && "직진 ↑"}
                {currentDirection === "left" && "← 좌회전"}
                {currentDirection === "right" && "우회전 →"}
                {(currentDirection === "unknown" || !currentDirection) && "—"}
              </span>
              {lastDecision?.goal_distance && lastDecision.goal_distance !== "unknown" && (
                <span className="text-sm text-gray-400">{lastDecision.goal_distance}</span>
              )}
            </div>
            <DirectionArrow direction={currentDirection} />
          </section>
        )}

        {/* ── 도착 완료 카드 ── */}
        {navState === "arrived" && (
          <div className="glass-card rounded-2xl px-6 py-6 text-center border border-blue-500/30 animate-slide-up">
            <div className="text-4xl mb-2">🎉</div>
            <p className="text-blue-300 text-xl font-bold">
              {targetInfo?.icon} {target}에 도착했습니다!
            </p>
            <p className="text-gray-500 text-sm mt-1">목적지에 도달했습니다</p>
            <button
              onClick={() => { setNavState("idle"); setLastDecision(null); setStatus("목적지를 선택하고 시작하세요"); }}
              className="mt-4 px-6 py-2 bg-blue-600 hover:bg-blue-500 rounded-full text-sm font-semibold transition-colors"
            >
              처음으로
            </button>
          </div>
        )}

        {/* ── 상태 카드 ── */}
        {lastDecision && navState !== "arrived" && <StatusCard decision={lastDecision} />}

        {/* ── idle 상태 안내 텍스트 ── */}
        {!isRunning && navState !== "arrived" && (
          <p className="text-center text-gray-500 text-sm px-4">
            {status}
          </p>
        )}

        {/* ── 신뢰도 바 ── */}
        {isRunning && lastDecision?.confidence != null && (
          <ConfidenceBar confidence={lastDecision.confidence} />
        )}

        {/* ── 컨트롤 ── */}
        <div className="flex items-center justify-center gap-6 mt-2">
          {/* 음성 목적지 입력 */}
          <div className="flex flex-col items-center gap-1.5">
            <VoiceButton
              isListening={isListening}
              onStart={startSTT}
              onStop={stopSTT}
              disabled={isRunning}
            />
            <span className="text-xs text-gray-600">음성 입력</span>
          </div>

          {/* 안내 시작/중지 */}
          <div className="flex flex-col items-center gap-1.5">
            <button
              onClick={isRunning ? stopNavigation : startNavigation}
              aria-label={isRunning ? "안내 중지" : "안내 시작"}
              disabled={navState === "arrived"}
              className={[
                "w-20 h-20 rounded-full font-bold text-base transition-all duration-200 active:scale-90",
                "shadow-lg flex items-center justify-center flex-col gap-1",
                isRunning
                  ? "bg-gray-700 hover:bg-gray-600 shadow-gray-900"
                  : navState === "arrived"
                  ? "bg-gray-800 opacity-40 cursor-not-allowed"
                  : "bg-green-600 hover:bg-green-500 shadow-green-950",
              ].join(" ")}
            >
              {isRunning ? (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
                    <path fillRule="evenodd" d="M4.5 7.5a3 3 0 013-3h9a3 3 0 013 3v9a3 3 0 01-3 3h-9a3 3 0 01-3-3v-9z" clipRule="evenodd" />
                  </svg>
                  <span className="text-xs">중지</span>
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
                    <path fillRule="evenodd" d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0 3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z" clipRule="evenodd" />
                  </svg>
                  <span className="text-xs">시작</span>
                </>
              )}
            </button>
            <span className="text-xs text-gray-600">{isRunning ? "안내 중" : "시작"}</span>
          </div>
        </div>

        {/* ── STT 인식 중 텍스트 ── */}
        {(transcript || isListening) && (
          <div className="glass-card rounded-xl px-4 py-3 text-center animate-fade-in">
            <p className="text-sm text-gray-300">
              {isListening
                ? <span className="text-blue-400 animate-pulse">🎙 듣는 중...</span>
                : `"${transcript}"`}
            </p>
          </div>
        )}

        {/* ── 오류 표시 ── */}
        {(cameraError || sttError) && (
          <div className="rounded-xl bg-red-950/60 border border-red-700/50 px-4 py-3">
            <p role="alert" className="text-red-400 text-sm text-center">
              {cameraError || sttError}
            </p>
          </div>
        )}

        {/* ── 디버그 패널 ── */}
        {lastDecision && (
          <details className="text-xs text-gray-600 mt-1 w-full max-w-lg">
            <summary className="cursor-pointer hover:text-gray-400 transition-colors select-none">
              🔧 디버그 패널
            </summary>
            <div className="mt-2 p-3 bg-gray-900/80 rounded-xl border border-gray-800 space-y-2">
              {lastDecision.debug && (
                <>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-400">
                    <span className="text-gray-600">VLM 호출됨</span>
                    <span className={lastDecision.debug.vlm_called ? "text-green-400" : "text-gray-600"}>
                      {lastDecision.debug.vlm_called ? "✅ Yes" : "⏳ 쿨다운"}
                    </span>
                    <span className="text-gray-600">VLM 방향</span>
                    <span className="text-white">{lastDecision.debug.vlm_direction || "—"}</span>
                    <span className="text-gray-600">VLM 신뢰도</span>
                    <span className={lastDecision.debug.vlm_confidence >= 0.6 ? "text-green-400" : "text-red-400"}>
                      {lastDecision.debug.vlm_confidence?.toFixed(2) ?? "—"}
                    </span>
                    <span className="text-gray-600">확정 방향</span>
                    <span className="text-blue-400">{lastDecision.debug.confirmed_direction}</span>
                    <span className="text-gray-600">필터 버퍼</span>
                    <span>{lastDecision.debug.filter_buffer_size}/3</span>
                    <span className="text-gray-600">unknown 연속</span>
                    <span className={lastDecision.debug.unknown_streak > 0 ? "text-amber-400" : "text-gray-400"}>
                      {lastDecision.debug.unknown_streak}회
                    </span>
                    <span className="text-gray-600">장애물 거리</span>
                    <span>{lastDecision.debug.obstacle_dist === 999 ? "없음" : `${lastDecision.debug.obstacle_dist?.toFixed(1)}m`}</span>
                  </div>
                  {lastDecision.debug.vlm_reasoning && (
                    <p className="text-gray-500 border-t border-gray-800 pt-2 leading-snug">
                      💬 {lastDecision.debug.vlm_reasoning}
                    </p>
                  )}
                  <p className="text-gray-700 border-t border-gray-800 pt-2">
                    YOLO: {lastDecision.yolo_context || "탐지 없음"}
                  </p>
                </>
              )}
            </div>
          </details>
        )}

      </div>
    </main>
  );
}
