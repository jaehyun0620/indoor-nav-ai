"use client";

/**
 * page.js
 * 메인 카메라 UI + STT + TTS 통합 페이지.
 *
 * 흐름:
 * 1. 사용자가 목적지 선택 (버튼 or 음성)
 * 2. "시작" 버튼 → WebSocket 연결 → 세션 시작 메시지 전송
 * 3. 1초마다 카메라 프레임 캡처 → WebSocket으로 전송
 * 4. 서버 응답 tts_text를 useTTS로 음성 출력
 * 5. message_type === "arrived" 수신 시 자동 종료
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useSTT } from "./hooks/useSTT";
import { useTTS } from "./hooks/useTTS";
import VoiceButton from "./components/VoiceButton";

const WS_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
  .replace(/^http/, "ws") + "/ws/navigate";

const CAPTURE_INTERVAL_MS = 1000; // 1초마다 프레임 전송
const TARGETS = ["화장실", "강의실", "엘리베이터"];

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
  const [navState, setNavState] = useState("idle"); // idle / navigating / arrived

  const { transcript, isListening, start: startSTT, stop: stopSTT, reset: resetSTT, error: sttError } = useSTT();
  const { speak, isSpeaking } = useTTS();

  // ── STT 결과로 목적지 변경 ───────────────────────────────────────────────
  useEffect(() => {
    if (!transcript) return;
    const found = TARGETS.find((t) => transcript.includes(t));
    if (found) {
      setTarget(found);
      speak(`목적지를 ${found}(으)로 설정했습니다`);
      resetSTT();
    }
  }, [transcript, speak, resetSTT]);

  // ── 카메라 시작 ──────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (e) {
      setCameraError(`카메라 오류: ${e.message}`);
    }
  }, []);

  // ── 카메라 종료 ──────────────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  // ── 프레임 캡처 → WebSocket 전송 ────────────────────────────────────────
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

  // ── WebSocket 메시지 처리 ────────────────────────────────────────────────
  const handleWsMessage = useCallback((data) => {
    setLastDecision(data);
    setStatus(data.tts_text);
    speak(data.tts_text);

    if (data.message_type === "arrived") {
      // 도착: 루프 정지, 카메라 종료
      clearInterval(intervalRef.current);
      stopCamera();
      setIsRunning(false);
      setNavState("arrived");
    }
  }, [speak, stopCamera]);

  // ── 안내 시작 ────────────────────────────────────────────────────────────
  const startNavigation = useCallback(async () => {
    await startCamera();

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      // 세션 시작 메시지
      ws.send(JSON.stringify({ action: "start", target }));
      setIsRunning(true);
      setNavState("navigating");

      // 1초마다 프레임 전송
      intervalRef.current = setInterval(captureAndSend, CAPTURE_INTERVAL_MS);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        handleWsMessage(data);
      } catch {}
    };

    ws.onerror = () => {
      setStatus("서버 연결 오류");
      setIsRunning(false);
      setNavState("idle");
    };

    ws.onclose = () => {
      clearInterval(intervalRef.current);
      setIsRunning(false);
    };
  }, [startCamera, captureAndSend, handleWsMessage, target]);

  // ── 안내 중지 ────────────────────────────────────────────────────────────
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
    setStatus("안내를 중지했습니다");
    speak("안내를 중지했습니다");
  }, [stopCamera, speak]);

  // 언마운트 시 정리
  useEffect(() => {
    return () => {
      clearInterval(intervalRef.current);
      wsRef.current?.close();
      stopCamera();
    };
  }, [stopCamera]);

  // ── 상태별 색상 / 레이블 ─────────────────────────────────────────────────
  const statusColor = {
    warning: "text-red-400",
    caution: "text-yellow-400",
    guidance: "text-green-400",
    arrived: "text-blue-400",
    unknown: "text-gray-400",
  }[lastDecision?.message_type] ?? "text-white";

  const navStateLabel = {
    idle: "",
    navigating: `${target} 안내 중`,
    arrived: `${target} 도착`,
  }[navState];

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-4 gap-4">
      {/* 제목 */}
      <h1 className="text-2xl font-bold mt-2">실내 길 안내</h1>

      {/* 네비게이션 상태 뱃지 */}
      {navStateLabel && (
        <span className={[
          "px-4 py-1 rounded-full text-sm font-semibold",
          navState === "navigating" ? "bg-green-700 text-green-100 animate-pulse" : "bg-blue-700 text-blue-100",
        ].join(" ")}>
          {navStateLabel}
        </span>
      )}

      {/* 목적지 선택 (안내 중에는 비활성) */}
      <section aria-label="목적지 선택" className="flex gap-3 flex-wrap justify-center">
        {TARGETS.map((t) => (
          <button
            key={t}
            onClick={() => !isRunning && setTarget(t)}
            aria-pressed={target === t}
            disabled={isRunning}
            className={[
              "px-5 py-2 rounded-full font-semibold text-lg transition-colors",
              target === t ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-300",
              isRunning ? "opacity-40 cursor-not-allowed" : "hover:bg-gray-600",
            ].join(" ")}
          >
            {t}
          </button>
        ))}
      </section>

      {/* 카메라 뷰 */}
      <div className="relative w-full max-w-lg aspect-video bg-black rounded-xl overflow-hidden">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          playsInline
          muted
          aria-label="카메라 화면"
        />
        <canvas ref={canvasRef} className="hidden" />
        {!isRunning && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-lg">
            카메라 꺼짐
          </div>
        )}
        {/* 안내 중 오버레이 */}
        {isRunning && (
          <div className="absolute top-2 right-2 bg-black/60 rounded-full px-3 py-1 text-xs text-green-400 font-semibold">
            ● 안내 중
          </div>
        )}
      </div>

      {/* 상태 메시지 */}
      <p
        role="status"
        aria-live="assertive"
        className={`text-xl font-semibold text-center px-4 min-h-[2rem] ${statusColor}`}
      >
        {status}
      </p>

      {/* 컨트롤 버튼 */}
      <div className="flex gap-6 items-center mt-2">
        {/* 음성 목적지 입력 */}
        <VoiceButton
          isListening={isListening}
          onStart={startSTT}
          onStop={stopSTT}
          disabled={!isRunning}
        />

        {/* 안내 시작/중지 */}
        <button
          onClick={isRunning ? stopNavigation : startNavigation}
          aria-label={isRunning ? "안내 중지" : "안내 시작"}
          className={[
            "w-24 h-24 rounded-full font-bold text-lg transition-all",
            isRunning
              ? "bg-gray-600 hover:bg-gray-500"
              : "bg-green-600 hover:bg-green-500",
          ].join(" ")}
        >
          {isRunning ? "중지" : "시작"}
        </button>
      </div>

      {/* STT 인식 텍스트 */}
      {(transcript || isListening) && (
        <p className="text-sm text-gray-400 text-center">
          {isListening ? "듣는 중..." : `"${transcript}"`}
        </p>
      )}

      {/* 오류 표시 */}
      {(cameraError || sttError) && (
        <p role="alert" className="text-red-400 text-sm text-center">
          {cameraError || sttError}
        </p>
      )}

      {/* 디버그 패널 */}
      {process.env.NODE_ENV === "development" && lastDecision && (
        <details className="w-full max-w-lg text-xs text-gray-500 mt-2">
          <summary className="cursor-pointer">디버그 정보</summary>
          <pre className="mt-1 p-2 bg-gray-900 rounded overflow-x-auto">
            {JSON.stringify(lastDecision, null, 2)}
          </pre>
        </details>
      )}
    </main>
  );
}
