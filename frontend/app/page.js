"use client";

/**
 * page.js
 * 메인 카메라 UI + STT + TTS 통합 페이지.
 *
 * 흐름:
 * 1. 카메라 스트림 시작 (후면 카메라 우선)
 * 2. 2.5초마다 프레임 캡처 → FastAPI /navigate 전송
 * 3. 응답 tts_text를 useTTS로 음성 출력
 * 4. VoiceButton(STT)으로 목적지 변경
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useSTT } from "./hooks/useSTT";
import { useTTS } from "./hooks/useTTS";
import VoiceButton from "./components/VoiceButton";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const CAPTURE_INTERVAL_MS = 2500; // 빠른 채널 2~3초 주기

const TARGETS = ["화장실", "강의실", "엘리베이터"];

export default function HomePage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [target, setTarget] = useState("화장실");
  const [status, setStatus] = useState("카메라를 시작해주세요");
  const [isRunning, setIsRunning] = useState(false);
  const [lastDecision, setLastDecision] = useState(null);
  const [cameraError, setCameraError] = useState(null);

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
          facingMode: { ideal: "environment" }, // 후면 카메라 우선
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

  // ── 프레임 캡처 → API 전송 ──────────────────────────────────────────────
  const captureAndSend = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const form = new FormData();
      form.append("frame", blob, "frame.jpg");
      form.append("target", target);

      try {
        const res = await fetch(`${API_URL}/navigate`, {
          method: "POST",
          body: form,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        setLastDecision(data);
        setStatus(data.tts_text);
        speak(data.tts_text);
      } catch (e) {
        setStatus(`통신 오류: ${e.message}`);
      }
    }, "image/jpeg", 0.8);
  }, [target, speak]);

  // ── 안내 시작 / 중지 ─────────────────────────────────────────────────────
  const startNavigation = useCallback(async () => {
    await startCamera();
    setIsRunning(true);
    speak(`${target} 안내를 시작합니다`);
    intervalRef.current = setInterval(captureAndSend, CAPTURE_INTERVAL_MS);
  }, [startCamera, captureAndSend, target, speak]);

  const stopNavigation = useCallback(() => {
    clearInterval(intervalRef.current);
    setIsRunning(false);
    setStatus("안내를 중지했습니다");
    speak("안내를 중지했습니다");
    // 스트림 종료
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
  }, [speak]);

  // 언마운트 시 정리
  useEffect(() => {
    return () => {
      clearInterval(intervalRef.current);
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // ── 메시지 타입별 색상 ──────────────────────────────────────────────────
  const statusColor = {
    warning: "text-red-400",
    caution: "text-yellow-400",
    guidance: "text-green-400",
    unknown: "text-gray-400",
  }[lastDecision?.message_type] ?? "text-white";

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-4 gap-4">
      {/* 제목 */}
      <h1 className="text-2xl font-bold mt-2">실내 길 안내</h1>

      {/* 목적지 선택 */}
      <section aria-label="목적지 선택" className="flex gap-3 flex-wrap justify-center">
        {TARGETS.map((t) => (
          <button
            key={t}
            onClick={() => setTarget(t)}
            aria-pressed={target === t}
            className={[
              "px-5 py-2 rounded-full font-semibold text-lg transition-colors",
              target === t
                ? "bg-blue-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600",
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

      {/* STT 인식 텍스트 표시 */}
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

      {/* 디버그 패널 (개발 중에만 표시) */}
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
