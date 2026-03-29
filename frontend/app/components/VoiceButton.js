"use client";

import { useEffect } from "react";

export default function VoiceButton({ isListening, onStart, onStop, disabled }) {
  // 진동 피드백 (모바일)
  useEffect(() => {
    if (!navigator.vibrate) return;
    if (isListening) {
      navigator.vibrate(50);
    } else {
      navigator.vibrate([30, 30, 30]);
    }
  }, [isListening]);

  return (
    <button
      aria-label={isListening ? "음성 인식 중 — 손을 떼면 인식 완료" : "음성으로 목적지 말하기"}
      aria-pressed={isListening}
      disabled={disabled}
      onPointerDown={(e) => { e.preventDefault(); onStart?.(); }}
      onPointerUp={(e) => { e.preventDefault(); onStop?.(); }}
      onPointerLeave={(e) => { if (isListening) { e.preventDefault(); onStop?.(); } }}
      className={[
        "relative flex items-center justify-center",
        "w-20 h-20 rounded-full select-none touch-none",
        "transition-all duration-150",
        isListening
          ? "bg-red-500 scale-110 shadow-lg shadow-red-500/40"
          : "bg-gray-800 border-2 border-gray-700 hover:border-blue-600 hover:bg-gray-700",
        disabled
          ? "opacity-30 cursor-not-allowed"
          : "cursor-pointer active:scale-90",
      ].join(" ")}
    >
      {/* 마이크 아이콘 */}
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="currentColor"
        className={`w-8 h-8 transition-colors ${isListening ? "text-white" : "text-gray-400"}`}
        aria-hidden="true"
      >
        <path d="M8.25 4.5a3.75 3.75 0 117.5 0v8.25a3.75 3.75 0 11-7.5 0V4.5z" />
        <path d="M6 10.5a.75.75 0 01.75.75v1.5a5.25 5.25 0 1010.5 0v-1.5a.75.75 0 011.5 0v1.5a6.751 6.751 0 01-6 6.709v2.291h3a.75.75 0 010 1.5h-7.5a.75.75 0 010-1.5h3v-2.291a6.751 6.751 0 01-6-6.709v-1.5A.75.75 0 016 10.5z" />
      </svg>

      {/* 청취 중 펄스 링 */}
      {isListening && (
        <>
          <span className="absolute inset-0 rounded-full animate-ping bg-red-400 opacity-25" />
          <span className="absolute -inset-2 rounded-full border-2 border-red-400/40 animate-ping" style={{ animationDelay: "0.3s" }} />
        </>
      )}
    </button>
  );
}
