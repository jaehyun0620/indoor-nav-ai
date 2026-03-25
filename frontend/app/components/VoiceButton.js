"use client";

/**
 * VoiceButton
 * 음성 입력 버튼 컴포넌트.
 * - 누르고 있는 동안 STT 실행 (Push-to-Talk 방식)
 * - 시각장애인 접근성: aria-label, 큰 터치 영역, 진동 피드백
 */

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
      onPointerDown={(e) => {
        e.preventDefault();
        onStart?.();
      }}
      onPointerUp={(e) => {
        e.preventDefault();
        onStop?.();
      }}
      onPointerLeave={(e) => {
        if (isListening) {
          e.preventDefault();
          onStop?.();
        }
      }}
      className={[
        // 기본 레이아웃
        "relative flex items-center justify-center",
        "w-24 h-24 rounded-full",
        "select-none touch-none",
        "text-white font-bold text-sm",
        "transition-all duration-150",
        // 상태별 색상
        isListening
          ? "bg-red-500 scale-110 shadow-lg shadow-red-400/60"
          : "bg-blue-600 hover:bg-blue-500 active:scale-95",
        disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer",
      ].join(" ")}
    >
      {/* 마이크 SVG 아이콘 */}
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="currentColor"
        className="w-10 h-10"
        aria-hidden="true"
      >
        <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z" />
        <path d="M19 10a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.92V19H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.08A7 7 0 0 0 19 10z" />
      </svg>

      {/* 청취 중 펄스 애니메이션 링 */}
      {isListening && (
        <span
          aria-hidden="true"
          className="absolute inset-0 rounded-full animate-ping bg-red-400 opacity-40"
        />
      )}
    </button>
  );
}
