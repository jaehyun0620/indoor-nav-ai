"use client";

import { useCallback, useRef, useState } from "react";

/**
 * useTTS
 * Naver Clova Voice TTS를 백엔드 프록시(/tts)를 통해 호출한다.
 *   - 백엔드가 Naver API를 대신 호출 → CORS 문제 없음, API 키 서버 측 보관
 *   - 백엔드 /tts 오류(키 미설정 등) 시 브라우저 내장 Web Speech API로 폴백
 *
 * 목소리 변경: backend/.env 의 NAVER_TTS_SPEAKER 값을 바꾸면 됨
 *   vara   — 차분한 남성 내레이션 (기본값)
 *   vdain  — 조용하고 낮은 남성
 *   vdoyun — 활기찬 남성
 *
 * @returns {{ speak, stop, isSpeaking, error }}
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError]           = useState(null);
  const audioRef    = useRef(null);
  const lastTextRef = useRef("");

  /**
   * 중복 발화 방지: 같은 텍스트를 500ms 이내에 다시 읽지 않는다.
   * (더 긴 쿨다운은 page.js의 speakIfNeeded 에서 message_type / cached 기준으로 제어)
   */
  const _isDuplicate = useCallback((text) => {
    if (text === lastTextRef.current) return true;
    lastTextRef.current = text;
    setTimeout(() => {
      if (lastTextRef.current === text) lastTextRef.current = "";
    }, 500);
    return false;
  }, []);

  /** 재생 중인 오디오 중단 */
  const _stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      try { URL.revokeObjectURL(audioRef.current.src); } catch {}
      audioRef.current = null;
    }
  }, []);

  /** Blob → Audio 재생 */
  const _playBlob = useCallback((blob) => {
    _stopAudio();
    const url   = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audioRef.current  = audio;
    audio.onplay  = () => setIsSpeaking(true);
    audio.onended = () => { setIsSpeaking(false); URL.revokeObjectURL(url); };
    audio.onerror = () => { setIsSpeaking(false); };
    audio.play().catch(() => setIsSpeaking(false));
  }, [_stopAudio]);

  /** Naver Clova Voice — 백엔드 프록시 경유 (기본 엔진) */
  const _speakNaver = useCallback(async (text) => {
    try {
      const form = new FormData();
      form.append("text", text);

      const res = await fetch(`${API_URL}/tts`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        console.warn(`[TTS] 백엔드 /tts 오류 ${res.status} → Web Speech 폴백`);
        return false;
      }

      const blob = await res.blob();
      _playBlob(blob);
      return true;
    } catch (e) {
      console.warn(`[TTS] 네트워크 오류: ${e.message} → Web Speech 폴백`);
      return false;
    }
  }, [_playBlob]);

  /** Web Speech API — 폴백 엔진 */
  const _speakNative = useCallback((text) => {
    const synth = window.speechSynthesis;
    if (!synth) return;
    synth.cancel();

    const utter    = new SpeechSynthesisUtterance(text);
    utter.lang     = "ko-KR";
    utter.rate     = 1.05;
    utter.pitch    = 0.9;  // 약간 낮게 → 남성스러운 톤
    utter.volume   = 1.0;

    const voices   = synth.getVoices();
    const koMale   = voices.find((v) => v.lang.startsWith("ko") && /male|남/i.test(v.name));
    const koAny    = voices.find((v) => v.lang.startsWith("ko"));
    if (koMale)     utter.voice = koMale;
    else if (koAny) utter.voice = koAny;

    utter.onstart  = () => setIsSpeaking(true);
    utter.onend    = () => setIsSpeaking(false);
    utter.onerror  = () => setIsSpeaking(false);
    synth.speak(utter);
  }, []);

  const speak = useCallback(async (text) => {
    if (!text || _isDuplicate(text)) return;
    setError(null);

    // Naver 백엔드 프록시 먼저 → 실패 시 Web Speech
    const ok = await _speakNaver(text);
    if (!ok) _speakNative(text);
  }, [_isDuplicate, _speakNaver, _speakNative]);

  const stop = useCallback(() => {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    _stopAudio();
    setIsSpeaking(false);
  }, [_stopAudio]);

  return { speak, stop, isSpeaking, error };
}
