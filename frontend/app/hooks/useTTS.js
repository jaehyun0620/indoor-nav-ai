"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * useTTS
 * Naver Clova Voice TTS를 백엔드 프록시(/tts)를 통해 호출한다.
 *   - 백엔드가 Naver API를 대신 호출 → CORS 문제 없음, API 키 서버 측 보관
 *   - 백엔드 /tts 오류(키 미설정 등) 시 브라우저 내장 Web Speech API로 폴백
 *
 * ── TTS 흐름 ────────────────────────────────────────────────────────────────
 * speak(text, urgent = false)
 *   urgent = false (방향 안내 등 일반 메시지):
 *     재생 중 or fetch 중이면 → pendingRef에 저장 (마지막 메시지만 유지)
 *     현재 재생이 끝나면 자동으로 pending 재생
 *   urgent = true (경고·주의·도착):
 *     pending 초기화 → 진행 중 fetch AbortController로 취소 → 오디오 중단 → 즉시 재생
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError]           = useState(null);

  const audioRef          = useRef(null);
  const lastTextRef       = useRef("");
  const isSpeakingRef     = useRef(false);  // isSpeaking state의 ref 미러
  const fetchingRef       = useRef(false);  // fetch 진행 중 여부
  const abortRef          = useRef(null);   // 진행 중인 fetch AbortController
  const pendingRef        = useRef(null);   // 재생 끝나면 바로 재생할 대기 메시지
  const speakRef          = useRef(null);   // speak 함수의 최신 참조 (onended 콜백에서 사용)
  const duplicateTimerRef = useRef(null);   // 중복 방지 타이머 (누적 방지용)

  /** isSpeaking state + ref 동시 업데이트 */
  const _setIsSpeaking = useCallback((val) => {
    isSpeakingRef.current = val;
    setIsSpeaking(val);
  }, []);

  /**
   * 중복 발화 방지: 같은 텍스트를 500ms 이내에 다시 읽지 않는다.
   * 이전 타이머를 취소하고 새 타이머로 교체해 누적을 방지한다.
   */
  const _isDuplicate = useCallback((text) => {
    if (text === lastTextRef.current) return true;
    lastTextRef.current = text;
    if (duplicateTimerRef.current) clearTimeout(duplicateTimerRef.current);
    duplicateTimerRef.current = setTimeout(() => {
      if (lastTextRef.current === text) lastTextRef.current = "";
      duplicateTimerRef.current = null;
    }, 500);
    return false;
  }, []);

  /** 재생 중인 오디오 중단 + 이벤트 핸들러 해제 */
  const _stopAudio = useCallback(() => {
    if (audioRef.current) {
      const audio = audioRef.current;
      // 이벤트 핸들러를 명시적으로 해제해 클로저 참조 누수 방지
      audio.onplay  = null;
      audio.onended = null;
      audio.onerror = null;
      audio.pause();
      try { URL.revokeObjectURL(audio.src); } catch {}
      audioRef.current = null;
    }
  }, []);

  /**
   * Blob → Audio 재생.
   * 재생이 끝나면 pendingRef에 대기 중인 메시지를 자동으로 재생한다.
   */
  const _playBlob = useCallback((blob) => {
    _stopAudio();
    const url   = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audioRef.current = audio;
    audio.onplay  = () => _setIsSpeaking(true);
    audio.onended = () => {
      _setIsSpeaking(false);
      URL.revokeObjectURL(url);
      const pending = pendingRef.current;
      if (pending) {
        pendingRef.current = null;
        lastTextRef.current = "";
        speakRef.current?.(pending, false);
      }
    };
    audio.onerror = () => { _setIsSpeaking(false); };
    audio.play().catch(() => _setIsSpeaking(false));
  }, [_stopAudio, _setIsSpeaking]);

  /** Naver Clova Voice — 백엔드 프록시 경유 */
  const _speakNaver = useCallback(async (text, signal) => {
    try {
      const form = new FormData();
      form.append("text", text);

      const res = await fetch(`${API_URL}/tts`, {
        method: "POST",
        body: form,
        signal,
      });

      if (!res.ok) {
        console.warn(`[TTS] 백엔드 /tts 오류 ${res.status} → Web Speech 폴백`);
        return false;
      }

      const blob = await res.blob();
      if (!blob || blob.size === 0) {
        console.warn("[TTS] 백엔드 /tts 빈 응답 → Web Speech 폴백");
        return false;
      }

      _playBlob(blob);
      return true;
    } catch (e) {
      if (e.name === "AbortError") return false;
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
    utter.pitch    = 0.9;
    utter.volume   = 1.0;

    const voices   = synth.getVoices();
    const koMale   = voices.find((v) => v.lang.startsWith("ko") && /male|남/i.test(v.name));
    const koAny    = voices.find((v) => v.lang.startsWith("ko"));
    if (koMale)     utter.voice = koMale;
    else if (koAny) utter.voice = koAny;

    utter.onstart  = () => _setIsSpeaking(true);
    utter.onend    = () => {
      _setIsSpeaking(false);
      const pending = pendingRef.current;
      if (pending) {
        pendingRef.current = null;
        lastTextRef.current = "";
        speakRef.current?.(pending, false);
      }
    };
    utter.onerror  = () => _setIsSpeaking(false);
    synth.speak(utter);
  }, [_setIsSpeaking]);

  /**
   * speak(text, urgent = false)
   *
   * urgent = false: 재생 중 or fetch 중 → pendingRef에 저장 후 리턴
   * urgent = true:  pending 초기화 → fetch 취소 → 오디오 중단 → 즉시 재생
   */
  const speak = useCallback(async (text, urgent = false) => {
    if (!text || _isDuplicate(text)) return;

    if (!urgent) {
      if (isSpeakingRef.current || fetchingRef.current) {
        pendingRef.current = text;
        return;
      }
    } else {
      pendingRef.current = null;
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
      _stopAudio();
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      _setIsSpeaking(false);
    }

    setError(null);
    fetchingRef.current = true;
    abortRef.current = new AbortController();

    try {
      const ok = await _speakNaver(text, abortRef.current.signal);
      if (!ok && !abortRef.current?.signal?.aborted) {
        _speakNative(text);
      }
    } finally {
      fetchingRef.current = false;
      abortRef.current = null;
    }
  }, [_isDuplicate, _speakNaver, _speakNative, _stopAudio, _setIsSpeaking]);

  // speak 최신 참조 유지 (onended 콜백에서 참조)
  useEffect(() => {
    speakRef.current = speak;
  }, [speak]);

  // 언마운트 시 오디오·타이머 정리
  useEffect(() => {
    return () => {
      if (duplicateTimerRef.current) clearTimeout(duplicateTimerRef.current);
      if (abortRef.current) abortRef.current.abort();
      if (audioRef.current) {
        audioRef.current.onplay = null;
        audioRef.current.onended = null;
        audioRef.current.onerror = null;
        audioRef.current.pause();
      }
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    };
  }, []);

  const stop = useCallback(() => {
    pendingRef.current = null;
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    _stopAudio();
    _setIsSpeaking(false);
    fetchingRef.current = false;
  }, [_stopAudio, _setIsSpeaking]);

  const clearPending = useCallback(() => {
    pendingRef.current = null;
  }, []);

  return { speak, stop, clearPending, isSpeaking, error };
}
