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

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "https://indoor-nav-ai-production.up.railway.app"
).replace(/\/+$/, "");

// iOS 오디오 요소 unlock용 짧은 무음 WAV
const SILENT_WAV =
  "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";

export function useTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError]           = useState(null);

  // ⚠️ iOS(Chrome 포함 전부 WebKit)는 사용자 제스처 내에서 play()한 오디오 요소만
  //    이후 프로그램 재생을 허용한다. 매번 new Audio()를 만들면 제스처 없는
  //    WebSocket 콜백 재생이 차단됨 → 단일 요소를 재사용한다.
  const audioRef          = useRef(null);   // 재사용하는 단일 HTMLAudioElement
  const currentUrlRef     = useRef(null);   // 현재 src의 blob URL (정리용)
  const audioCtxRef       = useRef(null);   // AudioContext (재사용)
  const lastTextRef       = useRef("");
  const isSpeakingRef     = useRef(false);  // isSpeaking state의 ref 미러
  const fetchingRef       = useRef(false);  // fetch 진행 중 여부
  const abortRef          = useRef(null);   // 진행 중인 fetch AbortController
  const pendingRef        = useRef(null);   // 재생 끝나면 바로 재생할 대기 메시지
  const speakRef          = useRef(null);   // speak 함수의 최신 참조 (onended 콜백에서 사용)
  const duplicateTimerRef = useRef(null);   // 중복 방지 타이머 (누적 방지용)
  const speakStartRef     = useRef(0);      // 재생 시작 timestamp (watchdog용)
  const watchdogRef       = useRef(null);   // isSpeaking stuck 방지 watchdog 타이머
  const audioUnlockedRef  = useRef(false);  // iOS audio 요소 unlock 여부

  /** 재사용할 단일 오디오 요소를 lazy 생성해 반환 */
  const _ensureAudioEl = useCallback(() => {
    if (!audioRef.current && typeof Audio !== "undefined") {
      audioRef.current = new Audio();
      audioRef.current.preload = "auto";
    }
    return audioRef.current;
  }, []);

  /** isSpeaking state + ref 동시 업데이트 */
  const _setIsSpeaking = useCallback((val) => {
    isSpeakingRef.current = val;
    setIsSpeaking(val);
    // watchdog: true로 설정될 때 타이머 시작, false면 타이머 해제
    if (val) {
      speakStartRef.current = Date.now();
      if (watchdogRef.current) clearTimeout(watchdogRef.current);
      watchdogRef.current = setTimeout(() => {
        // 15초 이상 isSpeaking이 true이면 강제 reset
        // (onended/onerror 미발화 → stuck 방지)
        if (isSpeakingRef.current) {
          console.warn("[TTS] watchdog: isSpeaking stuck → 강제 reset");
          isSpeakingRef.current = false;
          setIsSpeaking(false);
          fetchingRef.current = false;
          // pending 메시지 있으면 재생 시도
          const pending = pendingRef.current;
          if (pending) {
            pendingRef.current = null;
            lastTextRef.current = "";
            speakRef.current?.(pending, false);
          }
        }
        watchdogRef.current = null;
      }, 15000);
    } else {
      if (watchdogRef.current) {
        clearTimeout(watchdogRef.current);
        watchdogRef.current = null;
      }
    }
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

  /** 재생 중인 오디오 중단 + 이벤트 핸들러 해제 (요소 자체는 재사용 위해 유지) */
  const _stopAudio = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      // 이벤트 핸들러를 명시적으로 해제해 클로저 참조 누수 방지
      audio.onplay  = null;
      audio.onended = null;
      audio.onerror = null;
      try { audio.pause(); } catch {}
      // ⚠️ audioRef.current는 null로 만들지 않는다 — iOS unlock 상태를 보존하기 위함
    }
    if (currentUrlRef.current) {
      try { URL.revokeObjectURL(currentUrlRef.current); } catch {}
      currentUrlRef.current = null;
    }
  }, []);

  /** 브라우저 내장 Web Speech 중단 */
  const _stopNative = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  }, []);

  /**
   * iOS/Android 자동재생 잠금 해제.
   * 반드시 사용자 제스처(탭·버튼) 직접 핸들러 안에서 호출해야 효과 있음.
   *
   * 핵심: 실제 재생에 쓰는 단일 HTMLAudioElement를 제스처 내에서 한 번 play()해
   * unlock한다. 이후 WebSocket 콜백 등 제스처 밖에서도 같은 요소로 재생 가능.
   * (매번 new Audio()를 만들면 iOS가 차단 → 첫 메시지만 나오던 원인)
   *
   * 매 제스처마다 호출해도 안전(멱등) — 백그라운드 복귀 후 재-unlock에도 도움.
   */
  const unlockAudio = useCallback(() => {
    // 1) HTMLAudioElement unlock — 실제 TTS 재생 경로와 동일한 요소
    try {
      const el = _ensureAudioEl();
      if (el) {
        el.muted = true;
        el.src = SILENT_WAV;
        const p = el.play();
        if (p && p.then) {
          p.then(() => {
            try { el.pause(); el.currentTime = 0; } catch {}
            el.muted = false;
            audioUnlockedRef.current = true;
          }).catch(() => {
            el.muted = false;
          });
        } else {
          el.muted = false;
          audioUnlockedRef.current = true;
        }
      }
    } catch {}

    // 2) AudioContext도 resume (Web Audio 경로 보조 — 무해)
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        if (!audioCtxRef.current) audioCtxRef.current = new AudioCtx();
        const ctx = audioCtxRef.current;
        if (ctx.state === "suspended") ctx.resume().catch(() => {});
      }
    } catch {}
  }, [_ensureAudioEl]);

  /**
   * Blob → Audio 재생.
   * ⚠️ new Audio()를 만들지 않고 unlock된 단일 요소(audioRef)를 재사용한다.
   *    매번 새 요소를 만들면 iOS가 제스처 밖 재생을 차단함 (핵심 수정).
   * 재생이 끝나면 pendingRef에 대기 중인 메시지를 자동으로 재생한다.
   */
  const _playBlob = useCallback((blob) => {
    _stopNative();

    const audio = _ensureAudioEl();
    if (!audio) { _setIsSpeaking(false); return; }

    // 이전 재생/핸들러/URL 정리 (요소 자체는 유지)
    audio.onended = null;
    audio.onerror = null;
    try { audio.pause(); } catch {}
    if (currentUrlRef.current) {
      try { URL.revokeObjectURL(currentUrlRef.current); } catch {}
    }

    const url = URL.createObjectURL(blob);
    currentUrlRef.current = url;
    audio.src    = url;
    audio.muted  = false;
    audio.volume = 1.0;

    // ⚠️ isSpeaking을 동기로 즉시 true 설정 — onplay(비동기)를 기다리면
    //    speak()의 finally 블록이 먼저 실행돼 pending이 잘못 발화되는 경쟁 조건 발생.
    _setIsSpeaking(true);

    const _playNextPending = () => {
      const pending = pendingRef.current;
      if (pending) {
        pendingRef.current = null;
        lastTextRef.current = "";
        speakRef.current?.(pending, false);
      }
    };

    audio.onended = () => {
      _setIsSpeaking(false);
      _playNextPending();
    };
    audio.onerror = () => {
      _setIsSpeaking(false);
      _playNextPending();
    };
    const p = audio.play();
    if (p && p.catch) {
      p.catch((err) => {
        // iOS에서 unlock 안 된 경우 등 — Web Speech로 폴백되도록 false 처리
        console.warn(`[TTS] audio.play() 실패: ${err?.message || err}`);
        _setIsSpeaking(false);
        _playNextPending();
      });
    }
  }, [_stopNative, _ensureAudioEl, _setIsSpeaking]);

  /**
   * Naver Clova Voice — 백엔드 프록시 경유.
   * 네트워크 실패/타임아웃 시 1회 재시도 후에야 Web Speech로 폴백한다.
   * (LTE 환경에서 일시적 지연으로 여성 목소리(폴백)로 떨어지는 현상 방지)
   *
   * @returns {Promise<boolean>} Naver 재생 성공 여부 (false면 호출부가 Web Speech 폴백)
   */
  const _speakNaver = useCallback(async (text, signal) => {
    const MAX_ATTEMPTS = 2;     // 최초 1회 + 재시도 1회
    const TIMEOUT_MS   = 8000;  // LTE에서 mp3(50KB+) 다운로드 여유 확보

    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
      // 사용자 abort(urgent 인터럽트)면 즉시 중단 — 재시도하지 않음
      if (signal?.aborted) return false;

      const timeout      = new AbortController();
      const timeoutTimer = setTimeout(() => timeout.abort(), TIMEOUT_MS);

      try {
        const form = new FormData();
        form.append("text", text);

        // 사용자 abort + 타임아웃 abort 둘 다 감지 (미지원 브라우저는 user signal만)
        const combinedSignal =
          typeof AbortSignal !== "undefined" && AbortSignal.any
            ? AbortSignal.any([signal, timeout.signal].filter(Boolean))
            : signal;

        const res = await fetch(`${API_URL}/tts`, {
          method: "POST",
          body: form,
          signal: combinedSignal,
        });

        if (!res.ok) {
          console.warn(`[TTS] /tts ${res.status} (시도 ${attempt + 1})`);
          continue; // 재시도
        }

        const blob = await res.blob();
        if (!blob || blob.size === 0) {
          console.warn(`[TTS] /tts 빈 응답 (시도 ${attempt + 1})`);
          continue; // 재시도
        }

        _playBlob(blob);
        return true;
      } catch (e) {
        // 사용자가 의도적으로 abort한 경우(urgent 인터럽트)는 재시도 없이 종료
        if (signal?.aborted) return false;
        console.warn(`[TTS] 네트워크 오류 (시도 ${attempt + 1}): ${e.message}`);
        // 타임아웃 등 → 다음 attempt로 재시도
      } finally {
        clearTimeout(timeoutTimer);
      }
    }

    console.warn("[TTS] Naver 재시도 모두 실패 → Web Speech 폴백");
    return false;
  }, [_playBlob]);

  /** Web Speech API — 폴백 엔진 */
  const _speakNative = useCallback((text) => {
    const synth = window.speechSynthesis;
    if (!synth) return;
    _stopAudio();
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
  }, [_stopAudio, _setIsSpeaking]);

  /**
   * speak(text, urgent = false)
   *
   * urgent = false: 재생 중 or fetch 중 → pendingRef에 저장 후 리턴
   * urgent = true:  중복 체크 완화(100ms) → pending 초기화 → fetch 취소 → 즉시 재생
   */
  const speak = useCallback(async (text, urgent = false) => {
    if (!text) return;

    // urgent는 중복 체크를 완화 — 100ms 이내 완전히 동일한 경우만 막음
    // (경고음이 0.5초 쿨다운에 걸려 묵살되는 버그 수정)
    if (urgent) {
      if (text === lastTextRef.current && Date.now() - speakStartRef.current < 100) return;
      lastTextRef.current = text;
      speakStartRef.current = Date.now();
    } else {
      if (_isDuplicate(text)) return;
    }

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
      _stopNative();
      _setIsSpeaking(false);
    }

    setError(null);
    fetchingRef.current = true;
    // ⚠️ 이 speak 호출의 컨트롤러를 로컬 변수로 고정.
    //    abortRef.current를 직접 검사하면, 그 사이 urgent 인터럽트가 새 컨트롤러로
    //    교체했을 때 "abort 안 됨"으로 오판해 끊긴 메시지를 Web Speech(여성)로
    //    재생하는 버그 발생. → 반드시 로컬 myController로 판정한다.
    const myController = new AbortController();
    abortRef.current = myController;

    try {
      const ok = await _speakNaver(text, myController.signal);
      // 내 fetch가 abort된 경우(urgent 인터럽트 등)에는 Web Speech 폴백을 하지 않음
      if (!ok && !myController.signal.aborted) {
        _speakNative(text);
      }
    } finally {
      // abortRef가 여전히 내 컨트롤러일 때만 정리 (인터럽트로 교체됐으면 건드리지 않음)
      if (abortRef.current === myController) {
        fetchingRef.current = false;
        abortRef.current = null;
        // fetch 완료 후 대기 중인 pending이 있고 현재 재생 중이 아니면 즉시 실행
        const pending = pendingRef.current;
        if (pending && !isSpeakingRef.current) {
          pendingRef.current = null;
          lastTextRef.current = "";
          speakRef.current?.(pending, false);
        }
      }
    }
  }, [_isDuplicate, _speakNaver, _speakNative, _stopAudio, _stopNative, _setIsSpeaking]);

  // speak 최신 참조 유지 (onended 콜백에서 참조)
  useEffect(() => {
    speakRef.current = speak;
  }, [speak]);

  // 언마운트 시 오디오·타이머 정리
  useEffect(() => {
    return () => {
      if (duplicateTimerRef.current) clearTimeout(duplicateTimerRef.current);
      if (watchdogRef.current) clearTimeout(watchdogRef.current);
      if (abortRef.current) abortRef.current.abort();
      if (audioRef.current) {
        audioRef.current.onplay = null;
        audioRef.current.onended = null;
        audioRef.current.onerror = null;
        try { audioRef.current.pause(); } catch {}
      }
      if (currentUrlRef.current) {
        try { URL.revokeObjectURL(currentUrlRef.current); } catch {}
        currentUrlRef.current = null;
      }
      if (audioCtxRef.current) {
        try { audioCtxRef.current.close(); } catch {}
        audioCtxRef.current = null;
      }
      _stopNative();
    };
  }, [_stopNative]);

  const stop = useCallback(() => {
    pendingRef.current = null;
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    _stopNative();
    _stopAudio();
    _setIsSpeaking(false);
    fetchingRef.current = false;
  }, [_stopAudio, _stopNative, _setIsSpeaking]);

  const clearPending = useCallback(() => {
    pendingRef.current = null;
  }, []);

  return { speak, stop, clearPending, unlockAudio, isSpeaking, error };
}
