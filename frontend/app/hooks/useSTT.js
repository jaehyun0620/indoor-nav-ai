"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * useSTT
 * Web Speech API 기반 한국어 음성 인식 훅.
 *
 * 핵심: 시각장애인 사용자가 탭 후 말을 시작하기까지 시간이 걸리는데,
 * Web Speech(특히 iOS)는 continuous=false에서 무음이 잠깐만 있어도 금방 끝나버린다.
 * → "듣기 세션" 개념을 도입: 결과가 나오기 전까지는 인식이 끝나도(no-speech 등)
 *   최대 청취 시간(MAX_LISTEN_MS) 안에서 자동으로 다시 시작해 계속 듣는다.
 *   사용자에게는 하나의 연속된 듣기 상태로 보인다.
 *
 * @returns {{
 *   transcript: string,
 *   interimText: string,
 *   isListening: boolean,    // 세션 단위 (자동 재시작 중에도 true 유지)
 *   start: () => void,
 *   stop: () => void,
 *   reset: () => void,
 *   error: string | null
 * }}
 */

const MAX_LISTEN_MS = 8000;   // 한 번 시작하면 최대 8초까지 발화를 기다린다

export function useSTT() {
  const recognitionRef = useRef(null);
  const shouldListenRef = useRef(false);  // 세션 활성 여부 (자동 재시작 판단용)
  const gotResultRef    = useRef(false);  // 이번 세션에서 최종 결과를 받았는지
  const startTimeRef    = useRef(0);      // 세션 시작 시각
  const maxTimerRef     = useRef(null);   // 하드 종료 타이머
  const supportedRef    = useRef(false);

  const [transcript, setTranscript]   = useState("");
  const [interimText, setInterimText] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [error, setError]             = useState(null);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError("이 브라우저는 음성 인식을 지원하지 않습니다.");
      return;
    }
    supportedRef.current = true;

    const recognition = new SpeechRecognition();
    recognition.lang = "ko-KR";
    recognition.continuous = false;     // iOS 호환 — 자동 재시작으로 연속성 확보
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    let mounted = true;

    const _finalizeSession = () => {
      shouldListenRef.current = false;
      if (maxTimerRef.current) { clearTimeout(maxTimerRef.current); maxTimerRef.current = null; }
      if (mounted) setIsListening(false);
    };

    const _beginRecognition = () => {
      try {
        recognition.start();
      } catch {
        // 직전 인스턴스가 아직 정리 중일 수 있음 — 짧게 후 재시도
        setTimeout(() => {
          if (shouldListenRef.current) {
            try { recognition.start(); } catch {}
          }
        }, 150);
      }
    };
    // onend에서 참조할 수 있도록 보관
    recognition._begin = _beginRecognition;

    recognition.onstart = () => {
      if (!mounted) return;
      setError(null);
    };

    recognition.onend = () => {
      if (!mounted) return;
      const elapsed = Date.now() - startTimeRef.current;
      // 세션이 살아있고, 아직 결과가 없고, 최대 시간 안이면 → 자동 재시작 (계속 듣기)
      if (shouldListenRef.current && !gotResultRef.current && elapsed < MAX_LISTEN_MS) {
        _beginRecognition();
        return;
      }
      _finalizeSession();
    };

    recognition.onerror = (e) => {
      if (!mounted) return;
      // no-speech / aborted 는 정상 흐름 — onend가 이어서 자동 재시작/종료 처리
      if (e.error !== "aborted" && e.error !== "no-speech") {
        setError(`음성 인식 오류: ${e.error}`);
      }
    };

    recognition.onresult = (e) => {
      if (!mounted) return;
      let interim = "";
      let final = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const result = e.results[i];
        if (result.isFinal) final += result[0].transcript;
        else interim += result[0].transcript;
      }
      if (final) {
        gotResultRef.current = true;        // 결과 확보 → 더 이상 재시작 안 함
        shouldListenRef.current = false;
        setTranscript((prev) => (prev + " " + final).trim());
        setInterimText("");
        try { recognition.stop(); } catch {}
      } else {
        setInterimText(interim);
      }
    };

    recognitionRef.current = recognition;

    return () => {
      mounted = false;
      shouldListenRef.current = false;
      if (maxTimerRef.current) clearTimeout(maxTimerRef.current);
      try { recognition.abort(); } catch {}
    };
  }, []);

  const start = useCallback(() => {
    if (!recognitionRef.current || shouldListenRef.current) return;
    setTranscript("");
    setInterimText("");
    setError(null);
    gotResultRef.current = false;
    shouldListenRef.current = true;
    startTimeRef.current = Date.now();
    setIsListening(true);

    // 하드 종료 타이머 — 최대 시간 도달 시 강제 종료
    if (maxTimerRef.current) clearTimeout(maxTimerRef.current);
    maxTimerRef.current = setTimeout(() => {
      shouldListenRef.current = false;
      try { recognitionRef.current?.stop(); } catch {}
    }, MAX_LISTEN_MS);

    recognitionRef.current._begin?.();
  }, []);

  const stop = useCallback(() => {
    if (!recognitionRef.current || !shouldListenRef.current) return;
    shouldListenRef.current = false;
    if (maxTimerRef.current) { clearTimeout(maxTimerRef.current); maxTimerRef.current = null; }
    try { recognitionRef.current.stop(); } catch {}
  }, []);

  const reset = useCallback(() => {
    setTranscript("");
    setInterimText("");
  }, []);

  return { transcript, interimText, isListening, start, stop, reset, error };
}
