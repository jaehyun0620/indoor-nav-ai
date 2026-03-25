"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * useSTT
 * Web Speech API를 사용한 한국어 음성 인식 훅.
 *
 * @returns {{
 *   transcript: string,      // 인식된 텍스트 (누적)
 *   interimText: string,     // 중간 결과 텍스트
 *   isListening: boolean,    // 현재 청취 중 여부
 *   start: () => void,       // 인식 시작
 *   stop: () => void,        // 인식 중지
 *   reset: () => void,       // transcript 초기화
 *   error: string | null     // 오류 메시지
 * }}
 */
export function useSTT() {
  const recognitionRef = useRef(null);
  const [transcript, setTranscript] = useState("");
  const [interimText, setInterimText] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError("이 브라우저는 음성 인식을 지원하지 않습니다.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "ko-KR";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setError(null);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.onerror = (e) => {
      setError(`음성 인식 오류: ${e.error}`);
      setIsListening(false);
    };

    recognition.onresult = (e) => {
      let interim = "";
      let final = "";

      for (let i = e.resultIndex; i < e.results.length; i++) {
        const result = e.results[i];
        if (result.isFinal) {
          final += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }

      if (final) {
        setTranscript((prev) => (prev + " " + final).trim());
        setInterimText("");
      } else {
        setInterimText(interim);
      }
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.abort();
    };
  }, []);

  const start = useCallback(() => {
    if (!recognitionRef.current || isListening) return;
    setTranscript("");
    setInterimText("");
    recognitionRef.current.start();
  }, [isListening]);

  const stop = useCallback(() => {
    if (!recognitionRef.current || !isListening) return;
    recognitionRef.current.stop();
  }, [isListening]);

  const reset = useCallback(() => {
    setTranscript("");
    setInterimText("");
  }, []);

  return { transcript, interimText, isListening, start, stop, reset, error };
}
