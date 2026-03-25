"use client";

import { useCallback, useRef, useState } from "react";

/**
 * useTTS
 * 브라우저 내장 Web Speech API (SpeechSynthesis)로 한국어 TTS를 제공한다.
 * 브라우저가 한국어 음성을 지원하지 않는 경우 Kakao TTS API를 폴백으로 사용한다.
 *
 * @returns {{
 *   speak: (text: string) => void,  // 텍스트 읽기
 *   stop: () => void,               // 읽기 중지
 *   isSpeaking: boolean,            // 현재 읽는 중 여부
 *   error: string | null
 * }}
 */
export function useTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState(null);
  const audioRef = useRef(null);
  const lastTextRef = useRef("");

  /** 중복 발화 방지: 같은 텍스트를 2초 이내에 다시 읽지 않는다 */
  const _isDuplicate = useCallback((text) => {
    if (text === lastTextRef.current) return true;
    lastTextRef.current = text;
    setTimeout(() => {
      if (lastTextRef.current === text) lastTextRef.current = "";
    }, 2000);
    return false;
  }, []);

  /** Web Speech API TTS */
  const _speakNative = useCallback((text) => {
    const synth = window.speechSynthesis;
    if (!synth) return false;

    synth.cancel();

    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "ko-KR";
    utter.rate = 1.1;
    utter.pitch = 1.0;
    utter.volume = 1.0;

    // 한국어 음성 우선 선택
    const voices = synth.getVoices();
    const koVoice = voices.find((v) => v.lang.startsWith("ko"));
    if (koVoice) utter.voice = koVoice;

    utter.onstart = () => setIsSpeaking(true);
    utter.onend = () => setIsSpeaking(false);
    utter.onerror = () => setIsSpeaking(false);

    synth.speak(utter);
    return true;
  }, []);

  /** Kakao TTS API 폴백 */
  const _speakKakao = useCallback(async (text) => {
    const apiKey = process.env.NEXT_PUBLIC_KAKAO_API_KEY;
    if (!apiKey) {
      setError("Kakao TTS API 키가 없습니다.");
      return;
    }

    try {
      const res = await fetch(
        `https://kakaoi-newtone-openapi.kakao.com/v1/synthesize`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/xml",
            Authorization: `KakaoAK ${apiKey}`,
          },
          body: `<speak><voice name="MIN_YOUNG">${text}</voice></speak>`,
        }
      );
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      if (audioRef.current) {
        audioRef.current.pause();
        URL.revokeObjectURL(audioRef.current.src);
      }

      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onplay = () => setIsSpeaking(true);
      audio.onended = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(url);
      };
      audio.onerror = () => setIsSpeaking(false);
      audio.play();
    } catch (e) {
      setError(`TTS 오류: ${e.message}`);
      setIsSpeaking(false);
    }
  }, []);

  const speak = useCallback(
    (text) => {
      if (!text || _isDuplicate(text)) return;
      setError(null);

      const success = _speakNative(text);
      if (!success) {
        _speakKakao(text);
      }
    },
    [_isDuplicate, _speakNative, _speakKakao]
  );

  const stop = useCallback(() => {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    if (audioRef.current) audioRef.current.pause();
    setIsSpeaking(false);
  }, []);

  return { speak, stop, isSpeaking, error };
}
