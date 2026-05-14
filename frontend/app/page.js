"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSTT } from "./hooks/useSTT";
import { useTTS } from "./hooks/useTTS";

const WS_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
  .replace(/^http/, "ws") + "/ws/navigate";

const CAPTURE_INTERVAL_MS = 200;

const PRESET_TARGETS = ["화장실", "강의실", "엘리베이터"];

const DIRECTION = {
  left: { label: "왼쪽", mark: "<" },
  right: { label: "오른쪽", mark: ">" },
  straight: { label: "직진", mark: "^" },
  unknown: { label: "확인 중", mark: "-" },
};

const MESSAGE = {
  warning: { label: "위험", color: "#ef4444", bg: "#2a0c0c" },
  caution: { label: "주의", color: "#f59e0b", bg: "#261a07" },
  guidance: { label: "안내", color: "#22c55e", bg: "#071f12" },
  searching: { label: "탐색", color: "#3b82f6", bg: "#071629" },
  monitoring: { label: "감시", color: "#94a3b8", bg: "#0b1220" },
  arrived: { label: "도착", color: "#a855f7", bg: "#160d24" },
  stopped: { label: "중지", color: "#94a3b8", bg: "#0b1220" },
  ready: { label: "준비", color: "#94a3b8", bg: "#0b1220" },
};

const styles = {
  page: {
    minHeight: "100vh",
    background: "#080b10",
    color: "#f8fafc",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
    display: "flex",
    justifyContent: "center",
  },
  shell: {
    width: "100%",
    maxWidth: 460,
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    padding: "18px 16px 20px",
    gap: 14,
    boxSizing: "border-box",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  title: {
    margin: 0,
    fontSize: 21,
    lineHeight: 1.2,
    fontWeight: 800,
    letterSpacing: 0,
  },
  subtitle: {
    margin: "4px 0 0",
    fontSize: 13,
    lineHeight: 1.35,
    color: "#94a3b8",
  },
  pill: {
    minWidth: 72,
    height: 34,
    padding: "0 11px",
    borderRadius: 999,
    border: "1px solid #243244",
    background: "#0d1420",
    color: "#cbd5e1",
    fontSize: 12,
    fontWeight: 700,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 7,
    whiteSpace: "nowrap",
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#475569",
    flex: "0 0 auto",
  },
  targetPanel: {
    display: "grid",
    gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
    gap: 8,
  },
  targetButton: {
    height: 48,
    borderRadius: 8,
    border: "1px solid #243244",
    background: "#101722",
    color: "#cbd5e1",
    fontSize: 14,
    fontWeight: 750,
    cursor: "pointer",
  },
  inputRow: {
    display: "grid",
    gridTemplateColumns: "1fr 50px",
    gap: 8,
  },
  input: {
    width: "100%",
    height: 46,
    padding: "0 13px",
    borderRadius: 8,
    border: "1px solid #243244",
    background: "#0d1420",
    color: "#f8fafc",
    fontSize: 15,
    outline: "none",
    boxSizing: "border-box",
  },
  iconButton: {
    width: 50,
    height: 46,
    borderRadius: 8,
    border: "1px solid #243244",
    background: "#101722",
    color: "#e2e8f0",
    fontSize: 19,
    fontWeight: 800,
    cursor: "pointer",
  },
  cameraBox: {
    width: "100%",
    aspectRatio: "4 / 3",
    borderRadius: 8,
    overflow: "hidden",
    background: "#111827",
    border: "1px solid #243244",
    position: "relative",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    display: "block",
    background: "#111827",
  },
  cameraEmpty: {
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#64748b",
    fontSize: 15,
    fontWeight: 650,
  },
  overlayTop: {
    position: "absolute",
    top: 10,
    left: 10,
    right: 10,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    pointerEvents: "none",
  },
  overlayBadge: {
    height: 30,
    padding: "0 10px",
    borderRadius: 999,
    background: "rgba(8, 11, 16, 0.76)",
    border: "1px solid rgba(226, 232, 240, 0.14)",
    display: "flex",
    alignItems: "center",
    gap: 7,
    fontSize: 12,
    fontWeight: 800,
    color: "#e2e8f0",
  },
  directionOverlay: {
    position: "absolute",
    left: "50%",
    bottom: 12,
    transform: "translateX(-50%)",
    minWidth: 150,
    height: 48,
    borderRadius: 999,
    background: "rgba(8, 11, 16, 0.84)",
    border: "1px solid rgba(34, 197, 94, 0.45)",
    color: "#bbf7d0",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    fontSize: 15,
    fontWeight: 850,
    pointerEvents: "none",
  },
  directionMark: {
    width: 28,
    height: 28,
    borderRadius: 999,
    background: "#12351f",
    color: "#86efac",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 18,
    lineHeight: 1,
  },
  status: {
    minHeight: 88,
    borderRadius: 8,
    border: "1px solid #243244",
    background: "#0d1420",
    padding: 14,
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    gap: 8,
  },
  statusHead: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 10,
  },
  statusLabel: {
    fontSize: 12,
    fontWeight: 850,
    letterSpacing: 0,
  },
  statusTarget: {
    fontSize: 12,
    color: "#94a3b8",
    whiteSpace: "nowrap",
  },
  statusText: {
    margin: 0,
    color: "#e2e8f0",
    fontSize: 16,
    lineHeight: 1.45,
    fontWeight: 720,
  },
  actions: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 10,
  },
  primaryAction: {
    height: 58,
    borderRadius: 8,
    border: "1px solid #2563eb",
    background: "#1d4ed8",
    color: "#fff",
    fontSize: 16,
    fontWeight: 850,
    cursor: "pointer",
  },
  secondaryAction: {
    height: 58,
    borderRadius: 8,
    border: "1px solid #334155",
    background: "#111827",
    color: "#e2e8f0",
    fontSize: 16,
    fontWeight: 850,
    cursor: "pointer",
  },
  startAction: {
    height: 58,
    borderRadius: 8,
    border: "1px solid #16a34a",
    background: "#15803d",
    color: "#fff",
    fontSize: 16,
    fontWeight: 850,
    cursor: "pointer",
  },
  error: {
    borderRadius: 8,
    border: "1px solid #7f1d1d",
    background: "#2a0c0c",
    color: "#fecaca",
    padding: "11px 12px",
    fontSize: 13,
    lineHeight: 1.35,
  },
};

export default function HomePage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const frameReadyRef = useRef(false);
  const rvcHandleRef = useRef(null);
  const lastSpokenRef = useRef({ text: "", type: "", at: 0 });
  const tapCountRef = useRef(0);
  const tapTimerRef = useRef(null);

  const [target, setTarget] = useState("");
  const [navState, setNavState] = useState("idle");
  const [decision, setDecision] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [cameraError, setCameraError] = useState(null);

  const { transcript, isListening, start: startSTT, stop: stopSTT, reset: resetSTT } = useSTT();
  const { speak, stop: stopTTS, clearPending, isSpeaking } = useTTS();

  const isRunning = navState === "navigating";
  const direction = DIRECTION[decision?.direction] || DIRECTION.unknown;
  const messageType = decision?.message_type || (isRunning ? "monitoring" : "ready");
  const message = MESSAGE[messageType] || MESSAGE.monitoring;

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

  const stopCamera = useCallback(() => {
    videoRef.current?.srcObject?.getTracks().forEach((track) => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startFrameCache = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const capture = () => {
      if (video.readyState >= 2) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        frameReadyRef.current = true;
      }
    };

    if (video.requestVideoFrameCallback) {
      const loop = () => {
        capture();
        rvcHandleRef.current = video.requestVideoFrameCallback(loop);
      };
      rvcHandleRef.current = video.requestVideoFrameCallback(loop);
    } else {
      const loop = () => {
        capture();
        rvcHandleRef.current = requestAnimationFrame(loop);
      };
      rvcHandleRef.current = requestAnimationFrame(loop);
    }
  }, []);

  const stopFrameCache = useCallback(() => {
    const video = videoRef.current;
    if (rvcHandleRef.current != null) {
      video?.requestVideoFrameCallback
        ? video.cancelVideoFrameCallback(rvcHandleRef.current)
        : cancelAnimationFrame(rvcHandleRef.current);
      rvcHandleRef.current = null;
    }
    frameReadyRef.current = false;
  }, []);

  const handleMessage = useCallback((data) => {
    const { message_type, tts_text, query_response } = data;
    setDecision(data);

    if (message_type === "arrived") {
      stopTTS();
      speak(tts_text, true);
      setNavState("arrived");
      clearInterval(intervalRef.current);
      stopFrameCache();
      stopCamera();
      if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.close();
      wsRef.current = null;
      setWsConnected(false);
      setTimeout(() => {
        setNavState("idle");
        setDecision(null);
        setTarget("");
      }, 6000);
      return;
    }

    if (message_type === "stopped") return;

    if (query_response) {
      setIsQuerying(false);
      speak(tts_text, true);
      lastSpokenRef.current = { text: tts_text, type: message_type, at: Date.now() };
      return;
    }

    const now = Date.now();
    const prev = lastSpokenRef.current;
    const elapsed = now - prev.at;

    if (message_type === "warning") {
      // 같은 경고 텍스트는 5초 쿨다운 — 매 프레임 interrupt로 음성이 잘리는 현상 방지
      if (tts_text !== prev.text || elapsed > 5000) {
        speak(tts_text, true);
        lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      }
      return;
    }

    if (message_type === "caution") {
      // warning 직후 5초 이내에는 caution 억제 — warning↔caution 진동으로 음성 끊김 방지
      if (prev.type === "warning" && elapsed < 5000) return;
      if (tts_text !== prev.text || elapsed > 4000) {
        speak(tts_text, false);
        lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      }
      return;
    }

    if (message_type === "monitoring") {
      speak(tts_text, false);
      lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      return;
    }

    if (message_type === "guidance" || message_type === "searching") {
      if (tts_text !== prev.text || elapsed > 8000) {
        speak(tts_text, false);
        lastSpokenRef.current = { text: tts_text, type: message_type, at: now };
      }
    }
  }, [speak, stopTTS, stopFrameCache, stopCamera]);

  const startNavigation = useCallback(async (navTarget) => {
    const dest = (navTarget || target).trim();
    if (!dest) return;
    await startCamera();
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ action: "start", target: dest }));
      setNavState("navigating");
      setWsConnected(true);
      setDecision(null);
      lastSpokenRef.current = { text: "", type: "", at: 0 };
      startFrameCache();
      intervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN && frameReadyRef.current) {
          canvasRef.current?.toBlob((blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
          }, "image/jpeg", 0.8);
        }
      }, CAPTURE_INTERVAL_MS);
    };

    ws.onmessage = (e) => {
      try {
        handleMessage(JSON.parse(e.data));
      } catch {}
    };

    ws.onerror = () => {
      setCameraError("서버 연결 오류");
      setNavState("idle");
      setWsConnected(false);
    };

    ws.onclose = () => {
      clearInterval(intervalRef.current);
      stopFrameCache();
      setWsConnected(false);
    };
  }, [target, startCamera, startFrameCache, stopFrameCache, handleMessage]);

  // STT 인식 완료 → 목적지 설정 + 즉시 안내 시작
  // (startNavigation 정의 이후에 위치해야 const TDZ 오류가 발생하지 않음)
  useEffect(() => {
    if (!transcript || navState !== "idle") return;
    const cleaned = transcript
      .replace(/찾아줘|가고\s?싶어|어디야|알려줘|데려다줘|가자|가줘|보여줘|어디에|어디로/g, "")
      .trim();
    if (cleaned.length >= 1) {
      setTarget(cleaned);
      speak(`${cleaned} 안내를 시작합니다`);
      resetSTT();
      setTimeout(() => startNavigation(cleaned), 1400);
    }
  }, [transcript, navState, speak, resetSTT, startNavigation]);

  const stopNavigation = useCallback(() => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stop" }));
      ws.close();
    }
    clearInterval(intervalRef.current);
    stopFrameCache();
    stopCamera();
    stopTTS();
    clearPending();
    setNavState("idle");
    setWsConnected(false);
    setDecision(null);
    setIsQuerying(false);
    speak("안내를 중지했습니다");
  }, [stopCamera, stopFrameCache, stopTTS, clearPending, speak]);

  const queryDirection = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !frameReadyRef.current || isQuerying) return;
    const b64 = canvasRef.current?.toDataURL("image/jpeg", 0.8);
    if (!b64) return;
    ws.send(JSON.stringify({ action: "query", frame: b64, target }));
    setIsQuerying(true);
    speak("분석 중입니다", false);
  }, [target, speak, isQuerying]);

  useEffect(() => () => {
    clearInterval(intervalRef.current);
    if (tapTimerRef.current) clearTimeout(tapTimerRef.current);
    wsRef.current?.close();
    stopCamera();
  }, [stopCamera]);

  const statusText = isRunning
    ? decision?.tts_text || "화면을 한 번 터치하면 방향을 안내해 드립니다."
    : target
      ? `${target} 안내를 시작할 수 있습니다.`
      : "목적지를 선택하세요.";

  // ── 화면 탭 접근성 (시각장애인용) ──────────────────────────────────────────
  // 버튼/입력 영역을 제외한 화면 아무 곳이나 탭
  //   한 번  : 대기 중 → 목적지 음성 입력 / 안내 중 → 방향 즉시 조회
  //   두 번  : 대기 중(목적지 설정됨) → 안내 시작 / 안내 중 → 안내 중지
  const handleScreenTap = useCallback(
    (e) => {
      const tag = e.target.tagName;
      if (tag === "BUTTON" || tag === "INPUT") return;

      tapCountRef.current += 1;
      if (tapTimerRef.current) clearTimeout(tapTimerRef.current);

      tapTimerRef.current = setTimeout(() => {
        const count = tapCountRef.current;
        tapCountRef.current = 0;

        if (count === 1) {
          if (navState === "idle") {
            speak("목적지를 말씀하세요");
            setTimeout(startSTT, 600);
          } else if (navState === "navigating") {
            queryDirection();
          }
        } else if (count >= 2) {
          if (navState === "idle" && target.trim()) {
            speak("안내를 시작합니다");
            setTimeout(startNavigation, 500);
          } else if (navState === "navigating") {
            stopNavigation();
          } else if (navState === "idle" && !target.trim()) {
            speak("목적지를 먼저 설정해 주세요. 화면을 한 번 탭하면 음성으로 입력할 수 있습니다");
          }
        }
      }, 300);
    },
    [navState, target, speak, startSTT, queryDirection, startNavigation, stopNavigation]
  );

  return (
    <main style={styles.page} onClick={handleScreenTap}>
      <div style={styles.shell}>
        <header style={styles.header}>
          <div>
            <h1 style={styles.title}>실내 보조 안내</h1>
            <p style={styles.subtitle}>목적지: {target || "미선택"}</p>
          </div>
          <div style={{
            ...styles.pill,
            borderColor: wsConnected ? "#14532d" : "#243244",
            color: wsConnected ? "#86efac" : "#94a3b8",
          }}>
            <span style={{ ...styles.dot, background: wsConnected ? "#22c55e" : "#475569" }} />
            {wsConnected ? "연결됨" : "대기"}
            {isSpeaking ? " · 음성" : ""}
          </div>
        </header>

        {!isRunning && navState !== "arrived" && (
          <>
            <div style={styles.targetPanel}>
              {PRESET_TARGETS.map((item) => {
                const selected = target === item;
                return (
                  <button
                    key={item}
                    type="button"
                    onClick={() => {
                      setTarget(item);
                      speak(`${item} 안내를 시작합니다`);
                      setTimeout(() => startNavigation(item), 1000);
                    }}
                    style={{
                      ...styles.targetButton,
                      borderColor: selected ? "#2563eb" : "#243244",
                      background: selected ? "#10254a" : "#101722",
                      color: selected ? "#bfdbfe" : "#cbd5e1",
                    }}
                  >
                    {item}
                  </button>
                );
              })}
            </div>

            <div style={styles.inputRow}>
              <input
                type="text"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && target.trim()) startNavigation();
                }}
                placeholder="직접 입력 후 Enter"
                style={styles.input}
              />
              <button
                type="button"
                onPointerDown={startSTT}
                onPointerUp={stopSTT}
                style={{
                  ...styles.iconButton,
                  borderColor: isListening ? "#2563eb" : "#243244",
                  background: isListening ? "#10254a" : "#101722",
                }}
                aria-label="음성 입력"
              >
                {isListening ? "..." : "음성"}
              </button>
            </div>
          </>
        )}

        <section style={styles.cameraBox}>
          <video ref={videoRef} style={styles.video} playsInline muted />
          <canvas ref={canvasRef} style={{ display: "none" }} />

          {!isRunning && navState !== "arrived" && (
            <div style={styles.cameraEmpty}>카메라 대기</div>
          )}

          {isRunning && (
            <>
              <div style={styles.overlayTop}>
                <div style={styles.overlayBadge}>
                  <span style={{ ...styles.dot, background: "#ef4444" }} />
                  감시 중
                </div>
                <div style={styles.overlayBadge}>{target}</div>
              </div>

              {decision?.message_type === "warning" && (
                <div style={{
                  position: "absolute",
                  inset: 0,
                  border: "4px solid #ef4444",
                  borderRadius: 8,
                  pointerEvents: "none",
                }} />
              )}

              {isQuerying && (
                <div style={styles.directionOverlay}>
                  <span style={styles.directionMark}>...</span>
                  분석 중
                </div>
              )}

              {!isQuerying && decision?.query_response && decision?.direction !== "unknown" && (
                <div style={styles.directionOverlay}>
                  <span style={styles.directionMark}>{direction.mark}</span>
                  {direction.label}
                </div>
              )}
            </>
          )}

          {navState === "arrived" && (
            <div style={{
              position: "absolute",
              inset: 0,
              background: "rgba(8, 11, 16, 0.86)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#e9d5ff",
              fontSize: 20,
              fontWeight: 850,
            }}>
              도착했습니다
            </div>
          )}
        </section>

        <section style={{
          ...styles.status,
          background: message.bg,
          borderColor: message.color,
        }}>
          <div style={styles.statusHead}>
            <span style={{ ...styles.statusLabel, color: message.color }}>{message.label}</span>
            <span style={styles.statusTarget}>
              {isRunning ? "세션 유지 중" : isListening ? "듣는 중" : "준비"}
            </span>
          </div>
          <p style={styles.statusText}>{statusText}</p>
        </section>

        {cameraError && <div style={styles.error}>{cameraError}</div>}

        {isRunning ? (
          <div style={styles.actions}>
            <button
              type="button"
              onClick={queryDirection}
              disabled={isQuerying}
              style={{
                ...styles.primaryAction,
                opacity: isQuerying ? 0.62 : 1,
                cursor: isQuerying ? "not-allowed" : "pointer",
              }}
            >
              {isQuerying ? "확인 중" : "현재 위치 확인"}
            </button>
            <button type="button" onClick={stopNavigation} style={styles.secondaryAction}>
              중지
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={startNavigation}
            disabled={navState === "arrived" || !target.trim()}
            style={{
              ...styles.startAction,
              opacity: target.trim() && navState !== "arrived" ? 1 : 0.45,
              cursor: target.trim() && navState !== "arrived" ? "pointer" : "not-allowed",
            }}
          >
            보조 안내 시작
          </button>
        )}
      </div>
    </main>
  );
}
