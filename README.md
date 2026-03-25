# 🦯 indoor-nav-ai
> 시각장애인을 위한 실내 AI 길 안내 시스템 — 캡스톤디자인 2026

---

##  프로젝트 개요

스마트폰 카메라로 건물 복도를 비추면, AI가 실시간으로 장애물을 감지하고 목적지 방향을 **한국어 음성**으로 안내하는 시스템입니다.

단순히 GPT에 사진을 넣는 것이 아니라, **YOLOv8이 먼저 탐지한 정보를 프롬프트에 주입**해서 VLM의 오류를 구조적으로 통제하는 것이 핵심 설계입니다.

---

##  확정 범위

| 항목 | 내용 |
|------|------|
| 장소 | 학과 건물 1층 |
| 목표물 | 강의실 / 화장실 / 엘리베이터 |
| 방향 안내 | 직진 / 좌회전 / 우회전 |
| 플랫폼 | 웹 브라우저 (스마트폰) |

---

##  시스템 구조

카메라 프레임이 들어오면 두 채널이 병렬로 동작합니다.

```
카메라 프레임
    ├── 빠른 채널 (30fps) ──────────────────────────────────────┐
    │     YOLOv8 (장애물 탐지) + MiDaS (거리 추정)              │
    │     출력: { class, bbox, distance_m, conf }               │
    │                                                           ▼
    └── 느린 채널 (2~3초 주기)                        우선순위 판단 모듈
          YOLO 결과 주입 → VLM API → 일관성 필터  ──▶  최종 TTS 메시지
          출력: { confirmed_direction, tts_text }
```

### 빠른 채널 (안전 담당)
- **YOLOv8**: 사전학습 가중치로 장애물 실시간 탐지
- **MiDaS**: 단안 깊이 추정으로 장애물까지의 거리(m) 계산

### 느린 채널 (목표 담당)
- **컨텍스트 주입**: YOLO 탐지 결과를 텍스트로 변환해 VLM 프롬프트에 삽입
- **VLM API**: GPT-4o 또는 Gemini 1.5 Flash로 목적지 방향 판단
- **일관성 필터**: 최근 3회 응답 중 2회 이상 일치한 방향만 확정 (할루시네이션 억제)

### 우선순위 판단
```
장애물 1m 미만   →  "즉시 멈추세요. {장애물}이 있습니다."
장애물 1~2m      →  "주의: {장애물} {거리}m 앞"
안전 + 방향 확정 →  "목적지는 {방향} 방향입니다"
```

---

## 🔬 비교 실험 설계

같은 25개 시나리오를 3가지 조건으로 평가해 설계 기여를 수치로 증명합니다.

| 조건 | 방식 |
|------|------|
| Baseline | 사진 → VLM 자유응답 → TTS |
| +구조화 | 사진 → VLM JSON 강제 → TTS |
| **Proposed** | 사진 + YOLO·MiDaS 주입 + JSON 강제 + 일관성 필터 → TTS |

**목표 지표**
- 목표 방향 정확도: 70% 이상
- 장애물 경고 누락률: 0% (안전 요구사항)
- 할루시네이션 감소: 50% 이상
- 응답 지연: 3초 이하

---

## 🛠️ 기술 스택

### 백엔드
| 역할 | 기술 |
|------|------|
| 웹 서버 | FastAPI + uvicorn |
| 객체 탐지 | YOLOv8 (ultralytics) |
| 깊이 추정 | MiDaS (torch.hub) |
| VLM API | GPT-4o / Gemini 1.5 Flash |
| OCR | EasyOCR |
| TTS | gTTS / Kakao TTS |

### 프론트엔드
| 역할 | 기술 |
|------|------|
| UI 프레임워크 | Next.js (App Router) |
| 스타일 | Tailwind CSS |
| 음성 인식 | Web Speech API (브라우저 내장) |
| 음성 출력 | Web Speech API + Kakao TTS 폴백 |

---

## 📁 폴더 구조

```
capstone/
├── backend/
│   ├── main.py                   # FastAPI 서버 진입점
│   ├── channels/
│   │   ├── fast_channel.py       # YOLOv8 + MiDaS 빠른 채널
│   │   └── slow_channel.py       # VLM + 일관성 필터 느린 채널
│   ├── modules/
│   │   ├── context_builder.py    # YOLO 결과 → 프롬프트 텍스트 변환
│   │   ├── prompt_designer.py    # 구조화 프롬프트 설계
│   │   ├── consistency_filter.py # 일관성 필터 (deque + TTL)
│   │   ├── priority_module.py    # 우선순위 판단 (경로 A/B 분기)
│   │   ├── ocr_pipeline.py       # OCR 전처리 파이프라인
│   │   └── scene_memory.py       # Scene Memory 버퍼
│   ├── models/
│   │   └── yolo_midas.py         # YOLOv8 + MiDaS 래퍼
│   └── requirements.txt
│
└── frontend/
    ├── app/
    │   ├── page.js               # 메인 카메라 UI
    │   ├── hooks/
    │   │   ├── useSTT.js         # 음성 인식 훅
    │   │   └── useTTS.js         # 음성 출력 훅
    │   └── components/
    │       └── VoiceButton.js    # 음성 입력 버튼
    └── package.json
```


---

## 👥 팀 구성 및 역할

| 팀원 | 담당 |
|------|------|
| 팀원 A | YOLOv8·MiDaS 파이프라인, 빠른 채널, 컨텍스트 주입, 우선순위 모듈, 실험 |
| 팀원 B | VLM·프롬프트 설계, 일관성 필터, FastAPI 서버, Next.js UI·STT/TTS, OCR, 배포 |

---

## 📄 참고 논문

- **MARINE** (ICML 2025) — Visual Grounding 기반 할루시네이션 통제
- **M3ID** (CVPR 2024) — 이미지-언어 상호정보 최대화로 할루시네이션 감소
- **Nav-YOLO** (MDPI 2025) — YOLOv8 경량화 비교 실험 방법론
