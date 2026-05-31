# indoor-nav-ai
> 시각장애인을 위한 실내 AI 길 안내 시스템 — 캡스톤디자인 2026

스마트폰 카메라로 복도를 비추면, AI가 장애물을 감지하고 목적지 방향을 한국어 음성으로 안내합니다.

---

## 핵심 아이디어

GPT/Gemini 같은 VLM에 이미지만 넣으면 방향을 그냥 "추측"해서 틀리는 경우가 많습니다. 특히 목적지가 안 보이는 상황에서도 "왼쪽으로 가세요" 같은 잘못된 안내를 자신 있게 내놓는 문제가 있어요.

그래서 **YOLO가 먼저 탐지한 객체 정보를 VLM 프롬프트에 주입**하는 구조를 만들었습니다. VLM이 이미지만 보는 게 아니라 "계단이 정면 1.2m에 있습니다"라는 텍스트 정보도 함께 받아서 판단하게 됩니다. 장애물이 탐지되면 VLM 자체를 호출하지 않고 바로 경고를 냅니다.

---

## 구조

카메라 프레임이 들어오면 두 경로로 처리됩니다.

```
카메라 프레임 (WebSocket)
    │
    ├─ 빠른 채널 (매 프레임)
    │   YOLO → 거리 추정 → 장애물 거리 계산
    │   장애물 2m 이내: 즉각 경고, VLM 우회
    │
    └─ 느린 채널 (안내 시작 시 1회 + 사용자 요청 시)
        YOLO 결과 텍스트화 → Gemini 프롬프트 주입
        → JSON 응답 강제 → 일관성 필터 (deque 3회)
        → 방향 안내 TTS (시계 방위 포함)
```

장애물이 가까우면 VLM을 아예 건너뛰고 경고를 먼저 냅니다. VLM은 안내를 시작하는 순간(초기 장면 파악)과 사용자가 목적지를 물어볼 때만 호출해서 API 비용도 아끼고 응답 속도도 챙겼습니다.

**거리 추정** — 로컬에서는 Depth Anything V2로 단안 깊이를 추정하지만, 배포 환경(CPU)에서는 추론이 불안정하고 느려서 YOLO 바운딩 박스 높이 비율 기반 거리 근사(`FAST_MODE=yolo_only`)를 기본으로 씁니다. 정확한 미터 값보다 "가까움/중간/멀리" 구분이 목적이라 복도 환경에서 충분히 안정적으로 동작합니다.

**초기 장면 파악(Orientation)** — 안내를 시작하면 첫 프레임을 VLM이 한 번 분석해 현재 환경을 알려줍니다. 목적지가 보이면 바로 방향을, 안 보이면 "지금 복도에 계신 것 같아요. 앞으로 이동해 보세요"처럼 어느 쪽으로 움직일지 제안합니다.

**시계 방위 안내** — 시각장애인 보행의 표준 방식대로 "10시 방향에 화장실 표지판이 보여요"처럼 정면을 12시로 둔 시계 방위로 안내합니다.

---

## 접근성 설계 (시각장애인 관점)

화면을 보지 않고도 쓸 수 있도록 음성·촉각 중심으로 설계했습니다.

- **온보딩 음성 안내** — 첫 진입 시 화면을 탭하면 사용법을 음성으로 안내하고, 동시에 iOS 오디오 재생 잠금을 해제합니다.
- **탭 인터페이스** — 화면 어디든 한 번 탭하면 음성으로 목적지 입력(대기 중) 또는 방향 조회(안내 중), 두 번 탭하면 안내 시작/중지. 길게 누르면 사용법을 다시 안내합니다.
- **안심 멘트(heartbeat)** — 안내 중 일정 시간 조용하면 "계속 안내 중입니다"라고 알려 시스템이 멈춘 게 아님을 알립니다.
- **촉각 피드백** — 장애물 경고 시 진동으로도 알려 시끄러운 환경에 대비합니다.
- **상태 음성화** — 카메라·서버 오류, 음성 인식 실패도 화면뿐 아니라 음성으로 안내합니다.
- **자동 재연결** — 이동 중 네트워크가 끊겨도 자동으로 재연결하고 음성으로 알립니다.
- **강의실 번호 읽기(OCR)** — 강의실 안내 시 표지판의 호실 번호를 인식해 "111호가 보입니다"라고 덧붙입니다. (`ENABLE_OCR=true`)

---

## 실험 결과

25개 시나리오를 3가지 조건으로 비교했습니다.

| 지표 | Baseline | +구조화 | Proposed |
|------|:---:|:---:|:---:|
| 장애물 경고 누락률 | 100% | 100% | **0%** |
| 할루시네이션 발생률 | 12.5% | 6.2% | **0%** |
| 방향 정확도 | 56% | 64% | 65% |
| 평균 응답 지연 | 2.94s | 1.94s | 2.29s |

Baseline은 계단이 있는 사진에서도 "오른쪽으로 가세요" 같은 응답을 냈습니다. YOLO가 계단을 탐지하면 VLM을 우회하는 구조 덕분에 Proposed는 장애물 경고 누락이 없었습니다.

---

## YOLO 커스텀 파인튜닝

엘리베이터와 계단을 더 잘 잡기 위해 커스텀 데이터셋을 만들었습니다.

- 직접 촬영한 이미지 + Roboflow 어노테이션
- 8가지 증강 적용 (밝기, 대비, 블러, 노이즈, 회전, 좌우반전 등) → 57장 → 513장
- 파인튜닝 결과: **mAP@50 73.7%**

---

## 기술 스택

**백엔드**
- FastAPI + WebSocket — 실시간 프레임 수신
- YOLOv8 (커스텀 파인튜닝) — 엘리베이터, 계단 탐지
- 거리 추정 — bbox 높이 비율(배포) / Depth Anything V2 단안 깊이(로컬)
- Gemini 2.5 Flash Lite / GPT-4o — 장면 이해 및 방향 판단
- EasyOCR — 강의실 번호, 표지판 인식

**프론트엔드**
- Next.js 14 (App Router)
- Web Speech API — 음성 인식 (STT)
- Naver Clova TTS + Web Speech API 폴백 — 음성 안내

**배포**
- Railway (백엔드, Docker) · Vercel (프론트엔드)

---

## 실행 방법

**백엔드**
```bash
cd capstone
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# .env 설정 (backend/.env.example 참고)
# 배포/CPU 환경: bbox 기반 거리 추정 권장
FAST_MODE=yolo_only python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**프론트엔드**
```bash
cd capstone/frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

주요 환경변수:

| 변수 | 설명 | 기본값 |
|------|------|:---:|
| `FAST_MODE` | `yolo_only`(bbox 거리) / `full`(Depth Anything) | `yolo_only` |
| `VLM_PROVIDER` | `gemini` / `openai` | `openai` |
| `ENABLE_OCR` | 강의실 번호 OCR 사용 여부 | `false` |
| `NAVER_TTS_CLIENT_ID/SECRET` | Naver Clova TTS (미설정 시 Web Speech 폴백) | — |

### 배포

- **백엔드 → Railway** : `Dockerfile` 기반 빌드. YOLO 가중치를 빌드 시점에 미리 받아 런타임 다운로드 타임아웃을 피합니다. 영구 URL이 발급되어 ngrok이 필요 없습니다.
- **프론트엔드 → Vercel** : Root Directory를 `frontend`로 설정하고 `NEXT_PUBLIC_API_URL`에 Railway URL을 넣습니다.

> iOS Safari/Chrome은 사용자 제스처로 재생을 시작한 오디오 요소만 이후 자동 재생을 허용합니다. 시작 화면을 한 번 탭해 오디오를 활성화한 뒤 사용하세요. 무음 모드(측면 스위치)도 해제해야 소리가 납니다.

---

## 폴더 구조

```
capstone/
├── Dockerfile                    # Railway 배포 (YOLO 가중치 사전 다운로드)
├── railway.toml
├── backend/
│   ├── main.py                   # FastAPI 서버 (WebSocket, TTS 프록시, orientation)
│   ├── channels/
│   │   ├── fast_channel.py       # YOLO + 거리 추정 (매 프레임)
│   │   └── slow_channel.py       # VLM + 일관성 필터 + 초기 장면 파악
│   ├── modules/
│   │   ├── context_builder.py    # YOLO 결과 → 프롬프트 텍스트
│   │   ├── prompt_designer.py    # 구조화 프롬프트 + 시계 방위 + orientation
│   │   ├── consistency_filter.py # deque 기반 방향 확정
│   │   ├── priority_module.py    # 장애물 우선 / VLM 분기
│   │   ├── navigation_session.py # 세션 상태 관리
│   │   ├── ocr_pipeline.py       # 표지판/강의실 번호 OCR
│   │   └── scene_memory.py       # 이전 프레임 컨텍스트
│   ├── models/
│   │   └── yolo_midas.py         # YOLO + 거리 추정 래퍼
│   └── training/
│       ├── train.py              # YOLO 파인튜닝
│       ├── augment_dataset.py    # 데이터 증강
│       └── convert_seg_to_det.py # Roboflow 라벨 변환
│
└── frontend/
    └── app/
        ├── page.js               # 메인 UI (온보딩, 탭 인터페이스, 자동 재연결)
        ├── hooks/
        │   ├── useSTT.js         # 음성 인식
        │   └── useTTS.js         # 음성 출력 (Naver TTS + iOS 오디오 unlock)
        └── components/
            └── VoiceButton.js    # 음성 입력 버튼
```

---

## 참고 논문

- MARINE (ICML 2025) — Visual Grounding 기반 VLM 할루시네이션 통제
- M3ID (CVPR 2024) — 이미지-언어 상호정보 최대화
- Nav-YOLO (MDPI 2025) — 실내 환경 YOLOv8 경량화
