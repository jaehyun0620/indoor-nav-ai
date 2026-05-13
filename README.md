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
    │   YOLO → Depth Anything V2 → 장애물 거리 계산
    │   장애물 2m 이내: 즉각 경고, VLM 우회
    │
    └─ 느린 채널 (사용자 요청 시)
        YOLO 결과 텍스트화 → Gemini 프롬프트 주입
        → JSON 응답 강제 → 일관성 필터 (deque 3회)
        → 방향 안내 TTS
```

장애물이 가까우면 VLM을 아예 건너뛰고 경고를 먼저 냅니다. VLM은 사용자가 목적지를 물어볼 때만 호출해서 API 비용도 아끼고 응답 속도도 챙겼습니다.

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
- Depth Anything V2 — 단안 깊이 추정
- Gemini 2.5 Flash Lite / GPT-4o — 장면 이해 및 방향 판단
- EasyOCR — 강의실 번호, 표지판 인식

**프론트엔드**
- Next.js 14 (App Router)
- Web Speech API — 음성 인식 (STT)
- Web Speech API + Naver Clova TTS — 음성 안내

---

## 실행 방법

**백엔드**
```bash
cd capstone
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# .env 설정 (backend/.env.example 참고)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**프론트엔드**
```bash
cd capstone/frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

스마트폰에서 접근하려면 백엔드를 ngrok으로 터널링하고 환경변수에 ngrok URL을 넣으면 됩니다. 프론트는 Vercel에 올려두면 편합니다.

---

## 폴더 구조

```
capstone/
├── backend/
│   ├── main.py                   # FastAPI 서버
│   ├── channels/
│   │   ├── fast_channel.py       # YOLO + Depth Anything (매 프레임)
│   │   └── slow_channel.py       # VLM + 일관성 필터 (요청 시)
│   ├── modules/
│   │   ├── context_builder.py    # YOLO 결과 → 프롬프트 텍스트
│   │   ├── prompt_designer.py    # 구조화 프롬프트
│   │   ├── consistency_filter.py # deque 기반 방향 확정
│   │   ├── priority_module.py    # 장애물 우선 / VLM 분기
│   │   ├── ocr_pipeline.py       # 표지판 OCR
│   │   └── scene_memory.py       # 이전 프레임 컨텍스트
│   └── training/
│       ├── train.py              # YOLO 파인튜닝
│       ├── augment_dataset.py    # 데이터 증강
│       └── convert_seg_to_det.py # Roboflow 라벨 변환
│
└── frontend/
    └── app/
        ├── page.js               # 메인 카메라 UI
        └── hooks/
            ├── useSTT.js         # 음성 인식
            └── useTTS.js         # 음성 출력
```

---

## 참고 논문

- MARINE (ICML 2025) — Visual Grounding 기반 VLM 할루시네이션 통제
- M3ID (CVPR 2024) — 이미지-언어 상호정보 최대화
- Nav-YOLO (MDPI 2025) — 실내 환경 YOLOv8 경량화
