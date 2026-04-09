# 실내 길 안내 AI 시스템 — 코드 전체 보고서

> 작성일: 2026-04-01
> 대상: 캡스톤디자인 팀원 코드 학습 및 리뷰용

---

## 목차

1. [시스템 전체 흐름](#1-시스템-전체-흐름)
2. [백엔드 파일별 상세 설명](#2-백엔드-파일별-상세-설명)
   - 2-1. `backend/main.py`
   - 2-2. `backend/channels/fast_channel.py`
   - 2-3. `backend/channels/slow_channel.py`
   - 2-4. `backend/models/yolo_midas.py`
   - 2-5. `backend/modules/context_builder.py`
   - 2-6. `backend/modules/prompt_designer.py`
   - 2-7. `backend/modules/consistency_filter.py`
   - 2-8. `backend/modules/priority_module.py`
   - 2-9. `backend/modules/navigation_session.py`
   - 2-10. `backend/modules/scene_memory.py`
3. [프론트엔드 파일별 상세 설명](#3-프론트엔드-파일별-상세-설명)
   - 3-1. `frontend/app/page.js`
   - 3-2. `frontend/app/hooks/useSTT.js`
   - 3-3. `frontend/app/hooks/useTTS.js`
   - 3-4. `frontend/app/components/VoiceButton.js`
   - 3-5. `frontend/app/layout.js` / `globals.css` / `tailwind.config.js`
4. [환경 설정 파일](#4-환경-설정-파일)
5. [WebSocket 메시지 프로토콜 전체 정리](#5-websocket-메시지-프로토콜-전체-정리)
6. [주요 설계 결정과 이유](#6-주요-설계-결정과-이유)
7. [발생한 버그와 수정 이력](#7-발생한-버그와-수정-이력)
8. [현재 남은 한계점](#8-현재-남은-한계점)

---

## 1. 시스템 전체 흐름

```
[스마트폰 카메라]
      │  1초마다 JPEG 프레임 (Base64)
      ▼
[WebSocket /ws/navigate]
      │
      ├─ action: "frame" ─────────────────────────────────────────────────────┐
      │                                                                        │
      │   ⚡ 빠른 채널 (매 프레임 실행)                                        │
      │   YOLOv8 탐지 → MiDaS 거리 추정 → 장애물 있으면 경고 응답            │
      │                                                                        │
      └─ action: "query" ─────────────────────────────────────────────────────┤
                                                                               │
          🧠 느린 채널 (사용자 요청 시에만 실행)                               │
          YOLOv8 탐지 → YOLO 결과를 프롬프트에 주입 → Gemini API 호출        │
          → VLM JSON 파싱 → 방향 확정 → TTS 응답                             │
                                                                               │
[우선순위 판단 모듈]                                                           │
  • 1m 미만 장애물 → "즉시 멈추세요" (경로 A)                                │
  • 안전 + 방향 확정 → "목적지는 왼쪽" (경로 B)                              │
      │                                                                        │
[클라이언트 응답]◄─────────────────────────────────────────────────────────────┘
      │
[TTS 발화] → 사용자 음성 안내
```

---

## 2. 백엔드 파일별 상세 설명

---

### 2-1. `backend/main.py` — FastAPI 서버 진입점

#### 역할
서버의 모든 엔드포인트를 정의하고, 빠른/느린 채널과 우선순위 모듈을 통합한다.

#### 핵심 구조

```python
# 서버 시작 시 모델 1회 로드 (싱글톤)
@asynccontextmanager
async def lifespan(app: FastAPI):
    fast_channel = FastChannel(...)   # YOLOv8 + MiDaS
    slow_channel = SlowChannel(...)   # VLM 클라이언트 + 일관성 필터
    yield
```

> **왜 lifespan을 쓰나?**
> YOLOv8과 MiDaS 모델은 로드하는 데 수 초가 걸린다.
> 요청마다 로드하면 너무 느리기 때문에 서버 시작 시 1번만 로드하고 싱글톤으로 공유한다.

#### 엔드포인트 목록

| 엔드포인트 | 방식 | 용도 |
|-----------|------|------|
| `/ws/navigate` | WebSocket | 실제 서비스 (지속 스트리밍) |
| `/navigate` | POST (multipart) | 단발성 테스트용 |
| `/reset` | POST | REST 세션 초기화 |
| `/health` | GET | 헬스체크 |

#### WebSocket 핸들러 동작

```python
@app.websocket("/ws/navigate")
async def ws_navigate(websocket: WebSocket):
    session = NavigationSession()   # 세션 상태머신
    scene_memory = SceneMemory()    # 최근 장면 기억

    while True:
        data = await websocket.receive_json()
        action = data.get("action")

        if action == "start":   # 세션 시작
            ...
        if action == "stop":    # 세션 중지
            ...
        if action == "frame":   # YOLO만 실행 (장애물 감지)
            ...
        if action == "query":   # VLM 즉시 호출 (방향 분석)
            ...
```

> **설계 포인트**: `frame` 액션과 `query` 액션을 분리한 이유
> - `frame`: 1초마다 자동 전송, YOLO만 실행 → 빠르고 비용 없음
> - `query`: 사용자가 버튼 눌렀을 때만 전송, VLM 호출 → 느리지만 정확한 방향 제공
> 이렇게 분리하면 API 비용을 최소화하면서 안전성(장애물 감지)은 유지된다.

#### `_resize_for_vlm()` 헬퍼

```python
def _resize_for_vlm(image_bytes: bytes) -> bytes:
    size = int(os.getenv("VLM_IMAGE_SIZE", "320"))  # 기본 320px
    # 이미지를 width 320으로 축소 → JPEG quality 70으로 재압축
```

> VLM API에 보내는 이미지를 작게 줄여서 API 비용과 전송 시간을 줄인다.
> 현재 `.env`에서 `VLM_IMAGE_SIZE=320`으로 설정되어 있다.

---

### 2-2. `backend/channels/fast_channel.py` — 빠른 채널

#### 역할
카메라 프레임을 받아 YOLOv8으로 객체를 탐지하고, MiDaS로 거리를 추정한다.

#### 입출력

```
입력: BGR numpy 배열 또는 JPEG bytes 또는 Base64 문자열

출력: {
    "detections": [
        {"class": "person", "bbox": [x1,y1,x2,y2], "distance_m": 2.1, "conf": 0.91},
        ...
    ],
    "fast_result": {
        "class": "사람",       ← 가장 가까운 장애물 (한국어)
        "distance_m": 2.1,
        "direction": "정면",
        "has_obstacle": True,  ← 2m 이내 장애물 있으면 True
        "conf": 0.91,
        "bbox": [...]
    },
    "yolo_context": "사람: 정면 2.1m (신뢰도 0.91)\n의자: 왼쪽 1.2m (신뢰도 0.77)"
}
```

#### 메서드 3가지

```python
process_frame(frame_bgr)    # numpy 배열 직접 입력
process_bytes(image_bytes)  # JPEG/PNG bytes 입력 (WebSocket 용)
process_base64(b64_string)  # Base64 문자열 입력
```

> **설계 포인트**: 3가지 입력 방식을 제공하는 이유
> 카메라에서 직접 읽으면 numpy 배열, WebSocket으로 받으면 bytes, REST API로 받으면 Base64.
> 어디서 입력받든 내부적으로 `process_frame()`으로 수렴한다.

---

### 2-3. `backend/channels/slow_channel.py` — 느린 채널

#### 역할
VLM API(Gemini 또는 GPT-4o)를 호출해서 방향을 분석한다.
두 개의 핵심 클래스를 포함한다: `VLMClient`, `SlowChannel`

#### VLMClient — API 호출 담당

```python
class VLMClient:
    def __init__(self, provider="gemini"):
        # provider가 "gemini"면 Gemini API
        # provider가 "openai"면 GPT-4o API
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    def call(self, prompt, image_bytes) -> str:
        # 프롬프트 + 이미지를 API에 전송
        # 원본 텍스트 응답 반환 (JSON 문자열)
```

**Gemini API 호출 구조:**
```python
payload = {
    "contents": [{
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
        ]
    }],
    "generationConfig": {"maxOutputTokens": 150}  # 응답 토큰 제한
}
# timeout=20초 (Gemini가 느릴 수 있어서 넉넉하게)
```

> **왜 `maxOutputTokens: 150`?**
> JSON 응답은 보통 100토큰 이내다. 너무 크면 VLM이 불필요한 설명을 추가할 수 있고 비용도 증가한다.

**안전 필터 오류 처리:**
```python
candidates = body.get("candidates", [])
if not candidates:
    raise ValueError(f"Gemini candidates 없음: {body.get('promptFeedback', '')}")
```
> Gemini의 안전 필터가 응답을 차단하면 `candidates` 배열이 비어있다.
> 이전에 이 처리가 없어서 `KeyError`가 발생했던 버그가 있었다.

#### SlowChannel — 채널 실행 담당

```python
class SlowChannel:
    def process(self, image_bytes, yolo_context, target) -> dict:
        """일관성 필터를 통해 3회 중 2회 일치 방향을 반환한다."""
        prompt = build_prompt(yolo_context, target)
        raw_text = self.vlm.call(prompt, image_bytes)
        parsed = parse_vlm_response(raw_text)
        self.filter.add(parsed["goal_direction"], parsed["confidence"])
        confirmed_dir, tts_text = self.filter.get_guidance()
        return {"confirmed_direction": confirmed_dir, "tts_text": tts_text, "raw": parsed}

    def reset(self):
        """목적지 변경 시 필터 초기화"""
        self.filter.reset()
```

> **`process` vs 미구현 `process_instant`**
> `process`는 ConsistencyFilter를 통해 3회 중 2회 일치를 기다린다 (최소 ~10초).
> 사용자 요청 시 즉시 답변을 위한 `process_instant`(필터 없이 1회 즉시 반환)는
> 아직 구현되지 않았고 `main.py`의 `query` 액션에서 필요하다.

---

### 2-4. `backend/models/yolo_midas.py` — YOLO + MiDaS 래퍼

#### 역할
YOLOv8 객체 탐지와 MiDaS 단안 깊이 추정을 하나의 클래스로 묶는다.

#### MiDaS 로드 (전역 싱글톤)

```python
_MIDAS_MODEL = None
_MIDAS_TRANSFORM = None
_MIDAS_DEVICE = None

def _load_midas(model_type="MiDaS_small"):
    global _MIDAS_MODEL, _MIDAS_TRANSFORM, _MIDAS_DEVICE
    if _MIDAS_MODEL is not None:
        return  # 이미 로드됨, 스킵

    _MIDAS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", model_type)
    _MIDAS_MODEL.eval()  # 추론 모드 (학습 안 함)
```

> **왜 전역 변수로?**
> 모델 로드는 시간이 걸리므로 처음 한 번만 로드하고 재사용한다.
> Python에서는 이런 패턴을 "Lazy Initialization" 또는 "Singleton"이라고 부른다.

#### 거리 추정 원리

```python
def inverse_depth_to_meters(inv_depth, depth_map, scale_factor=5.0):
    d_max = float(depth_map.max())
    normalized = inv_depth / d_max    # 0.0~1.0 (클수록 = 가까울수록)
    distance_m = scale_factor / normalized  # 가까울수록 distance 작음
    return distance_m
```

> **핵심**: MiDaS는 절대 거리를 모른다. 상대적인 "역깊이 맵"만 출력한다.
> `scale_factor`(기본 5.0m)를 곱해서 실제 미터로 변환하는데,
> 이 값은 실제 환경에서 측정해서 보정해야 한다 (`MIDAS_SCALE` 환경변수).

#### bbox 중심 깊이 샘플링

```python
def bbox_center_depth(depth_map, bbox):
    # bbox 중심 좌표 계산
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # 중심 주변 20×20 픽셀 패치의 중앙값 사용
    patch = depth_map[cy-10:cy+10, cx-10:cx+10]
    return float(np.median(patch))
```

> **왜 중앙값(median)?**
> 평균이면 노이즈에 취약하다. 중앙값은 극단값의 영향을 받지 않는다.
> 20×20 패치를 쓰는 이유는 단일 픽셀보다 안정적이기 때문이다.

#### YOLOMiDaSWrapper.run() 전체 흐름

```python
def run(self, frame_bgr):
    # 1. MiDaS 깊이 맵 계산
    depth_map = estimate_depth_map(frame_bgr)

    # 2. YOLOv8 탐지
    results = self.model(frame_bgr, conf=self.conf_threshold, verbose=False)

    # 3. 각 탐지 객체마다 거리 추정
    for box in result.boxes:
        inv_d = bbox_center_depth(depth_map, bbox)
        distance_m = inverse_depth_to_meters(inv_d, depth_map, self.scale_factor)
        detections.append({"class": cls_name, "bbox": bbox, "distance_m": distance_m, "conf": conf})

    # 4. 가장 가까운 장애물 요약 (priority_module용)
    summary = build_obstacle_summary(detections)
    fast_result = {
        "class": summary["closest_class"],
        "distance_m": summary["closest_distance"],
        ...
    }
    return detections, fast_result
```

---

### 2-5. `backend/modules/context_builder.py` — 컨텍스트 주입 모듈

#### 역할
YOLO 탐지 결과(영문 클래스 + 픽셀 좌표 + 거리)를 VLM이 이해할 수 있는 **한국어 텍스트**로 변환한다.
이것이 논문에서 말하는 "컨텍스트 주입(Context Injection)"의 핵심이다.

#### 클래스 매핑

```python
CLASS_KO = {
    "person": "사람",
    "chair": "의자",
    "door": "문",
    "stairs": "계단",
    "elevator": "엘리베이터",
    ...  # 총 40여 개 COCO 클래스 → 한국어
}
```

> **왜 한국어로 변환?**
> VLM 프롬프트 자체가 한국어이므로, 탐지 정보도 한국어로 맞춰야 VLM이 일관되게 처리한다.

#### 방향 결정 로직

```python
def _bbox_to_direction(bbox, frame_width=640):
    cx = (x1 + x2) / 2
    ratio = cx / frame_width
    if ratio < 0.35:    return "왼쪽"
    elif ratio > 0.65:  return "오른쪽"
    else:               return "정면"
```

> 화면을 3등분해서 객체의 x좌표 중심이 어느 구간에 있는지로 방향을 결정한다.
> 좌 35% / 중앙 30% / 우 35%

#### build_context() 출력 예시

```
입력: [
    {"class": "person", "bbox": [200,100,300,400], "distance_m": 2.1, "conf": 0.91},
    {"class": "chair",  "bbox": [50,100,150,300],  "distance_m": 1.2, "conf": 0.77}
]

출력:
"의자: 왼쪽 1.2m (신뢰도 0.77)
사람: 정면 2.1m (신뢰도 0.91)"
```

> 가까운 것부터 정렬하여 VLM이 중요한 정보를 먼저 읽도록 한다.

---

### 2-6. `backend/modules/prompt_designer.py` — 구조화 프롬프트 설계

#### 역할
VLM에 전달할 프롬프트를 생성하고, VLM이 반환한 JSON 응답을 파싱한다.

#### 3가지 실험 조건 (비교 실험용)

| 조건 | 코드 | 설명 |
|------|------|------|
| Baseline | `"baseline"` | 자유 응답 ("~어디 있나요?") |
| +구조화 | `"structured"` | JSON 강제, YOLO 주입 없음 |
| Proposed | `"proposed"` | JSON 강제 + YOLO 주입 (기본값) |

```python
# .env에서 선택
EXPERIMENT_CONDITION=proposed
```

#### Proposed 프롬프트 구조 (핵심)

```
당신은 시각장애인을 돕는 실내 안내 시스템입니다.

[탐지된 객체 정보 - 신뢰도 높음]      ← YOLO 결과 주입
사람: 정면 2.1m (신뢰도 0.91)
의자: 왼쪽 1.2m (신뢰도 0.77)

위 탐지 정보와 이미지를 함께 분석하여 반드시 아래 JSON 형식으로만 응답하세요.

목표물: 화장실 (남자화장실 / 여자화장실 표시)

{
  "goal_visible": true 또는 false,
  "goal_direction": "left" 또는 "right" 또는 "straight" 또는 "unknown",
  "goal_distance": "약 Xm" 또는 "unknown",
  "confidence": 0.0에서 1.0 사이 숫자,
  "reasoning": "판단 근거 한 문장"
}

규칙:
1. 목표물이 직접 보이지 않더라도, 복도 구조·표지판·화살표 등을 근거로 방향을 추론
2. confidence가 0.4 미만일 때만 goal_direction을 "unknown"으로 설정
3. 탐지된 객체 정보와 모순되는 응답 금지
4. 장애물이 1~2m 이내에 있을 경우 confidence 기준을 0.6으로 높여서 판단
5. reasoning은 반드시 한국어 한 문장으로 작성
```

> **핵심 설계 원칙**: VLM에게 "맥락(YOLO 결과)"을 주입하면 더 정확해진다.
> YOLO가 "사람이 정면 2m에 있다"고 알려주면, VLM이 "화장실은 그 사람 뒤에 있다"고 추론할 수 있다.

#### parse_vlm_response() — JSON 파싱

```python
def parse_vlm_response(response_text):
    # 1. 마크다운 코드블록 포함해서 JSON 추출
    json_match = re.search(r"\{[\s\S]*\}", response_text)

    # 2. JSON 파싱
    parsed = json.loads(json_match.group())

    # 3. confidence < 0.4 이면 direction 강제 unknown (핵심 안전장치)
    if confidence < 0.4:
        direction = "unknown"

    return {"goal_visible": ..., "goal_direction": ..., ...}
```

> **왜 regex로 JSON을 추출?**
> VLM이 가끔 JSON을 마크다운 코드블록(` ```json ... ``` `)으로 감싸서 반환하기 때문이다.
> `re.search(r"\{[\s\S]*\}")` 패턴이 중괄호 안을 추출해서 이를 처리한다.

---

### 2-7. `backend/modules/consistency_filter.py` — 일관성 필터

#### 역할
VLM의 단일 응답을 믿지 않고, 최근 3회 응답 중 2회 이상 동일한 방향을 "확정"으로 처리한다.
할루시네이션(잘못된 응답)이 1회 들어와도 무시된다.

#### 알고리즘

```python
class ConsistencyFilter:
    def __init__(self):
        self.buffer = deque(maxlen=3)   # 최근 3개 VLM 응답 저장
        self.agree_threshold = 2        # 2/3 이상 일치해야 확정
        self.conf_min = 0.4             # 이 미만이면 unknown으로 처리
        self.ttl = 30.0                 # 30초 지난 응답은 무효
        self.unknown_streak = 0         # 연속 unknown 횟수

    def add(self, direction, confidence):
        if confidence < self.conf_min:
            direction = "unknown"       # 신뢰도 낮으면 강제 unknown
        self.buffer.append({
            "direction": direction,
            "confidence": confidence,
            "timestamp": time.time()
        })

    def get_guidance(self):
        now = time.time()
        valid = [r for r in self.buffer if now - r["timestamp"] < self.ttl]

        if len(valid) < self.agree_threshold:
            return "unknown", "아직 분석 중입니다"

        counter = Counter([r["direction"] for r in valid])
        top_dir, top_count = counter.most_common(1)[0]

        if top_count >= self.agree_threshold:
            if top_dir != "unknown":
                return top_dir, "목적지는 {방향} 방향입니다"
        return self._handle_unknown()
```

#### TTL 설계 이유

```
VLM 호출 간격: 5초
버퍼 크기: 3개
필요한 최소 버퍼 채우기 시간: 5초 × 3 = 15초

TTL: 30초 (15초 + 여유 15초)
```

> **TTL이 VLM 호출 간격보다 짧으면** 버퍼가 항상 비어서 방향이 확정되지 않는다.
> 이전에 TTL=3초로 설정했을 때 "아직 분석 중" 무한 반복 버그가 있었다.

#### unknown_streak 단계별 유도 메시지

```
1회 연속 unknown → "잠시 기다려주세요"
2회 연속 unknown → "카메라를 천천히 움직여주세요"
3회 연속 unknown → "주변을 천천히 둘러보세요" (초기화)
```

---

### 2-8. `backend/modules/priority_module.py` — 우선순위 판단 모듈

#### 역할
빠른 채널(장애물 정보)과 느린 채널(방향 정보)을 합쳐서 최종 TTS 메시지를 결정한다.

#### 결정 로직

```python
def decide(self, fast_result, slow_result):
    distance = fast_result.get("distance_m", 999.0)
    direction = slow_result.get("confirmed_direction", "unknown")

    # ── 경로 A: 즉각 경고 (VLM 결과 무시) ──
    if distance < 1.0:          # 1m 미만
        return {
            "message_type": "warning",
            "tts_text": "즉시 멈추세요. {장애물}이 있습니다.",
            "priority": 1,
            "suppress_guidance": True   # 방향 안내 억제
        }

    if 1.0 <= distance < 2.0:   # 1~2m
        return {
            "message_type": "caution",
            "tts_text": "주의: {장애물} 1.5m 앞, {방향}",
            "priority": 1,
            "suppress_guidance": False
        }

    # ── 경로 B: 방향 안내 ──
    if direction != "unknown":
        return {"message_type": "guidance", "tts_text": "{방향} 방향입니다", "priority": 2}

    # ── 방향 미확정 ──
    return {"message_type": "unknown", "tts_text": "...", "priority": 3}
```

> **`suppress_guidance: True`의 의미**
> 1m 미만 장애물이 있으면 방향 안내를 억제한다.
> "왼쪽으로 가세요"와 "즉시 멈추세요"가 동시에 나오면 혼란스럽기 때문이다.

---

### 2-9. `backend/modules/navigation_session.py` — 세션 관리

#### 역할
목적지 도착까지의 세션 상태와 도착 판정을 관리한다.

#### 상태머신

```
idle ──(start)──► navigating ──(check_arrival)──► arrived
  ▲                    │
  └──────(stop)────────┘
```

#### 도착 판정 로직

```python
def check_arrival(self, goal_visible, goal_distance_str, confidence):
    distance_m = self._parse_distance(goal_distance_str)  # "약 1.2m" → 1.2

    condition_met = (
        goal_visible          # VLM이 목적지를 화면에서 봄
        and confidence >= 0.75  # 신뢰도 75% 이상
        and distance_m <= 1.5   # 1.5m 이내
    )

    if condition_met:
        self._arrival_count += 1
    else:
        self._arrival_count = max(0, self._arrival_count - 1)  # 1 감소 (오인식 허용)

    if self._arrival_count >= 2:    # 2회 연속 충족
        self.state = "arrived"
        return True
    return False
```

> **왜 2회 연속?**
> VLM이 한 번 "목적지가 1m 앞에 보인다"고 말해도 할루시네이션일 수 있다.
> 2회 연속 같은 조건을 충족해야 진짜 도착으로 본다.
> 반면, 1회 실패 시 카운트를 0이 아니라 1만 감소시켜서 일시적 오인식을 허용한다.

#### 진행 피드백

```python
def get_progress_feedback(self, current_direction):
    recent = list(self._direction_history)[-3:]
    if all(d == current_direction for d in recent):
        return "왼쪽 방향으로 잘 가고 있습니다"  # 3회 연속 같은 방향
    return ""
```

---

### 2-10. `backend/modules/scene_memory.py` — 장면 메모리

#### 역할
최근 10프레임의 탐지 결과와 VLM 응답을 버퍼에 저장해서,
다음 VLM 호출 시 "이전에 목적지가 왼쪽으로 확인됐으니 다시 확인해줘" 같은 힌트를 프롬프트에 추가한다.

#### 핵심 메서드

```python
def get_context_for_prompt(self):
    """이전 방향 이력을 프롬프트 보강 문자열로 반환"""
    last = self.get_last_direction()
    if not last:
        return ""
    # 예: "이전 분석에서 목적지는 왼쪽으로 확인됨. 현재도 왼쪽인지 재확인해줘."
    return f"이전 분석에서 목적지는 {label}으로 확인됨. 현재도 {label}인지 재확인해줘."
```

> VLM이 이전 분석 결과를 참고하면 방향 일관성이 높아진다.
> TTL=5초: 5초 이상 지난 기억은 무시한다.

---

## 3. 프론트엔드 파일별 상세 설명

---

### 3-1. `frontend/app/page.js` — 메인 UI 페이지

#### 역할
카메라 스트리밍, WebSocket 통신, TTS 발화, UI 렌더링을 모두 담당하는 핵심 파일이다.

#### 주요 상태(State)

```javascript
const [target, setTarget] = useState("");          // 목적지 ("화장실")
const [isRunning, setIsRunning] = useState(false); // 안내 중 여부
const [lastDecision, setLastDecision] = useState(null); // 최신 서버 응답
const [navState, setNavState] = useState("idle");  // idle/navigating/arrived
const [wsConnected, setWsConnected] = useState(false); // WebSocket 연결 상태
```

#### 주요 Ref(참조, 리렌더링 없이 유지되는 값)

```javascript
const videoRef = useRef(null);      // <video> DOM 엘리먼트
const canvasRef = useRef(null);     // 프레임 캡처용 숨긴 <canvas>
const wsRef = useRef(null);         // WebSocket 인스턴스
const intervalRef = useRef(null);   // 1초 타이머 ID

// TTS 쿨다운 관리
const lastSpokenTextRef = useRef(""); // 마지막 발화 텍스트
const lastSpokenAtRef = useRef(0);    // 마지막 발화 시각 (ms)
```

> **왜 Ref?** useState는 바뀔 때 리렌더링을 유발한다.
> WebSocket 인스턴스, 타이머 ID 같은 것은 바뀌어도 화면이 바뀔 필요가 없으므로 ref 사용.

#### 자유 목적지 입력 (STT → 목적지 추출)

```javascript
const NAV_VERBS = ["찾아줘", "가고싶어", "어디야", "알려줘", ...];

function extractTarget(text) {
    let t = text.trim();
    for (const v of NAV_VERBS) t = t.replace(v, "").trim();
    return t.length >= 1 ? t : null;
}

// 사용 예:
// "화장실 찾아줘" → "화장실"
// "301호 어디야"  → "301호"
// "엘리베이터"   → "엘리베이터"
```

#### 프레임 캡처 및 전송

```javascript
const captureAndSend = useCallback(() => {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, ...);

    const b64 = canvas.toDataURL("image/jpeg", 0.8);  // JPEG 품질 80%
    ws.send(JSON.stringify({ action: "frame", frame: b64, target }));
}, [target]);

// 1초마다 실행
intervalRef.current = setInterval(captureAndSend, 1000);
```

> `canvas.toDataURL("image/jpeg", 0.8)` → `"data:image/jpeg;base64,/9j/..."` 형태
> 백엔드에서는 `,` 뒤의 Base64 부분만 디코딩해서 사용한다.

#### TTS 쿨다운 (speakIfNeeded)

```javascript
const speakIfNeeded = useCallback((tts_text, message_type) => {
    const elapsed = Date.now() - lastSpokenAtRef.current;

    // 경고/주의/도착: 무조건 즉시 발화 (안전 최우선)
    if (["warning", "caution", "arrived"].includes(message_type)) {
        speak(tts_text);
        return;
    }

    // 방향 안내: 텍스트가 바뀌거나 8초 지나면 발화
    if (message_type === "guidance") {
        if (tts_text !== lastSpokenTextRef.current || elapsed > 8000) {
            speak(tts_text);
        }
        return;
    }

    // unknown: 발화 안 함 (시작 시 "분석 후 안내해드리겠습니다" 1회로 대체)
}, [speak]);
```

> **왜 guidance는 8초 쿨다운?**
> "목적지는 왼쪽 방향입니다"를 1초마다 계속 읽으면 시끄럽다.
> 방향이 바뀌거나 8초가 지나야 다시 읽는다.

#### 안내 시작 흐름

```javascript
const startNavigation = useCallback(async () => {
    await startCamera();          // 1. 카메라 켜기

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        ws.send({ action: "start", target }); // 2. 서버에 시작 알림
        speak("분석 후 안내해드리겠습니다"); // 3. 시작 안내 1회
        intervalRef.current = setInterval(captureAndSend, 1000); // 4. 1초마다 프레임 전송
    };

    ws.onmessage = (e) => handleWsMessage(JSON.parse(e.data)); // 5. 응답 처리
}, [...]);
```

#### 방향 표시

```javascript
// 서버 응답의 debug.confirmed_direction을 사용 (이전 버그: goal_direction 사용 → 항상 null)
const currentDirection = lastDecision?.debug?.confirmed_direction ?? null;
```

#### 신뢰도 바

```javascript
// 서버 응답의 debug.vlm_confidence를 사용 (이전 버그: confidence 사용 → 항상 undefined)
{isRunning && lastDecision?.debug?.vlm_confidence != null && (
    <ConfidenceBar confidence={lastDecision.debug.vlm_confidence} />
)}
```

---

### 3-2. `frontend/app/hooks/useSTT.js` — 음성 인식 훅

#### 역할
Web Speech API를 래핑해서 한국어 음성 인식 기능을 제공한다.

#### 핵심 설정

```javascript
recognition.lang = "ko-KR";         // 한국어
recognition.continuous = false;      // 한 번만 듣고 끝 (true면 계속 듣는 모드)
recognition.interimResults = true;   // 중간 결과도 제공 (듣는 중 미리보기)
recognition.maxAlternatives = 1;     // 대안 후보 1개만
```

#### 반환값

```javascript
return {
    transcript,    // 최종 인식 텍스트 (완전히 인식된 결과 누적)
    interimText,   // 중간 결과 (아직 확정 안 된 텍스트)
    isListening,   // 현재 듣는 중 여부
    start,         // 인식 시작
    stop,          // 인식 중지
    reset,         // transcript 초기화
    error          // 오류 메시지
}
```

> **`continuous: false`인 이유**
> 목적지 하나를 말하면 끝나는 용도이므로 연속 인식이 필요 없다.
> 연속 모드(true)면 배터리 소모가 크고 오인식이 누적될 수 있다.

---

### 3-3. `frontend/app/hooks/useTTS.js` — 음성 출력 훅

#### 역할
한국어 TTS를 제공한다. Web Speech API를 우선 사용하고, 없으면 Kakao TTS를 폴백으로 사용한다.

#### 2단계 폴백 구조

```javascript
const speak = useCallback((text) => {
    if (_isDuplicate(text)) return;  // 2초 이내 같은 텍스트 중복 방지

    const success = _speakNative(text);  // 1차: 브라우저 내장 TTS
    if (!success) {
        _speakKakao(text);               // 2차: Kakao TTS API
    }
}, [...]);
```

#### _speakNative() — 브라우저 내장 TTS

```javascript
const utter = new SpeechSynthesisUtterance(text);
utter.lang = "ko-KR";
utter.rate = 1.1;    // 속도 1.1배 (약간 빠르게)
utter.pitch = 1.0;
utter.volume = 1.0;

const voices = synth.getVoices();
const koVoice = voices.find((v) => v.lang.startsWith("ko"));  // 한국어 음성 우선
if (koVoice) utter.voice = koVoice;

synth.speak(utter);
```

#### 중복 방지 로직

```javascript
const _isDuplicate = useCallback((text) => {
    if (text === lastTextRef.current) return true;  // 같은 텍스트 → 중복
    lastTextRef.current = text;
    setTimeout(() => {
        if (lastTextRef.current === text) lastTextRef.current = "";
    }, 2000);  // 2초 후 초기화
    return false;
}, []);
```

> useTTS 자체 중복 방지(2초) + speakIfNeeded의 쿨다운(8초/15초)이 이중으로 작동한다.

---

### 3-4. `frontend/app/components/VoiceButton.js` — 음성 버튼

#### 역할
마이크 버튼 UI. 누르는 동안 STT, 떼면 인식 종료.

```javascript
export default function VoiceButton({ isListening, onStart, onStop, disabled }) {
    return (
        <button
            onPointerDown={onStart}   // 누르기 시작 → STT 시작
            onPointerUp={onStop}      // 떼기 → STT 종료
            onPointerLeave={onStop}   // 버튼 밖으로 나가도 종료
        >
            {/* 듣는 중이면 펄스 애니메이션 */}
            {isListening && <div className="animate-ping-slow ..." />}
            {/* 마이크 아이콘 */}
        </button>
    );
}
```

> **`onPointerDown/Up`을 쓰는 이유**
> `onClick`은 마우스/터치 구분이 없다.
> `onPointerDown/Up`은 터치 스크린에서도 일관되게 동작한다.

---

### 3-5. Tailwind 설정 파일들

#### `frontend/tailwind.config.js` — 커스텀 애니메이션

```javascript
animation: {
    "ping-slow": "ping 1.5s cubic-bezier(0,0,0.2,1) infinite",   // REC 점, 연결 표시
    "pulse-fast": "pulse 0.8s ... infinite",                       // 음성 출력 표시
    "fade-in": "fadeIn 0.3s ease-out",                            // 카드 등장
    "slide-up": "slideUp 0.3s ease-out",                          // 상태 카드 등장
    "bounce-subtle": "bounceSubtle 1s ease-in-out infinite",       // 방향 화살표
},
```

#### `frontend/app/globals.css` — 글로벌 스타일

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* 반투명 배경 카드 */
.glass-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(75, 85, 99, 0.3);
}
```

---

## 4. 환경 설정 파일

### `backend/.env`

```env
# VLM 설정
VLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-2.5-flash-lite
VLM_IMAGE_SIZE=320           # VLM에 보내는 이미지 최대 너비 (픽셀)

OPENAI_API_KEY=sk-...        # GPT-4o 사용 시
OPENAI_MODEL=gpt-4o

# 실험 조건
EXPERIMENT_CONDITION=proposed  # baseline / structured / proposed

# YOLO / MiDaS
YOLO_MODEL=yolov8n.pt
YOLO_CONF=0.4               # YOLO 신뢰도 임계값 (이 미만 탐지 무시)
MIDAS_MODEL=MiDaS_small
MIDAS_SCALE=5.0             # 거리 보정값 (현장에서 실측 필요!)

# 채널 설정
SLOW_CHANNEL_INTERVAL=5.0   # VLM 자동 호출 간격 (초) — REST 엔드포인트용
```

> **`MIDAS_SCALE=5.0` 중요!**
> 이 값은 현재 임시값이다. 실제 복도에서 사람을 세워두고 1m, 2m, 3m 지점에서 측정해서
> 실제 거리와 MiDaS 추정값의 비율로 보정해야 한다.

### `backend/requirements.txt`

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0   # WebSocket 지원 포함
python-multipart>=0.0.9     # 파일 업로드 파싱
httpx>=0.27.0               # VLM API HTTP 클라이언트
python-dotenv>=1.0.0
ultralytics>=8.1.0          # YOLOv8
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.9.0
numpy>=1.26.0
Pillow>=10.2.0
easyocr>=1.7.1              # OCR (아직 미구현)
gTTS>=2.5.0                 # 구글 TTS (선택)
```

---

## 5. WebSocket 메시지 프로토콜 전체 정리

### 클라이언트 → 서버

```json
// 안내 시작
{ "action": "start", "target": "화장실" }

// 매 프레임 전송 (YOLO 전용, 1초마다)
{ "action": "frame", "frame": "data:image/jpeg;base64,/9j/...", "target": "화장실" }

// 방향 즉시 분석 요청 (사용자가 버튼 눌렀을 때)
{ "action": "query", "frame": "data:image/jpeg;base64,/9j/...", "target": "화장실" }

// 안내 중지
{ "action": "stop" }
```

### 서버 → 클라이언트

```json
// 시작 확인
{ "message_type": "started", "tts_text": "화장실 안내를 시작합니다", "arrived": false }

// 장애물 경고
{
    "message_type": "warning",
    "tts_text": "즉시 멈추세요. 사람이 있습니다.",
    "priority": 1,
    "suppress_guidance": true,
    "arrived": false,
    "yolo_context": "사람: 정면 0.8m (신뢰도 0.93)",
    "debug": { "vlm_called": false, "obstacle_dist": 0.8 }
}

// 방향 안내 (query 응답)
{
    "message_type": "guidance",
    "tts_text": "목적지는 왼쪽 방향입니다",
    "priority": 2,
    "suppress_guidance": false,
    "arrived": false,
    "progress": "왼쪽 방향으로 잘 가고 있습니다",
    "yolo_context": "사람: 정면 3.0m (신뢰도 0.88)",
    "query_response": true,      ← 사용자 요청 응답임을 표시
    "debug": {
        "vlm_direction": "left",
        "vlm_confidence": 0.88,
        "vlm_reasoning": "왼쪽에 화장실 표지판이 보입니다",
        "vlm_goal_distance": "약 5m",
        "vlm_called": true,
        "confirmed_direction": "left",
        "filter_buffer_size": 1,
        "unknown_streak": 0,
        "obstacle_dist": 3.0
    }
}

// 도착
{
    "message_type": "arrived",
    "tts_text": "화장실에 도착했습니다. 45초 만에 안내를 완료했습니다.",
    "priority": 1,
    "arrived": true
}
```

---

## 6. 주요 설계 결정과 이유

### 결정 1: VLM에 YOLO 결과 주입

| 방식 | 장점 | 단점 |
|------|------|------|
| 사진만 VLM에 보냄 | 단순함 | VLM 할루시네이션 가능성 높음 |
| **YOLO 결과 + 사진** | VLM이 더 정확하게 판단 | 복잡함 |

> YOLO가 먼저 "사람이 정면 2m에 있음"을 감지하고 그 정보를 프롬프트에 주입하면,
> VLM이 이미지만 볼 때보다 더 정확하게 판단한다.
> 이것이 이 프로젝트의 핵심 기여(Technical Contribution)다.

### 결정 2: ConsistencyFilter (3회 중 2회 다수결)

> 단일 VLM 응답을 바로 사용하면 할루시네이션 한 번에 잘못된 방향으로 안내된다.
> 3회 중 2회 일치 조건으로 이를 방지한다.

### 결정 3: 비동기 WebSocket 대신 동기 httpx 사용

> VLM API 호출(`httpx.Client`)이 동기(sync)다. FastAPI는 async인데 동기 코드를 호출하면 이벤트 루프를 블로킹한다.
> 즉, VLM 호출 중(최대 20초) 다른 WebSocket 메시지를 처리 못 한다.
> 단일 사용자 캡스톤 프로젝트에서는 무방하지만, 다중 사용자라면 `asyncio.to_thread()`로 감싸야 한다.

### 결정 4: YOLO 거리 추정에 MiDaS 사용

> RGB 카메라 1대로 거리를 측정하는 것은 원래 불가능하다.
> MiDaS는 딥러닝으로 단안(단일 카메라) 깊이를 추정한다.
> 절대값이 아닌 상대값이라 `scale_factor` 보정이 필요하다.

---

## 7. 발생한 버그와 수정 이력

| # | 버그 | 원인 | 수정 |
|---|------|------|------|
| 1 | 방향 화살표가 항상 "분석 중" | `goal_direction` 필드가 서버 응답에 없음 | `debug.confirmed_direction` 사용 |
| 2 | 신뢰도 바가 안 보임 | `confidence` 필드가 서버 응답에 없음 | `debug.vlm_confidence` 사용 |
| 3 | 거리 정보가 안 보임 | `goal_distance` 필드 미전송 | `debug.vlm_goal_distance` 추가 |
| 4 | VLM 첫 프레임 이후 호출 안 됨 | `context_changed` 조건이 같은 YOLO 결과면 VLM 스킵 | 조건 제거 |
| 5 | "아직 분석 중" 무한 반복 | TTL=3초 < VLM 호출 간격 5초 → 버퍼 항상 비어있음 | TTL=30초로 변경 |
| 6 | Gemini candidates 오류 | 안전 필터 차단 시 candidates 없이 응답 | 명시적 오류 처리 추가 |
| 7 | VLM 항상 conf=0.0 | confidence 임계값 0.6이 너무 엄격 | 0.4로 완화 |
| 8 | VLM 항상 unknown | `goal_visible=false → unknown 강제 규칙`이 방향 추론 차단 | 규칙 제거 |
| 9 | OCR 정확도 0% | CLAHE+이진화가 깨끗한 이미지를 파괴 | 원본 우선, fallback 전략 |
| 10 | TTS 매초 반복 발화 | 모든 서버 응답마다 무조건 speak() 호출 | 타입별 쿨다운 적용 |
| 11 | Tailwind CSS 미작동 | globals.css, postcss.config.js 누락 | 파일 신규 생성 |
| 12 | CLASS_KO bicycle 중복 키 | 딕셔너리에 동일 키 2개 | 중복 제거 |
| 13 | "아직 분석 중" 반복 TTS | unknown 타입도 15초마다 발화 | unknown 발화 완전 제거 |

---

## 8. 현재 남은 한계점

### 기능 미완성

| 항목 | 상태 | 비고 |
|------|------|------|
| `process_instant()` | ❌ 미구현 | query 액션에서 필터 없이 즉시 응답이 필요 |
| `action: "query"` | ❌ 미구현 | main.py에 query 분기 추가 필요 |
| VoiceButton → query 전송 | ❌ 미구현 | 안내 중 버튼 누르면 방향 쿼리 전송 필요 |
| OCR 파이프라인 통합 | ⚠️ 모듈만 존재 | 메인 흐름에 연결 안 됨 |
| 배포 (Vercel + Railway) | ❌ 미완 | 로컬에서만 동작 |

### 성능 한계

| 항목 | 현재 | 목표 |
|------|------|------|
| VLM 응답 속도 | 3~10초 | 5초 이내 |
| MiDaS 거리 정확도 | 보정값 임시 | 현장 실측 필요 |
| 방향 확정까지 시간 | 최소 10초 (필터 3회) | query 방식으로 1회 즉시 |
| YOLO 장애물 거리 | 실내에서 부정확 | scale_factor 현장 보정 |

### 다음에 해야 할 작업 (우선순위 순)

1. **`slow_channel.py`**: `process_instant()` 메서드 추가 (필터 없이 VLM 1회 즉시 반환)
2. **`main.py`**: `action: "query"` 분기 추가 (YOLO + VLM 즉시 호출 → `query_response: True`)
3. **`page.js`**: VoiceButton 안내 중 누르면 query 액션 전송하도록 변경
4. **MIDAS_SCALE 보정**: 실제 복도에서 거리 실측 후 `.env` 값 조정
5. **배포**: Vercel(프론트) + Railway(백엔드) 연결
