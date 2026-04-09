"""
calibrate.py
MiDaS scale_factor 캘리브레이션 도우미.

사용법:
  python calibrate.py <이미지파일> <실제거리_m>

예시:
  python calibrate.py photo_1m.jpg 1.0
  python calibrate.py photo_2m.jpg 2.0
  python calibrate.py photo_3m.jpg 3.0

측정 후 권장 scale_factor를 출력합니다.
"""

import sys
import json
import requests

API_URL = "http://localhost:8000/navigate"
CURRENT_SCALE = 5.0  # backend/.env 의 MIDAS_SCALE 값

def measure(image_path: str, real_distance_m: float):
    with open(image_path, "rb") as f:
        resp = requests.post(
            API_URL,
            files={"frame": ("photo.jpg", f, "image/jpeg")},
            data={"target": "사람"},
        )

    if resp.status_code != 200:
        print(f"[오류] 서버 응답 {resp.status_code}: {resp.text}")
        return None

    result = resp.json()
    detections = result.get("detections", [])

    if not detections:
        print(f"[{image_path}] ⚠️  탐지된 객체 없음 — 사람이나 물체가 프레임에 보여야 합니다")
        return None

    # 가장 가까운 탐지 객체 사용
    nearest = min(detections, key=lambda d: d.get("distance_m", 999))
    measured_m = nearest.get("distance_m", 0)
    cls = nearest.get("class", "unknown")
    conf = nearest.get("conf", 0)

    if measured_m <= 0:
        print(f"[{image_path}] ⚠️  거리 측정 실패 (distance_m={measured_m})")
        return None

    corrected_scale = CURRENT_SCALE * (real_distance_m / measured_m)

    print(f"\n📸  {image_path}")
    print(f"    탐지 객체  : {cls} (신뢰도 {conf:.2f})")
    print(f"    실제 거리  : {real_distance_m:.1f} m")
    print(f"    측정 출력  : {measured_m:.2f} m")
    print(f"    보정 계수  : {corrected_scale:.3f}")

    return corrected_scale


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    results = []
    i = 1
    while i + 1 <= len(sys.argv) - 1:
        image_path = sys.argv[i]
        real_dist = float(sys.argv[i + 1])
        scale = measure(image_path, real_dist)
        if scale is not None:
            results.append(scale)
        i += 2

    if not results:
        print("\n측정 결과 없음. 이미지와 실제 거리를 확인하세요.")
        return

    avg_scale = sum(results) / len(results)
    print(f"\n{'='*45}")
    print(f"  측정 횟수       : {len(results)}회")
    print(f"  개별 보정값     : {[f'{s:.3f}' for s in results]}")
    print(f"  권장 MIDAS_SCALE: {avg_scale:.2f}")
    print(f"{'='*45}")
    print(f"\n  backend/.env 에서 아래 값으로 변경하세요:")
    print(f"  MIDAS_SCALE={avg_scale:.2f}")


if __name__ == "__main__":
    main()
