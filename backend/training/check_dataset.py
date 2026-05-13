"""
check_dataset.py — 학습 전 데이터셋 상태 점검
============================================================
실행:
    cd /Users/jaehyun/ai_capstone/capstone
    python backend/training/check_dataset.py
"""

from pathlib import Path
from collections import Counter
import yaml

BASE   = Path(__file__).parent / "dataset"
YAML   = Path(__file__).parent / "dataset.yaml"

with open(YAML) as f:
    cfg = yaml.safe_load(f)

CLASS_NAMES = cfg["names"]  # {0: 'person', 1: 'chair', ...}

print("=" * 60)
print("데이터셋 점검")
print("=" * 60)

total_images = 0
total_labels = 0
class_count  = Counter()
missing      = []

for split in ("train", "val"):
    img_dir = BASE / "images" / split
    lbl_dir = BASE / "labels" / split

    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.JPG"))
    lbls = list(lbl_dir.glob("*.txt"))

    print(f"\n[{split}]")
    print(f"  이미지: {len(imgs)}장")
    print(f"  라벨:   {len(lbls)}개")

    # 라벨 없는 이미지 찾기
    for img in imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing.append(str(img.relative_to(BASE)))

    # 클래스별 bbox 수
    for lbl in lbls:
        for line in lbl.read_text().strip().splitlines():
            if line:
                cls_idx = int(line.split()[0])
                class_count[cls_idx] += 1

    total_images += len(imgs)
    total_labels += len(lbls)

print(f"\n[전체]")
print(f"  총 이미지: {total_images}장")
print(f"  총 라벨:   {total_labels}개")

print(f"\n[클래스별 bbox 수]")
for idx, name in CLASS_NAMES.items():
    cnt = class_count.get(idx, 0)
    bar = "█" * min(cnt // 5, 30)
    status = "⚠️  부족" if cnt < 50 else ("✅" if cnt >= 100 else "🟡 보통")
    print(f"  {idx} {name:15s} {cnt:4d}개  {bar}  {status}")

if missing:
    print(f"\n⚠️  라벨 없는 이미지 {len(missing)}개:")
    for m in missing[:5]:
        print(f"   {m}")
    if len(missing) > 5:
        print(f"   ... 외 {len(missing)-5}개")
else:
    print("\n✅ 모든 이미지에 라벨 있음")

print("\n[권장 최소 수량]")
print("  클래스당 100개 bbox 이상 → 안정적 학습")
print("  클래스당 50개 이상    → 기본 학습 가능")
print("  50개 미만             → 공개 데이터셋 추가 필요")
