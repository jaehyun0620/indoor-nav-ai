"""
augment_dataset.py — 기존 이미지/라벨로 데이터 증강
=======================================================
각 이미지에 7가지 변환을 적용해서 데이터를 약 7배로 늘린다.

적용 변환:
  1. 좌우 반전 (bbox cx 반전)
  2. 밝기 증가 (+40%)
  3. 밝기 감소 (−30%, 어두운 복도 시뮬레이션)
  4. 대비 강화
  5. 가우시안 블러 (카메라 흔들림)
  6. 가우시안 노이즈
  7. 소각도 회전 +10°
  8. 소각도 회전 −10°

⚠️ 상하 반전은 계단 방향이 바뀌므로 제외

실행:
    cd /Users/jaehyun/ai_capstone/capstone
    python backend/training/augment_dataset.py
"""

import cv2
import numpy as np
from pathlib import Path
import shutil

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "dataset"
SPLITS = ["train", "val"]   # test는 증강 제외

# ── bbox 변환 유틸 ──────────────────────────────────────────────────────────────

def flip_lr_bbox(cx, cy, w, h):
    """좌우 반전: cx만 뒤집힘"""
    return 1.0 - cx, cy, w, h


def rotate_bbox(cx, cy, w, h, angle_deg, img_w, img_h):
    """
    이미지 중심 기준 소각도 회전 후 bbox 재계산.
    4개 꼭짓점을 회전시켜 새 외접 bbox 반환.
    """
    # 정규화 → 픽셀 좌표
    px = cx * img_w
    py = cy * img_h
    pw = w  * img_w
    ph = h  * img_h

    # 4 꼭짓점 (픽셀)
    corners = np.array([
        [px - pw/2, py - ph/2],
        [px + pw/2, py - ph/2],
        [px + pw/2, py + ph/2],
        [px - pw/2, py + ph/2],
    ])

    # 이미지 중심 기준 회전 행렬
    cx_img, cy_img = img_w / 2, img_h / 2
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # 회전
    corners_shifted = corners - [cx_img, cy_img]
    corners_rot = (R @ corners_shifted.T).T + [cx_img, cy_img]

    # 외접 bbox
    x_min = np.clip(corners_rot[:, 0].min(), 0, img_w)
    x_max = np.clip(corners_rot[:, 0].max(), 0, img_w)
    y_min = np.clip(corners_rot[:, 1].min(), 0, img_h)
    y_max = np.clip(corners_rot[:, 1].max(), 0, img_h)

    new_cx = ((x_min + x_max) / 2) / img_w
    new_cy = ((y_min + y_max) / 2) / img_h
    new_w  = (x_max - x_min) / img_w
    new_h  = (y_max - y_min) / img_h

    return new_cx, new_cy, new_w, new_h


def read_labels(label_path: Path):
    """라벨 파일 읽기 → [(cls, cx, cy, w, h), ...]"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    labels = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            labels.append((cls, cx, cy, w, h))
    return labels


def write_labels(label_path: Path, labels):
    """라벨 리스트 → 파일 저장"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in labels]
    label_path.write_text("\n".join(lines) + "\n" if lines else "")


# ── 이미지 변환 함수들 ─────────────────────────────────────────────────────────

def aug_flip_lr(img):
    return cv2.flip(img, 1)

def aug_brightness_up(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.4, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def aug_brightness_down(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.6, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def aug_contrast(img):
    alpha = 1.5   # 대비
    beta  = -30   # 밝기 보정
    return np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

def aug_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 1.2)

def aug_noise(img):
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def aug_rotate_pos(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 10, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_rotate_neg(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -10, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ── 변환 목록 (이름, 이미지변환함수, 라벨변환함수 or None) ────────────────────
AUGMENTATIONS = [
    ("flip",       aug_flip_lr,       lambda labels, iw, ih: [(c, *flip_lr_bbox(cx,cy,w,h)) for c,cx,cy,w,h in labels]),
    ("bright_up",  aug_brightness_up, None),
    ("bright_dn",  aug_brightness_down, None),
    ("contrast",   aug_contrast,      None),
    ("blur",       aug_blur,          None),
    ("noise",      aug_noise,         None),
    ("rot_pos",    aug_rotate_pos,    lambda labels, iw, ih: [(c, *rotate_bbox(cx,cy,w,h, 10,iw,ih)) for c,cx,cy,w,h in labels]),
    ("rot_neg",    aug_rotate_neg,    lambda labels, iw, ih: [(c, *rotate_bbox(cx,cy,w,h,-10,iw,ih)) for c,cx,cy,w,h in labels]),
]


# ── 메인 ──────────────────────────────────────────────────────────────────────

def augment_split(split: str):
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split

    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"\n[{split}] 원본 이미지 {len(imgs)}장 → 증강 시작")

    added = 0
    for img_path in imgs:
        # 원본이 이미 증강된 파일이면 건너뜀
        if any(f"_aug_" in img_path.stem for f in [img_path]):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        labels   = read_labels(lbl_path)

        for aug_name, img_fn, lbl_fn in AUGMENTATIONS:
            out_stem = f"{img_path.stem}_aug_{aug_name}"
            out_img  = img_dir / f"{out_stem}.jpg"
            out_lbl  = lbl_dir / f"{out_stem}.txt"

            # 이미 존재하면 건너뜀 (재실행 안전)
            if out_img.exists():
                continue

            # 이미지 변환
            aug_img = img_fn(img)
            cv2.imwrite(str(out_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # 라벨 변환
            if lbl_fn and labels:
                aug_labels = lbl_fn(labels, iw, ih)
            else:
                aug_labels = labels  # 라벨 그대로

            write_labels(out_lbl, aug_labels)
            added += 1

    total = len(list(img_dir.glob("*.jpg")))
    print(f"  증강 완료: +{added}장 추가 → 총 {total}장")
    return added


def count_bboxes_per_class(split: str):
    from collections import Counter
    lbl_dir = DATASET_DIR / "labels" / split
    counter = Counter()
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                counter[int(parts[0])] += 1
    return counter


if __name__ == "__main__":
    print("=" * 60)
    print("데이터 증강 시작")
    print("=" * 60)

    for split in SPLITS:
        augment_split(split)

    print("\n" + "=" * 60)
    print("증강 후 클래스별 bbox 수")
    print("=" * 60)

    class_names = {0: "elevator", 1: "stairs"}
    for split in SPLITS:
        counter = count_bboxes_per_class(split)
        print(f"\n[{split}]")
        for idx, name in class_names.items():
            cnt = counter.get(idx, 0)
            bar = "█" * min(cnt // 10, 30)
            status = "✅ 충분" if cnt >= 100 else ("🟡 보통" if cnt >= 50 else "⚠️  부족")
            print(f"  {name:15s} {cnt:4d}개  {bar}  {status}")

    print("\n✅ 완료! 다음 단계:")
    print("  python backend/training/check_dataset.py")
    print("  python backend/training/train.py")
