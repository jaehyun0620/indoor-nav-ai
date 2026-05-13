"""
convert_seg_to_det.py — Roboflow Segmentation 라벨 → YOLO Detection 라벨 변환
=============================================================================
Roboflow에서 내보낸 폴리곤(segmentation) 형식을 YOLOv8 detection bbox 형식으로 변환.

입력:  class_id x1 y1 x2 y2 x3 y3 ... (정규화 좌표)
출력:  class_id cx cy w h             (정규화 좌표)

실행:
    cd /Users/jaehyun/ai_capstone/capstone
    python backend/training/convert_seg_to_det.py
"""

from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent / "indoor-obstacle"
DST_ROOT = Path(__file__).parent / "dataset"

SPLITS = {
    "train": ("train/labels", "images/train"),
    "val":   ("valid/labels", "images/val"),
    "test":  ("test/labels",  "images/val"),   # test는 val에 합침
}

def poly_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """폴리곤 좌표 리스트(x1,y1,x2,y2,...) → (cx, cy, w, h) 정규화값"""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    return cx, cy, w, h


def convert_label_file(src_path: Path, dst_path: Path):
    """라벨 파일 하나 변환."""
    lines_in  = src_path.read_text().strip().splitlines()
    lines_out = []

    for line in lines_in:
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        if len(coords) < 4:
            # 이미 bbox 형식이거나 데이터 부족 — 건너뜀
            continue

        if len(coords) == 4:
            # 이미 cx cy w h 형식
            cx, cy, w, h = coords
        else:
            # 폴리곤 → bbox 변환
            cx, cy, w, h = poly_to_bbox(coords)

        lines_out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if lines_out:
        dst_path.write_text("\n".join(lines_out) + "\n")
    else:
        # 배경 이미지 — 빈 라벨 파일 생성
        dst_path.write_text("")


def copy_images(src_img_dir: Path, dst_img_dir: Path):
    """이미지 파일을 dst로 복사 (심볼릭 링크 대신 실제 복사)."""
    import shutil
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for img in src_img_dir.glob("*.jpg"):
        shutil.copy2(img, dst_img_dir / img.name)
    for img in src_img_dir.glob("*.png"):
        shutil.copy2(img, dst_img_dir / img.name)


def main():
    total_converted = 0
    total_skipped   = 0

    for split, (lbl_rel, img_dst_rel) in SPLITS.items():
        src_lbl_dir = SRC_ROOT / lbl_rel
        src_img_dir = SRC_ROOT / lbl_rel.replace("labels", "images")
        dst_lbl_dir = DST_ROOT / "labels" / split.replace("val", "val")
        dst_img_dir = DST_ROOT / img_dst_rel

        if not src_lbl_dir.exists():
            print(f"[SKIP] {src_lbl_dir} 없음")
            continue

        print(f"\n[{split}] {src_lbl_dir} → {dst_lbl_dir}")

        # 이미지 복사
        if src_img_dir.exists():
            copy_images(src_img_dir, dst_img_dir)
            print(f"  이미지 복사: {len(list(src_img_dir.glob('*.jpg'))) + len(list(src_img_dir.glob('*.png')))}장")

        # 라벨 변환
        label_files = list(src_lbl_dir.glob("*.txt"))
        for lbl_file in label_files:
            dst_file = dst_lbl_dir / lbl_file.name
            try:
                convert_label_file(lbl_file, dst_file)
                total_converted += 1
            except Exception as e:
                print(f"  [ERR] {lbl_file.name}: {e}")
                total_skipped += 1

        print(f"  라벨 변환: {len(label_files)}개")

    print(f"\n✅ 변환 완료: {total_converted}개 성공, {total_skipped}개 실패")
    print(f"   출력 경로: {DST_ROOT}")
    print("\n다음 단계:")
    print("  python backend/training/check_dataset.py  # 데이터셋 점검")
    print("  python backend/training/train.py          # 학습 시작")


if __name__ == "__main__":
    main()
