from pathlib import Path
from PIL import Image

# =========================
# Config
# =========================
ROOT_DIR = r"D:\TileMatch_PixelFont"  # 你的根目录：包含很多字体子目录
OUTPUT_SUBDIR_NAME = "output"

# trim 规则
WHITE_THRESHOLD = 10   # >10 认为是“白/非黑内容”（L模式 0-255） 
PADDING = 0            # bbox 外扩 1px
INPLACE = True         # True 覆盖 output 里的 png；False 输出到 output_trimmed
OUT_DIR_NAME = "output_trimmed"
# =========================


def trim_by_white(img: Image.Image, white_threshold: int, padding: int) -> Image.Image:
    # 转 L，拿到 bbox
    g = img.convert("L")
    # 用 point 做一个二值 mask（白色内容为 255，否则 0）
    mask = g.point(lambda p: 255 if p > white_threshold else 0)

    bbox = mask.getbbox()
    if bbox is None:
        return img  # 全黑，原样返回

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(img.width, x1 + padding)
    y1 = min(img.height, y1 + padding)

    return img.crop((x0, y0, x1, y1))


def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root}")

    total = 0
    trimmed = 0
    skipped = 0

    for font_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        out_dir = font_dir / OUTPUT_SUBDIR_NAME
        if not out_dir.exists():
            continue

        if not INPLACE:
            out2 = font_dir / OUT_DIR_NAME
            out2.mkdir(parents=True, exist_ok=True)
        else:
            out2 = out_dir

        pngs = sorted(out_dir.glob("*.png"))
        if not pngs:
            continue

        for p in pngs:
            total += 1
            try:
                img = Image.open(p).convert("RGBA")
                cropped = trim_by_white(img, WHITE_THRESHOLD, PADDING)

                # 如果 bbox 没变化也写一下无所谓；你也可以加判断优化
                save_path = out2 / p.name
                cropped.save(save_path)
                trimmed += 1
            except Exception as e:
                print(f"[SKIP] {p} -> {e}")
                skipped += 1

        print(f"[TRIM] {font_dir.name}: {len(pngs)} pngs")

    print(f"[DONE] total={total}, trimmed={trimmed}, skipped={skipped}, inplace={INPLACE}")


if __name__ == "__main__":
    main()
