import os
from pathlib import Path
from PIL import Image
import pillow_avif

# =========================
INPUT_IMAGE = r"D:\游戏开发\TileMatch\1.avif"
OUTPUT_DIR  = r"D:\游戏开发\TileMatch\1_sliced"
PREFIX = "icon"

ICON_W = 132
ICON_H = 132
OFFSET_X = 0
OFFSET_Y = 0
GAP_X = 0
GAP_Y = 0

SKIP_EMPTY = True
MIN_NONEMPTY_PIXELS = 10
BLACK_THRESHOLD = 220

EXPORT_DOWNSCALED = True
DOWNSCALE_SIZE = 32

EXPORT_GRID_PREVIEW = True
GRID_PREVIEW_NAME = "_grid_preview.png"
GRID_COLOR = (255, 0, 0, 255)

# ✅ 新增：自动内容居中校正
AUTO_CENTER = True
# 内容识别阈值（越大越“把深色都算内容”）
CONTENT_DARK_THRESHOLD = 240
# 如果内容 bbox 太小（像噪点），不做居中
MIN_CONTENT_AREA = 20
# =========================


def resolve_output_dir(input_path: Path) -> Path:
    out = Path(OUTPUT_DIR)
    return out if out.is_absolute() else (input_path.parent / out)


def is_empty_icon(img: Image.Image) -> bool:
    if not SKIP_EMPTY:
        return False
    if img.mode in ("RGBA", "LA"):
        alpha = img.split()[-1]
        non_empty = sum(1 for a in alpha.getdata() if a > 0)
        return non_empty < MIN_NONEMPTY_PIXELS
    g = img.convert("L")
    dark = sum(1 for p in g.getdata() if p < BLACK_THRESHOLD)
    return dark < MIN_NONEMPTY_PIXELS


def draw_rect_rgba(img_rgba: Image.Image, x0, y0, x1, y1, color):
    px = img_rgba.load()
    w, h = img_rgba.size
    for x in range(x0, x1):
        if 0 <= x < w and 0 <= y0 < h: px[x, y0] = color
        if 0 <= x < w and 0 <= (y1 - 1) < h: px[x, y1 - 1] = color
    for y in range(y0, y1):
        if 0 <= x0 < w and 0 <= y < h: px[x0, y] = color
        if 0 <= (x1 - 1) < w and 0 <= y < h: px[x1 - 1, y] = color


def autocenter_icon(icon: Image.Image) -> Image.Image:
    """
    在单个切片内，根据内容 bbox 做居中回填到 ICON_W×ICON_H
    适合纠正“格子里图标没对齐”的情况
    """
    # 用灰度找内容 bbox（不依赖透明）
    g = icon.convert("L")
    data = g.load()
    w, h = g.size

    xs = []
    ys = []
    for y in range(h):
        for x in range(w):
            if data[x, y] < CONTENT_DARK_THRESHOLD:
                xs.append(x)
                ys.append(y)

    if not xs:
        return icon  # 无内容

    x0, x1 = min(xs), max(xs) + 1
    y0, y1 = min(ys), max(ys) + 1
    bw, bh = x1 - x0, y1 - y0

    if bw * bh < MIN_CONTENT_AREA:
        return icon  # 内容太小，可能是噪点

    content = icon.crop((x0, y0, x1, y1))

    # 回填到同尺寸画布中心
    base = Image.new(icon.mode, (ICON_W, ICON_H), (0, 0, 0, 0) if "A" in icon.getbands() else (255, 255, 255))
    ox = (ICON_W - bw) // 2
    oy = (ICON_H - bh) // 2
    base.paste(content, (ox, oy))
    return base


def slice_sheet(input_path: str):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    sheet = Image.open(input_path)
    sheet_w, sheet_h = sheet.size

    out_dir = resolve_output_dir(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_x = ICON_W + GAP_X
    step_y = ICON_H + GAP_Y

    cols = (sheet_w - OFFSET_X + GAP_X) // step_x
    rows = (sheet_h - OFFSET_Y + GAP_Y) // step_y
    if cols <= 0 or rows <= 0:
        raise ValueError("Invalid slicing params.")

    preview = sheet.convert("RGBA").copy() if EXPORT_GRID_PREVIEW else None

    saved, idx = 0, 1
    for r in range(rows):
        for c in range(cols):
            x0 = OFFSET_X + c * step_x
            y0 = OFFSET_Y + r * step_y
            x1 = x0 + ICON_W
            y1 = y0 + ICON_H
            if x1 > sheet_w or y1 > sheet_h:
                continue

            if preview is not None:
                draw_rect_rgba(preview, x0, y0, x1, y1, GRID_COLOR)

            icon = sheet.crop((x0, y0, x1, y1))
            if is_empty_icon(icon):
                idx += 1
                continue

            if AUTO_CENTER:
                icon = autocenter_icon(icon)

            out_name = f"{PREFIX}_{idx:04d}_r{r:02d}_c{c:02d}.png"
            icon.save(out_dir / out_name)

            if EXPORT_DOWNSCALED:
                small = icon.resize((DOWNSCALE_SIZE, DOWNSCALE_SIZE), resample=Image.Resampling.NEAREST)
                small.save(out_dir / out_name.replace(".png", f"_{DOWNSCALE_SIZE}.png"))

            saved += 1
            idx += 1

    if preview is not None:
        preview.save(out_dir / GRID_PREVIEW_NAME)

    print(f"[DONE] saved={saved} -> {out_dir}")
    if EXPORT_GRID_PREVIEW:
        print(f"[DONE] preview -> {out_dir / GRID_PREVIEW_NAME}")


if __name__ == "__main__":
    slice_sheet(INPUT_IMAGE)
