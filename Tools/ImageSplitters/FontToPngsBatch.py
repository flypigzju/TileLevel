import shutil
from pathlib import Path

import freetype
from fontTools.ttLib import TTFont
from PIL import Image

# =========================
# Config
# =========================
ROOT_DIR = r"D:\TileMatch_PixelFont"   # <-- 根目录：包含很多字体子目录
PX_SIZE  = 512                          # glyph 渲染像素大小：8/16/32/64...
FIXED_CANVAS = True                    # True: 固定画布输出；False: tight crop
CANVAS_W = 512
CANVAS_H = 512
PADDING  = 0                           # tight crop 后加边
OUTPUT_SUBDIR_NAME = "output"          # 输出到每个字体子目录下 output/
SAVE_MAPPING = True                    # 输出 mapping.txt：序号 -> codepoint
# =========================


def mono_bitmap_to_L(bitmap) -> Image.Image:
    w, h = bitmap.width, bitmap.rows
    pitch = bitmap.pitch
    buf = bitmap.buffer

    img = Image.new("L", (w, h), 0)
    px = img.load()
    for y in range(h):
        row = buf[y * pitch:(y + 1) * pitch]
        for x in range(w):
            byte = row[x >> 3]
            bit = 7 - (x & 7)
            v = 255 if (byte >> bit) & 1 else 0
            px[x, y] = v
    return img


def gray_bitmap_to_L(bitmap) -> Image.Image:
    w, h = bitmap.width, bitmap.rows
    pitch = bitmap.pitch
    buf = bitmap.buffer

    img = Image.new("L", (w, h), 0)
    px = img.load()
    for y in range(h):
        row = buf[y * pitch:(y + 1) * pitch]
        for x in range(w):
            px[x, y] = row[x]
    return img


def tight_crop(img: Image.Image, padding: int = 0) -> Image.Image:
    bbox = img.getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(img.width, x1 + padding)
    y1 = min(img.height, y1 + padding)
    return img.crop((x0, y0, x1, y1))


def paste_center(src: Image.Image, w: int, h: int) -> Image.Image:
    dst = Image.new("L", (w, h), 0)
    ox = (w - src.width) // 2
    oy = (h - src.height) // 2
    dst.paste(src, (ox, oy))
    return dst


def find_font_file(folder: Path) -> Path | None:
    ttf = sorted(folder.glob("*.ttf"))
    otf = sorted(folder.glob("*.otf"))
    if ttf:
        return ttf[0]
    if otf:
        return otf[0]

    # 有些会放在更深一层
    ttf = sorted(folder.rglob("*.ttf"))
    otf = sorted(folder.rglob("*.otf"))
    if ttf:
        return ttf[0]
    if otf:
        return otf[0]
    return None


def clear_output_dir(out_dir: Path):
    """清空 output 目录（存在就删掉重建）"""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def export_font_glyphs(font_path: Path, out_dir: Path) -> int:
    clear_output_dir(out_dir)

    # cmap（字符 -> glyph）
    tt = TTFont(str(font_path))
    cmap = tt.getBestCmap() or {}
    codepoints = sorted(cmap.keys())

    face = freetype.Face(str(font_path))
    face.set_pixel_sizes(0, PX_SIZE)

    mapping_lines = []
    exported = 0

    for idx, cp in enumerate(codepoints, start=1):
        ch = chr(cp)

        # 渲染 glyph（MONO target 更像硬像素）
        try:
            face.load_char(ch, flags=freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO)
        except Exception:
            continue

        bm = face.glyph.bitmap
        if bm.width == 0 or bm.rows == 0:
            continue

        if bm.pixel_mode == freetype.FT_PIXEL_MODE_MONO:
            img = mono_bitmap_to_L(bm)
        else:
            img = gray_bitmap_to_L(bm)

        img = tight_crop(img, padding=PADDING)
        if FIXED_CANVAS:
            img = paste_center(img, CANVAS_W, CANVAS_H)

        # 按编号保存：0001.png, 0002.png...
        filename = f"{idx:04d}.png"
        img.save(out_dir / filename)
        exported += 1

        if SAVE_MAPPING:
            mapping_lines.append(f"{idx:04d}\tU+{cp:04X}")

    if SAVE_MAPPING:
        (out_dir / "mapping.txt").write_text("\n".join(mapping_lines), encoding="utf-8")

    return exported


def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root}")

    total_fonts = 0
    total_exported = 0
    skipped = 0

    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        font_file = find_font_file(sub)
        if font_file is None:
            print(f"[SKIP] no .ttf/.otf found in: {sub}")
            skipped += 1
            continue

        out_dir = sub / OUTPUT_SUBDIR_NAME
        print(f"[FONT] {sub.name} -> {font_file.name} -> {out_dir} (clear first)")

        try:
            exported = export_font_glyphs(font_file, out_dir)
            print(f"  exported={exported}")
            total_fonts += 1
            total_exported += exported
        except Exception as e:
            print(f"[ERROR] failed on {font_file}: {e}")
            skipped += 1

    print(f"[DONE] fonts={total_fonts}, total_pngs={total_exported}, skipped_folders={skipped}, root={root}")


if __name__ == "__main__":
    main()
