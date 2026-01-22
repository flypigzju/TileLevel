import os
import re
from pathlib import Path

import freetype
from fontTools.ttLib import TTFont
from PIL import Image

# =========================
# Config
# =========================
TTF_PATH = r"D:\public_pixel\PublicPixel.ttf"
OUT_DIR  = r"D:\public_pixel\output"   # 输出目录
PX_SIZE  = 32                               # 渲染像素大小：推荐 8/16/32/64...
FIXED_CANVAS = True                         # True: 输出固定画布大小；False: tight crop
CANVAS_W = 32
CANVAS_H = 32
PADDING  = 1                                # tight crop 后加点边
# =========================


def safe_filename(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = s.strip()
    return s if s else "char"


def mono_bitmap_to_L(bitmap) -> Image.Image:
    """FreeType MONO 位图(1-bit) -> PIL 'L'"""
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
    """FreeType GRAY 位图(8-bit) -> PIL 'L'"""
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


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 取 cmap（字符 -> glyph）
    tt = TTFont(TTF_PATH)
    cmap = tt.getBestCmap()  # {codepoint:int -> glyphName:str}
    codepoints = sorted(cmap.keys())

    # 2) 用 FreeType 渲染
    face = freetype.Face(TTF_PATH)
    face.set_pixel_sizes(0, PX_SIZE)

    exported = 0
    for cp in codepoints:
        ch = chr(cp)
        # 文件名：U+XXXX + 可读字符（可选）
        name_hint = safe_filename(ch)
        filename = f"U+{cp:04X}_{name_hint}.png"

        # 渲染 glyph
        # 用 MONO target 更像“硬像素”；但有些字体会输出 GRAY，也兼容
        face.load_char(ch, flags=freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO)
        g = face.glyph
        bm = g.bitmap

        if bm.width == 0 or bm.rows == 0:
            continue

        # bitmap -> PIL
        if bm.pixel_mode == freetype.FT_PIXEL_MODE_MONO:
            img = mono_bitmap_to_L(bm)
        else:
            img = gray_bitmap_to_L(bm)

        img = tight_crop(img, padding=PADDING)

        if FIXED_CANVAS:
            img = paste_center(img, CANVAS_W, CANVAS_H)

        img.save(out_dir / filename)
        exported += 1

    print(f"[DONE] exported={exported} glyphs -> {out_dir}")


if __name__ == "__main__":
    main()
