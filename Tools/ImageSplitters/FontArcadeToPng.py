import os
import re
from pathlib import Path

import freetype
from fontTools.ttLib import TTFont
from PIL import Image

# =========================
# Config
# =========================
TTF_PATH = r"D:\TileMatch_PixelFont\ARCADE.TTF"
OUT_DIR  = r"D:\游戏开发\TileMatch\pixelfonts\arcade\output"   # 输出目录
PX_SIZE  = 512                               # 渲染像素大小：推荐 8/16/32/64...
FIXED_CANVAS = True                          # True: 输出固定画布大小；False: tight crop
CANVAS_W = 512
CANVAS_H = 512
PADDING  = 1                                 # tight crop 后加点边
# =========================


def safe_filename(s: str, fallback: str = "char") -> str:
    """
    Make a Windows-safe filename fragment.
    - remove NULL and control chars
    - replace reserved characters <>:"/\\|?*
    - strip trailing dots/spaces (Windows invalid)
    """
    if s is None:
        return fallback

    # remove NULL explicitly
    s = s.replace("\x00", "")

    # remove control chars 0x00-0x1F and 0x7F
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)

    # replace windows forbidden characters
    s = re.sub(r'[\\/:*?"<>|]', "_", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # windows forbids trailing dot/space
    s = s.strip(" .")

    return s if s else fallback


def is_printable_safe_char(ch: str) -> bool:
    """Visible and not a path/reserved char; avoids weird names."""
    if not ch or len(ch) != 1:
        return False
    o = ord(ch)
    if o < 32 or o == 127:
        return False
    if ch in '<>:"/\\|?*':
        return False
    # 你也可以选择把空格过滤掉
    if ch.isspace():
        return False
    return True


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
    skipped = 0

    for cp in codepoints:
        ch = chr(cp)

        # 文件名建议：永远以 codepoint 为主，hint 只是辅助
        if is_printable_safe_char(ch):
            name_hint = safe_filename(ch, fallback=f"cp{cp:04X}")
        else:
            name_hint = f"cp{cp:04X}"

        filename = f"U+{cp:04X}_{name_hint}.png"

        # 终极保险：防止任何地方混入 NULL
        filename = filename.replace("\x00", "")
        if "\x00" in filename:
            # 理论不会进来，除非上面被改坏
            print("[WARN] filename contains NULL:", repr(filename))
            skipped += 1
            continue

        # 渲染 glyph
        try:
            face.load_char(ch, flags=freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO)
        except Exception as e:
            # 有些 cmap codepoint 可能 freetype load 失败，跳过
            skipped += 1
            continue

        g = face.glyph
        bm = g.bitmap

        if bm.width == 0 or bm.rows == 0:
            skipped += 1
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

    print(f"[DONE] exported={exported} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
