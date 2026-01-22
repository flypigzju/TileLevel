# font_png_to_level_txt.py
# ------------------------------------------------------------
# Convert pixel-font glyph PNGs into TileMatch level patterns
# + rename pngs to 1.png,2.png,... before parsing
# + robust grid inference via pitch estimation (no clustering tol)
# + DEBUG outputs (overlay centers, grid view, final 9x9 view)
# ------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# =========================
# Hardcoded paths
# =========================
INPUT_DIR = r"D:\游戏开发\TileMatch\pixelfonts\arcade\output"
OUT_TXT   = r"D:\游戏开发\TileMatch\pixelfonts\arcade\level_pattern_simple_arcade.txt"

# =========================
# Debug output
# =========================
DEBUG = True
DEBUG_DIR = r"D:\游戏开发\TileMatch\pixelfonts\arcade\debug_out"  # <-- 改成你想保存debug的目录
DEBUG_CELL_SCALE = 24  # grid 可视化放大倍数

# =========================
# Rename settings
# =========================
RENAME_FILES = True
RENAME_PAD_WIDTH = 0  # 0=1.png；4=0001.png

# =========================
# Output board setting
# =========================
BOARD_TILES = 9
TYPE_LINE_VALUE = 3
LAYER_INDEX = 0

# =========================
# Image / mask settings
# =========================
WHITE_LUM_THRESHOLD = 200
ALPHA_THRESHOLD = 10

# =========================
# Pattern constraints
# =========================
REJECT_IF_ON_LEQ = 4
DOWNSAMPLE_IF_GT_BOARD = True
MIN_COMPONENT_AREA = 6   # ✅ 这类像素块通常更大，提高点更稳（你也可按实际调）

# ============================================================
# Utils: mask + trim
# ============================================================

def load_rgba(p: Path) -> Image.Image:
    return Image.open(p).convert("RGBA")

def load_white_mask(p: Path) -> np.ndarray:
    img = load_rgba(p)
    arr = np.array(img).astype(np.uint8)
    rgb = arr[..., :3].astype(np.float32)
    a   = arr[..., 3].astype(np.float32)

    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    mask = (lum >= WHITE_LUM_THRESHOLD) & (a >= ALPHA_THRESHOLD)
    return mask

def trim_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())
    return mask[miny:maxy+1, minx:maxx+1]

# ============================================================
# Connected components (8-neighbor)
# ============================================================

def components_bboxes(mask: np.ndarray) -> List[Tuple[int,int,int,int,int,int,int]]:
    """
    Return list of components:
      (minx, miny, maxx, maxy, area, cx, cy)
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    comps = []

    neigh = [(-1,-1), (-1,0), (-1,1),
             (0,-1),          (0,1),
             (1,-1),  (1,0),  (1,1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            q = [(x, y)]
            visited[y, x] = 1
            minx = maxx = x
            miny = maxy = y
            area = 0
            sx = sy = 0

            while q:
                cx, cy = q.pop()
                area += 1
                sx += cx
                sy += cy
                minx = min(minx, cx); maxx = max(maxx, cx)
                miny = min(miny, cy); maxy = max(maxy, cy)

                for dx, dy in neigh:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((nx, ny))

            if area >= MIN_COMPONENT_AREA:
                ccx = int(round(sx / area))
                ccy = int(round(sy / area))
                comps.append((minx, miny, maxx, maxy, area, ccx, ccy))

    return comps

# ============================================================
# Robust grid inference via pitch estimation (NO clustering)
# ============================================================

def infer_grid_from_components_debug(mask_trim: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    - comps centers -> estimate pitchX/pitchY via neighbor diffs median
    - map each center to integer (ix,iy) via round((c-c0)/pitch)
    """
    dbg: Dict[str, Any] = {}
    comps = components_bboxes(mask_trim)
    dbg["comps"] = comps

    if not comps:
        return None, dbg

    xs = sorted([c[5] for c in comps])
    ys = sorted([c[6] for c in comps])

    def estimate_pitch(vals: List[int]) -> float:
        if len(vals) < 2:
            return 1.0
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        diffs = [d for d in diffs if d > 0]
        if not diffs:
            return 1.0
        med = float(np.median(diffs))
        good = [d for d in diffs if d >= med * 0.6]
        if not good:
            good = diffs
        return float(np.median(good))

    pitchX = estimate_pitch(xs)
    pitchY = estimate_pitch(ys)

    x0 = float(min(xs))
    y0 = float(min(ys))

    pts = []
    for c in comps:
        cx, cy = c[5], c[6]
        ix = int(round((cx - x0) / pitchX))
        iy = int(round((cy - y0) / pitchY))
        pts.append((ix, iy, cx, cy))

    if not pts:
        return None, dbg

    max_ix = max(p[0] for p in pts)
    max_iy = max(p[1] for p in pts)

    W = max_ix + 1
    H = max_iy + 1
    if W <= 0 or H <= 0:
        return None, dbg

    grid = np.zeros((H, W), dtype=np.uint8)
    for ix, iy, _, _ in pts:
        if 0 <= ix < W and 0 <= iy < H:
            grid[iy, ix] = 1

    dbg.update({
        "pitchX": pitchX, "pitchY": pitchY,
        "x0": x0, "y0": y0,
        "mapped": pts,
        "gridW": W, "gridH": H
    })
    return grid, dbg

# ============================================================
# Downsample + trim
# ============================================================

def downsample_grid_any(src: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    in_h, in_w = src.shape
    dst = np.zeros((out_h, out_w), dtype=np.uint8)

    for oy in range(out_h):
        y0 = int(round(oy * in_h / out_h))
        y1 = int(round((oy + 1) * in_h / out_h))
        y1 = max(y1, y0 + 1)
        y1 = min(y1, in_h)

        for ox in range(out_w):
            x0 = int(round(ox * in_w / out_w))
            x1 = int(round((ox + 1) * in_w / out_w))
            x1 = max(x1, x0 + 1)
            x1 = min(x1, in_w)

            if np.any(src[y0:y1, x0:x1]):
                dst[oy, ox] = 1
    return dst

def trim_grid(grid: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return grid
    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())
    return grid[miny:maxy+1, minx:maxx+1]

# ============================================================
# Center into 9x9 micro coords
# ============================================================

def grid_to_micro_points_centered(grid: np.ndarray, board_tiles: int) -> List[Tuple[int,int]]:
    grid = trim_grid(grid)
    h, w = grid.shape
    if w == 0 or h == 0:
        return []

    if DOWNSAMPLE_IF_GT_BOARD and (w > board_tiles or h > board_tiles):
        grid = downsample_grid_any(grid, min(w, board_tiles), min(h, board_tiles))
        grid = trim_grid(grid)
        h, w = grid.shape

    offx = (board_tiles - w) / 2.0
    offy = (board_tiles - h) / 2.0

    pts = []
    for iy in range(h):
        for ix in range(w):
            if grid[iy, ix] == 0:
                continue
            gy = (h - 1) - iy  # flip to bottom-origin
            gx = ix

            mx = int(round((gx + offx) * 2 + 1))
            my = int(round((gy + offy) * 2 + 1))
            pts.append((mx, my))

    pts.sort(key=lambda t: (t[1], t[0]))
    return pts

# ============================================================
# Rename helpers (two-pass safe rename)
# ============================================================

def _list_pngs(in_dir: Path) -> List[Path]:
    return [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]

def _safe_unlink_if_exists(p: Path):
    if p.exists():
        p.unlink()

def rename_pngs_sequential(in_dir: Path) -> List[Path]:
    files = sorted(_list_pngs(in_dir), key=lambda p: p.name.lower())
    if not files:
        return []

    def final_name(i: int) -> str:
        if RENAME_PAD_WIDTH and RENAME_PAD_WIDTH > 0:
            return f"{i:0{RENAME_PAD_WIDTH}d}.png"
        return f"{i}.png"

    # pass1 temp
    temp_paths = []
    for idx, p in enumerate(files, start=1):
        tmp = in_dir / f"__tmp__{idx:08d}.png"
        if tmp.exists():
            _safe_unlink_if_exists(tmp)
        p.rename(tmp)
        temp_paths.append(tmp)

    # pass2 final
    final_paths = []
    for idx, tmp in enumerate(temp_paths, start=1):
        dst = in_dir / final_name(idx)
        if dst.exists():
            _safe_unlink_if_exists(dst)
        tmp.rename(dst)
        final_paths.append(dst)

    final_paths.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    return final_paths

def list_pngs_numeric(in_dir: Path) -> List[Path]:
    files = _list_pngs(in_dir)
    def key(p: Path):
        return (0, int(p.stem)) if p.stem.isdigit() else (1, p.name.lower())
    return sorted(files, key=key)

# ============================================================
# Debug drawing helpers
# ============================================================

def _get_font():
    try:
        return ImageFont.load_default()
    except:
        return None

def debug_overlay_components(img_rgba: Image.Image, dbg: Dict[str, Any], out_path: Path):
    """
    在原图上画：
    - 每个 component bbox（绿框）
    - 中心点（红十字）
    - 映射到的 (ix,iy)（黄字）
    """
    im = img_rgba.copy()
    draw = ImageDraw.Draw(im)
    font = _get_font()

    comps = dbg.get("comps", [])
    mapped = dbg.get("mapped", [])

    # bbox + center
    for c in comps:
        minx, miny, maxx, maxy, area, cx, cy = c
        draw.rectangle([minx, miny, maxx, maxy], outline=(0, 255, 0, 255), width=1)
        draw.line([cx-3, cy, cx+3, cy], fill=(255, 0, 0, 255), width=1)
        draw.line([cx, cy-3, cx, cy+3], fill=(255, 0, 0, 255), width=1)

    # label mapped idx near center
    for ix, iy, cx, cy in mapped:
        draw.text((cx+4, cy-6), f"{ix},{iy}", fill=(255, 255, 0, 255), font=font)

    # header info
    pitchX = dbg.get("pitchX", None)
    pitchY = dbg.get("pitchY", None)
    txt = f"pitchX={pitchX:.2f} pitchY={pitchY:.2f}  comps={len(comps)}"
    draw.text((4, 4), txt, fill=(0, 255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def debug_grid_image(grid: np.ndarray, out_path: Path, scale: int = 24):
    """
    输出 occupancy grid 的可视化（白=1 黑=0），并画网格线
    """
    h, w = grid.shape
    im = Image.new("RGBA", (w*scale, h*scale), (0, 0, 0, 255))
    draw = ImageDraw.Draw(im)

    for y in range(h):
        for x in range(w):
            if grid[y, x]:
                draw.rectangle([x*scale, y*scale, (x+1)*scale-1, (y+1)*scale-1],
                               fill=(255, 255, 255, 255))

    # grid lines
    for x in range(w+1):
        draw.line([x*scale, 0, x*scale, h*scale], fill=(60, 60, 60, 255), width=1)
    for y in range(h+1):
        draw.line([0, y*scale, w*scale, y*scale], fill=(60, 60, 60, 255), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def debug_board9_image(micro_pts: List[Tuple[int,int]], out_path: Path, tile_scale: int = 48):
    """
    输出最终 9x9 board 的可视化（按 tile 显示），用于对照 Editor
    micro -> tile: ix=(mx-1)/2, iy=(my-1)/2 （这里 micro_pts 本身是中心点）
    """
    w = BOARD_TILES
    h = BOARD_TILES
    im = Image.new("RGBA", (w*tile_scale, h*tile_scale), (30, 30, 30, 255))
    draw = ImageDraw.Draw(im)

    occ = np.zeros((h, w), dtype=np.uint8)
    for mx, my in micro_pts:
        ix = int(round((mx - 1) * 0.5))
        iy = int(round((my - 1) * 0.5))
        if 0 <= ix < w and 0 <= iy < h:
            occ[iy, ix] = 1

    # draw cells (iy 0 at bottom for your editor usually; for view we draw top-down,
    # so flip y for visualization)
    for iy in range(h):
        for ix in range(w):
            vy = (h - 1 - iy)
            x0 = ix * tile_scale
            y0 = vy * tile_scale
            x1 = x0 + tile_scale - 1
            y1 = y0 + tile_scale - 1

            if occ[iy, ix]:
                draw.rectangle([x0, y0, x1, y1], fill=(230, 230, 230, 255))
            draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80, 255), width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

def debug_dump_txt(dbg: Dict[str, Any], micro_pts: List[Tuple[int,int]], out_path: Path):
    lines = []
    lines.append(f"pitchX={dbg.get('pitchX', None)}")
    lines.append(f"pitchY={dbg.get('pitchY', None)}")
    lines.append(f"x0={dbg.get('x0', None)} y0={dbg.get('y0', None)}")
    lines.append(f"comps={len(dbg.get('comps', []))}")
    lines.append(f"mapped_count={len(dbg.get('mapped', []))}")
    lines.append("")
    lines.append("mapped(ix,iy,cx,cy):")
    for ix, iy, cx, cy in dbg.get("mapped", []):
        lines.append(f"  {ix},{iy}  center=({cx},{cy})")
    lines.append("")
    lines.append(f"micro_pts({len(micro_pts)}):")
    lines.append("; ".join([f"{x},{y}" for x, y in micro_pts]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

# ============================================================
# Main
# ============================================================

def main():
    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {in_dir}")

    # rename first
    if RENAME_FILES:
        renamed = rename_pngs_sequential(in_dir)
        print(f"[RENAME] renamed_pngs={len(renamed)} (pad={RENAME_PAD_WIDTH})")

    files = list_pngs_numeric(in_dir)

    levels = []
    level_id = 1
    skipped = 0

    debug_root = Path(DEBUG_DIR)

    for p in files:
        mask = load_white_mask(p)
        mask_t = trim_mask(mask)
        if mask_t is None:
            skipped += 1
            continue

        grid, dbg = infer_grid_from_components_debug(mask_t)
        if grid is None:
            # fallback: coarse downsample
            coarse = downsample_grid_any(mask_t.astype(np.uint8), 16, 16)
            grid = trim_grid(coarse)
            dbg["fallback"] = "coarse16"

        pts = grid_to_micro_points_centered(grid, BOARD_TILES)

        # DEBUG outputs (before reject) - so you can see what's going on even when skipped
        if DEBUG:
            img = load_rgba(p)
            stem = p.stem
            debug_overlay_components(img, dbg, debug_root / "01_overlay" / f"{stem}_overlay.png")
            debug_grid_image(grid, debug_root / "02_grid" / f"{stem}_grid.png", scale=DEBUG_CELL_SCALE)
            debug_board9_image(pts, debug_root / "03_board9" / f"{stem}_board9.png", tile_scale=48)
            debug_dump_txt(dbg, pts, debug_root / "00_info" / f"{stem}_info.txt")

        if len(pts) <= REJECT_IF_ON_LEQ:
            skipped += 1
            continue

        line_cells = "; ".join([f"{x},{y}" for x, y in pts])

        block = []
        block.append(f"# --- Level {level_id} ---")
        block.append(f"level={level_id}")
        block.append(f"type={TYPE_LINE_VALUE}")
        block.append(f"{LAYER_INDEX}= {line_cells}")
        block.append("")
        levels.append("\n".join(block))
        level_id += 1

    out_path = Path(OUT_TXT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(levels), encoding="utf-8")

    print(f"[DONE] pngs={len(files)} exported_levels={len(levels)} skipped={skipped}")
    print(f"[DONE] wrote: {out_path}")
    if DEBUG:
        print(f"[DONE] debug wrote to: {debug_root}")

if __name__ == "__main__":
    main()
