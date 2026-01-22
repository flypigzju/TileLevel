# tile_tetris_repeat_generator.py
# ------------------------------------------------------------
# Standalone symmetric tetris-like pattern generator
# - Run this file directly to generate PNG previews + contact_sheet
#
# Output:
#   OUT_DIR/icons/px_00001_nXX.png ...
#   OUT_DIR/contact_sheet.png
#
# Requirements:
#   pip install numpy pillow
# ------------------------------------------------------------

from __future__ import annotations
import os
import random
from typing import List, Tuple, Dict, Iterable
import numpy as np
from PIL import Image, ImageDraw
from collections import deque

Coord = Tuple[int, int]

# =========================
# CONFIG (edit here)
# =========================
OUT_DIR = r"D:\TileMatch_TetrisPreview"
NUM_PATTERNS = 500
SEED = 12345

CANVAS = 9               # final output grid is 9x9
ICON_SCALE = 28          # each cell => 28px (9*28=252px)
ICON_BG = (255, 255, 255, 255)
ICON_FG = (0, 0, 0, 255)

CONTACT_SHEET_COLS = 20
CONTACT_SHEET_NAME = "contact_sheet.png"

# density limit: based on 9*9 (20% => 16)
MAX_FILL_RATIO = 0.20

# avoid tiny junk pixels
FORBID_ISOLATED = True                 # forbid 8-neighborhood isolated single pixels
REMOVE_TINY_COMPONENTS = True          # remove connected components smaller than:
TINY_COMPONENT_MIN_SIZE = 2            # 2 => remove 1-pixel components

# spacing / overlap ranges (you asked: can be 1..4 etc; overlap allowed)
GAP_CHOICES = [0, 1, 1, 2, 2, 3, 4]    # farther apart possible
OVERLAP_CHOICES = [0, 0, 1, 1, 2, 2, 3]

# generation attempts
ATTEMPTS_PER_PATTERN = 600

# dedupe (canonical by rot/mirror), plus optional hamming min distance
ENABLE_DEDUPE = True
ENABLE_HAMMING_FILTER = True
HAMMING_MIN = 10               # bigger => fewer "too similar"
HAMMING_CHECK_LAST = 2000      # compare only last N patterns to keep it fast
# =========================


# =========================
# Base Shapes (local coords)
# (You can add your 4 new starter shapes here later)
# =========================
SHAPES: Dict[str, List[Coord]] = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "L": [(0, 0), (0, 1), (0, 2), (1, 0)],
    "J": [(1, 0), (1, 1), (1, 2), (0, 0)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
}

# weighted pool
DEFAULT_SHAPE_POOL = ["T","T","L","L","J","S","Z","I","O","O","I"]


# -------------------------
# Coord transforms
# -------------------------
def rot90(c: List[Coord]) -> List[Coord]:
    return [(y, -x) for (x, y) in c]

def rot180(c: List[Coord]) -> List[Coord]:
    return rot90(rot90(c))

def rot270(c: List[Coord]) -> List[Coord]:
    return rot90(rot180(c))

def flip_lr(c: List[Coord]) -> List[Coord]:
    return [(-x, y) for (x, y) in c]

def flip_ud(c: List[Coord]) -> List[Coord]:
    return [(x, -y) for (x, y) in c]

def normalize(c: List[Coord]) -> List[Coord]:
    minx = min(x for x, _ in c)
    miny = min(y for _, y in c)
    return [(x - minx, y - miny) for (x, y) in c]

def bbox_coords(c: List[Coord]) -> Tuple[int, int, int, int]:
    xs = [x for x, _ in c]
    ys = [y for _, y in c]
    return min(xs), min(ys), max(xs), max(ys)

def size_xy(c: List[Coord]) -> Tuple[int, int]:
    mnx, mny, mxx, mxy = bbox_coords(c)
    return (mxx - mnx + 1, mxy - mny + 1)

def translate(c: List[Coord], dx: int, dy: int) -> List[Coord]:
    return [(x + dx, y + dy) for (x, y) in c]

def union(parts: List[List[Coord]]) -> List[Coord]:
    s = set()
    for p in parts:
        for t in p:
            s.add(t)
    return list(s)

def random_base_variant(rng: random.Random, base: List[Coord]) -> List[Coord]:
    # regular: rotation + optional LR mirror (less "weird")
    variants = [base, rot90(base), rot180(base), rot270(base)]
    c = variants[rng.randint(0, 3)]
    if rng.random() < 0.35:
        c = flip_lr(c)
    return normalize(c)


# -------------------------
# Grid helpers
# -------------------------
def coords_to_grid(coords: List[Coord], size: int) -> np.ndarray:
    g = np.zeros((size, size), dtype=np.uint8)
    for (x, y) in coords:
        if 0 <= x < size and 0 <= y < size:
            g[y, x] = 1
    return g

def center_bbox_in_canvas(g: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return g
    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())

    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    n = g.shape[0]
    tx = (n - 1) / 2.0
    ty = (n - 1) / 2.0

    dx = int(round(tx - cx))
    dy = int(round(ty - cy))
    if dx == 0 and dy == 0:
        return g

    out = np.zeros_like(g)
    for y in range(n):
        ny = y + dy
        if ny < 0 or ny >= n:
            continue
        for x in range(n):
            nx = x + dx
            if nx < 0 or nx >= n:
                continue
            if g[y, x]:
                out[ny, nx] = 1
    return out

def count_isolated_pixels(g: np.ndarray) -> int:
    n = g.shape[0]
    ys, xs = np.nonzero(g)
    iso = 0
    for y, x in zip(ys, xs):
        ok = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < n and 0 <= nx < n and g[ny, nx]:
                    ok = True
                    break
            if ok:
                break
        if not ok:
            iso += 1
    return iso

def remove_small_components(g: np.ndarray, min_size: int = 2) -> np.ndarray:
    h, w = g.shape
    vis = np.zeros_like(g, dtype=np.uint8)
    out = g.copy()

    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def bfs(sy, sx):
        q = deque([(sy, sx)])
        vis[sy, sx] = 1
        comp = [(sy, sx)]
        while q:
            y, x = q.popleft()
            for dy, dx in nbrs:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and out[yy, xx] and not vis[yy, xx]:
                    vis[yy, xx] = 1
                    q.append((yy, xx))
                    comp.append((yy, xx))
        return comp

    ys, xs = np.nonzero(out)
    for y, x in zip(ys, xs):
        if out[y, x] and not vis[y, x]:
            comp = bfs(y, x)
            if len(comp) < min_size:
                for yy, xx in comp:
                    out[yy, xx] = 0
    return out


# -------------------------
# Spacing
# -------------------------
def step_from_size(base_len: int, gap: int, overlap: int) -> int:
    # step = base_len + gap - overlap
    return max(1, int(base_len) + int(gap) - int(overlap))


# -------------------------
# Symmetric builders
# -------------------------
def build_two_shapes_symmetric(
    rng: random.Random,
    base: List[Coord],
    axis: str,        # "x" LR or "y" UD
    gap: int,
    overlap: int,
    mirror_mode: str  # "same" or "mirror"
) -> List[Coord]:
    A = random_base_variant(rng, base)
    w, h = size_xy(A)

    if axis == "x":
        B = A if mirror_mode == "same" else normalize(flip_lr(A))
        stepx = step_from_size(w, gap, overlap)
        coords = union([A, translate(B, stepx, 0)])
        return normalize(coords)
    else:
        B = A if mirror_mode == "same" else normalize(flip_ud(A))
        stepy = step_from_size(h, gap, overlap)
        coords = union([A, translate(B, 0, stepy)])
        return normalize(coords)

def build_four_shapes_perfect_symmetry(
    rng: random.Random,
    base: List[Coord],
    gapx: int, gapy: int,
    overlapx: int, overlapy: int,
    top_mirror_lr: bool
) -> List[Coord]:
    A = random_base_variant(rng, base)
    w, _ = size_xy(A)

    TR = normalize(flip_lr(A)) if top_mirror_lr else A
    stepx = step_from_size(w, gapx, overlapx)
    top = normalize(union([A, translate(TR, stepx, 0)]))

    bottom = normalize(flip_ud(top))

    _, _, _, top_maxy = bbox_coords(top)
    top_h = top_maxy + 1
    stepy = step_from_size(top_h, gapy, overlapy)

    coords = union([top, translate(bottom, 0, stepy)])
    return normalize(coords)

def build_four_shapes_two_rows(
    rng: random.Random,
    baseA: List[Coord],
    baseB: List[Coord],
    gapx: int, gapy: int,
    overlapx: int, overlapy: int,
    enforce_ud_symmetry: bool
) -> List[Coord]:
    A = random_base_variant(rng, baseA)
    wA, _ = size_xy(A)
    stepxA = step_from_size(wA, gapx, overlapx)

    AR = normalize(flip_lr(A)) if (rng.random() < 0.7) else A
    top = normalize(union([A, translate(AR, stepxA, 0)]))

    if enforce_ud_symmetry:
        bottom = normalize(flip_ud(top))
    else:
        B = random_base_variant(rng, baseB)
        wB, _ = size_xy(B)
        stepxB = step_from_size(wB, gapx, overlapx)
        BR = normalize(flip_lr(B)) if (rng.random() < 0.7) else B
        bottom = normalize(union([B, translate(BR, stepxB, 0)]))

    _, _, _, top_maxy = bbox_coords(top)
    top_h = top_maxy + 1
    stepy = step_from_size(top_h, gapy, overlapy)

    coords = union([top, translate(bottom, 0, stepy)])
    return normalize(coords)

def build_four_shapes_two_cols(
    rng: random.Random,
    baseA: List[Coord],
    baseB: List[Coord],
    gapx: int, gapy: int,
    overlapx: int, overlapy: int,
    enforce_lr_symmetry: bool
) -> List[Coord]:
    A = random_base_variant(rng, baseA)
    _, hA = size_xy(A)
    stepyA = step_from_size(hA, gapy, overlapy)

    AB = normalize(flip_ud(A)) if (rng.random() < 0.7) else A
    col1 = normalize(union([A, translate(AB, 0, stepyA)]))

    if enforce_lr_symmetry:
        col2 = normalize(flip_lr(col1))
    else:
        B = random_base_variant(rng, baseB)
        _, hB = size_xy(B)
        stepyB = step_from_size(hB, gapy, overlapy)
        BB = normalize(flip_ud(B)) if (rng.random() < 0.7) else B
        col2 = normalize(union([B, translate(BB, 0, stepyB)]))

    _, _, col1_maxx, _ = bbox_coords(col1)
    col1_w = col1_maxx + 1
    stepx = step_from_size(col1_w, gapx, overlapx)

    coords = union([col1, translate(col2, stepx, 0)])
    return normalize(coords)


# -------------------------
# Dedupe (rot/mirror canonical) + hamming
# -------------------------
def transforms8_grid(g: np.ndarray) -> Iterable[np.ndarray]:
    r0 = g
    r1 = np.rot90(g, 3)
    r2 = np.rot90(g, 2)
    r3 = np.rot90(g, 1)
    for r in (r0, r1, r2, r3):
        yield r
        yield np.fliplr(r)

def canonical_key(g: np.ndarray) -> bytes:
    return min(t.tobytes() for t in transforms8_grid(g))

def bitset81(g: np.ndarray) -> int:
    # flatten y-major into bits
    bits = 0
    idx = 0
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            if g[y, x]:
                bits |= (1 << idx)
            idx += 1
    return bits

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# -------------------------
# Render
# -------------------------
def grid_to_icon(g: np.ndarray) -> Image.Image:
    n = g.shape[0]
    w = n * ICON_SCALE
    h = n * ICON_SCALE
    img = Image.new("RGBA", (w, h), ICON_BG)
    draw = ImageDraw.Draw(img)

    # y=0 at bottom visually: flip y for display
    for gy in range(n):
        for gx in range(n):
            if g[gy, gx]:
                x0 = gx * ICON_SCALE
                y0 = (n - 1 - gy) * ICON_SCALE
                draw.rectangle([x0, y0, x0 + ICON_SCALE - 1, y0 + ICON_SCALE - 1], fill=ICON_FG)
    return img

def make_contact_sheet(images: List[Image.Image], cols: int) -> Image.Image:
    if not images:
        return Image.new("RGBA", (64, 64), ICON_BG)
    iw, ih = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * iw, rows * ih), ICON_BG)
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        sheet.paste(im, (c * iw, r * ih))
    return sheet


# -------------------------
# Core generator (single pattern)
# -------------------------
def make_tetris_pattern_9x9(
    rng: random.Random,
    max_fill_ratio: float = MAX_FILL_RATIO,
    min_on: int = 1,
    forbid_isolated: bool = FORBID_ISOLATED,
    remove_tiny_components: bool = REMOVE_TINY_COMPONENTS,
    tiny_component_min_size: int = TINY_COMPONENT_MIN_SIZE,
    shape_pool: List[str] | None = None,
    attempts: int = ATTEMPTS_PER_PATTERN
) -> np.ndarray:
    size = CANVAS
    max_on = int(round(size * size * max_fill_ratio))  # 9*9*0.2 => 16
    pool = shape_pool[:] if shape_pool else DEFAULT_SHAPE_POOL

    for _ in range(attempts):
        nshape = rng.choices([2, 4], [40, 60], k=1)[0]
        gapx = rng.choice(GAP_CHOICES)
        gapy = rng.choice(GAP_CHOICES)
        overlapx = rng.choice(OVERLAP_CHOICES)
        overlapy = rng.choice(OVERLAP_CHOICES)

        if nshape == 2:
            key = rng.choice(pool)
            base = SHAPES[key]
            axis = rng.choice(["x", "y"])
            mirror_mode = "mirror" if rng.random() < 0.75 else "same"
            coords = build_two_shapes_symmetric(
                rng, base,
                axis=axis,
                gap=(gapx if axis == "x" else gapy),
                overlap=(overlapx if axis == "x" else overlapy),
                mirror_mode=mirror_mode
            )
        else:
            keyA = rng.choice(pool)
            baseA = SHAPES[keyA]
            mode = rng.choices(["perfect", "two_rows", "two_cols"], [55, 30, 15], k=1)[0]

            if mode == "perfect":
                coords = build_four_shapes_perfect_symmetry(
                    rng, baseA,
                    gapx=gapx, gapy=gapy,
                    overlapx=overlapx, overlapy=overlapy,
                    top_mirror_lr=(rng.random() < 0.85)
                )
            elif mode == "two_rows":
                keyB = rng.choice(pool)
                baseB = SHAPES[keyB]
                coords = build_four_shapes_two_rows(
                    rng, baseA, baseB,
                    gapx=gapx, gapy=gapy,
                    overlapx=overlapx, overlapy=overlapy,
                    enforce_ud_symmetry=(rng.random() < 0.60)
                )
            else:
                keyB = rng.choice(pool)
                baseB = SHAPES[keyB]
                coords = build_four_shapes_two_cols(
                    rng, baseA, baseB,
                    gapx=gapx, gapy=gapy,
                    overlapx=overlapx, overlapy=overlapy,
                    enforce_lr_symmetry=(rng.random() < 0.50)
                )

        g = coords_to_grid(normalize(coords), size=size)
        g = center_bbox_in_canvas(g)

        if remove_tiny_components:
            g = remove_small_components(g, min_size=tiny_component_min_size)
            g = center_bbox_in_canvas(g)

        on = int(np.count_nonzero(g))
        if on < min_on or on > max_on:
            continue

        if forbid_isolated and count_isolated_pixels(g) > 0:
            continue

        # final re-center
        g = center_bbox_in_canvas(g)
        return g

    return np.zeros((size, size), dtype=np.uint8)


# -------------------------
# MAIN (batch generate)
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    rng = random.Random(SEED)
    out_dir = OUT_DIR
    icons_dir = os.path.join(out_dir, "icons")
    ensure_dir(icons_dir)

    max_on = int(round(CANVAS * CANVAS * MAX_FILL_RATIO))
    seen = set()
    bit_pool: List[int] = []
    icons: List[Image.Image] = []

    saved = 0
    attempts = 0
    hard_cap = NUM_PATTERNS * ATTEMPTS_PER_PATTERN

    while saved < NUM_PATTERNS and attempts < hard_cap:
        attempts += 1
        g = make_tetris_pattern_9x9(rng)

        # dedupe
        if ENABLE_DEDUPE:
            key = canonical_key(g)
            if key in seen:
                continue

        if ENABLE_HAMMING_FILTER:
            bs = bitset81(g)
            too_close = False
            for old in bit_pool[-HAMMING_CHECK_LAST:]:
                if hamming(bs, old) < HAMMING_MIN:
                    too_close = True
                    break
            if too_close:
                continue

        if ENABLE_DEDUPE:
            seen.add(key)
        if ENABLE_HAMMING_FILTER:
            bit_pool.append(bs)

        on = int(np.count_nonzero(g))
        img = grid_to_icon(g)
        name = f"px_{saved+1:05d}_n{CANVAS:02d}_o{on:02d}.png"
        img.save(os.path.join(icons_dir, name))
        icons.append(img)
        saved += 1

    sheet = make_contact_sheet(icons, CONTACT_SHEET_COLS)
    sheet.save(os.path.join(out_dir, CONTACT_SHEET_NAME))

    print(f"[DONE] out_dir={out_dir}")
    print(f"[DONE] icons={icons_dir}")
    print(f"[DONE] sheet={os.path.join(out_dir, CONTACT_SHEET_NAME)}")
    print(f"[DONE] saved={saved} attempts={attempts} max_on(20% of 9x9)={max_on} dedupe={ENABLE_DEDUPE} hamming={ENABLE_HAMMING_FILTER}")


if __name__ == "__main__":
    main()
