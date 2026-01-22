# tile_tetris_repeat_generator.py
# ------------------------------------------------------------
# Standalone 9x9 centered Tetris/Polyomino pattern generator
# - output png previews + contact sheet
# - windmill mode: deterministic precomputed bank (fast, diverse)
# - dedup: EXACT only (no canonical / similarity dedup)
# - REQUIREMENT:
#   * windmill: no extra symmetry requirement
#   * NON-windmill 4-shape modes: MUST be LEFT-RIGHT symmetric
#     (enforced at generation time by symmetrizing + trim+center)
# ------------------------------------------------------------

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw
import time
import uuid

# =========================
# Config
# =========================
OUT_DIR = r"D:\TileMatch_TetrisPreview"
NUM_PATTERNS = 2000
SEED = -1 #12345

CANVAS = 9
CENTER = (CANVAS // 2, CANVAS // 2)  # (4,4)
ICON_SCALE = 16
ICON_BG = (255, 255, 255, 255)
ICON_FG = (0, 0, 0, 255)

CONTACT_SHEET_COLS = 20
CONTACT_SHEET_NAME = "contact_sheet.png"

MAX_FILL_RATIO = 0.20
MAX_ON = int(CANVAS * CANVAS * MAX_FILL_RATIO)  # 16
MIN_ON = 4

# ---- for non-windmill spacing ----
DIST_CHOICES = [-2, -1, 0, 1, 2, 3]
DIST_WEIGHTS = [1, 2, 5, 5, 2, 1]  # prefer 0/1

# ---- windmill precompute ----
WINDMILL_MAX_OFFSET = 4  # for 9x9, max safe offset around center is 4
WINDMILL_ANCHOR_MODE = "square"  # "square" => all (dx,dy) with max(|dx|,|dy|)<=R
WINDMILL_REQUIRE_INBOUNDS = True

# As requested: default off (windmill won't create isolated surprises; fixed shapes)
FORBID_ISOLATED = False

# ---- random base shapes (4x4) ----
ENABLE_RANDOM_BASE_SHAPES = True
RANDOM_BASE_GRID = 4
RANDOM_BASE_COUNT = 120          # 想要多少个随机基础shape
RANDOM_BASE_K_CHOICES = [5, 6]   # 在16格里选多少点
RANDOM_BASE_REQUIRE_CONNECTED = True   # 8邻域连通（含对角线）
RANDOM_BASE_MAX_TRIES = 20000    # 生成时最多尝试次数（防止死循环）
RANDOM_BASE_KEY_PREFIX = "RND"
RANDOM_BASE_SEED_SALT = 99991    # 让随机shape库在同一个SEED下也更稳定/可控


# Mode weights (windmill bigger chunk)
MODE_WEIGHTS = {
    "two_sym": 16,
    "four_full": 16,
    "four_two_rows": 16,
    "four_two_cols": 16,
    "diag_sym": 16,
    "windmill": 20,
}

Coord = Tuple[int, int]

# =========================
# Shapes
# =========================
SHAPES: Dict[str, List[Coord]] = {
    # --- 4 basic starters ---
    "MONO": [(0, 0)],
    "DOMINO": [(0, 0), (1, 0)],
    "TRIO_I": [(0, 0), (1, 0), (2, 0)],
    "TRIO_L": [(0, 0), (0, 1), (1, 0)],

    # --- your custom small patterns ---
    "2_SEP": [(0, 0), (1, 1)],
    "3_SEP": [(0, 0), (1, 1), (2, 2)],
    "4_SEP": [(0, 0), (1, 1), (2, 2), (3, 3)],
    "4_T": [(0, 0), (1, 1), (2, 0)],
    "3_A": [(0, 0), (1, 1), (1, 2)],
    "3_B": [(0, 0), (1, 0), (2, 1)],
    "3_C": [(0, 0), (1, 0), (1, 2)],
    "4_A": [(0, 0), (1, 1), (1, 2), (0, 3)],
    "4_B": [(1, 0), (1, 1), (1, 3), (0, 3)],
    "4_C": [(1, 0), (2, 1), (0, 2), (1, 2)],
    "4_D": [(0, 0), (1, 0), (1, 1), (2, 2)],
    "4_E": [(0, 2), (1, 2), (2, 0), (2, 1)],
    "4_F": [(0, 1), (1, 0), (2, 1), (3, 0)],

    "5_A": [(1,0),(0,1),(0,2),(1,2),(2,2)],
    "5_B": [(0,0),(1,1),(2,2),(1,3),(0,4)],
    "5_C": [(0,3),(1,3),(2,2),(3,1),(2,0)],
    "5_D": [(0,0),(0,1),(1,0),(2,1),(3,2),(3,3)],

    # --- classic tetromino ---
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "L": [(0, 0), (0, 1), (0, 2), (1, 0)],
    "J": [(1, 0), (1, 1), (1, 2), (0, 0)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
}

# Non-windmill sampling pool
SHAPE_POOL = [
    "2_SEP", "3_SEP", "4_SEP", "4_T",
    "3_A", "3_B", "3_C", "4_A", "4_B", "4_C", "4_D", "4_E", "4_F",
    "5_A", "5_B", "5_C", "5_D",
    "DOMINO", "DOMINO", "TRIO_L", "TRIO_I",
    "O", "O",
    "T", "T",
    "L", "J",
    "S", "Z",
    "I",
    "MONO",
]

# ============================================================
# Utils (coords)
# ============================================================

def transforms8_coords(coords: List[Coord]) -> List[List[Coord]]:
    """
    返回 coords 的 8 种等价变换（4旋转 * 可选镜像），用于去重。
    coords 是任意坐标集合（不要求在正方形里），会先 normalize。
    """
    def rot90_local(c):
        return [(y, -x) for (x, y) in c]

    def rotk(c, k):
        out = c
        for _ in range(k % 4):
            out = rot90_local(out)
        return out

    def flip_lr_local(c):
        return [(-x, y) for (x, y) in c]

    base = normalize(coords)
    outs = []
    for k in range(4):
        r = normalize(rotk(base, k))
        outs.append(normalize(r))
        outs.append(normalize(flip_lr_local(r)))
    return outs


def canonical_key_coords(coords: List[Coord]) -> Tuple[Coord, ...]:
    """
    把 coords 归一化后，对8种变换取字典序最小的，作为 canonical key。
    """
    variants = transforms8_coords(coords)
    # 把每个 variant 变成排序后的 tuple
    keys = []
    for v in variants:
        vv = tuple(sorted(v))
        keys.append(vv)
    return min(keys)


def coords_from_bitmask_4x4(mask: int, n: int = 4) -> List[Coord]:
    coords = []
    for i in range(n * n):
        if (mask >> i) & 1:
            x = i % n
            y = i // n
            coords.append((x, y))
    return coords


def is_connected_8(coords: List[Coord]) -> bool:
    if not coords:
        return False
    s = set(coords)
    stack = [coords[0]]
    vis = set([coords[0]])
    while stack:
        x, y = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in s and (nx, ny) not in vis:
                    vis.add((nx, ny))
                    stack.append((nx, ny))
    return len(vis) == len(s)


def generate_random_base_shapes(
    rng: random.Random,
    count: int,
    grid: int = 4,
    k_choices: List[int] = [5, 6],
    require_connected: bool = True,
    max_tries: int = 20000,
    key_prefix: str = "RND"
) -> Dict[str, List[Coord]]:
    """
    在 grid x grid 内随机选 k 个点生成基础shape，并做 canonical 去重。
    返回 dict: key -> coords(list)
    """
    out: Dict[str, List[Coord]] = {}
    seen = set()

    tries = 0
    while len(out) < count and tries < max_tries:
        tries += 1
        k = rng.choice(k_choices)

        # 从 grid*grid 里选 k 个不同格
        picks = rng.sample(range(grid * grid), k)
        mask = 0
        for p in picks:
            mask |= (1 << p)

        coords = coords_from_bitmask_4x4(mask, n=grid)

        # 过滤：连通性（8邻域，含对角）
        if require_connected and not is_connected_8(coords):
            continue

        # canonical 去重（旋转/镜像视为同一形）
        key = canonical_key_coords(coords)
        if key in seen:
            continue
        seen.add(key)

        # 存为 coords（normalize 后更方便后面变换）
        coords_norm = normalize(list(key))
        name = f"{key_prefix}_{len(out)+1:03d}"
        out[name] = coords_norm

    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def bbox(coords: List[Coord]) -> Tuple[int, int, int, int]:
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return min(xs), min(ys), max(xs), max(ys)

def normalize(coords: List[Coord]) -> List[Coord]:
    mnx, mny, _, _ = bbox(coords)
    return [(x - mnx, y - mny) for (x, y) in coords]

def size_xy(coords: List[Coord]) -> Tuple[int, int]:
    mnx, mny, mxx, mxy = bbox(coords)
    return (mxx - mnx + 1, mxy - mny + 1)

def translate(coords: List[Coord], dx: int, dy: int) -> List[Coord]:
    return [(x + dx, y + dy) for (x, y) in coords]

def union(parts: List[List[Coord]]) -> List[Coord]:
    s = set()
    for p in parts:
        for c in p:
            s.add(c)
    return list(s)

def mirror_lr_in_bbox(coords: List[Coord]) -> List[Coord]:
    coords = normalize(coords)
    w, _ = size_xy(coords)
    return [(w - 1 - x, y) for (x, y) in coords]

def mirror_ud_in_bbox(coords: List[Coord]) -> List[Coord]:
    coords = normalize(coords)
    _, h = size_xy(coords)
    return [(x, h - 1 - y) for (x, y) in coords]

def random_oriented_variant(rng: random.Random, base: List[Coord]) -> List[Coord]:
    # rotate about origin in local coords: (x,y)->(y,-x)
    k = rng.randint(0, 3)
    coords = base[:]
    for _ in range(k):
        coords = [(y, -x) for (x, y) in coords]
    coords = normalize(coords)

    if rng.random() < 0.35:
        coords = mirror_lr_in_bbox(coords)
        coords = normalize(coords)
    if rng.random() < 0.20:
        coords = mirror_ud_in_bbox(coords)
        coords = normalize(coords)
    return normalize(coords)


# ============================================================
# Inject random base shapes into SHAPES / SHAPE_POOL
# ============================================================
if ENABLE_RANDOM_BASE_SHAPES:
    # 用一个独立 RNG，避免影响你主生成的随机序列
    rng_shapes = random.Random(SEED if SEED != -1 else None)
    # 加一点 salt，让 shape 库跟图案生成随机性分离
    rng_shapes.seed((SEED if SEED != -1 else random.randrange(1<<30)) + RANDOM_BASE_SEED_SALT)

    rnd_shapes = generate_random_base_shapes(
        rng=rng_shapes,
        count=RANDOM_BASE_COUNT,
        grid=RANDOM_BASE_GRID,
        k_choices=RANDOM_BASE_K_CHOICES,
        require_connected=RANDOM_BASE_REQUIRE_CONNECTED,
        max_tries=RANDOM_BASE_MAX_TRIES,
        key_prefix=RANDOM_BASE_KEY_PREFIX
    )

    # 写入 SHAPES
    SHAPES.update(rnd_shapes)

    # 加入 non-windmill 采样池（你可以加权：比如重复添加几次）
    # 建议：随机shape权重不要太高，否则你会觉得“全是随机块”
    # 这里给一个温和权重：每个随机shape进池一次
    SHAPE_POOL.extend(list(rnd_shapes.keys()))

    print(f"[INFO] injected random base shapes: {len(rnd_shapes)} (grid={RANDOM_BASE_GRID}, k={RANDOM_BASE_K_CHOICES})")

# ============================================================
# Grid helpers
# ============================================================
def coords_to_grid(coords: List[Coord], n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.uint8)
    for (x, y) in coords:
        if 0 <= x < n and 0 <= y < n:
            g[y, x] = 1
    return g

def count_on(g: np.ndarray) -> int:
    return int(np.count_nonzero(g))

def count_isolated_pixels(g: np.ndarray) -> int:
    n = g.shape[0]
    ys, xs = np.nonzero(g)
    iso = 0
    for y, x in zip(ys, xs):
        ok = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
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

def internal_empty_band_max(g: np.ndarray) -> int:
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return 999
    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())
    sub = g[miny:maxy+1, minx:maxx+1]

    row_empty = [int(sub[i, :].sum() == 0) for i in range(sub.shape[0])]
    col_empty = [int(sub[:, j].sum() == 0) for j in range(sub.shape[1])]

    def max_run(a):
        best = cur = 0
        for v in a:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    return max(max_run(row_empty), max_run(col_empty))

def trim_bbox(g: np.ndarray):
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return None
    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())
    return (minx, miny, maxx, maxy)

def trim_and_center_grid(g: np.ndarray, n: int) -> np.ndarray:
    """
    Crop to bbox then paste back centered into n×n.
    """
    bb = trim_bbox(g)
    if bb is None:
        return np.zeros((n, n), dtype=np.uint8)

    minx, miny, maxx, maxy = bb
    crop = g[miny:maxy+1, minx:maxx+1]
    ch, cw = crop.shape

    out = np.zeros((n, n), dtype=np.uint8)
    if cw > n or ch > n:
        crop = crop[:min(ch, n), :min(cw, n)]
        ch, cw = crop.shape

    ox = (n - cw) // 2
    oy = (n - ch) // 2
    ox = max(0, min(ox, n - cw))
    oy = max(0, min(oy, n - ch))

    out[oy:oy+ch, ox:ox+cw] = crop
    return out

def enforce_lr_symmetry_grid(g: np.ndarray, n: int) -> np.ndarray:
    """
    Make the whole 9x9 pattern left-right symmetric, then trim+center.
    """
    gg = np.maximum(g, np.fliplr(g))
    return trim_and_center_grid(gg, n)

def min_chebyshev_dist_to_center(g: np.ndarray, cx: int, cy: int) -> int:
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return 999
    # Chebyshev distance = max(|dx|,|dy|)，适合方格“圈”
    return int(min(max(abs(int(x) - cx), abs(int(y) - cy)) for x, y in zip(xs, ys)))

def center_square_count(g: np.ndarray, cx: int, cy: int, r: int) -> int:
    # 统计中心 (2r+1)x(2r+1) 方块里有多少像素
    x0, x1 = max(0, cx - r), min(g.shape[0] - 1, cx + r)
    y0, y1 = max(0, cy - r), min(g.shape[0] - 1, cy + r)
    return int(np.count_nonzero(g[y0:y1+1, x0:x1+1]))


# ============================================================
# Diagonal symmetry helpers
# ============================================================
def reflect_main_diag(coords: List[Coord]) -> List[Coord]:
    return [(y, x) for (x, y) in coords]

def reflect_anti_diag(coords: List[Coord], n: int) -> List[Coord]:
    return [(n - 1 - y, n - 1 - x) for (x, y) in coords]

def place_near_corner(rng: random.Random, coords: List[Coord], n: int, corner: str, margin_max: int = 3) -> Optional[List[Coord]]:
    coords = normalize(coords)
    w, h = size_xy(coords)
    if w > n or h > n:
        return None

    m = min(margin_max, n - 1)

    if corner == "TL":
        x0 = rng.randint(0, min(m, n - w))
        y0 = rng.randint(max(0, n - h - m), n - h)
    elif corner == "TR":
        x0 = rng.randint(max(0, n - w - m), n - w)
        y0 = rng.randint(max(0, n - h - m), n - h)
    elif corner == "BL":
        x0 = rng.randint(0, min(m, n - w))
        y0 = rng.randint(0, min(m, n - h))
    else:  # "BR"
        x0 = rng.randint(max(0, n - w - m), n - w)
        y0 = rng.randint(0, min(m, n - h))

    placed = translate(coords, x0, y0)
    for x, y in placed:
        if x < 0 or x >= n or y < 0 or y >= n:
            return None
    return placed

def center_square_count(g: np.ndarray, cx: int, cy: int, r: int = 1) -> int:
    """count pixels in center (2r+1)x(2r+1) square"""
    n = g.shape[0]
    x0 = max(0, cx - r)
    x1 = min(n - 1, cx + r)
    y0 = max(0, cy - r)
    y1 = min(n - 1, cy + r)
    return int(np.count_nonzero(g[y0:y1 + 1, x0:x1 + 1]))


def min_chebyshev_dist_to_center(g: np.ndarray, cx: int, cy: int) -> int:
    """min max(|dx|,|dy|) from any ON pixel to center"""
    ys, xs = np.nonzero(g)
    if len(xs) == 0:
        return 999
    return int(min(max(abs(int(x) - cx), abs(int(y) - cy)) for x, y in zip(xs, ys)))


def place_near_center(
    rng: random.Random,
    coords: List[Coord],
    n: int,
    radius: int = 2,
    tries: int = 80
) -> Optional[List[Coord]]:
    """
    Place shape by choosing any pivot point on the shape and snapping it to
    a random anchor near board center (within [-radius..radius]).
    This greatly reduces 'two corners far apart' artifacts.
    """
    coords = normalize(coords)
    pivots = coords[:]  # any point can be pivot
    cx, cy = (n // 2, n // 2)

    for _ in range(tries):
        px, py = rng.choice(pivots)
        ax = cx + rng.randint(-radius, radius)
        ay = cy + rng.randint(-radius, radius)

        placed = [(x - px + ax, y - py + ay) for (x, y) in coords]
        if all(0 <= x < n and 0 <= y < n for (x, y) in placed):
            return placed
    return None


def build_diag_symmetric_pattern(rng: random.Random, n: int) -> Optional[np.ndarray]:
    """
    Diagonal symmetry patterns.
    FIXES:
    - do NOT place at corners; place near center to avoid huge empty middle
    - filter: must touch center area / not too far from center
    - use a diag-specific pool to avoid ultra sparse shapes
    """
    # --- diag specific pool: remove very sparse shapes that cause corner artifacts ---
    DIAG_EXCLUDE = {"MONO", "DOMINO", "2_SEP", "3_SEP", "4_SEP"}
    diag_pool = [k for k in SHAPE_POOL if k in SHAPES and k not in DIAG_EXCLUDE]
    if not diag_pool:
        diag_pool = [k for k in SHAPE_POOL if k in SHAPES]  # fallback

    key = rng.choice(diag_pool)
    base = SHAPES[key]
    A = random_oriented_variant(rng, base)

    diag_mode = rng.choice(["main", "anti"])  # y=x or x+y=n-1

    # place near center instead of corner
    placed = place_near_center(rng, A, n, radius=2, tries=80)
    if placed is None:
        return None

    mirrored = reflect_main_diag(placed) if diag_mode == "main" else reflect_anti_diag(placed, n)
    allc = union([placed, mirrored])

    # bounds safety
    for x, y in allc:
        if x < 0 or x >= n or y < 0 or y >= n:
            return None

    g = coords_to_grid(allc, n)

    # --------- anti "big empty middle" filters ----------
    cx, cy = (n // 2, n // 2)

    # (1) center 3x3 must have at least one pixel (very effective)
    if center_square_count(g, cx, cy, r=1) == 0:
        return None

    # (2) also ensure closest pixel to center is not too far (Chebyshev distance)
    # 2 is a good default for 9x9; 1 is even tighter
    if min_chebyshev_dist_to_center(g, cx, cy) > 2:
        return None

    return g

# ============================================================
# Distance sampling (non-windmill)
# ============================================================
def sample_dist(rng: random.Random, w_or_h: int) -> int:
    dmin = max(-2, 1 - w_or_h)
    choices = [d for d in DIST_CHOICES if d >= dmin]
    weights = [DIST_WEIGHTS[DIST_CHOICES.index(d)] for d in choices]
    return rng.choices(choices, weights, k=1)[0]

def sample_dist_tight(rng: random.Random, w_or_h: int) -> int:
    """
    tighter distance for stacking rows/cols:
    prefer 0/1, allow small overlaps, forbid big gaps.
    """
    # 只允许 [-2..1]，基本杜绝中间空很多行
    tight_choices = [-2, -1, 0, 1]
    tight_weights = [1, 3, 8, 5]  # 强烈偏向 0/1

    dmin = max(-2, 1 - w_or_h)
    choices = [d for d in tight_choices if d >= dmin]
    weights = [tight_weights[tight_choices.index(d)] for d in choices]
    return rng.choices(choices, weights, k=1)[0]

# ============================================================
# Windmill precompute (deterministic enumeration)
# ============================================================
def rot90_about(x: int, y: int, cx: int, cy: int) -> Tuple[int, int]:
    dx = x - cx
    dy = y - cy
    return (cx + dy, cy - dx)

def rotk_about(coords: List[Coord], k: int, cx: int, cy: int) -> List[Coord]:
    out = coords
    for _ in range(k % 4):
        out = [rot90_about(x, y, cx, cy) for (x, y) in out]
    return out

def iter_anchor_points(n: int, cx: int, cy: int, R: int):
    if WINDMILL_ANCHOR_MODE == "square":
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < n and 0 <= y < n:
                    yield (x, y)
    else:
        yield (cx, cy)

def all_windmill_patterns_for_shape(shape_key: str, n: int) -> List[np.ndarray]:
    cx, cy = CENTER
    base = normalize(SHAPES[shape_key])
    pivots = list(base)

    patterns: List[np.ndarray] = []
    seen = set()

    R = min(WINDMILL_MAX_OFFSET, cx, cy, n - 1 - cx, n - 1 - cy)

    for (px, py) in pivots:
        for (ax, ay) in iter_anchor_points(n, cx, cy, R):
            placed0 = [(x - px + ax, y - py + ay) for (x, y) in base]

            allc: List[Coord] = []
            ok = True
            for k in range(4):
                rk = rotk_about(placed0, k, cx, cy)
                allc.extend(rk)

            if WINDMILL_REQUIRE_INBOUNDS:
                for (x, y) in allc:
                    if x < 0 or x >= n or y < 0 or y >= n:
                        ok = False
                        break
            if not ok:
                continue

            g = coords_to_grid(allc, n)
            # ---- reduce big center hole windmills ----
            dmin = min_chebyshev_dist_to_center(g, cx, cy)
            if dmin > 2:     # 2 是很舒服的阈值；想更紧凑用 1
                continue
            # 可选：中心 3x3 至少要有一点像素（更强）
#            if center_square_count(g, cx, cy, r=1) == 0:
#                continue

            on = count_on(g)
            if on < MIN_ON or on > MAX_ON:
                continue

            key = g.tobytes()
            if key in seen:
                continue
            seen.add(key)
            patterns.append(g)

    return patterns

def precompute_windmill_bank(n: int) -> List[np.ndarray]:
    bank: List[np.ndarray] = []
    for k in SHAPES.keys():
        bank.extend(all_windmill_patterns_for_shape(k, n))
    random.shuffle(bank)
    return bank

# ============================================================
# Structured symmetric builders (non-windmill)
# ============================================================
def build_two_shapes(rng: random.Random) -> List[Coord]:
    key = rng.choice(SHAPE_POOL)
    A = random_oriented_variant(rng, SHAPES[key])
    w, h = size_xy(A)

    axis = rng.choice(["LR", "UD"])
    if axis == "LR":
        B = mirror_lr_in_bbox(A)
        d = sample_dist(rng, w)
        stepx = w + d
        coords = union([A, translate(B, stepx, 0)])
    else:
        B = mirror_ud_in_bbox(A)
        d = sample_dist_tight(rng, h)
        stepy = h + d
        coords = union([A, translate(B, 0, stepy)])

    return normalize(coords)

def build_four_shapes_full_sym(rng: random.Random) -> List[Coord]:
    key = rng.choice(SHAPE_POOL)
    A = random_oriented_variant(rng, SHAPES[key])
    w, _ = size_xy(A)

    right = mirror_lr_in_bbox(A)
    d_x = sample_dist(rng, w)
    stepx = w + d_x
    top = normalize(union([A, translate(right, stepx, 0)]))

    top_ud = mirror_ud_in_bbox(top)

    _, top_h = size_xy(top)
    d_y = sample_dist_tight(rng, top_h)
    stepy = top_h + d_y

    coords = union([top, translate(top_ud, 0, stepy)])
    return normalize(coords)

def build_four_shapes_lr_sym_two_rows(rng: random.Random) -> List[Coord]:
    keyA = rng.choice(SHAPE_POOL)
    keyB = rng.choice(SHAPE_POOL)
    A = random_oriented_variant(rng, SHAPES[keyA])
    B = random_oriented_variant(rng, SHAPES[keyB])

    wA, _ = size_xy(A)
    wB, _ = size_xy(B)

    A2 = mirror_lr_in_bbox(A)
    d_x1 = sample_dist(rng, wA)
    row1 = normalize(union([A, translate(A2, wA + d_x1, 0)]))

    B2 = mirror_lr_in_bbox(B)
    d_x2 = sample_dist(rng, wB)
    row2 = normalize(union([B, translate(B2, wB + d_x2, 0)]))

    _, row1_h = size_xy(row1)
    d_y = sample_dist(rng, row1_h)
    coords = union([row1, translate(row2, 0, row1_h + d_y)])
    return normalize(coords)

def build_four_shapes_lr_sym_two_cols(rng: random.Random) -> List[Coord]:
    key = rng.choice(SHAPE_POOL)
    A = random_oriented_variant(rng, SHAPES[key])
    _, h = size_xy(A)

    A2 = mirror_ud_in_bbox(A)
    d_y = sample_dist(rng, h)
    col = normalize(union([A, translate(A2, 0, h + d_y)]))

    col_r = mirror_lr_in_bbox(col)

    col_w, _ = size_xy(col)
    d_x = sample_dist(rng, col_w)
    coords = union([col, translate(col_r, col_w + d_x, 0)])
    return normalize(coords)

def center_in_canvas(coords: List[Coord], n: int) -> Optional[np.ndarray]:
    coords = normalize(coords)
    mnx, mny, mxx, mxy = bbox(coords)
    w = mxx - mnx + 1
    h = mxy - mny + 1
    if w > n or h > n:
        return None

    dx = (n - w) // 2 - mnx
    dy = (n - h) // 2 - mny
    placed = translate(coords, dx, dy)

    for x, y in placed:
        if x < 0 or x >= n or y < 0 or y >= n:
            return None
    return coords_to_grid(placed, n)

# ============================================================
# Render
# ============================================================
def grid_to_icon(g: np.ndarray) -> Image.Image:
    w = CANVAS * ICON_SCALE
    h = CANVAS * ICON_SCALE
    img = Image.new("RGBA", (w, h), ICON_BG)
    draw = ImageDraw.Draw(img)
    for gy in range(CANVAS):
        for gx in range(CANVAS):
            if g[gy, gx]:
                x0 = gx * ICON_SCALE
                y0 = (CANVAS - 1 - gy) * ICON_SCALE
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

# ============================================================
# Filters / finalize
# ============================================================
def passes_filters(g: np.ndarray, allow_band: bool = True) -> bool:
    on = count_on(g)

    # 新增：<=4 直接丢弃（等价于要求 on >= 5）
    if on <= 4:
        return False

    if on < MIN_ON or on > MAX_ON:
        return False

    if FORBID_ISOLATED and count_isolated_pixels(g) > 0:
        return False

    if allow_band:
        if internal_empty_band_max(g) > 4:
            return False

    return True

def weighted_choice(rng: random.Random, table: Dict[str, int]) -> str:
    keys = list(table.keys())
    weights = [table[k] for k in keys]
    return rng.choices(keys, weights, k=1)[0]

def finalize_grid(g: np.ndarray, *, require_lr_sym: bool) -> np.ndarray:
    """
    统一的“保存前处理”：
    - 如果需要左右对称：强制 LR 对称 + trim+center
    - 否则：trim+center
    """
    if require_lr_sym:
        return enforce_lr_symmetry_grid(g, CANVAS)
    return trim_and_center_grid(g, CANVAS)

# ============================================================
# Main
# ============================================================
def main():
    # 每次运行唯一ID：时间戳 + 8位随机
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    
    # SEED = -1  # -1 means random every run
    if SEED == -1:
        rng = random.Random()   # system-seeded
    else:
        rng = random.Random(SEED)    
    # rng = random.Random(SEED)
    out_dir = Path(OUT_DIR) / f"run_{run_id}"
    icons_dir = out_dir / "icons"
    ensure_dir(str(icons_dir))

    # ---- precompute windmill bank ----
    windmill_bank = precompute_windmill_bank(CANVAS)

    # exact dedup across shapes in bank
    bank_seen = set()
    windmill_bank_unique = []
    for gg in windmill_bank:
        kk = gg.tobytes()
        if kk in bank_seen:
            continue
        bank_seen.add(kk)
        windmill_bank_unique.append(gg)
    windmill_bank = windmill_bank_unique

    if not windmill_bank:
        print("[WARN] windmill bank is empty. Check MAX_ON / MIN_ON constraints.")
    else:
        print(f"[INFO] windmill bank size = {len(windmill_bank)}")

    # ---- generation loop ----
    seen_exact = set()
    icons: List[Image.Image] = []
    saved = 0
    attempts = 0

    while saved < NUM_PATTERNS and attempts < NUM_PATTERNS * 200:
        attempts += 1
        mode = weighted_choice(rng, MODE_WEIGHTS)

        g_try: Optional[np.ndarray] = None
        tag = mode
        allow_band = True

        if mode == "windmill" and windmill_bank:
            g_try = windmill_bank[rng.randrange(len(windmill_bank))]
            allow_band = False  # don't band-filter windmill

        elif mode == "two_sym":
            coords = build_two_shapes(rng)
            g_try = center_in_canvas(coords, CANVAS)

        elif mode == "four_full":
            coords = build_four_shapes_full_sym(rng)
            g_try = center_in_canvas(coords, CANVAS)

        elif mode == "four_two_rows":
            coords = build_four_shapes_lr_sym_two_rows(rng)
            g_try = center_in_canvas(coords, CANVAS)

        elif mode == "four_two_cols":
            coords = build_four_shapes_lr_sym_two_cols(rng)
            g_try = center_in_canvas(coords, CANVAS)

        elif mode == "diag_sym":
            g_try = build_diag_symmetric_pattern(rng, CANVAS)
            tag = "diag"

        if g_try is None:
            continue

        # ✅ Requirement: non-windmill 4-shape must be LR symmetric
        require_lr_sym = (mode in ("four_full", "four_two_rows", "four_two_cols"))

        # ✅ Save-finalize: trim then center (and enforce LR symmetry when required)
        g = finalize_grid(g_try, require_lr_sym=require_lr_sym)

        # filters AFTER finalize (because finalize can change pixel count)
        if not passes_filters(g, allow_band=allow_band):
            continue

        # exact-only dedup
        k = g.tobytes()
        if k in seen_exact:
            continue
        seen_exact.add(k)

        img = grid_to_icon(g)
        img.save(icons_dir / f"px_{saved+1:05d}_{tag}_n{CANVAS:02d}_o{count_on(g):02d}.png")
        icons.append(img)
        saved += 1

    sheet = make_contact_sheet(icons, CONTACT_SHEET_COLS)
    ensure_dir(str(out_dir))
    sheet.save(out_dir / f"contact_sheet_{run_id}.png")

    print(f"[DONE] saved={saved} attempts={attempts}")
    print(f"[DONE] icons: {icons_dir}")
    print(f"[DONE] sheet: {out_dir / CONTACT_SHEET_NAME}")
    print(f"[CFG ] CANVAS={CANVAS} MAX_ON={MAX_ON} MIN_ON={MIN_ON} windmill_bank={len(windmill_bank)}")
    print(f"[CFG ] FORBID_ISOLATED={FORBID_ISOLATED} exact_dedup_only=True")
    print("[CFG ] LR-sym enforced for non-windmill 4-shape modes: four_full/four_two_rows/four_two_cols")

if __name__ == "__main__":
    main()
