# geometry.py
from typing import List, Tuple

Point = Tuple[int, int]  # micro coords


def bbox(points: List[Point]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def center_of_bbox(points: List[Point]) -> Tuple[float, float]:
    mnx, mny, mxx, mxy = bbox(points)
    return (mnx + mxx) * 0.5, (mny + mxy) * 0.5


def shift(points: List[Point], dx: int, dy: int) -> List[Point]:
    return [(x + dx, y + dy) for x, y in points]


def trim_to_board(points: List[Point], bw: int, bh: int) -> List[Point]:
    maxx = bw * 2 - 1
    maxy = bh * 2 - 1
    pts = [(x, y) for (x, y) in points if 1 <= x <= maxx and 1 <= y <= maxy]
    return sorted(set(pts), key=lambda t: (t[1], t[0]))


def span_tiles_from_micro_union(union: List[Point]) -> Tuple[int, int]:
    mnx, mny, mxx, mxy = bbox(union)
    span_w = int(round((mxx - mnx) / 2.0)) + 1
    span_h = int(round((mxy - mny) / 2.0)) + 1
    return max(1, span_w), max(1, span_h)


def suggest_board_size(union: List[Point]) -> Tuple[int, int]:
    # 保持你原来的 8/9 策略
    sw, sh = span_tiles_from_micro_union(union)
    bw = 9 if sw > 8 else (8 if sw % 2 == 0 else 9)
    bh = 9 if sh > 8 else (8 if sh % 2 == 0 else 9)
    return bw, bh


def global_normalize_center(layers: List[List[Point]], bw: int, bh: int) -> List[List[Point]]:
    union: List[Point] = []
    for ly in layers:
        union.extend(ly)
    if not union:
        return layers

    ucx, ucy = center_of_bbox(union)
    target_cx = float(bw)
    target_cy = float(bh)

    dx = int(round(target_cx - ucx))
    dy = int(round(target_cy - ucy))

    return [shift(ly, dx, dy) for ly in layers]


# -----------------------------
# total%3 fix (delete edge tiles)
# -----------------------------

def total_tile_count(layers: List[List[Point]]) -> int:
    return sum(len(ly) for ly in layers)


def _edge_score(p: Point, bw: int, bh: int):
    """
    Higher score => more edge => remove first.
    Priority:
    1) on boundary
    2) Chebyshev distance to center (bw,bh)
    3) Manhattan distance to center
    """
    x, y = p
    maxx = bw * 2 - 1
    maxy = bh * 2 - 1
    on_boundary = 1 if (x == 1 or x == maxx or y == 1 or y == maxy) else 0

    cx, cy = bw, bh  # board center micro (per your convention)
    dx = abs(x - cx)
    dy = abs(y - cy)
    cheb = max(dx, dy)
    manh = dx + dy
    return (on_boundary, cheb, manh)


def _remove_k_edge_tiles(points: List[Point], k: int, bw: int, bh: int):
    if k <= 0 or not points:
        return points, []

    pts = sorted(set(points), key=lambda t: (t[1], t[0]))
    pts_sorted = sorted(pts, key=lambda p: _edge_score(p, bw, bh), reverse=True)

    removed = pts_sorted[:k]
    removed_set = set(removed)
    kept = [p for p in pts if p not in removed_set]
    return kept, removed


def ensure_divisible_by_3(layers: List[List[Point]], comments: List[str], bw: int, bh: int):
    cnt = total_tile_count(layers)
    rem = cnt % 3
    if rem == 0:
        return layers, comments

    need_remove = rem  # rem=1删1, rem=2删2
    layer_indices = list(range(len(layers) - 1, 0, -1)) + [0]  # 优先底层，最后才动 layer0

    for li in layer_indices:
        if need_remove <= 0:
            break
        if not layers[li]:
            continue

        take = min(need_remove, len(layers[li]))
        new_pts, removed = _remove_k_edge_tiles(layers[li], take, bw, bh)
        if removed:
            layers[li] = new_pts
            need_remove -= len(removed)
            if li < len(comments):
                base = comments[li] or ""
                comments[li] = (base + f" | 删{len(removed)}补3").strip()

    new_cnt = total_tile_count(layers)
    if new_cnt % 3 != 0:
        print(f"[WARN] ensure_divisible_by_3 failed: before={cnt}, after={new_cnt} (level may be too small)")

    return layers, comments
