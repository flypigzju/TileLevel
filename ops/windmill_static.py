# ops/windmill_static.py
# ------------------------------------------------------------
# Windmill "static-base" generator:
# - Build 4 quadrant base lists from TOP pattern (exclude center + axis)
# - Each layer i uses the SAME base lists, shifted by i micro-cells (step=i)
# - Direction: CCW or CW
# ------------------------------------------------------------

from typing import Dict, List, Tuple

Point = Tuple[int, int]


def _bbox(points: List[Point]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _center2_from_bbox(points: List[Point]) -> Tuple[int, int]:
    mnx, mny, mxx, mxy = _bbox(points)
    return mnx + mxx, mny + mxy  # cx2, cy2


def _is_center(x: int, y: int, cx2: int, cy2: int) -> bool:
    return (2 * x == cx2) and (2 * y == cy2)


def _is_on_axis(x: int, y: int, cx2: int, cy2: int) -> bool:
    # axis points are not arms
    return (2 * x == cx2) or (2 * y == cy2)


def _quadrant(x: int, y: int, cx2: int, cy2: int) -> int:
    # assumes not on axis
    dx2 = 2 * x - cx2
    dy2 = 2 * y - cy2
    if dx2 > 0 and dy2 > 0:
        return 1
    if dx2 < 0 and dy2 > 0:
        return 2
    if dx2 < 0 and dy2 < 0:
        return 3
    return 4


def build_windmill_base_quadrants(top: List[Point]) -> Dict[int, List[Point]]:
    """
    Returns dict: {1:[...],2:[...],3:[...],4:[...]}
    Excludes:
      - exact center point (if exists)
      - axis points (x==cx or y==cy)
    """
    if not top:
        return {1: [], 2: [], 3: [], 4: []}

    cx2, cy2 = _center2_from_bbox(top)
    q = {1: [], 2: [], 3: [], 4: []}

    s = set(top)
    for (x, y) in s:
        if _is_center(x, y, cx2, cy2):
            continue
        if _is_on_axis(x, y, cx2, cy2):
            continue
        qi = _quadrant(x, y, cx2, cy2)
        q[qi].append((x, y))

    # stable order (optional)
    for k in q.keys():
        q[k] = sorted(q[k], key=lambda t: (t[1], t[0]))

    return q


def _vec_ccw(q: int) -> Tuple[int, int]:
    # 你定义的“逆时针”：Q1下 Q2右 Q3上 Q4左
    if q == 1: return (0, -1)
    if q == 2: return (+1, 0)
    if q == 3: return (0, +1)
    return (-1, 0)


def _vec_cw(q: int) -> Tuple[int, int]:
    # 与上面相反
    if q == 1: return (0, +1)
    if q == 2: return (-1, 0)
    if q == 3: return (0, -1)
    return (+1, 0)


def windmill_layer_from_base(
    base_quads: Dict[int, List[Point]],
    step: int,
    direction: str,  # "ccw" or "cw"
) -> List[Point]:
    out = set()
    for qi, pts in base_quads.items():
        vx, vy = (_vec_ccw(qi) if direction == "ccw" else _vec_cw(qi))
        dx = vx * step
        dy = vy * step
        for x, y in pts:
            out.add((x + dx, y + dy))
    return sorted(out, key=lambda t: (t[1], t[0]))
