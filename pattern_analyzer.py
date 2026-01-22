# pattern_analyzer.py
# ------------------------------------------------------------
# Analyze a "top pattern" (list of micro points) and produce features:
# - LR / UD symmetry
# - Rot90 symmetry ("windmill")
# - Connectedness / components
# - Holes (enclosed empty regions) count
# - Quadrant distribution (incl. arm-only counts excluding center + axis)
# - Windmill-arm constraint: excluding center, each quadrant count >= MIN
#
# Notes:
# - All computations are on "micro coords" (int,int).
# - Rotation/symmetry use bbox center (cx2, cy2) = (minx+maxx, miny+maxy)
#   so we can stay integer-safe.
# ------------------------------------------------------------

from typing import Dict, List, Tuple
from collections import deque

Point = Tuple[int, int]


# =======================
# Basic bbox / centers
# =======================

def bbox(points: List[Point]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def center2_from_bbox(points: List[Point]) -> Tuple[int, int]:
    """
    Return (cx2, cy2) where center = (cx2/2, cy2/2)
    Using cx2 = minx + maxx, cy2 = miny + maxy so reflection/rotation stays integer-safe.
    """
    mnx, mny, mxx, mxy = bbox(points)
    return mnx + mxx, mny + mxy


def is_exact_center(p: Point, cx2: int, cy2: int) -> bool:
    x, y = p
    return (2 * x == cx2) and (2 * y == cy2)


# =======================
# Symmetry checks
# =======================

def is_lr_sym(points: List[Point]) -> bool:
    s = set(points)
    if not s:
        return False
    cx2, _ = center2_from_bbox(points)
    for x, y in s:
        mx = cx2 - x
        if (mx, y) not in s:
            return False
    return True


def is_ud_sym(points: List[Point]) -> bool:
    s = set(points)
    if not s:
        return False
    _, cy2 = center2_from_bbox(points)
    for x, y in s:
        my = cy2 - y
        if (x, my) not in s:
            return False
    return True


def is_rot90_sym(points: List[Point]) -> bool:
    """
    90° rotation symmetry about bbox center.
    Use doubled coordinates to avoid float:

      let cx2 = 2*cx, cy2 = 2*cy
      x2 = 2*x, y2 = 2*y
      dx2 = x2 - cx2, dy2 = y2 - cy2
      rotate 90°: (dx2,dy2) -> (-dy2, dx2)
      x2' = cx2 - dy2
      y2' = cy2 + dx2
      x' = x2'/2, y' = y2'/2

    If x2' or y2' odd -> not possible in integer grid.
    """
    s = set(points)
    if not s:
        return False
    cx2, cy2 = center2_from_bbox(points)

    for x, y in s:
        x2 = 2 * x
        y2 = 2 * y
        dx2 = x2 - cx2
        dy2 = y2 - cy2

        x2r = cx2 - dy2
        y2r = cy2 + dx2

        if (x2r & 1) != 0 or (y2r & 1) != 0:
            return False

        xr = x2r // 2
        yr = y2r // 2
        if (xr, yr) not in s:
            return False

    return True


# =======================
# Grid compression for connectivity & holes
# =======================

def compress_axes(points: List[Point]):
    xs = sorted(set(p[0] for p in points))
    ys = sorted(set(p[1] for p in points))
    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_j = {y: j for j, y in enumerate(ys)}
    return xs, ys, x_to_i, y_to_j


def build_occupancy(points: List[Point]):
    """
    Occupancy on compressed grid:
      occ[j][i] = True if point exists at that (x_i, y_j)
    """
    xs, ys, x_to_i, y_to_j = compress_axes(points)
    w = len(xs)
    h = len(ys)
    occ = [[False] * w for _ in range(h)]
    for x, y in points:
        occ[y_to_j[y]][x_to_i[x]] = True
    return occ, xs, ys


def neighbors4(i: int, j: int, w: int, h: int):
    if i > 0:     yield i - 1, j
    if i + 1 < w: yield i + 1, j
    if j > 0:     yield i, j - 1
    if j + 1 < h: yield i, j + 1


# =======================
# Connectivity (components)
# =======================

def count_components(points: List[Point]) -> Dict[str, int]:
    """
    Components computed on compressed grid with 4-neighbor adjacency.
    """
    if not points:
        return {"components": 0, "largest": 0}

    occ, xs, ys = build_occupancy(points)
    h = len(occ)
    w = len(occ[0]) if h else 0

    vis = [[False] * w for _ in range(h)]
    comps = 0
    largest = 0

    for j in range(h):
        for i in range(w):
            if not occ[j][i] or vis[j][i]:
                continue
            comps += 1
            q = deque([(i, j)])
            vis[j][i] = True
            size = 0
            while q:
                ci, cj = q.popleft()
                size += 1
                for ni, nj in neighbors4(ci, cj, w, h):
                    if occ[nj][ni] and not vis[nj][ni]:
                        vis[nj][ni] = True
                        q.append((ni, nj))
            largest = max(largest, size)

    return {"components": comps, "largest": largest}


# =======================
# Holes (enclosed empty regions)
# =======================

def count_holes(points: List[Point]) -> Dict[str, int]:
    """
    Hole detection on compressed grid:
    - Treat occ cells as "solid"
    - Flood fill empty cells from the BORDER -> those are outside air
    - Remaining empty cells are enclosed -> holes
    Count connected components of enclosed empty.
    """
    if not points:
        return {"has_hole": 0, "holes_count": 0}

    occ, xs, ys = build_occupancy(points)
    h = len(occ)
    w = len(occ[0]) if h else 0

    vis = [[False] * w for _ in range(h)]  # visited empty
    q = deque()

    def try_push(i, j):
        if 0 <= i < w and 0 <= j < h and (not occ[j][i]) and (not vis[j][i]):
            vis[j][i] = True
            q.append((i, j))

    # start from border empty cells
    for i in range(w):
        try_push(i, 0)
        try_push(i, h - 1)
    for j in range(h):
        try_push(0, j)
        try_push(w - 1, j)

    # flood fill outside air
    while q:
        ci, cj = q.popleft()
        for ni, nj in neighbors4(ci, cj, w, h):
            try_push(ni, nj)

    # count enclosed empty components (holes)
    holes = 0
    for j in range(h):
        for i in range(w):
            if occ[j][i] or vis[j][i]:
                continue
            holes += 1
            qq = deque([(i, j)])
            vis[j][i] = True
            while qq:
                ci, cj = qq.popleft()
                for ni, nj in neighbors4(ci, cj, w, h):
                    if (not occ[nj][ni]) and (not vis[nj][ni]):
                        vis[nj][ni] = True
                        qq.append((ni, nj))

    return {"has_hole": 1 if holes > 0 else 0, "holes_count": holes}


# =======================
# Quadrants (distribution + arm-only counts)
# =======================

def quadrant_stats(points: List[Point]) -> Dict:
    """
    Quadrants relative to bbox center (cx,cy).
    Use doubled center (cx2,cy2) to avoid float.
    Q1: x>cx, y>cy
    Q2: x<cx, y>cy
    Q3: x<cx, y<cy
    Q4: x>cx, y<cy
    Axis (x==cx or y==cy) => axis bucket (not counted into quadrants)
    """
    if not points:
        return {
            "q_mask": 0,
            "q1": 0, "q2": 0, "q3": 0, "q4": 0,
            "axis": 0,
            "quadrants_4": 0,
        }

    cx2, cy2 = center2_from_bbox(points)

    q1 = q2 = q3 = q4 = axis = 0
    for x, y in points:
        x2 = 2 * x
        y2 = 2 * y

        if x2 == cx2 or y2 == cy2:
            axis += 1
            continue

        if x2 > cx2 and y2 > cy2:
            q1 += 1
        elif x2 < cx2 and y2 > cy2:
            q2 += 1
        elif x2 < cx2 and y2 < cy2:
            q3 += 1
        elif x2 > cx2 and y2 < cy2:
            q4 += 1
        else:
            axis += 1

    mask = 0
    if q1 > 0: mask |= 1
    if q2 > 0: mask |= 2
    if q3 > 0: mask |= 4
    if q4 > 0: mask |= 8

    return {
        "q_mask": mask,
        "q1": q1, "q2": q2, "q3": q3, "q4": q4,
        "axis": axis,
        "quadrants_4": 1 if mask == 0b1111 else 0,
    }


def _quadrant_by_center2(x: int, y: int, cx2: int, cy2: int) -> int:
    """
    Quadrant by sign of dx2/dy2 (EXCLUDING AXIS):
      Q1 (+,+) top-right
      Q2 (-,+) top-left
      Q3 (-,-) bottom-left
      Q4 (+,-) bottom-right

    If on axis (dx2==0 or dy2==0) => return 0
    """
    dx2 = 2 * x - cx2
    dy2 = 2 * y - cy2
    if dx2 == 0 or dy2 == 0:
        return 0
    if dx2 > 0 and dy2 > 0:
        return 1
    if dx2 < 0 and dy2 > 0:
        return 2
    if dx2 < 0 and dy2 < 0:
        return 3
    return 4


def quadrant_counts_excluding_center(points: List[Point]) -> Dict[str, int]:
    """
    Count tiles in each quadrant, excluding exact center point (if exists),
    and excluding axis tiles (x==cx or y==cy).

    Returns:
      {
        "q1","q2","q3","q4",
        "axis",
        "has_center"
      }
    """
    if not points:
        return {"q1": 0, "q2": 0, "q3": 0, "q4": 0, "axis": 0, "has_center": 0}

    cx2, cy2 = center2_from_bbox(points)
    s = set(points)

    has_center = 1 if any(is_exact_center(p, cx2, cy2) for p in s) else 0
    q = {1: 0, 2: 0, 3: 0, 4: 0}
    axis = 0

    for (x, y) in s:
        if is_exact_center((x, y), cx2, cy2):
            continue
        qi = _quadrant_by_center2(x, y, cx2, cy2)
        if qi == 0:
            axis += 1
            continue
        q[qi] += 1

    return {"q1": q[1], "q2": q[2], "q3": q[3], "q4": q[4], "axis": axis, "has_center": has_center}


def windmill_arm_ok(points: List[Point], min_per_quadrant: int = 3) -> Tuple[bool, str]:
    """
    Windmill-arm constraint:
    - excluding center, each quadrant count >= min_per_quadrant
    - axis points do NOT contribute to arms
    """
    qc = quadrant_counts_excluding_center(points)
    m = min(qc["q1"], qc["q2"], qc["q3"], qc["q4"])
    ok = m >= min_per_quadrant
    reason = f"q=({qc['q1']},{qc['q2']},{qc['q3']},{qc['q4']}),axis={qc['axis']},min={m},need>={min_per_quadrant}"
    return ok, reason


# =======================
# Main analyze API
# =======================

def analyze_top_pattern(top: List[Point], windmill_arm_min: int = 3) -> Dict:
    """
    Return a feature dict for your generator to decide which ops are suitable.
    """
    if not top:
        return {"ok": 0, "reason": "empty"}

    s = set(top)
    mnx, mny, mxx, mxy = bbox(top)
    cx2, cy2 = center2_from_bbox(top)

    lr = is_lr_sym(top)
    ud = is_ud_sym(top)
    lrud = lr and ud
    rot90 = is_rot90_sym(top)

    comp = count_components(list(s))
    holes = count_holes(list(s))

    total = len(s)
    xs, ys, _, _ = compress_axes(list(s))
    grid_w = len(xs)
    grid_h = len(ys)
    density = total / max(1, grid_w * grid_h)

    comps = comp["components"]

    quad_all = quadrant_stats(list(s))
    qc_arm = quadrant_counts_excluding_center(list(s))
    arm_ok, arm_reason = windmill_arm_ok(list(s), min_per_quadrant=windmill_arm_min)

    # "windmill candidate" (you can use this directly in main gating)
    windmill_candidate = bool(rot90) and (comps > 1) and bool(quad_all["quadrants_4"]) and bool(arm_ok)

    return {
        "ok": 1,

        # bbox / center
        "bbox": (mnx, mny, mxx, mxy),
        "center2": (cx2, cy2),

        # counts
        "tiles": total,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "density": density,

        # symmetry
        "lr_sym": 1 if lr else 0,
        "ud_sym": 1 if ud else 0,
        "lr_ud_sym": 1 if lrud else 0,
        "rot90_sym": 1 if rot90 else 0,

        # topology
        "components": comps,
        "largest_component": comp["largest"],
        "is_connected": 1 if comps == 1 else 0,
        "is_discrete": 1 if comps > 1 else 0,

        # holes
        "has_hole": holes["has_hole"],
        "holes_count": holes["holes_count"],

        # quadrants (all points)
        "q_mask": quad_all["q_mask"],
        "q1": quad_all["q1"], "q2": quad_all["q2"], "q3": quad_all["q3"], "q4": quad_all["q4"],
        "axis": quad_all["axis"],
        "quadrants_4": quad_all["quadrants_4"],

        # windmill-arm (excluding center + axis)
        "arm_has_center": qc_arm["has_center"],
        "arm_axis": qc_arm["axis"],
        "arm_q1": qc_arm["q1"], "arm_q2": qc_arm["q2"], "arm_q3": qc_arm["q3"], "arm_q4": qc_arm["q4"],
        "arm_min": min(qc_arm["q1"], qc_arm["q2"], qc_arm["q3"], qc_arm["q4"]),
        "windmill_arm_ok": 1 if arm_ok else 0,
        "windmill_arm_reason": arm_reason,

        # convenience flag
        "windmill_candidate": 1 if windmill_candidate else 0,
    }


# =======================
# Compatibility helpers
# =======================

def allow_ring_expand_for_top(top: List[Point], enable_ring_expand: bool = True, require_features: bool = True):
    """
    RingExpand gating:
    - enable switch
    - if require_features: must be LR+UD symmetric and have a hole
    """
    if not enable_ring_expand:
        return False, "RingOff"

    feat = analyze_top_pattern(top)

    if not require_features:
        return True, "RingAny"

    ok = bool(feat["lr_sym"]) and bool(feat["ud_sym"]) and bool(feat["has_hole"])
    reason = f"lr={int(feat['lr_sym'])},ud={int(feat['ud_sym'])},hole={int(feat['has_hole'])}"
    return ok, reason
