# ops/expandshrink_ring.py
from typing import List, Tuple

Point = Tuple[int, int]


def bbox(points: List[Point]):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def center2_from_bbox(points: List[Point]) -> Tuple[int, int]:
    mnx, mny, mxx, mxy = bbox(points)
    return (mnx + mxx), (mny + mxy)  # cx2, cy2


def op_expand_ring(prev: List[Point], rng):
    """
    环形外扩（保持“中空趋势”的扩张方式）：
    - 先用 bbox 中心把点划分为 上/下/左/右
    - 上侧点：生成 (x-1,y+1) (x+1,y+1)
    - 下侧点：生成 (x-1,y-1) (x+1,y-1)
    - 左侧点：生成 (x-1,y-1) (x-1,y+1)
    - 右侧点：生成 (x+1,y-1) (x+1,y+1)
    """
    if not prev:
        return [], "环形扩张"

    cx2, cy2 = center2_from_bbox(prev)
    out = set()

    for x, y in prev:
        dx2 = 2 * x - cx2
        dy2 = 2 * y - cy2

        # 更偏垂直就归到 上/下，否则归到 左/右
        if abs(dy2) >= abs(dx2):
            if dy2 >= 0:
                # 上侧：往上扩
                out.add((x - 1, y + 1))
                out.add((x + 1, y + 1))
            else:
                # 下侧：往下扩
                out.add((x - 1, y - 1))
                out.add((x + 1, y - 1))
        else:
            if dx2 >= 0:
                # 右侧：往右扩
                out.add((x + 1, y - 1))
                out.add((x + 1, y + 1))
            else:
                # 左侧：往左扩
                out.add((x - 1, y - 1))
                out.add((x - 1, y + 1))

    return list(out), "环形扩张"
