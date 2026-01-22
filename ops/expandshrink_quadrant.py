# ops/expandshrink_quadrant.py
import random
from typing import List, Tuple

Point = Tuple[int, int]


def _center2_from_bbox(points: List[Point]) -> Tuple[int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) + max(xs)), (min(ys) + max(ys))  # cx2, cy2


def op_expand_quadrant(prev: List[Point], rng: random.Random):
    """
    象限外扩（示例版本）：
    - 根据点相对中心属于哪个象限，每个点只往“该象限的外侧”扩 2 个点
    - Q1(右上): (x+1,y+1) & (x+1,y-1)? -> 这里我们让它更“向上/向右”
    - Q2(左上): 更向上/向左
    - Q3(左下): 更向下/向左
    - Q4(右下): 更向下/向右
    你后续要做“风车四象限不同策略”就在这里改。
    """
    if not prev:
        return [], "象限扩张"

    cx2, cy2 = _center2_from_bbox(prev)
    out = set()

    for x, y in prev:
        dx2 = 2 * x - cx2
        dy2 = 2 * y - cy2

        # 归象限
        if dx2 >= 0 and dy2 >= 0:
            # 右上：上+右
            out.add((x + 1, y + 1))
            out.add((x - 1, y + 1))
        elif dx2 < 0 and dy2 >= 0:
            # 左上：上+左
            out.add((x - 1, y + 1))
            out.add((x + 1, y + 1))
        elif dx2 < 0 and dy2 < 0:
            # 左下：下+左
            out.add((x - 1, y - 1))
            out.add((x + 1, y - 1))
        else:
            # 右下：下+右
            out.add((x + 1, y - 1))
            out.add((x - 1, y - 1))

    return list(out), "象限扩张"
