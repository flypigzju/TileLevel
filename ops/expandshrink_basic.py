# ops/expandshrink_basic.py
import random
from typing import List, Tuple

Point = Tuple[int, int]


def op_expand(prev: List[Point], rng: random.Random):
    out = set()
    for x, y in prev:
        out.add((x - 1, y - 1))
        out.add((x - 1, y + 1))
        out.add((x + 1, y - 1))
        out.add((x + 1, y + 1))
    return list(out), "扩张"


def op_shrink(prev: List[Point], rng: random.Random):
    s = set(prev)
    out = set()

    candidates = set()
    for x, y in s:
        candidates.add((x + 1, y + 1))
        candidates.add((x + 1, y - 1))
        candidates.add((x - 1, y + 1))
        candidates.add((x - 1, y - 1))

    for cx, cy in candidates:
        if ((cx - 1, cy - 1) in s and
            (cx + 1, cy - 1) in s and
            (cx - 1, cy + 1) in s and
            (cx + 1, cy + 1) in s):
            out.add((cx, cy))

    return list(out), "收缩"


def count_shrinkable_centers(points: List[Point]) -> int:
    s = set(points)
    if not s:
        return 0

    candidates = set()
    for x, y in s:
        candidates.add((x + 1, y + 1))
        candidates.add((x + 1, y - 1))
        candidates.add((x - 1, y + 1))
        candidates.add((x - 1, y - 1))

    cnt = 0
    for cx, cy in candidates:
        if ((cx - 1, cy - 1) in s and
            (cx + 1, cy - 1) in s and
            (cx - 1, cy + 1) in s and
            (cx + 1, cy + 1) in s):
            cnt += 1
    return cnt
