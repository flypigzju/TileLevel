# ops/shifts_basic.py
from typing import List, Tuple
from geometry import shift

Point = Tuple[int, int]


def op_shift_dir(prev: List[Point], dx: int, dy: int, cmt: str):
    return shift(prev, dx, dy), cmt


def op_left(prev, rng):
    return op_shift_dir(prev, -1, 0, "左移")


def op_right(prev, rng):
    return op_shift_dir(prev, +1, 0, "右移")


def op_up(prev, rng):
    return op_shift_dir(prev, 0, +1, "上移")


def op_down(prev, rng):
    return op_shift_dir(prev, 0, -1, "下移")


def op_lu(prev, rng):
    return op_shift_dir(prev, -1, +1, "左上移")


def op_ld(prev, rng):
    return op_shift_dir(prev, -1, -1, "左下移")


def op_ru(prev, rng):
    return op_shift_dir(prev, +1, +1, "右上移")


def op_rd(prev, rng):
    return op_shift_dir(prev, +1, -1, "右下移")
