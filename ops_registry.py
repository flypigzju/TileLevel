# ops_registry.py
import random
from typing import Optional, Tuple, Callable

from ops.expandshrink_basic import op_expand
from ops.expandshrink_ring import op_expand_ring

# 你可以未来把 quadrant expand 也接进来
# from ops.expandshrink_quadrant import op_expand_quadrant

ExpandOp = Callable  # (prev, rng) -> (points, comment)


def choose_group_for_level(
    rng: random.Random,
    group_weights: dict,
    force_group_name: Optional[str] = None
) -> str:
    if force_group_name is not None:
        return force_group_name

    items = [(k, v) for k, v in group_weights.items() if v > 0]
    if not items:
        return "ExpandShrink"

    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for name, w in items:
        acc += w
        if r <= acc:
            return name
    return items[-1][0]


def pick_expand_op(
    rng: random.Random,
    allow_ring: bool,
    ring_prob_in_expand: float,
    plan_e_can_use_ring: bool,
    in_plan: bool
) -> Tuple[ExpandOp, str]:
    """
    在 “E(扩张)” 这个动作内部，选择：标准扩张 或 环形扩张
    """
    if not allow_ring:
        return op_expand, "扩张"

    if in_plan and not plan_e_can_use_ring:
        return op_expand, "扩张(PlanNoRing)"

    if rng.random() < ring_prob_in_expand:
        return op_expand_ring, "环形扩张"

    return op_expand, "扩张"
