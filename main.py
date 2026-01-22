# main.py
import random
from pathlib import Path
from typing import List, Tuple, Optional

from parse_patterns import Pattern, parse_patterns, build_patterns_text
from geometry import (
    trim_to_board,
    suggest_board_size,
    global_normalize_center,
    ensure_divisible_by_3,
)
from pattern_analyzer import analyze_top_pattern, allow_ring_expand_for_top

from ops.expandshrink_basic import op_shrink, count_shrinkable_centers
from ops.shifts_basic import (
    op_left, op_right, op_up, op_down,
    op_lu, op_ld, op_ru, op_rd,
)

# Windmill base-driven generator
from ops.windmill_static import build_windmill_base_quadrants, windmill_layer_from_base

from ops_registry import choose_group_for_level, pick_expand_op

Point = Tuple[int, int]

# =========================
# HARD CODED CONFIG
# =========================

INPUT_TXT = r"D:\TileMatch_Levels\base_pattern.txt"
OUTPUT_TXT = r"D:\TileMatch_Levels\output_levels.txt"

TARGET_LEVEL = 0   # 0 = all levels; else specific id
LAYER_COUNT  = 6   # total layers including 0 (layers: 0..LAYER_COUNT-1)

LEVEL_OP_PLAN = {
    1: ["E","E","E","E","E"],
    2: ["E","E","E","E","E"],
    3: ["E","E","E","E","E"],
}

SEED = 20260102
REROLL_MAX = 30

# 你要的：ExpandShrink 50%（其它均分）
GROUP_WEIGHTS = {
    "ExpandShrink": 0.50,
    "HShift":       0.1667,
    "VShift":       0.1667,
    "DiagShift":    0.1667,
}

# -------- quick switches --------
FORCE_GROUP_NAME = None
# FORCE_GROUP_NAME = "ExpandShrink"   # 测试：100% ExpandShrink
# FORCE_GROUP_NAME = "WindmillShift"  # 测试：100% WindmillShift

# -------- RingExpand controls --------
ENABLE_RING_EXPAND = True
RING_EXPAND_REQUIRE_FEATURES = True     # True：只有“对称+中空”才允许 ring
RING_EXPAND_PROB_IN_EXPAND = 1.00       # 测试先 100% ring（你要 50% 就改回 0.50）
# RING_EXPAND_PROB_IN_EXPAND = 0.50
PLAN_E_CAN_USE_RING = True

# -------- Windmill controls --------
WINDMILL_SHIFT_PROB = 1.0               # 测试用 100%，之后改 0.3 之类
WINDMILL_DIR_RANDOM = True              # True: 随机 cw/ccw；False: 用默认 ccw
WINDMILL_DEFAULT_DIR = "ccw"            # "ccw" or "cw"

SHRINK_FIRST_MIN_CENTERS = 8
SHRINK_MIN_RESULT_POINTS = 1

# =========================

OP_GROUPS = {
    "HShift":    [op_left, op_right],
    "VShift":    [op_up, op_down],
    "DiagShift": [op_lu, op_ld, op_ru, op_rd],
    # 注意：WindmillShift 不在这里！它是 base-driven，不走 prev->next op
}

# -------------------------
# Helpers
# -------------------------

def _sorted_pts(pts):
    return sorted(set(pts), key=lambda t: (t[1], t[0]))


def generate_layers(
    top: List[Point],
    layer_count: int,
    rng: random.Random,
    group_name: str,
    op_plan: Optional[List[str]] = None,
    allow_ring_expand: bool = False,
    ring_reason: str = "",
    windmill_dir: Optional[str] = None,
):
    """
    Return layers + comments.

    WindmillShift is special:
      - build base arms from TOP only
      - layer i = shift(base, step=i, dir)
      - does NOT depend on prev
    """
    layers: List[List[Point]] = []
    comments: List[str] = []

    # layer0
    layers.append(_sorted_pts(top))
    comments.append("")

    # ---------- SPECIAL PATH: WindmillShift ----------
    if group_name == "WindmillShift":
        # direction
        d = (windmill_dir or "ccw").lower()
        if d not in ("ccw", "cw"):
            d = "ccw"

        base = build_windmill_base_quadrants(layers[0])

        for li in range(1, layer_count):
            nxt = windmill_layer_from_base(base, step=li, direction=d)
            layers.append(_sorted_pts(nxt))
            comments.append(f"风车旋转{d.upper()} step={li} (Base)")
        return layers, comments
    # -----------------------------------------------

    shrink_centers0 = 0
    if group_name == "ExpandShrink":
        shrink_centers0 = count_shrinkable_centers(layers[0])

    for li in range(1, layer_count):
        prev = layers[-1]
        prev_set = set(prev)
        if not prev_set:
            layers.append([])
            comments.append("空")
            continue

        # ---------- plan ----------
        forced_op = None
        forced_cmt = None
        in_plan = False
        if op_plan is not None:
            step_idx = li - 1
            if step_idx < len(op_plan):
                tag = op_plan[step_idx].upper()
                in_plan = True
                if tag == "E":
                    expand_op, expand_cmt = pick_expand_op(
                        rng,
                        allow_ring=allow_ring_expand,
                        ring_prob_in_expand=RING_EXPAND_PROB_IN_EXPAND,
                        plan_e_can_use_ring=PLAN_E_CAN_USE_RING,
                        in_plan=True,
                    )
                    forced_op = expand_op
                    forced_cmt = f"{expand_cmt}(Plan)"
                elif tag == "S":
                    forced_op = op_shrink
                    forced_cmt = "收缩(Plan)"
                else:
                    forced_op = None
                    forced_cmt = None
                    in_plan = False
        # --------------------------

        forced_first_expand = False

        # ExpandShrink 第一层 shrink 门槛（仅随机模式生效）
        if group_name == "ExpandShrink" and li == 1 and op_plan is None:
            if shrink_centers0 < SHRINK_FIRST_MIN_CENTERS:
                expand_op, expand_cmt = pick_expand_op(
                    rng,
                    allow_ring=allow_ring_expand,
                    ring_prob_in_expand=RING_EXPAND_PROB_IN_EXPAND,
                    plan_e_can_use_ring=PLAN_E_CAN_USE_RING,
                    in_plan=False,
                )
                candidate_ops = [expand_op]
                forced_first_expand = True
            else:
                candidate_ops = [None]  # placeholder
        else:
            candidate_ops = [None]

        # 选择候选 op 列表
        if forced_op is not None:
            ops = [forced_op]
        else:
            if group_name == "ExpandShrink":
                ops = ["_EXPAND_", op_shrink]
            else:
                ops = OP_GROUPS[group_name]

        # 如果刚刚触发了“强制扩张”，覆盖 ops
        if group_name == "ExpandShrink" and li == 1 and op_plan is None and forced_first_expand:
            ops = candidate_ops

        best_pts = None
        best_cmt = None

        for _ in range(REROLL_MAX):
            op = rng.choice(ops)

            if op == "_EXPAND_":
                expand_op, expand_cmt = pick_expand_op(
                    rng,
                    allow_ring=allow_ring_expand,
                    ring_prob_in_expand=RING_EXPAND_PROB_IN_EXPAND,
                    plan_e_can_use_ring=PLAN_E_CAN_USE_RING,
                    in_plan=False,
                )
                nxt, cmt = expand_op(prev, rng)
                cmt = expand_cmt
            else:
                nxt, cmt = op(prev, rng)

            nxt_set = set(nxt)
            if nxt_set == prev_set:
                continue

            ok = True
            if group_name == "ExpandShrink":
                if cmt.startswith("扩张") or cmt.startswith("环形扩张"):
                    ok = len(nxt_set) > len(prev_set)
                elif cmt.startswith("收缩"):
                    ok = len(nxt_set) < len(prev_set) and len(nxt_set) >= SHRINK_MIN_RESULT_POINTS

            # plan 收缩失败兜底：改为 expand
            if forced_op == op_shrink and (not ok):
                expand_op, expand_cmt = pick_expand_op(
                    rng,
                    allow_ring=allow_ring_expand,
                    ring_prob_in_expand=RING_EXPAND_PROB_IN_EXPAND,
                    plan_e_can_use_ring=PLAN_E_CAN_USE_RING,
                    in_plan=True,
                )
                nxt2, _ = expand_op(prev, rng)
                nxt_set2 = set(nxt2)
                if len(nxt_set2) > len(prev_set):
                    best_pts = _sorted_pts(nxt_set2)
                    best_cmt = f"{expand_cmt}(PlanFallback)"
                    break

            if ok:
                best_pts = _sorted_pts(nxt_set)
                best_cmt = forced_cmt if forced_cmt is not None else cmt
                break

            best_pts = _sorted_pts(nxt_set)
            best_cmt = forced_cmt if forced_cmt is not None else cmt

        if best_pts is None:
            # 最终兜底：保持在本组
            if group_name == "ExpandShrink":
                expand_op, expand_cmt = pick_expand_op(
                    rng,
                    allow_ring=allow_ring_expand,
                    ring_prob_in_expand=RING_EXPAND_PROB_IN_EXPAND,
                    plan_e_can_use_ring=PLAN_E_CAN_USE_RING,
                    in_plan=in_plan,
                )
                forced, _ = expand_op(prev, rng)
                best_pts = _sorted_pts(forced)
                best_cmt = forced_cmt or expand_cmt
            else:
                forced, forced_cmt2 = ops[0](prev, rng)
                best_pts = _sorted_pts(forced)
                best_cmt = forced_cmt2

        # comment
        if group_name == "ExpandShrink":
            extra = f"shrinkCenters0={shrink_centers0}"
            extra += f", ringOK={int(allow_ring_expand)}({ring_reason})"
            if forced_first_expand and op_plan is None:
                comments.append(f"{best_cmt} (强制) ({group_name}, {extra})")
            else:
                comments.append(f"{best_cmt} ({group_name}, {extra})")
        else:
            comments.append(f"{best_cmt} ({group_name})")

        layers.append(best_pts)

    return layers, comments


def main():
    rng = random.Random(SEED)

    in_path = Path(INPUT_TXT)
    out_path = Path(OUTPUT_TXT)

    txt = in_path.read_text(encoding="utf-8")
    patterns = parse_patterns(txt)

    ids = sorted(patterns.keys())
    if TARGET_LEVEL > 0:
        ids = [TARGET_LEVEL] if TARGET_LEVEL in patterns else []

    out_patterns: List[Pattern] = []

    expected_steps = max(0, LAYER_COUNT - 1)

    for lid in ids:
        p = patterns[lid]
        top = p.layers.get(0, [])
        if not top:
            continue

        plan = LEVEL_OP_PLAN.get(lid)

        # plan 长度提醒
        if plan is not None and len(plan) != expected_steps:
            print(f"[WARN] level={lid} planLen={len(plan)} but expected {expected_steps} (extra will be ignored)")

        # ---------- analyze ----------
        feat = analyze_top_pattern(top)

        # RingExpand gating
        ring_ok, ring_reason = allow_ring_expand_for_top(
            top,
            enable_ring_expand=ENABLE_RING_EXPAND,
            require_features=RING_EXPAND_REQUIRE_FEATURES,
        )

        # WindmillShift gating
        windmill_ok = (
            feat.get("is_discrete", 0) == 1 and
            feat.get("quadrants_4", 0) == 1 and
            feat.get("rot90_sym", 0) == 1 and
            feat.get("windmill_arm_ok", 0) == 1
        )

        # ---------- choose group ----------
        if plan is not None:
            group_name = "ExpandShrink"
        else:
            if FORCE_GROUP_NAME is not None:
                group_name = FORCE_GROUP_NAME
            else:
                if windmill_ok and rng.random() < WINDMILL_SHIFT_PROB:
                    group_name = "WindmillShift"
                else:
                    group_name = choose_group_for_level(
                        rng,
                        GROUP_WEIGHTS,
                        force_group_name=None
                    )

        # pick windmill dir (only used when WindmillShift)
        windmill_dir = None
        if group_name == "WindmillShift":
            if WINDMILL_DIR_RANDOM:
                windmill_dir = "cw" if rng.random() < 0.5 else "ccw"
            else:
                windmill_dir = WINDMILL_DEFAULT_DIR

        layers, comments = generate_layers(
            top=top,
            layer_count=max(1, LAYER_COUNT),
            rng=rng,
            group_name=group_name,
            op_plan=plan,
            allow_ring_expand=ring_ok,
            ring_reason=ring_reason,
            windmill_dir=windmill_dir,
        )

        # ---------- board size ----------
        union: List[Point] = []
        for ly in layers:
            union.extend(ly)
        if not union:
            continue

        bw, bh = suggest_board_size(union)

        # normalize + trim
        layers = global_normalize_center(layers, bw, bh)
        layers = [trim_to_board(ly, bw, bh) for ly in layers]

        # enforce %3
        layers, comments = ensure_divisible_by_3(layers, comments, bw, bh)

        newp = Pattern(level_id=lid, w=bw, h=bh, type_=p.type_)
        newp.layer_comments = comments
        for i, ly in enumerate(layers):
            newp.layers[i] = ly
        out_patterns.append(newp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_patterns_text(out_patterns), encoding="utf-8")
    print(f"[DONE] wrote: {out_path}")
    print(f"[DONE] patterns={len(out_patterns)} layers={LAYER_COUNT} seed={SEED}")


if __name__ == "__main__":
    main()
