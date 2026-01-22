# parse_patterns.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Point = Tuple[int, int]  # micro coords


@dataclass
class Pattern:
    level_id: int
    w: int = 9
    h: int = 9
    type_: int = 3
    layers: Dict[int, List[Point]] = None

    def __post_init__(self):
        if self.layers is None:
            self.layers = {}


def clamp_8_9(v: int) -> int:
    return 8 if v <= 8 else 9


def parse_size(s: str) -> Tuple[int, int]:
    s = s.strip().replace("x", ",").replace("X", ",").replace("*", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2:
        try:
            return clamp_8_9(int(parts[0])), clamp_8_9(int(parts[1]))
        except:
            pass
    return 9, 9


def parse_patterns(txt: str) -> Dict[int, Pattern]:
    out: Dict[int, Pattern] = {}
    cur: Optional[Pattern] = None

    for raw in txt.splitlines():
        raw = raw.split("#", 1)[0]  # strip inline comment
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.lower().startswith("level="):
            if cur is not None:
                out[cur.level_id] = cur
            try:
                lid = int(line.split("=", 1)[1].strip())
            except:
                continue
            cur = Pattern(level_id=lid)
            continue

        if cur is None:
            continue

        if line.lower().startswith("size="):
            cur.w, cur.h = parse_size(line.split("=", 1)[1])
            continue

        if line.lower().startswith("type="):
            try:
                cur.type_ = int(line.split("=", 1)[1].strip())
            except:
                cur.type_ = 3
            continue

        if "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            try:
                li = int(lhs)
            except:
                continue

            pts: List[Point] = []
            if rhs:
                for chunk in [c.strip() for c in rhs.split(";") if c.strip()]:
                    xy = [t.strip() for t in chunk.split(",")]
                    if len(xy) >= 2:
                        try:
                            pts.append((int(xy[0]), int(xy[1])))
                        except:
                            pass

            cur.layers[li] = sorted(set(pts), key=lambda t: (t[1], t[0]))

    if cur is not None:
        out[cur.level_id] = cur
    return out


def build_patterns_text(patterns: List[Pattern]) -> str:
    patterns = sorted(patterns, key=lambda p: p.level_id)
    lines: List[str] = []
    for p in patterns:
        lines.append(f"# --- Level {p.level_id} ---")
        lines.append(f"level={p.level_id}")
        lines.append(f"size={p.w},{p.h}")
        lines.append(f"type={p.type_}")

        max_layer = max(p.layers.keys()) if p.layers else 0
        for li in range(max_layer + 1):
            pts = p.layers.get(li, [])
            pts = sorted(set(pts), key=lambda t: (t[1], t[0]))
            s = "; ".join([f"{x},{y}" for x, y in pts])

            cmt = ""
            if hasattr(p, "layer_comments") and li < len(p.layer_comments):
                cmt = p.layer_comments[li] or ""

            if li >= 1 and cmt:
                lines.append(f"{li}= {s} # {cmt}")
            else:
                lines.append(f"{li}= {s}")
        lines.append("")
    return "\n".join(lines)
