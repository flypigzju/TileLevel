import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# =========================
# Hardcoded paths 
# =========================
INPUT_JSON = r"D:\UnityProjects\ZenMatchURP\Assets\Resources\TileMatch\Levels\level_pattern_1.json"
OUT_TXT    = r"D:\UnityProjects\ZenMatchURP\Assets\Resources\TileMatch\Levels\level_pattern_simple_1.txt"

# 固定写入的 tile 种类数（你说先统一写 3）
FIXED_TYPE_COUNT = 3

# 输出排序方式
SORT_BY_ID = True          # True: 按 pattern.id 排序；False: 按 JSON 原顺序
RENUMBER_LEVELS = False    # True: level 从 1..N 重新编号；False: level=pattern.id

# cell 输出排序（让文本更稳定）
CELL_SORT = "yx"           # "yx" / "xy" / None
# =========================


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sort_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if CELL_SORT is None:
        return cells
    if CELL_SORT == "yx":
        return sorted(cells, key=lambda c: (int(c["y"]), int(c["x"])))
    if CELL_SORT == "xy":
        return sorted(cells, key=lambda c: (int(c["x"]), int(c["y"])))
    return cells


def format_level_block(level_id: int, cells: List[Dict[str, Any]]) -> str:
    # 只保留 micro x,y，不要 type id
    parts = [f'{int(c["x"])},{int(c["y"])}' for c in sort_cells(cells)]
    line0 = "0= " + "; ".join(parts) if parts else "0="

    return "\n".join([
        f"# --- Level {level_id} ---",
        f"level={level_id}",
        f"type={FIXED_TYPE_COUNT}",
        line0,
        ""
    ])


def main():
    data = load_json(INPUT_JSON)
    patterns = data.get("patterns", [])
    if not isinstance(patterns, list):
        raise ValueError("Invalid json: patterns is not a list")

    # 排序
    if SORT_BY_ID:
        patterns = sorted(patterns, key=lambda p: int(p.get("id", 0)))

    out_lines: List[str] = []

    for idx, p in enumerate(patterns):
        pid = int(p.get("id", idx + 1))
        level_id = (idx + 1) if RENUMBER_LEVELS else pid

        cells = p.get("cells", [])
        if not isinstance(cells, list):
            cells = []

        out_lines.append(format_level_block(level_id, cells))

    out_text = "\n".join(out_lines)

    Path(os.path.dirname(OUT_TXT)).mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write(out_text)

    print(f"[DONE] patterns={len(patterns)}")
    print(f"[DONE] wrote: {OUT_TXT}")


if __name__ == "__main__":
    main()
