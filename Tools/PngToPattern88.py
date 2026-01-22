import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

# =========================
SELECTED_DIR = r"D:\TileMatch_TruchetPreview\selected"   # 你筛选后的图标放这里
OUT_JSON     = r"D:\TileMatch_TruchetPreview\selected_patterns.json"

GRID = 8
CENTER_IN_9 = True

# 黑色判定阈值（越小越黑）
BLACK_THRESHOLD = 80
MIN_ON_CELLS = 1
# =========================


def micro_from_grid(gx: int, gy: int) -> Tuple[int, int]:
    if CENTER_IN_9:
        return gx * 2 + 2, gy * 2 + 2
    return gx * 2 + 1, gy * 2 + 1


def icon_png_to_cells(p: Path) -> List[Tuple[int, int]]:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    # 先缩到 8x8（用 nearest 保持像素块）
    small = Image.fromarray(lum.astype(np.uint8), mode="L").resize((GRID, GRID), resample=Image.Resampling.NEAREST)
    s = np.array(small)

    # 黑 = 有效
    mask = s <= BLACK_THRESHOLD

    cells = []
    for gy_img in range(GRID):      # 图像坐标 y=0 在上
        for gx in range(GRID):
            if mask[gy_img, gx]:
                gy = (GRID - 1) - gy_img  # 转成 y=0 在下
                cells.append((gx, gy))
    return cells


def main():
    in_dir = Path(SELECTED_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".png"])
    patterns = []
    pid = 1

    for p in files:
        cells = icon_png_to_cells(p)
        if len(cells) < MIN_ON_CELLS:
            continue

        out_cells = [{"x": micro_from_grid(gx, gy)[0], "y": micro_from_grid(gx, gy)[1]} for gx, gy in cells]
        patterns.append({
            "id": pid,
            "gridSize": 9 if CENTER_IN_9 else GRID,
            "cells": out_cells,
            "savedUtcTicks": 0,
            "sourcePng": p.name
        })
        pid += 1

    store = {"nextAutoId": pid, "patterns": patterns}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

    print(f"[DONE] selected_pngs={len(files)} patterns={len(patterns)} -> {OUT_JSON}")


if __name__ == "__main__":
    main()
