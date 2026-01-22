import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

# =========================
# HARD CODED CONFIG
# =========================
INPUT_DIR = r"D:\TileMatch_TetrisPreview\run_20260101_151403_2c7840f6\icons"              # 你的 144x144 icon 输出目录
OUT_JSON  = r"D:\TileMatch_TetrisPreview\level_pattern.json" # 输出配置文件（JSON）

TARGET_GRID = 9
BLACK_THRESHOLD = 80     # 0~255，越大越容易判定为黑
DROP_IF_ON_LEQ = 4       # 像素点 <=4 丢弃
# =========================


def save_pattern_file(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def micro_from_grid(gx: int, gy: int) -> Tuple[int, int]:
    # 你的编辑器微坐标规则：centerMicro = gx*2+1, gy*2+1
    return gx * 2 + 1, gy * 2 + 1


def png_to_grid_cells_black(png_path: Path, grid_n: int) -> List[Tuple[int, int]]:
    """
    读一张 icon PNG，把“黑色块”转成 grid_n x grid_n 占用格 (gx,gy)
    gy=0 在下（符合你 board 的坐标习惯）
    """
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)  # H,W,4
    h, w = arr.shape[0], arr.shape[1]

    rgb = arr[..., :3].astype(np.float32)
    a   = arr[..., 3].astype(np.float32)

    # 亮度（越小越黑）
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])

    black_mask = (lum <= BLACK_THRESHOLD) & (a > 0)

    grid = np.zeros((grid_n, grid_n), dtype=np.uint8)

    # 比例切块（兼容不是 144x144 的情况）
    for gy in range(grid_n):
        for gx in range(grid_n):
            x0 = int(round(gx * w / grid_n))
            x1 = int(round((gx + 1) * w / grid_n))
            y0 = int(round(gy * h / grid_n))
            y1 = int(round((gy + 1) * h / grid_n))

            x1 = max(x1, x0 + 1)
            y1 = max(y1, y0 + 1)

            if black_mask[y0:y1, x0:x1].any():
                grid[gy, gx] = 1

    # 图片 y=0 在上；你要 gy=0 在下 → 翻转
    cells: List[Tuple[int, int]] = []
    for gy in range(grid_n):
        for gx in range(grid_n):
            if grid[gy, gx] == 1:
                gy_bottom = (grid_n - 1) - gy
                cells.append((gx, gy_bottom))
    return cells


def main():
    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {in_dir}")

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".png"])

    # ✅ 每次运行都从 1 开始，且覆盖输出文件
    patterns: List[Dict[str, Any]] = []
    next_id = 1

    added = 0
    skipped_small = 0

    for p in files:
        cells = png_to_grid_cells_black(p, TARGET_GRID)
        on = len(cells)

        if on <= DROP_IF_ON_LEQ:
            skipped_small += 1
            continue

        pid = next_id
        next_id += 1

        out_cells = [{"x": mx, "y": my} for (mx, my) in (micro_from_grid(gx, gy) for gx, gy in cells)]

        patterns.append({
            "id": int(pid),
            "gridSize": TARGET_GRID,   # 9
            "cells": out_cells,
            "savedUtcTicks": 0
        })
        added += 1

    data = {
        "nextAutoId": int(next_id),
        "patterns": patterns
    }
    save_pattern_file(OUT_JSON, data)

    print(f"[DONE] input pngs={len(files)} added_patterns={added} skipped_on<= {DROP_IF_ON_LEQ} : {skipped_small}")
    print(f"[DONE] wrote(overwrite): {OUT_JSON}")
    print(f"[DONE] nextAutoId={next_id}")


if __name__ == "__main__":
    main()
