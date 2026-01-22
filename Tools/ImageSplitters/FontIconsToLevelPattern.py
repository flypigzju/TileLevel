import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

# =========================
# User Config
# =========================
INPUT_DIR = r"D:\游戏开发\TileMatch\public_pixel\output"  # 你的 glyph png 目录
OUT_JSON  = r"D:\游戏开发\TileMatch\public_pixel\level_pattern"

# 目标 pattern 尺寸
TARGET_GRID = 8

# 放到 9x9 board 的方式：居中（8x8 放进 9x9 会左右/上下各留 0.5 格）
CENTER_IN_9 = True  # True: gx/gy 加 0.5 格偏移；False: 从(0,0)开始贴左下

# 白色判定阈值：像素越接近白越“有效”
WHITE_THRESHOLD = 200   # 0~255，>200 视为白
MIN_ON_CELLS = 1        # 至少多少个格子为 1 才导出

# 若你想把 patternId 与 unicode 对应，可启用从文件名解析 U+XXXX
USE_UNICODE_AS_ID = False  # True: pattern.id = codepoint (十进制)；False: 自动递增
# =========================


def load_pattern_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"nextAutoId": 1, "patterns": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pattern_file(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_codepoint_from_name(name: str) -> int:
    # 文件名形如：U+00A1_¡.png
    m = re.search(r"U\+([0-9A-Fa-f]{4,6})", name)
    if not m:
        return -1
    return int(m.group(1), 16)


def micro_from_grid(gx: int, gy: int, center_in_9: bool) -> Tuple[int, int]:
    """
    你的编辑器微坐标规则：centerMicro = gx*2+1, gy*2+1
    若把 8x8 居中到 9x9：等价于 gx,gy + 0.5
      micro = (gx + 0.5)*2 + 1 = 2*gx + 2
    """
    if center_in_9:
        return gx * 2 + 2, gy * 2 + 2
    else:
        return gx * 2 + 1, gy * 2 + 1


def glyph_png_to_8x8_cells(png_path: Path) -> List[Tuple[int, int]]:
    """
    读一张 glyph PNG，把“白色像素”转成 8x8 占用格 (gx,gy)
    gy=0 在下（符合你 board 的坐标习惯）
    """
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)  # H,W,4

    # 用亮度判断白色（忽略 alpha）
    rgb = arr[..., :3].astype(np.float32)
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])

    white_mask = lum >= WHITE_THRESHOLD  # True 表示白

    # 缩到 8x8：用“块内是否存在白像素”来决定格子是否点亮
    h, w = white_mask.shape
    grid = np.zeros((TARGET_GRID, TARGET_GRID), dtype=np.uint8)

    # 为了更稳，按比例映射：每个 8x8 cell 对应源图一块区域
    for gy in range(TARGET_GRID):
        for gx in range(TARGET_GRID):
            x0 = int(round(gx * w / TARGET_GRID))
            x1 = int(round((gx + 1) * w / TARGET_GRID))
            y0 = int(round(gy * h / TARGET_GRID))
            y1 = int(round((gy + 1) * h / TARGET_GRID))

            x1 = max(x1, x0 + 1)
            y1 = max(y1, y0 + 1)

            block = white_mask[y0:y1, x0:x1]
            if block.any():
                grid[gy, gx] = 1

    # 转成 (gx,gy) 列表，注意：图片 y=0 在上，我们要 gy=0 在下 → 翻转
    cells = []
    for gy in range(TARGET_GRID):
        for gx in range(TARGET_GRID):
            if grid[gy, gx] == 1:
                gy_bottom = (TARGET_GRID - 1) - gy
                cells.append((gx, gy_bottom))
    return cells


def main():
    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {in_dir}")

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".png"])

    pf = load_pattern_file(OUT_JSON)
    patterns = pf.get("patterns", [])
    next_id = int(pf.get("nextAutoId", 1))

    added = 0
    for p in files:
        cells = glyph_png_to_8x8_cells(p)
        if len(cells) < MIN_ON_CELLS:
            continue

        if USE_UNICODE_AS_ID:
            code = parse_codepoint_from_name(p.name)
            if code <= 0:
                # 没解析到就跳过或用自增，你自己选
                pid = next_id
                next_id += 1
            else:
                pid = int(code)  # 十进制存
                if pid >= next_id:
                    next_id = pid + 1
        else:
            pid = next_id
            next_id += 1

        # 组装成 PatternStore 的 cells（micro 坐标）
        out_cells = []
        for gx, gy in cells:
            mx, my = micro_from_grid(gx, gy, CENTER_IN_9)
            out_cells.append({"x": int(mx), "y": int(my)})

        patterns.append({
            "id": int(pid),
            "gridSize": 9 if CENTER_IN_9 else TARGET_GRID,  # 这里 gridSize 你也可以固定写 9
            "cells": out_cells,
            "savedUtcTicks": 0
        })

        added += 1

    pf["patterns"] = patterns
    pf["nextAutoId"] = next_id
    save_pattern_file(OUT_JSON, pf)

    print(f"[DONE] input pngs={len(files)} added_patterns={added}")
    print(f"[DONE] wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
