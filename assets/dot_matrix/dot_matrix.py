#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dot_matrix.py -- 把 input/ 里的所有图片渲染成 LED 点阵屏（无命令行参数）

实现方式：
  1. 不接受任何命令行参数，所有参数都是文件顶部的常量；
  2. 每个通道的「单个 cell 蒙版」从 masks/ 目录读取 PNG 文件：
         masks/<LAYOUT>_B.png
         masks/<LAYOUT>_G.png
         masks/<LAYOUT>_R.png
     换蒙版就是换文件，不改代码。PNG 里存 0~255 灰度，读进来除以 255 当权重。
  3. 批量处理 input/ 里的**所有图片**，只输出最终结果到 output/，
     **不写任何中间文件**。

处理链（每张图独立运行一遍）：
  1. 原图 INTER_AREA 下采样到 (ROWS, COLS) 点阵屏分辨率
  2. 建一整屏纯黑画布 (COLS*cell_w  x  ROWS*cell_h)
  3. 读每个通道的单个 cell 蒙版，缩放到 cell 大小
  4. np.tile 把单个 cell 蒙版平铺到每一个点阵像素，再按通道填色：
         out[..., c] = 最近邻放大后的通道 c * mask_tiled[c]
  5. uint8 -> float32（可以超过 255）
  6. 乘 GAIN 提亮
  7. 高斯模糊，模拟发光/扩散（SIGMA 以 cell 边长为单位，取 cell 短边；0 = 不模糊）
  8. clip 回 [0, 255] 并转回 uint8

用法（唯一的运行方式，不接受参数）：
    python3 assets/dot_matrix/dot_matrix.py

    input/  放要处理的图（png/jpg/jpeg/bmp，只看这一层，不递归）
    output/ 每张图输出一张同名 PNG，中间结果只在内存里，不写入文件
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 全部参数硬编码在这里 —— 不接受命令行参数，改这几行就行
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent

INPUT_DIR = HERE / "input"     # 待点阵化的图片目录（处理这里面所有图片）
MASK_DIR = HERE / "masks"      # 单个 cell 蒙版目录，读 <LAYOUT>_B/G/R.png
OUT_DIR = HERE / "output"      # 只放最终结果，不写中间文件

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")   # input/ 里认哪些扩展名

LAYOUT = "stripe"      # 用哪一套蒙版："circle"（圆点）或 "stripe"（RGB 竖条子像素）
ROWS = 64              # 点阵屏行数
COLS = 64              # 点阵屏列数
OUT_SIZE = 2048        # 输出分辨率（取整为 cell 的整数倍）
GAIN = 1.0             # 提亮增益，越大越容易过曝成白
SIGMA = 0.0           # 高斯 sigma，单位是 cell 边长（长短边不一致时取短边）；0 = 不模糊

CHANNEL_NAMES = ("B", "G", "R")   # 蒙版文件的通道后缀，也是数组的第 0/1/2 维


# --------------------------------------------------------------------------- #
# 1) 把原图下采样到点阵屏分辨率
# --------------------------------------------------------------------------- #
def downsample_to_matrix(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """任意尺寸 BGR uint8 图像 -> (rows, cols) 的点阵屏像素。"""
    return cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# 2)+3) cell 大小 + 从 PNG 读每通道的单个 cell 蒙版
# --------------------------------------------------------------------------- #
def cell_size(rows: int, cols: int, out_size: int) -> tuple:
    """输出分辨率取整为 cell 整数倍时，单个 cell 的 (高, 宽)。"""
    cell_w = max(1, int(round(out_size / cols)))
    cell_h = max(1, int(round(out_size / rows)))
    return cell_h, cell_w


def load_cell_masks(layout: str, cell_h: int, cell_w: int,
                    mask_dir: Path = MASK_DIR) -> np.ndarray:
    """从 mask_dir 读 (3, cell_h, cell_w) float32 蒙版：只描述一个 cell 内部。

    PNG 是 0~255 灰度，读进来先归一化到 0~1；尺寸和当前 cell 不一致时自动缩放
    （缩小用 INTER_AREA，放大用 INTER_LINEAR），所以换 ROWS/COLS/OUT_SIZE 不用重画蒙版。
    """
    masks = np.empty((3, cell_h, cell_w), dtype=np.float32)
    for index, name in enumerate(CHANNEL_NAMES):
        path = mask_dir / f"{layout}_{name}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f"读不到蒙版：{path}（LAYOUT={layout!r} 需要 {layout}_B/G/R.png）")
        if mask.shape != (cell_h, cell_w):
            shrinking = mask.shape[0] > cell_h or mask.shape[1] > cell_w
            mask = cv2.resize(mask, (cell_w, cell_h),
                              interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
        masks[index] = mask.astype(np.float32) / 255.0
    return masks


def tile_masks(cell_masks: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(3, cell_h, cell_w) --np.tile--> (3, rows*cell_h, cols*cell_w)，每个点阵像素一张。"""
    return np.tile(cell_masks, (1, rows, cols))


# --------------------------------------------------------------------------- #
# 4) 按蒙版填充纯黑画布
# --------------------------------------------------------------------------- #
def fill_by_channel_masks(screen: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """screen: (rows, cols, 3) 点阵屏内容；masks: (3, H, W) 平铺后的蒙版。"""
    out_h, out_w = masks.shape[1], masks.shape[2]
    # 最近邻放大：一个 cell 里的所有子像素都拿到该点阵像素的颜色
    up = cv2.resize(screen, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for c in range(3):
        canvas[..., c] = up[..., c] * masks[c]
    return np.clip(canvas, 0.0, 255.0).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# 5)~8) 提亮 / 发光 / 截断
# --------------------------------------------------------------------------- #
def brighten(filled_u8: np.ndarray, gain: float) -> np.ndarray:
    """uint8 -> float32（可超过 255），再乘增益。"""
    return filled_u8.astype(np.float32) * gain


def gaussian_glow(bright_f: np.ndarray, sigma: float) -> np.ndarray:
    """高斯模糊；sigma 单位是**像素**，<= 0 时原样返回（0 就是 0，不做任何自动推断）。

    调用方负责把"cell 倍数"换算成像素（见 render_dot_matrix）；
    OpenCV 在 ksize=(0,0) 且 sigma=0 时会断言失败，所以 0 必须显式跳过。
    """
    if sigma <= 0:
        return bright_f
    return cv2.GaussianBlur(bright_f, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REPLICATE)


def clip_to_uint8(f: np.ndarray) -> np.ndarray:
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 完整流程（单张图，结果只在内存里，不写入文件）
# --------------------------------------------------------------------------- #
def render_dot_matrix(img: np.ndarray, rows: int = ROWS, cols: int = COLS,
                      out_size: int = OUT_SIZE, gain: float = GAIN, sigma: float = SIGMA,
                      layout: str = LAYOUT) -> Dict[str, object]:
    """返回各阶段结果（只返回给调用者，不写文件），其中 final 是最终 uint8 图像。

    sigma 以 cell 边长为单位（cell 长短边不一致时取较短边），内部换算成像素
    sigma_px = sigma * min(cell_w, cell_h)，两个值都会放在返回的 dict 里。
    """
    cell_h, cell_w = cell_size(rows, cols, out_size)
    out_w, out_h = cols * cell_w, rows * cell_h

    screen = downsample_to_matrix(img, rows, cols)                     # 1
    cell_masks = load_cell_masks(layout, cell_h, cell_w)               # 2 + 3（读 PNG）
    masks = tile_masks(cell_masks, rows, cols)                         # 4a 平铺
    filled = fill_by_channel_masks(screen, masks)                      # 4b 填充
    bright = brighten(filled, gain)                                    # 5 + 6
    sigma_px = sigma * min(cell_w, cell_h)                             # cell 边长倍数 -> 像素
    blurred = gaussian_glow(bright, sigma_px)                          # 7（0 = 不模糊）
    final = clip_to_uint8(blurred)                                     # 8

    return dict(screen=screen, cell_masks=cell_masks, masks=masks, filled=filled,
                bright=bright, blurred=blurred, final=final,
                sigma=float(sigma), sigma_px=float(sigma_px),
                cell_w=cell_w, cell_h=cell_h)


# --------------------------------------------------------------------------- #
# 批量入口（没有命令行参数）
# --------------------------------------------------------------------------- #
def list_input_images(input_dir: Path = INPUT_DIR) -> List[Path]:
    """input/ 里的所有图片，按文件名排序（不递归）。"""
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{input_dir}")
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> int:
    images = list_input_images()
    if not images:
        raise SystemExit(f"输入目录里没有图片（{' / '.join(IMAGE_EXTS)}）：{INPUT_DIR}")

    cell_h, cell_w = cell_size(ROWS, COLS, OUT_SIZE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"输入目录      : {INPUT_DIR}（{len(images)} 张图）")
    print(f"点阵屏        : {COLS} x {ROWS} 个点，单个 cell {cell_w} x {cell_h} px"
          f" -> 每张输出 {COLS * cell_w} x {ROWS * cell_h}")
    sigma_px = SIGMA * min(cell_w, cell_h)
    print(f"蒙版/增益/模糊: {LAYOUT}（masks/{LAYOUT}_B|G|R.png），gain={GAIN}，"
          f"sigma={SIGMA} cell（{sigma_px:.2f} px）" + ("，不模糊" if sigma_px <= 0 else ""))
    print(f"输出目录      : {OUT_DIR}（只写最终结果，不写中间文件）")

    done = 0
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"跳过（读不到）：{path.name}")
            continue
        stages = render_dot_matrix(img)          # 中间结果只在内存里
        out_path = OUT_DIR / f"{path.stem}.png"
        if not cv2.imwrite(str(out_path), stages["final"]):
            print(f"写出失败：{out_path}")
            continue
        done += 1
        h, w = stages["final"].shape[:2]
        print(f"  {path.name:<24s} {img.shape[1]}x{img.shape[0]} -> {out_path.name}  "
              f"{w}x{h}  sigma={stages['sigma']:g} cell / {stages['sigma_px']:.2f} px  "
              f"提亮峰值={float(stages['bright'].max()):.1f}")

    print(f"完成：{done}/{len(images)} 张 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
