#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dot_matrix.py -- 用 OpenCV 模拟点阵屏(LED dot-matrix)的显示效果

需求对应的处理链:
  1. 下采样        原图 -> 点阵屏分辨率 (rows x cols), 面积平均 INTER_AREA
  2. 建纯黑画布    输出分辨率 (cols*cell_w x rows*cell_h)
  3. 每通道一个蒙版 蒙版只描述"一个 cell"里的内容 -> (3, cell_h, cell_w)
  4. 平铺 + 填充   把单 cell 蒙版平铺到点阵屏的每一个 cell, 再按通道填颜色:
                   out[..., c] = 最近邻放大后的通道 c * mask_tiled[c]
  5. 换更大范围类型 uint8 -> float32 (可以表示 > 255 的亮度)
  6. 拉高亮度      * gain
  7. 高斯模糊      模拟点阵的发光/扩散
  8. 截断回原范围  clip 到 [0, 255] 后返回 uint8

用法:
  python dot_matrix.py assets/apple.png --cols 64 --rows 64 --size 1024
  python dot_matrix.py assets/apple.png --gain 2.5 --sigma 2.4 --outdir output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 1) 把原图下采样到点阵屏分辨率
# --------------------------------------------------------------------------- #
def downsample_to_matrix(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """任意尺寸 BGR uint8 图像 -> (rows, cols) 的点阵屏像素。"""
    return cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# 2)+3) 纯黑画布 + 每个通道一张"单 cell"蒙版
# --------------------------------------------------------------------------- #
def _round_rect_sdf(dx: np.ndarray, dy: np.ndarray, half_w: float, half_h: float,
                    radius: float) -> np.ndarray:
    """圆角矩形的有符号距离场(SDF), 用于生成带 1px 抗锯齿的蒙版。"""
    qx = np.abs(dx) - (half_w - radius)
    qy = np.abs(dy) - (half_h - radius)
    return (
        np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
        + np.minimum(np.maximum(qx, qy), 0.0)
        - radius
    )


def build_cell_masks(
    cell_h: int,
    cell_w: int,
    layout: str = "circle",
    dot_w: float = 0.78,
    bar_h: float = 0.62,
    gap: float = 0.05,
) -> np.ndarray:
    """生成 (3, cell_h, cell_w) float32 蒙版: 只描述一个 cell(一个点阵像素)内部。

    每个 cell = 一个"点", 点内部由 R/G/B 三个子像素拼成:
      * circle: 三个几乎重合的圆(色彩轻微错位), 更接近普通 LED 屏;
      * stripe: 三个竖直圆角条并排(像 RGB 背光/LED 模组)。
    cell 内的子像素用距离场(SDF)生成, clip(0.5 - d, 0, 1) 自带 1px 抗锯齿边。
    """
    # cell 内的像素中心坐标, 原点在 cell 中心: [-cell/2, cell/2)
    xs = (np.arange(cell_w, dtype=np.float32) + 0.5 - cell_w / 2.0)[None, :]
    ys = (np.arange(cell_h, dtype=np.float32) + 0.5 - cell_h / 2.0)[:, None]

    masks = np.empty((3, cell_h, cell_w), dtype=np.float32)
    names = ("B", "G", "R")

    if layout == "stripe":
        # 三个竖条: 总宽度 = 3*条宽 + 2*缝隙 = dot_w
        bar_w = (dot_w - 2.0 * gap) / 3.0 * cell_w
        step = (dot_w - 2.0 * gap) / 3.0 + gap          # 条中心间距(以 cell 计)
        bar_h_px = bar_h * cell_h
        radius = min(bar_w, bar_h_px) / 2.0
        offset = {"R": -step * cell_w, "G": 0.0, "B": +step * cell_w}
        for ch, name in enumerate(names):
            dist = _round_rect_sdf(xs - offset[name], ys, bar_w / 2.0,
                                   bar_h_px / 2.0, radius)
            masks[ch] = np.clip(0.5 - dist, 0.0, 1.0)

    elif layout == "circle":
        # 三个通道各是一张略小于 cell 的圆, 相互错开一点点 -> 一个彩色亮点带轻微色边
        radius = dot_w * 0.5 * min(cell_w, cell_h) - 0.025 * cell_w
        offset = {"R": (-0.035 * cell_w, -0.02 * cell_h),
                  "G": (0.0, 0.0),
                  "B": (+0.035 * cell_w, +0.02 * cell_h)}
        for ch, name in enumerate(names):
            ox, oy = offset[name]
            dist = np.sqrt((xs - ox) ** 2 + (ys - oy) ** 2) - radius
            masks[ch] = np.clip(0.5 - dist, 0.0, 1.0)

    else:
        raise ValueError(f"未知 layout: {layout}")

    return masks


def tile_masks(cell_masks: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """把单 cell 蒙版平铺到点阵屏的每一个像素上。

    (3, cell_h, cell_w) --np.tile--> (3, rows*cell_h, cols*cell_w)
    平铺后, 第 (i, j) 个 cell 覆盖的那块区域, 拿到的就是同一张 cell 蒙版。
    """
    return np.tile(cell_masks, (1, rows, cols))


# --------------------------------------------------------------------------- #
# 4) 按蒙版填充纯黑画布: 每个通道用自己那张(平铺后的)蒙版
# --------------------------------------------------------------------------- #
def fill_by_channel_masks(screen: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """screen: (rows, cols, 3) 点阵屏内容; masks: (3, H, W) 平铺后的蒙版。"""
    out_h, out_w = masks.shape[1], masks.shape[2]

    # 最近邻放大: 一个 cell 里的所有子像素都拿到该点阵像素的颜色
    up = cv2.resize(screen, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    # 输出分辨率的纯黑画布; 每个通道按自己那张蒙版把颜色填进去
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for c in range(3):
        canvas[..., c] = up[..., c] * masks[c]

    return np.clip(canvas, 0.0, 255.0).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# 5)+6) 换更大范围的数据类型并拉高亮度
# --------------------------------------------------------------------------- #
def brighten(filled_u8: np.ndarray, gain: float) -> np.ndarray:
    """uint8 -> float32(可超过 255), 再乘以增益。"""
    return filled_u8.astype(np.float32) * gain


# --------------------------------------------------------------------------- #
# 7) 高斯模糊
# --------------------------------------------------------------------------- #
def gaussian_glow(bright_f: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        bright_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
    )


# --------------------------------------------------------------------------- #
# 8) 截断回原范围, 返回 uint8
# --------------------------------------------------------------------------- #
def clip_to_uint8(f: np.ndarray) -> np.ndarray:
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 完整流程
# --------------------------------------------------------------------------- #
def render_dot_matrix(
    img: np.ndarray,
    rows: int = 64,
    cols: int = 64,
    out_size: int = 1024,
    gain: float = 2.2,
    sigma: float = 0.0,
    layout: str = "circle",
    dot_w: float = 0.78,
    bar_h: float = 0.62,
    gap: float = 0.05,
) -> Dict[str, object]:
    """返回各阶段结果, 其中 final 是最终 uint8 图像。"""
    # cell 必须是整数像素, 这样单 cell 蒙版才能整块平铺; 输出分辨率 = cell * 点数
    cell_w = max(1, int(round(out_size / cols)))
    cell_h = max(1, int(round(out_size / rows)))
    out_w, out_h = cols * cell_w, rows * cell_h

    screen = downsample_to_matrix(img, rows, cols)                     # 1
    cell_masks = build_cell_masks(cell_h, cell_w, layout,              # 2 + 3
                                  dot_w=dot_w, bar_h=bar_h, gap=gap)
    masks = tile_masks(cell_masks, rows, cols)                         # 4a 平铺
    filled = fill_by_channel_masks(screen, masks)                      # 4b 填充
    bright = brighten(filled, gain)                                    # 5 + 6
    if sigma <= 0:                                                     # 7
        sigma = 0.15 * min(cell_w, cell_h)
    blurred = gaussian_glow(bright, sigma)
    final = clip_to_uint8(blurred)                                     # 8

    return dict(screen=screen, cell_masks=cell_masks, masks=masks,
                filled=filled, bright=bright, blurred=blurred, final=final,
                sigma=float(sigma), cell_w=cell_w, cell_h=cell_h)


# --------------------------------------------------------------------------- #
# 可视化 / 保存
# --------------------------------------------------------------------------- #
def _label(img: np.ndarray, text: str, scale: float = 0.9) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _hstack(imgs: list, pad: int = 18, bg: int = 32) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    w = sum(i.shape[1] for i in imgs) + pad * (len(imgs) + 1)
    canvas = np.full((h + 2 * pad, w, 3), bg, np.uint8)
    x = pad
    for im in imgs:
        canvas[pad:pad + im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + pad
    return canvas


def save_stage_images(
    stages: Dict[str, object], input_img: np.ndarray, outdir: Path
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    screen = stages["screen"]
    cell_masks = stages["cell_masks"]      # (3, cell_h, cell_w) 单个 cell
    masks = stages["masks"]                # (3, H, W) 平铺到整屏
    cell_h, cell_w = stages["cell_h"], stages["cell_w"]
    out_h, out_w = stages["final"].shape[:2]

    def save(name: str, im: np.ndarray) -> None:
        cv2.imwrite(str(outdir / name), im)

    # 01 下采样后的点阵屏内容: 原始小图 + 最近邻放大预览
    save("01_screen_native.png", screen)
    save("01_screen_preview.png", cv2.resize(screen, (out_w, out_h), interpolation=cv2.INTER_NEAREST))

    # 02 单 cell 蒙版: 原始大小 + 放大预览(否则只有 cell_h x cell_w 像素, 看不清)
    zoom = max(1, int(round(512 / max(cell_h, cell_w))))
    for ch, name in enumerate(("B", "G", "R")):
        save(f"02_cell_mask_{name}.png",
             (cell_masks[ch] * 255.0).round().astype(np.uint8))
    cell_rgb = np.stack([cell_masks[0], cell_masks[1], cell_masks[2]], axis=-1)  # BGR
    cell_rgb_u8 = (np.clip(cell_rgb, 0, 1) * 255).round().astype(np.uint8)
    save("02_cell_mask_rgb.png",
         cv2.resize(cell_rgb_u8, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST))

    panels = []
    for ch, name in enumerate(("B", "G", "R")):
        mask_u8 = (cell_masks[ch] * 255.0).round().astype(np.uint8)
        big = cv2.resize(cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR), None, fx=zoom, fy=zoom,
                         interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, f"cell mask {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (60, 60, 60), 3, cv2.LINE_AA)
        cv2.putText(big, f"cell mask {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(big)
    save("02_cell_masks_montage.png", _hstack(panels))

    # 平铺后的整屏蒙版(合成成彩色) + 局部放大, 看得见"每格都是同一张 cell 蒙版"
    masks_rgb = np.stack([masks[0], masks[1], masks[2]], axis=-1)
    masks_bgr8 = (np.clip(masks_rgb, 0, 1) * 255).round().astype(np.uint8)
    save("02_masks_rgb.png", masks_bgr8)
    z, pad = 4, 6
    y0, x0 = cell_h * pad, cell_w * pad
    zoom_masks = masks_bgr8[y0:y0 + cell_h * z, x0:x0 + cell_w * z]
    save("02_masks_zoom.png", cv2.resize(zoom_masks, None, fx=8, fy=8,
                                         interpolation=cv2.INTER_NEAREST))

    # 03 按蒙版填充后的图像
    save("03_filled.png", stages["filled"])

    # 04 拉高亮度后(float, 这里为可视化先截断; 实际值可 > 255)
    save("04_brightened.png", clip_to_uint8(stages["bright"]))

    # 05 高斯模糊 + 截断 = 最终 uint8 结果
    save("05_final.png", stages["final"])

    # 06 原图 vs 点阵屏效果 对比
    inp = cv2.resize(input_img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    save("06_compare.png", _hstack([_label(inp, "input"),
                                    _label(stages["final"], "dot-matrix")]))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OpenCV 点阵屏(LED dot-matrix)效果模拟")
    ap.add_argument("image", help="输入图片路径")
    ap.add_argument("--rows", type=int, default=64, help="点阵屏行数, 默认 64")
    ap.add_argument("--cols", type=int, default=64, help="点阵屏列数, 默认 64")
    ap.add_argument("--size", type=int, default=1024, help="输出分辨率(正方形), 默认 1024")
    ap.add_argument("--gain", type=float, default=2.2, help="亮度增益, 默认 2.2")
    ap.add_argument("--sigma", type=float, default=0.0, help="高斯模糊 sigma, 0=自动(约 0.15 个 cell)")
    ap.add_argument("--layout", choices=("circle", "stripe"), default="circle",
                    help="子像素排布: circle=圆点(默认), stripe=RGB 竖条")
    ap.add_argument("--dot-w", type=float, default=0.78, help="每个点占 cell 的尺寸比例")
    ap.add_argument("--bar-h", type=float, default=0.62, help="stripe 布局下子像素高度比例")
    ap.add_argument("--gap", type=float, default=0.05, help="stripe 布局下子像素之间的缝隙(以 cell 计)")
    ap.add_argument("--outdir", default="output", help="输出目录, 默认 output")
    args = ap.parse_args(argv)

    img = cv2.imread(str(Path(args.image).expanduser()), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"无法读取图片: {args.image}")

    stages = render_dot_matrix(
        img, rows=args.rows, cols=args.cols, out_size=args.size,
        gain=args.gain, sigma=args.sigma, layout=args.layout,
        dot_w=args.dot_w, bar_h=args.bar_h, gap=args.gap,
    )
    outdir = Path(args.outdir)
    save_stage_images(stages, img, outdir)

    h, w = stages["final"].shape[:2]
    print(f"输入图片      : {args.image}  {img.shape[1]}x{img.shape[0]}")
    print(f"点阵屏分辨率  : {args.cols} x {args.rows}")
    print(f"单 cell 大小  : {stages['cell_w']} x {stages['cell_h']}  （蒙版只描述这一个 cell）")
    print(f"平铺后分辨率  : {w} x {h}  = {args.cols} x {args.rows} 个 cell")
    print(f"蒙版布局      : {args.layout} (dot_w={args.dot_w})")
    print(f"亮度增益      : {args.gain}")
    print(f"高斯 sigma    : {stages['sigma']:.3f}")
    print(f"提亮后 float  : min={stages['bright'].min():.1f} max={stages['bright'].max():.1f} dtype={stages['bright'].dtype}")
    print(f"最终图像      : shape={stages['final'].shape} dtype={stages['final'].dtype} range=[{stages['final'].min()}, {stages['final'].max()}]")
    print(f"中间结果目录  : {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
