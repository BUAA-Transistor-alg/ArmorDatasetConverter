#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moire_demo.py -- 点阵屏 + 透视变换: 复现"相机拍屏幕"的摩尔纹

流程:
  1. 用 dot_matrix.py 把简单图片渲染成高分辨率点阵屏 (默认 256x256 个点, 2048x2048 像素)
  2. 生成一个带轻微倾斜/透视的四边形, 代表"屏幕在照片里占据的位置"
  3. 用不同插值 (最近邻 / 双线性 / 双三次 / Lanczos4) 做透视变换
     -> 采样方式和缩放比例不同, 摩尔纹的强弱/形态也不同
  4. 另外做一份"先 4 倍超采样再面积平均"的正确抗锯齿结果作为对照(基本无摩尔纹)

用法:
  python moire_demo.py assets/apple.png --cols 256 --rows 256 --size 2048 --outdir output_moire
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import dot_matrix as dm

INTERPS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def screen_quad(photo_w: int, photo_h: int, angle: float = 2.5,
                margin: float = 0.05, perspective: float = 0.03) -> np.ndarray:
    """屏幕四个角在照片里的位置: 几乎铺满画面, 带一点点透视 + 旋转."""
    w, h = float(photo_w), float(photo_h)
    pts = np.float32([
        [w * margin, h * (margin + 0.03)],
        [w * (1 - margin), h * (margin - 0.02)],
        [w * (1 - margin + perspective), h * (1 - margin + 0.02)],
        [w * (margin - perspective), h * (1 - margin - 0.03)],
    ])
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)


def perspective_photo(screen: np.ndarray, photo_w: int, photo_h: int, angle: float,
                      interp: int, margin: float = 0.05,
                      perspective: float = 0.03) -> np.ndarray:
    sh, sw = screen.shape[:2]
    src = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
    dst = screen_quad(photo_w, photo_h, angle, margin, perspective)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(screen, M, (photo_w, photo_h), flags=interp,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(18, 18, 18))


def prefiltered_photo(screen: np.ndarray, photo_w: int, photo_h: int, angle: float,
                      supersample: int = 4, margin: float = 0.05,
                      perspective: float = 0.03) -> np.ndarray:
    """正确抗锯齿: 先 4 倍超采样 warp, 再 INTER_AREA 面积平均降采样."""
    big = perspective_photo(screen, photo_w * supersample, photo_h * supersample, angle,
                            cv2.INTER_LINEAR, margin, perspective)
    return cv2.resize(big, (photo_w, photo_h), interpolation=cv2.INTER_AREA)


def _vstack(imgs: list, pad: int = 18, bg: int = 32) -> np.ndarray:
    w = max(i.shape[1] for i in imgs)
    h = sum(i.shape[0] for i in imgs) + pad * (len(imgs) + 1)
    canvas = np.full((h, w + 2 * pad, 3), bg, np.uint8)
    y = pad
    for im in imgs:
        canvas[y:y + im.shape[0], pad:pad + im.shape[1]] = im
        y += im.shape[0] + pad
    return canvas


def _grid(imgs: list, cols: int, pad: int = 18, bg: int = 32) -> np.ndarray:
    if isinstance(imgs[0], tuple):
        imgs = [i for i, _ in imgs]
    rows = []
    for i in range(0, len(imgs), cols):
        chunk = imgs[i:i + cols]
        rows.append(dm._hstack(chunk, pad=pad, bg=bg))
    return _vstack(rows, pad=pad, bg=bg)


def center_crop(photo: np.ndarray, frac: float = 0.45) -> np.ndarray:
    h, w = photo.shape[:2]
    cw, ch = int(w * frac), int(h * frac)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return photo[y0:y0 + ch, x0:x0 + cw]


def run_sweep(screen: np.ndarray, outdir: Path, angle: float,
              photo_widths=(480, 640, 800, 1000), cell=(400, 320)) -> None:
    """固定角度, 扫一遍"屏幕在照片里占多大"(等价于采样率), 每种插值一列."""
    cells = []
    for pw in photo_widths:
        ph = int(round(pw * 0.8))
        for name, flag in INTERPS.items():
            photo = perspective_photo(screen, pw, ph, angle, flag)
            crop = cv2.resize(center_crop(photo, 0.45), cell, interpolation=cv2.INTER_NEAREST)
            cells.append(dm._label(crop, f"{pw}px {name}", 0.55))
    grid = _grid(cells, cols=len(INTERPS))
    cv2.imwrite(str(outdir / "04_sweep_grid.png"), grid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="点阵屏 + 透视变换 -> 摩尔纹")
    ap.add_argument("image", help="输入图片路径")
    ap.add_argument("--cols", type=int, default=256, help="点阵屏列数, 默认 256")
    ap.add_argument("--rows", type=int, default=256, help="点阵屏行数, 默认 256")
    ap.add_argument("--size", type=int, default=2048, help="点阵屏输出分辨率, 默认 2048")
    ap.add_argument("--gain", type=float, default=2.2, help="亮度增益, 默认 2.2")
    ap.add_argument("--sigma", type=float, default=1.0, help="高斯 sigma, 默认 1.0")
    ap.add_argument("--layout", choices=("stripe", "circle"), default="stripe")
    ap.add_argument("--photo-w", type=int, default=1000, help="照片宽, 默认 1000")
    ap.add_argument("--photo-h", type=int, default=800, help="照片高, 默认 800")
    ap.add_argument("--angle", type=float, default=2.5, help="照片里屏幕的旋转角, 默认 2.5 度")
    ap.add_argument("--sweep", action="store_true", help="额外跑一遍不同照片尺寸 x 插值的扫描网格")
    ap.add_argument("--outdir", default="output_moire", help="输出目录")
    args = ap.parse_args(argv)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"无法读取图片: {args.image}")

    stages = dm.render_dot_matrix(img, rows=args.rows, cols=args.cols, out_size=args.size,
                                  gain=args.gain, sigma=args.sigma, layout=args.layout,
                                  dot_w=0.9, bar_h=0.9, gap=0.02)
    screen = stages["final"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "00_screen.png"),
                cv2.resize(screen, (1024, 1024), interpolation=cv2.INTER_AREA))

    photos = {}
    for name, flag in INTERPS.items():
        photos[name] = perspective_photo(screen, args.photo_w, args.photo_h, args.angle, flag)
        cv2.imwrite(str(outdir / f"01_photo_{name}.png"), photos[name])
    photos["prefiltered"] = prefiltered_photo(screen, args.photo_w, args.photo_h, args.angle)
    cv2.imwrite(str(outdir / "01_photo_prefiltered.png"), photos["prefiltered"])

    # 裁剪同一块区域放大 2.5 倍(最近邻, 只放大不改变摩尔纹), 方便看清细节
    cw, ch = 340, 272
    x0 = int(args.photo_w * 0.34)
    y0 = int(args.photo_h * 0.28)
    crops, labels = [], []
    for name, p in photos.items():
        crop = p[y0:y0 + ch, x0:x0 + cw]
        big = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
        crops.append(dm._label(big, name))
        labels.append(name)
    cv2.imwrite(str(outdir / "02_crops_2x2.png"), _grid(crops, cols=3))

    # 整张照片拼图
    full = [dm._label(cv2.resize(p, (420, 336), interpolation=cv2.INTER_AREA), name, 0.7)
            for name, p in photos.items()]
    cv2.imwrite(str(outdir / "03_photos_2x3.png"), _grid(full, cols=3))

    if args.sweep:
        run_sweep(screen, outdir, args.angle)

    print(f"输入图片      : {args.image}  {img.shape[1]}x{img.shape[0]}")
    print(f"点阵屏        : {args.cols} x {args.rows} 个点, 输入 {args.size}px -> 每 cell {stages['cell_w']}px")
    print(f"布局/增益/模糊: {args.layout}, gain={args.gain}, sigma={stages['sigma']:.2f}")
    print(f"照片          : {args.photo_w} x {args.photo_h}, 屏幕旋转 {args.angle} 度")
    print(f"输出目录      : {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
