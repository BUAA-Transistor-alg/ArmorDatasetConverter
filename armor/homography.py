# -*- coding: utf-8 -*-
"""装甲板平面(米) <-> 图像(像素) 的单应变换。

矩形所在平面到图像严格是一个 3x3 单应 H，四个关键点即可用四点 DLT 精确求出；
三个目标矩形都与装甲板同心，因此共用同一个 H，只需要各自的「纹理 -> 平面」矩阵 A。
"""
import numpy as np


def plane_corners(half_w, half_h):
    """矩形四角在板平面内的坐标，顺序与检测关键点一致：P0 左上, P1 左下, P2 右下, P3 右上。

    平面坐标：X 向右、Y 向上，单位米，原点在装甲板中心。
    """
    return np.array([[-half_w, +half_h],
                     [-half_w, -half_h],
                     [+half_w, -half_h],
                     [+half_w, +half_h]], float)


def solve_homography(plane_pts, image_pts):
    """四点 DLT（h33 固定为 1），四个不共线的对应点给出唯一解。"""
    A = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((X, Y), (u, v)) in enumerate(zip(plane_pts, image_pts)):
        A[2 * i] = [X, Y, 1, 0, 0, 0, -u * X, -u * Y]
        A[2 * i + 1] = [0, 0, 0, X, Y, 1, -v * X, -v * Y]
        b[2 * i], b[2 * i + 1] = u, v
    h = np.linalg.solve(A, b)
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def texture_to_plane(width_m, height_m, tex_w, tex_h):
    """纹理像素坐标(x 右, y 下) -> 板平面坐标(X 右, Y 上)。

    纹理左上角对应平面左上角，保证贴图不会上下/左右翻转。
    """
    return np.array([[width_m / tex_w, 0.0, -width_m / 2.0],
                     [0.0, -height_m / tex_h, +height_m / 2.0],
                     [0.0, 0.0, 1.0]], float)
