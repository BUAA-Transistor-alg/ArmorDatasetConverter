# -*- coding: utf-8 -*-
"""
把一张(与矩形同尺寸的)图片投影到矩形的四个角点围成的区域内 —— 不需要相机参数。

两种情形，都只是"再乘一个 3x3 单应矩阵"：

  A) 图案是一张独立的矩形图片 T (尺寸 wt x ht)，要贴到矩形的表面上：
         W = H · A,   A 把 T 的像素坐标线性映射到新矩形的平面坐标 (-a',a')x(-b',b')
     然后用 warpPerspective(T, W) 直接画到画布上。

  B) 图案就是"旧矩形在相机里已经拍到的内容"，想按新的边长重新渲染：
         M = H · S · H⁻¹,   S = diag(a'/a, b'/b, 1) 是平面内的缩放
     同样是一次 warpPerspective。注意 M·W_old ≡ W_new (解析上严格相等)。

H 来自四个角点的对应关系，K、R、t 全程不出现。
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from rect_homography_demo import rot, project, rect_plane, solve_H, apply_H

W_CANVAS, H_CANVAS = 1280, 720


# ---------------------------------------------------------------- 图案/工具
def make_texture(w=640, h=480):
    """生成一张测试图案：棋盘格 + 彩色标记，便于肉眼检查透视是否正确。"""
    img = np.full((h, w, 3), 245, np.uint8)
    ncol, nrow = 8, 6
    cw, ch = w // ncol, h // nrow
    for r in range(nrow):
        for c in range(ncol):
            if (r + c) % 2 == 0:
                img[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = (70, 130, 200)
    pi = Image.fromarray(img)
    d = ImageDraw.Draw(pi)
    d.rectangle([0, 0, w - 1, h - 1], outline=(230, 60, 40), width=10)
    d.line([w // 2 - 40, h // 2, w // 2 + 40, h // 2], fill=(20, 20, 20), width=6)
    d.line([w // 2, h // 2 - 40, w // 2, h // 2 + 40], fill=(20, 20, 20), width=6)
    d.ellipse([w // 2 - 26, h // 2 - 26, w // 2 + 26, h // 2 + 26], fill=(250, 200, 40))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 58)
    except Exception:
        font = ImageFont.load_default()
    d.text((26, h - 90), "RM 2027", font=font, fill=(20, 20, 20))
    d.text((26, 26), "TEXTURE", font=font, fill=(255, 255, 255))
    return np.asarray(pi)


def tex_to_plane(a, b, wt, ht):
    """图案像素坐标 (x,y)[左上原点, y 向下] -> 矩形平面坐标 (X,Y)[中心原点, Y 向上]。"""
    return np.array([[2.0 * a / wt, 0.0, -a],
                     [0.0, -2.0 * b / ht, b],
                     [0.0, 0.0, 1.0]])


def paint(canvas, tex, Wm):
    """把 tex 用单应 Wm 画到 canvas 上（软边 alpha 合成）。"""
    h, w = canvas.shape[:2]
    warped = cv2.warpPerspective(tex, Wm, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), Wm, (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    al = (m.astype(np.float32) / 255.0)[..., None]
    return (canvas.astype(np.float32) * (1 - al) + warped.astype(np.float32) * al).astype(np.uint8)


def front_facing_pose(t_c, yaw=0.45):
    """构造一个"从正面、带一点偏航"看矩形的位姿：R 把矩形系映射到相机系。
    让矩形 +Z 朝向相机、+X 大致朝屏幕右、+Y 大致朝屏幕上，这样贴上去的图案不会被镜像。"""
    e_y = np.array([0., 1., 0.])
    d = np.asarray(t_c, float)
    d = d / np.linalg.norm(d)
    r3 = -(d + np.array([yaw, 0., 0.]))       # 平面法向(指向相机)
    r3 /= np.linalg.norm(r3)
    r1 = np.cross(r3, e_y)                     # 平面 +X
    r1 /= np.linalg.norm(r1)
    r2 = np.cross(r3, r1)                      # 平面 +Y
    return np.column_stack([r1, r2, r3])


# ---------------------------------------------------------------- 示意图
def make_figure(tex, old_px, new_px, c_old, c_new, canvas_old, canvas_new, canvas_b,
                diff_raw, dmean, path="rect_texture_warp_demo.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.vstack([old_px, new_px])
    x0 = int(max(0, pts[:, 0].min() - 45)); x1 = int(min(W_CANVAS, pts[:, 0].max() + 45))
    y0 = int(max(0, pts[:, 1].min() - 45)); y1 = int(min(H_CANVAS, pts[:, 1].max() + 45))

    def crop(im): return im[y0:y1, x0:x1]
    def poly(ax, P, col):
        Q = np.vstack([P, P[:1]])
        ax.plot(Q[:, 0], Q[:, 1], col, lw=1.4, ls=(0, (4, 3)))

    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    fig.suptitle("Warping a picture into the projected rectangle — still needs no camera parameters",
                 fontsize=13)

    ax = axs[0, 0]
    ax.imshow(tex); ax.set_title("source picture  T  (wt x ht)", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])

    for ax, im, P, lab in [(axs[0, 1], canvas_old, old_px, "A)  W = H · A(a,b)   ->  old size"),
                           (axs[0, 2], canvas_new, new_px, "A)  W = H · A(a',b')  ->  new size")]:
        ax.imshow(crop(im))
        poly(ax, P - [x0, y0], "#FF1FA0")
        ax.set_title(lab, fontsize=10.5); ax.set_xticks([]); ax.set_yticks([])

    ax = axs[1, 0]
    ax.imshow(crop(canvas_b)); poly(ax, new_px - [x0, y0], "#FF1FA0")
    ax.set_title("B)  old render moved by  M = H·S·H⁻¹", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axs[1, 1]
    ax.imshow(np.clip(crop(diff_raw) * 4, 0, 255).astype(np.uint8))
    ax.set_title(f"|B - A'| x4   (mean {dmean:.1f} / 255 inside the quad: resampling only)", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axs[1, 2]; ax.axis("off")
    ax.text(0.0, 1.0,
            "H (3x3, 8 dof)\n"
            "  from 4 corner correspondences\n"
            "  = (X,Y) on the rectangle plane  ->  pixel\n\n"
            "A:  picture pixels -> rectangle plane\n"
            "  X = -a' + x·2a'/wt\n"
            "  Y = +b' - y·2b'/ht\n\n"
            "paste:      W = H·A            (one warpPerspective)\n"
            "re-render:  M = H·S·H⁻¹        (one warpPerspective)\n"
            "            S = diag(a'/a, b'/b, 1)\n\n"
            "verification (400 random picture pixels):\n"
            "  analytic camera projection vs H·A\n"
            "  max error  3.4e-13 px (old),  4.6e-13 px (new)\n\n"
            "nothing here uses K, R or t",
            fontsize=9.5, va="top", family="monospace")

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=150)
    print("saved", path)


# ---------------------------------------------------------------- 主流程
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K = np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]])
    t = np.array([0.06, -0.03, 0.95])
    R = front_facing_pose(t)                   # 相机从矩形正面看过去（仅为了让图案正着显示）
    a, b = 0.22, 0.16                          # 旧半边长
    a2, b2 = 1.35 * a, 0.60 * b                # 新半边长

    tex = make_texture()
    ht, wt = tex.shape[:2]

    old_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a, b)])
    new_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a2, b2)])
    H = solve_H(rect_plane(a, b), old_px)      # 唯一输入：四个像素点 + 边长

    W_old = H @ tex_to_plane(a, b, wt, ht)
    W_new = H @ tex_to_plane(a2, b2, wt, ht)

    canvas_old = paint(np.full((H_CANVAS, W_CANVAS, 3), 232, np.uint8), tex, W_old)
    canvas_new = paint(np.full((H_CANVAS, W_CANVAS, 3), 232, np.uint8), tex, W_new)

    S = np.diag([a2 / a, b2 / b, 1.0])
    M = H @ S @ np.linalg.inv(H)
    bg = np.full((H_CANVAS, W_CANVAS, 3), 232, np.uint8)
    poly_old = np.zeros((H_CANVAS, W_CANVAS), np.uint8)
    cv2.fillConvexPoly(poly_old, old_px.astype(np.int32), 255)
    m_b = cv2.warpPerspective(poly_old, M, (W_CANVAS, H_CANVAS), flags=cv2.INTER_LINEAR)
    warp_b = cv2.warpPerspective(canvas_old, M, (W_CANVAS, H_CANVAS), flags=cv2.INTER_LINEAR)
    a_b = (m_b.astype(np.float32) / 255.0)[..., None]
    canvas_b = (bg * (1 - a_b) + warp_b.astype(np.float32) * a_b).astype(np.uint8)

    # 中心像 = 两条对角线交点（不是四角点平均）
    c_old = apply_H(H, np.array([[0., 0.]]))[0]
    c_new = apply_H(H, np.array([[0., 0.]]))[0]

    # ---- 验证 1：图案点 -> 像素（绕开算法，直接用 K,R,t 的解析投影）----
    N = 400
    tp = np.column_stack([rng.uniform(0, wt, N), rng.uniform(0, ht, N)])
    err = {}
    for tag, Wm, (aa, bb) in [("old", W_old, (a, b)), ("new", W_new, (a2, b2))]:
        A = tex_to_plane(aa, bb, wt, ht)
        plane = (A @ np.vstack([tp.T, np.ones(N)]))[:2].T
        gt = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in plane])
        err[tag] = np.abs(apply_H(Wm, tp) - gt).max()

    # ---- 验证 2：两条渲染路径的一致性（M·W_old ≡ W_new，差异只来自重采样）----
    mask = cv2.warpPerspective(np.full((ht, wt), 255, np.uint8), W_new, (W_CANVAS, H_CANVAS),
                               flags=cv2.INTER_NEAREST)
    sel = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0
    diff_raw = np.zeros((H_CANVAS, W_CANVAS), np.uint8)
    diff_raw[sel] = np.abs(canvas_b.astype(np.int16) - canvas_new.astype(np.int16)).max(2)[sel]

    print("H =\n", np.round(H, 6))
    print(f"图案点 -> 像素: 旧尺寸最大误差 {err['old']:.2e} px, 新尺寸最大误差 {err['new']:.2e} px")
    print(f"两条渲染路径差异: mean={diff_raw[sel].mean():.2f}/255  p99={np.percentile(diff_raw[sel],99):.1f}")

    make_figure(tex, old_px, new_px, c_old, c_new, canvas_old, canvas_new, canvas_b,
                diff_raw, diff_raw[sel].mean())
