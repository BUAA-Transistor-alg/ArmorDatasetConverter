# -*- coding: utf-8 -*-
"""
已知矩形四角点的像素坐标，不知相机参数，
预测"矩形边长改变、而矩形中心位姿相对相机不变"后的四角点像素坐标。

原理：矩形所在平面 -> 图像平面 是一个单应矩阵 H (3x3, 8 自由度)。
      四个角点的对应关系唯一确定 H；边长改变等价于在矩形平面内绕中心做一次
      仿射缩放，所以  新角点像素 = H * (新角点平面坐标)。
      全程不需要 K、R、t。

只依赖一个前提：相机光心不在矩形所在平面内（否则平面被投影成一条直线，退化）。
"""
import numpy as np

# ---------------------------------------------------------------- 基础工具
def rot(w):
    """Rodrigues: 旋转向量 -> 旋转矩阵"""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def project(K, R, t, Xw):
    """针孔模型: s*x ~ K (R Xw + t)；返回像素 (u, v)。"""
    p = K @ (R @ Xw + t)
    return p[:2] / p[2]


def rect_plane(a, b):
    """矩形在自身平面内的四个角点坐标 (X, Y)，逆时针；需与像素角点一一对应。"""
    return np.array([[-a, -b], [a, -b], [a, b], [-a, b]], float)


def solve_H(plane_pts, img_pts):
    """四点 DLT: (X,Y) -> (u,v)。固定 h33 = 1，解 8x8 线性方程组。"""
    A, rhs = [], []
    for (X, Y), (u, v) in zip(plane_pts, img_pts):
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y]); rhs.append(u)
        A.append([0, 0, 0, X, Y, 1, -v * X, -v * Y]); rhs.append(v)
    h = np.linalg.solve(np.array(A), np.array(rhs))
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def apply_H(H, pts):
    P = np.hstack([pts, np.ones((len(pts), 1))])
    q = (H @ P.T).T
    return q[:, :2] / q[:, 2:3]


def predict_new_corners(corners_px, a, b, a2, b2):
    """主函数：旧四角像素 + 旧半边长 (a,b) + 新半边长 (a2,b2) -> 新四角像素。"""
    H = solve_H(rect_plane(a, b), np.asarray(corners_px, float))
    return apply_H(H, rect_plane(a2, b2)), H


# ---------------------------------------------------------------- 1) 单次验证
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    K = np.array([[800., 0, 640], [0, 800, 360], [0, 0, 1]])
    R = rot(np.array([0.45, -0.30, 0.12]))
    t = np.array([0.10, -0.06, 1.30])
    a, b = 0.135, 0.055                        # 旧半边长（RM 装甲板量级）

    old_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a, b)])

    print("旧角点像素:\n", np.round(old_px, 3))
    print(f"{'a2':>8}{'b2':>8}{'最大像素误差':>16}")
    for a2, b2 in [(a, b), (1.3 * a, 0.8 * b), (2.0 * a, 2.0 * b), (0.5 * a, 2.5 * b)]:
        new_px, _ = predict_new_corners(old_px, a, b, a2, b2)
        gt_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a2, b2)])
        print(f"{a2:8.4f}{b2:8.4f}{np.abs(new_px - gt_px).max():16.3e}")

    # ------------------------------------------------------------ 2) 随机压力测试
    worst = 0.0
    for _ in range(2000):
        K = np.array([[rng.uniform(300, 2000), 0, rng.uniform(200, 1000)],
                      [0, rng.uniform(300, 2000), rng.uniform(200, 700)],
                      [0, 0, 1.]])
        R = rot(rng.normal(size=3) * rng.uniform(0.05, 1.2))
        t = np.array([rng.uniform(-.5, .5), rng.uniform(-.5, .5), rng.uniform(.3, 3.)])
        a, b = rng.uniform(.02, .5), rng.uniform(.02, .5)
        old_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a, b)])
        H = solve_H(rect_plane(a, b), old_px)
        for _ in range(3):
            a2, b2 = a * rng.uniform(.3, 3), b * rng.uniform(.3, 3)
            new_px = apply_H(H, rect_plane(a2, b2))
            gt_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a2, b2)])
            err = np.abs(new_px - gt_px).max() / np.linalg.norm(gt_px[2] - gt_px[0])
            worst = max(worst, err)
    print(f"\n2000 组随机相机/位姿/矩形: 最大相对误差 = {worst:.2e}（机器精度）")

    # ------------------------------------------------------------ 3) 噪声敏感度
    def noisy_trial(scale, sigma=1.0):
        K = np.array([[800., 0, 640], [0, 800, 360], [0, 0, 1]])
        R = rot(rng.normal(size=3) * 0.35)
        t = np.array([rng.uniform(-.1, .1), rng.uniform(-.1, .1), 1.5])
        old_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a, b)])
        H = solve_H(rect_plane(a, b), old_px + rng.normal(0, sigma, old_px.shape))
        a2, b2 = a * scale, b * scale
        new_px = apply_H(H, rect_plane(a2, b2))
        gt_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a2, b2)])
        return np.abs(new_px - gt_px).max()

    print("\n角点像素加 1px 高斯噪声后，新角点误差 vs 边长放大倍数:")
    for s in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        es = np.array([noisy_trial(s) for _ in range(300)])
        print(f"  s={s:3.1f}  median={np.median(es):7.2f}px  p95={np.percentile(es, 95):8.2f}px")


# ---------------------------------------------------------------- 4) 示意图
def make_figure(path="rect_homography_demo.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = np.array([[760., 0, 640], [0, 760, 360], [0, 0, 1]])
    R = rot(np.array([0.42, -0.34, 0.10]))
    t = np.array([0.09, -0.05, 1.25])
    a, b = 0.135, 0.055
    a2, b2 = 1.45 * a, 0.55 * b

    old_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a, b)])
    new_px, H = predict_new_corners(old_px, a, b, a2, b2)
    gt_px = np.array([project(K, R, t, np.array([X, Y, 0.])) for X, Y in rect_plane(a2, b2)])
    err = np.abs(new_px - gt_px).max()
    # 中心像 = (0,0) 的像 = 两条对角线交点（注意：不是四个角点的算术平均！）
    c_old = apply_H(H, np.array([[0., 0.]]))[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.0, 7.6))
    fig.suptitle("Rescale a planar rectangle, same centre pose: new corners from a 4-point homography",
                 fontsize=12.5)

    # ---- 上：矩形自身平面 (X, Y)，单位 m ----
    for (aa, bb, col, ls, lab) in [(a, b, "tab:blue", "-", "old  a x b"),
                                   (a2, b2, "tab:green", "--", "new  a' x b'")]:
        P = rect_plane(aa, bb)
        ax1.add_patch(plt.Polygon(np.vstack([P, P[:1]]), closed=True, fill=False,
                                  ec=col, ls=ls, lw=1.9, label=lab))
    Po, Pn = rect_plane(a, b), rect_plane(a2, b2)
    for i, (p, q) in enumerate(zip(Po, Pn), start=1):
        ax1.plot(*p, "o", color="tab:blue", ms=6)
        ax1.plot(*q, "s", color="tab:green", ms=5, mfc="none")
        ax1.annotate(f"P{i}", p, textcoords="offset points",
                     xytext=(-22, -4) if p[0] < 0 else (10, -4), color="tab:blue", fontsize=9)
        ax1.annotate(f"Q{i}", q, textcoords="offset points",
                     xytext=(-22, 4) if q[0] < 0 else (10, 4), color="tab:green", fontsize=9)
    ax1.plot(0, 0, "k+", ms=13, mew=2.2)
    ax1.annotate("centre (fixed)", (0, 0), textcoords="offset points", xytext=(6, -22), fontsize=9)
    ax1.annotate("", xy=(1.10 * a2, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))
    ax1.annotate("", xy=(0, 1.55 * b2), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))
    ax1.text(1.11 * a2, 0.0025, "X", color="0.35", fontsize=10)
    ax1.text(0.004, 1.58 * b2, "Y", color="0.35", fontsize=10)
    ax1.set_aspect("equal"); ax1.grid(alpha=.25)
    ax1.legend(loc="lower center", fontsize=8.5, ncol=2, framealpha=.9)
    ax1.set_title("rectangle plane (metres): the new rectangle is an affine scaling about the fixed centre",
                  fontsize=10.5)
    ax1.set_xlabel("X  [m]"); ax1.set_ylabel("Y  [m]")
    ax1.set_xlim(-1.45 * a2, 1.45 * a2); ax1.set_ylim(-2.75 * b2, 2.1 * b2)

    # ---- 下：图像平面 (u, v) ----
    ax2.plot(gt_px[[0, 1, 2, 3, 0], 0], gt_px[[0, 1, 2, 3, 0], 1], "x", color="red",
             ms=12, mew=2.4, label="new corners: ground truth (unknown to the algorithm)")
    ax2.plot(old_px[[0, 1, 2, 3, 0], 0], old_px[[0, 1, 2, 3, 0], 1], "-o",
             color="tab:blue", lw=2.0, ms=7, label="old corners (the only input)")
    ax2.plot(new_px[[0, 1, 2, 3, 0], 0], new_px[[0, 1, 2, 3, 0], 1], "--",
             color="tab:green", lw=1.7, label="new corners:  H x (a', b')   <- prediction")
    ax2.plot([old_px[0, 0], old_px[2, 0]], [old_px[0, 1], old_px[2, 1]], ":", color="0.55", lw=1.1)
    ax2.plot([old_px[1, 0], old_px[3, 0]], [old_px[1, 1], old_px[3, 1]], ":", color="0.55", lw=1.1)
    ax2.plot(*c_old, "+", color="k", ms=14, mew=2.2, label="centre image = diagonals' intersection")
    ax2.invert_yaxis(); ax2.set_aspect("equal"); ax2.grid(alpha=.25)
    ax2.legend(loc="lower left", fontsize=9, framealpha=.92)
    ax2.set_title("image plane (pixels): 4 point correspondences -> H -> new corners", fontsize=10.5)
    ax2.set_xlabel("u  [px]"); ax2.set_ylabel("v  [px]")

    allp = np.vstack([old_px, new_px, gt_px])
    pad = 0.12 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]))
    ax2.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax2.set_ylim(allp[:, 1].max() + 2.6 * pad, allp[:, 1].min() - pad)  # y 轴反向

    ax2.text(0.02, 0.975, f"max |prediction - ground truth| = {err:.1e} px",
             transform=ax2.transAxes, fontsize=9.5, ha="left", va="top",
             bbox=dict(boxstyle="round", fc="lemonchiffon", ec="0.7"))

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=150)
    print("saved", path)
    return err


if __name__ == "__main__":
    make_figure()
