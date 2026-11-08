# -*- coding: utf-8 -*-
"""矩形 (3)（贴图区）的尺寸、内容、关键点、类型和显示参数由这个函数提供。

候选目标写在 armor/config.yaml 的 paste_candidates 里，每次调用随机挑一个：
    paste_content(armor_type: int, color: int)
        -> (width_m, height_m, image, keypoints, obj_type, gain, blur_sigma)

    armor_type : 检测到的装甲板类别 label（0~8，含义同 C++ 的类别定义）
    color      : 颜色，0 = 红，1 = 蓝（与 C++ Object::color 约定一致）

    width_m    : 矩形 (3) 的长边（米）
    height_m   : 矩形 (3) 的短边（米）
    image      : 该候选的占位图（从 config 里的 image 路径直接读入的原始图像，
                 程序不做任何修改，只会在贴图时按矩形尺寸缩放）
    keypoints  : 该图像内的归一化关键点 [(x, y), ...]：左上 (0.0, 0.0)、右下 (1.0, 1.0)
    obj_type   : 该物体的类型（写进 YOLO 标签的类别 id）
    gain       : 贴图区亮度增益，贴图内容先乘该值
    blur_sigma : 主图像上的贴图区高斯模糊标准差（像素），0 表示不模糊

占位图是项目里预先生成的图片文件（深色底 + 红/蓝/绿边框 + 中心十字 + PLACEHOLDER 文字），
配置里当前使用的是点阵化后的 assets/placeholder_t*_dot.png，这里只负责按配置加载。
"""
import random
from pathlib import Path

import cv2
import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _PACKAGE_DIR.parent
_CONFIG_PATH = _PACKAGE_DIR / "config.yaml"
_RNG = random.Random(0)  # 固定种子：同样的运行顺序得到同样的随机选择


def _load_candidates(path):
    """从 config.yaml 读候选目标列表，并直接加载每个候选的占位图文件。"""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = config.get("paste_candidates")
    if not items:
        raise ValueError("config.yaml 里缺少 paste_candidates 列表")
    candidates = []
    for item in items:
        width, height = float(item["width"]), float(item["height"])
        if not (width > 0 and height > 0):
            raise ValueError("paste_candidates 的宽高必须大于 0：%r" % (item,))
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            image_path = _ROOT_DIR / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("读不到占位图：%s" % image_path)
        keypoints = [(float(x), float(y)) for x, y in item["keypoints"]]
        gain = float(item["gain"])
        blur_sigma = float(item["blur_sigma"])
        if not gain > 0:
            raise ValueError("paste_candidates 的 gain 必须大于 0：%r" % (item,))
        if not blur_sigma >= 0:
            raise ValueError("paste_candidates 的 blur_sigma 必须大于等于 0：%r" % (item,))
        candidates.append({
            "width": width,
            "height": height,
            "type": int(item["type"]),
            "keypoints": keypoints,
            "image": image,
            "gain": gain,
            "blur_sigma": blur_sigma,
        })
    return candidates


_CANDIDATES = _load_candidates(_CONFIG_PATH)


def set_seed(seed):
    """设置候选随机选择的种子；由 main.py 统一调用，保证整个数据集可复现。"""
    _RNG.seed(int(seed))


def paste_content(armor_type: int, color: int):
    """从候选目标里随机选一个，返回
    (width_m, height_m, image, keypoints, obj_type, gain, blur_sigma)。
    """
    candidate = _RNG.choice(_CANDIDATES)
    return (candidate["width"], candidate["height"], candidate["image"],
            list(candidate["keypoints"]), candidate["type"],
            candidate["gain"], candidate["blur_sigma"])
