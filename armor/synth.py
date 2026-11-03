# -*- coding: utf-8 -*-
"""矩形提取与合成：PlateSynth。

对一张图里的每个装甲板：
  1. 用四个关键点求出「装甲板平面 -> 图像」的单应 H；
  2. (1) 采样区：取区内像素，按亮度排序取最暗的 dark_ratio，统计各通道均值/方差；
  3. (2) 填充区：用该均值 + 同方差高斯噪声填充；
  4. (3) 贴图区：尺寸和内容来自构造时传入的 paste 函数，贴完后把该函数给出的
     关键点投影到图像上，得到归一化的物体框和关键点，一起返回给调用者。
三个矩形都与装甲板同心，共用同一个 H。
"""
from pathlib import Path

import cv2
import numpy as np
import yaml

from .detector import ArmorDetector
from .homography import plane_corners, solve_homography, texture_to_plane

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
DEFAULT_CONFIG = PACKAGE_DIR / "config.yaml"
DEFAULT_MODEL = ROOT_DIR / "models" / "0726.onnx"
MIN_TEXTURE_PX = 8      # 目标区纹理边长下限（像素）
MAX_TEXTURE_PX = 4096   # 目标区纹理边长上限（像素）


def _load_config(path):
    """读取 config.yaml；只做最基本的合法性检查。"""
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    for name, spec in config["plate_types"].items():
        for key in ("size", "sample", "fill"):
            rect = spec[key]
            if not (float(rect["width"]) > 0 and float(rect["height"]) > 0):
                raise ValueError("config：%s.%s 的宽高必须大于 0" % (name, key))
    return config


def _rect(spec, key):
    """取出配置里的一个矩形，返回 (宽, 高)，单位米。"""
    rect = spec[key]
    return float(rect["width"]), float(rect["height"])


class PlateSynth:
    """把一张图里所有装甲板按配置替换掉，并返回替换后的图像和物体标注。

    构造参数：
        paste_fn : 贴图区 (3) 的提供函数，形式为
                   paste_fn(armor_type: int, color: int) ->
                       (width_m, height_m, image, keypoints, obj_type)
                   armor_type 是类别 label（0~8），color 是 0=红 / 1=蓝；
                   keypoints 是 image 内的归一化关键点，obj_type 是 YOLO 类别 id。
    """

    def __init__(self, paste_fn, model_path=DEFAULT_MODEL, config_path=DEFAULT_CONFIG):
        if not callable(paste_fn):
            raise TypeError("paste_fn 必须是可调用的函数")
        self.paste_fn = paste_fn
        self.config = _load_config(config_path)
        self.detector = ArmorDetector(model_path)
        self.rng = np.random.default_rng(0)  # 固定噪声种子：同样的输入得到同样的输出

    # ------------------------------------------------------------------ 对外接口
    def process(self, image):
        """输入一张 BGR 图像，返回 (替换后的图像, 物体列表)。

        物体列表里每个元素是：
            {"type": int,                       # paste 函数给的类别 id
             "bbox": [x, y, w, h],              # 归一化，中心点 + 宽高（YOLO 格式）
             "keypoints": [[x, y], ...]}        # 归一化关键点，顺序与 paste 函数返回一致
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image 必须是 HxWx3 的 BGR 图像")

        detections = self.detector.detect(image)

        # 先从原图给每块板采样；一块板的填充区经常盖住相邻板的采样区，
        # 所以必须全部采样完，再统一往图上写。
        jobs = []
        for detection in detections:
            try:
                plan = self._plan(detection)
            except np.linalg.LinAlgError:
                continue  # 关键点退化（三点共线等），跳过这块板
            stats = self._sample(image, plan["sample"])
            if stats is not None:
                jobs.append((plan, stats))

        result = image.copy()
        objects = []
        for plan, stats in jobs:
            self._fill(result, plan["fill"], stats)
            self._paste(result, plan["paste"], plan["paste_image"], stats)
            obj = self._make_object(plan, result.shape)
            if obj is not None:
                objects.append(obj)
        return result, objects

    # ------------------------------------------------------------------ 几何
    def _plan(self, detection):
        """求 H、三个目标矩形、贴图内容以及贴图区的关键点/类型。"""
        spec = self._spec_for_label(detection.label)
        plate_w, plate_h = _rect(spec, "size")

        points = np.asarray(detection.landmarks, dtype=np.float64)
        H = solve_homography(plane_corners(plate_w / 2, plate_h / 2), points)

        # 图像上每米多少像素：用检测框长边/短边的平均长度除以真实边长
        ppm_w = 0.5 * (np.linalg.norm(points[0] - points[3]) + np.linalg.norm(points[1] - points[2])) / plate_w
        ppm_h = 0.5 * (np.linalg.norm(points[0] - points[1]) + np.linalg.norm(points[3] - points[2])) / plate_h

        sample_w, sample_h = _rect(spec, "sample")
        fill_w, fill_h = _rect(spec, "fill")
        paste_w, paste_h, paste_image, paste_keypoints, obj_type = self._paste_content(detection)

        return {
            "sample": self._target(H, ppm_w, ppm_h, sample_w, sample_h),
            "fill": self._target(H, ppm_w, ppm_h, fill_w, fill_h),
            "paste": self._target(H, ppm_w, ppm_h, paste_w, paste_h),
            "paste_image": paste_image,
            "paste_keypoints": paste_keypoints,
            "paste_type": obj_type,
        }

    def _paste_content(self, detection):
        """调用 paste 函数并检查返回值：(宽, 高, 图, 关键点, 类型)。"""
        result = self.paste_fn(int(detection.label), int(detection.color_id))
        if not isinstance(result, (tuple, list)) or len(result) != 5:
            raise TypeError("paste 函数必须返回 (width_m, height_m, image, keypoints, type) 五个值")
        width_m, height_m, image, keypoints, obj_type = result
        width_m, height_m = float(width_m), float(height_m)
        if not (width_m > 0 and height_m > 0):
            raise ValueError("paste 函数返回的尺寸必须大于 0，实际为 (%r, %r)" % (width_m, height_m))
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("paste 函数返回的图像必须是 HxWx3 的 BGR 图像，实际形状 %s" % (image.shape,))
        keypoints = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
        if keypoints.size and not np.isfinite(keypoints).all():
            raise ValueError("paste 函数返回的关键点必须是有限数值")
        return width_m, height_m, image, keypoints, int(obj_type)

    def _spec_for_label(self, label):
        """按类别 label 找大/小装甲板配置（label 1、8 是大装甲板）。"""
        for spec in self.config["plate_types"].values():
            if int(label) in spec["labels"]:
                return spec
        raise ValueError("config 里没有为类别 %r 配置装甲板类型" % (label,))

    def _target(self, H, ppm_w, ppm_h, width_m, height_m):
        """一个目标矩形：纹理分辨率 + 「纹理像素 -> 图像像素」的单应 W。"""
        tex_w = int(round(width_m * ppm_w))
        tex_h = int(round(height_m * ppm_h))
        tex_w = min(MAX_TEXTURE_PX, max(MIN_TEXTURE_PX, tex_w))
        tex_h = min(MAX_TEXTURE_PX, max(MIN_TEXTURE_PX, tex_h))
        W = H @ texture_to_plane(width_m, height_m, tex_w, tex_h)
        return {"width_m": width_m, "height_m": height_m,
                "tex_w": tex_w, "tex_h": tex_h, "W": W}

    # ------------------------------------------------------------------ 三步
    def _sample(self, image, target):
        """(1) 采样：把采样区反投影成纹理，取最暗的 dark_ratio 像素统计均值/方差。

        落在图像外的像素会被剔除；有效像素太少时返回 None（跳过该装甲板）。
        """
        inv_W = np.linalg.inv(target["W"])
        size = (target["tex_w"], target["tex_h"])
        texture = cv2.warpPerspective(image, inv_W, size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        ones = np.full(image.shape[:2], 255, dtype=np.uint8)
        mask = cv2.warpPerspective(ones, inv_W, size, flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0

        pixels = texture[mask].astype(np.float64)              # (N, 3)，BGR
        if pixels.shape[0] < 8:
            return None
        brightness = pixels.mean(axis=1)
        count = max(1, int(round(self.config["sampling"]["dark_ratio"] * pixels.shape[0])))
        darkest = pixels[np.argsort(brightness, kind="stable")[:count]]
        return {"mean": darkest.mean(axis=0), "var": darkest.var(axis=0)}

    def _noise(self, target, stats):
        """与 (2) 区域相同的白噪声：逐通道 N(0, sqrt(采样方差) * noise_gain)。"""
        gain = float(self.config["fill"]["noise_gain"])
        std = np.sqrt(np.maximum(stats["var"], 0.0)) * gain
        return self.rng.normal(0.0, 1.0, size=(target["tex_h"], target["tex_w"], 3)) * std[None, None, :]

    def _fill(self, image, target, stats):
        """(2) 填充：通道均值 + 上述白噪声，截断到 clip 范围后贴进填充区。"""
        low, high = (float(v) for v in self.config["fill"]["clip"])
        texture = np.clip(stats["mean"][None, None, :] + self._noise(target, stats), low, high)
        self._blend(image, np.rint(texture).astype(np.uint8), target["W"])

    def _paste(self, image, target, content, stats):
        """(3) 贴图：缩放内容图 -> 叠加上与 (2) 区域相同的噪声 -> 单应映射进贴图区。"""
        if content.shape[1] != target["tex_w"] or content.shape[0] != target["tex_h"]:
            shrinking = content.shape[1] > target["tex_w"] or content.shape[0] > target["tex_h"]
            content = cv2.resize(content, (target["tex_w"], target["tex_h"]),
                                 interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
        noisy = content.astype(np.float32) + self._noise(target, stats)
        self._blend(image, np.clip(np.rint(noisy), 0, 255).astype(np.uint8), target["W"])

    @staticmethod
    def _blend(image, texture, W):
        """把 texture 通过单应 W 贴到 image 上（带软边 alpha 合成，原地修改 image）。"""
        h, w = image.shape[:2]
        warped = cv2.warpPerspective(texture, W, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = np.full(texture.shape[:2], 255, dtype=np.uint8)
        alpha = cv2.warpPerspective(mask, W, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        alpha = (alpha.astype(np.float32) / 255.0)[..., None]
        blended = image.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
        image[...] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ 标注
    def _make_object(self, plan, image_shape):
        """把贴图区的关键点投影到输出图像，返回归一化的物体框 + 关键点。"""
        keypoints = plan["paste_keypoints"]
        if keypoints.shape[0] == 0:
            return None
        target = plan["paste"]
        # 1) 内容图归一化坐标 -> 内容图像素坐标；2) 单应映射到输出图像像素
        texture_px = keypoints * np.array([target["tex_w"], target["tex_h"]], dtype=np.float64)
        homogeneous = np.hstack([texture_px, np.ones((texture_px.shape[0], 1))])
        image_px = (target["W"] @ homogeneous.T).T
        image_px = image_px[:, :2] / image_px[:, 2:3]
        # 3) 归一化到图像（左上 (0,0)、右下 (1,1)），并裁剪到图像内
        height, width = image_shape[:2]
        normalized = image_px / np.array([width, height], dtype=np.float64)
        x0, y0 = normalized.min(axis=0)
        x1, y1 = normalized.max(axis=0)
        x0, y0 = max(float(x0), 0.0), max(float(y0), 0.0)
        x1, y1 = min(float(x1), 1.0), min(float(y1), 1.0)
        if x1 <= x0 or y1 <= y0:
            return None  # 整块贴图都在图像外，不产生标注
        return {
            "type": int(plan["paste_type"]),
            "bbox": [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0],
            "keypoints": np.clip(normalized, 0.0, 1.0).tolist(),
        }
