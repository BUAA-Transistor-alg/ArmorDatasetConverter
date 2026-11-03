# -*- coding: utf-8 -*-
"""0726.onnx 推理，只用 onnxruntime：letterbox -> 推理 -> 解码 -> NMS。

模型输出约定（与 UnifiedAutoAimPipeline 的 C++ 后处理 postprocess0726 一致）：
    output0: float32 [1, 21, 6300]    6300 = 60*80 + 30*40 + 15*20（480x640 下的网格数）
      通道 0..3   颜色分数（图内已经 sigmoid）：0=蓝 1=红 2=None 3=紫
      通道 4..12  9 个类别分数（图内已经 sigmoid）
      通道 13..20 4 个关键点 x0,y0,x1,y1,...（letterbox 输入下的像素坐标）
关键点顺序：P0 左上 / P1 左下 / P2 右下 / P3 右上。
这个模型没有 box 回归分支：框就是四个关键点的外接矩形。
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

NUM_COLOR = 4
NUM_CLASSES = 9
NUM_KPTS = 4
OUT_CH = NUM_COLOR + NUM_CLASSES + NUM_KPTS * 2  # 21
DEFAULT_INPUT_HW = (480, 640)                    # 0726.onnx 的输入 (高, 宽)
PAD_VALUE = 114                                  # letterbox 灰边


@dataclass
class Detection:
    """一个装甲板检测结果（坐标已是原图像素）。"""

    landmarks: np.ndarray  # (4, 2) float32：左上/左下/右下/右上
    label: int             # 类别 0~8
    prob: float            # 该类别的概率
    color_id: int          # 0 = 红, 1 = 蓝（C++ Object 约定）
    color_name: str        # "red" / "blue"


def _letterbox(image, target_hw):
    """等比缩放并居中补灰边，返回 (图, scale, left, top)；反变换用同样这三个数。"""
    h0, w0 = image.shape[:2]
    th, tw = target_hw
    scale = min(th / h0, tw / w0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    if (nw, nh) != (w0, h0):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (tw - nw) / 2.0, (th - nh) / 2.0
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
    return padded, float(scale), left, top


def _preprocess(image, target_hw):
    """BGR 图像 -> (1, 3, H, W) float32 RGB/255 张量 + letterbox 参数。"""
    padded, scale, left, top = _letterbox(image, target_hw)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), scale, left, top


def _nms(boxes, scores, iou_threshold):
    """贪心 NMS（与 cv::dnn::NMSBoxes 行为一致），boxes 为 (N,4) 的 x,y,w,h。"""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = np.maximum(boxes[:, 2], 0) * np.maximum(boxes[:, 3], 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _polygon_area(points):
    """多边形面积（Shoelace），points 形状 (N, K, 2)。"""
    x, y = points[..., 0], points[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1))


def _decode(raw, scale, left, top, orig_hw, conf_threshold, nms_threshold):
    """原始输出 -> Detection 列表：取类别/颜色，置信度过滤，反 letterbox，NMS。"""
    out = np.asarray(raw, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if out.shape[0] != OUT_CH:
        raise ValueError("输出通道数应为 %d，实际 %d" % (OUT_CH, out.shape[0]))

    finite = np.isfinite(out).all(axis=0)
    color_scores = out[:NUM_COLOR]
    cls_scores = out[NUM_COLOR:NUM_COLOR + NUM_CLASSES]
    kpts = np.transpose(out[NUM_COLOR + NUM_CLASSES:].reshape(NUM_KPTS, 2, -1), (2, 0, 1))

    label = cls_scores.argmax(axis=0)
    prob = np.take_along_axis(cls_scores, label[None], axis=0)[0]
    model_color = color_scores.argmax(axis=0)                     # 0=蓝 1=红
    keep = np.flatnonzero((prob >= conf_threshold) & finite & (model_color < 2))
    if keep.size == 0:
        return []
    label, prob, model_color = label[keep], prob[keep], model_color[keep]

    # 关键点从 letterbox 像素映射回原图
    points = (kpts[keep] - np.array([left, top], np.float32)) / scale
    orig_h, orig_w = orig_hw
    inside = ((points[..., 0] >= 0) & (points[..., 0] <= orig_w)
              & (points[..., 1] >= 0) & (points[..., 1] <= orig_h)).all(axis=1)
    points, label, prob, model_color = points[inside], label[inside], prob[inside], model_color[inside]
    if points.shape[0] == 0:
        return []

    length = np.linalg.norm(points[:, 0] - points[:, 3], axis=1)
    width = np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
    valid = (length > 0) & (width > 0) & (_polygon_area(points) > 1e-3)
    points, label, prob, model_color = points[valid], label[valid], prob[valid], model_color[valid]
    if points.shape[0] == 0:
        return []

    x0, y0 = points[..., 0].min(axis=1), points[..., 1].min(axis=1)
    x1, y1 = points[..., 0].max(axis=1), points[..., 1].max(axis=1)
    boxes = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1).astype(np.float32)

    detections = []
    for i in _nms(boxes, prob, nms_threshold):
        detections.append(Detection(
            landmarks=points[i].astype(np.float32),
            label=int(label[i]),
            prob=float(prob[i]),
            color_id=int(1 - model_color[i]),                     # 模型 0=蓝/1=红 -> Object 0=红/1=蓝
            color_name="blue" if int(model_color[i]) == 0 else "red",
        ))
    return detections


class ArmorDetector:
    """0726.onnx 检测器；输入 BGR 图像，输出 Detection 列表（坐标已还原到原图）。"""

    CONF_THRESHOLD = 0.65  # 与 C++ ArmorPipeline 默认值相同
    NMS_THRESHOLD = 0.45

    def __init__(self, model_path, providers=("CPUExecutionProvider",)):
        self.session = ort.InferenceSession(str(model_path), providers=list(providers))
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        shape = model_input.shape
        if len(shape) == 4 and all(isinstance(v, int) and v > 0 for v in shape[2:4]):
            self.input_hw = (int(shape[2]), int(shape[3]))
        else:
            self.input_hw = DEFAULT_INPUT_HW

    def detect(self, image):
        """输入 BGR 图像，返回该图中所有装甲板的检测结果。"""
        tensor, scale, left, top = _preprocess(image, self.input_hw)
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return _decode(raw, scale, left, top, image.shape[:2],
                       self.CONF_THRESHOLD, self.NMS_THRESHOLD)
