# -*- coding: utf-8 -*-
"""装甲板数据集转换：推理 + 三个矩形的采样/填充/贴图 + YOLO-Pose 写入。"""
from .detector import ArmorDetector, Detection
from .synth import PlateSynth
from .yolo_pose import YoloPoseWriter

__all__ = ["ArmorDetector", "Detection", "PlateSynth", "YoloPoseWriter"]
