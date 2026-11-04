#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 input/ 里的所有图片处理成 YOLO-Pose 数据集（输出到 output/）。

流程：读取 INPUT_DIR 下所有图片 -> PlateSynth.process() 得到（合成图, 物体列表）
      -> YoloPoseWriter.add() 写一条数据 -> 全部写完后 writer.finish() 生成 dataset.yaml。
没有命令行参数；要换目录/模型/处理逻辑，直接改下面这些常量或 armor/config.yaml。
"""
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from armor import PlateSynth, YoloPoseWriter          # noqa: E402
from armor.paste_providers import paste_content, set_seed   # noqa: E402

INPUT_DIR = ROOT / "input"      # 输入图片文件夹
DATASET_DIR = ROOT / "output"   # 输出 YOLO-Pose 数据集文件夹
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VAL_RATIO = 0.2                 # 验证集比例：val 数量 = round(总数 * VAL_RATIO)
# 统一随机种子：train/val 划分、贴图候选选择、(2) 的随机分块与拉伸方向、噪声都从它来
SEED = 0


def main():
    images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS) if INPUT_DIR.is_dir() else []
    if not images:
        print("输入文件夹里没有图片：%s" % INPUT_DIR)
        return

    # 按比例算验证集数量，训练集数量 = 总数 - 验证集数量；再随机从全部图片里抽训练集
    total = len(images)
    val_count = min(total, int(round(total * VAL_RATIO)))
    train_count = total - val_count
    train_indices = set(random.Random(SEED).sample(range(total), train_count))
    print("输入 %s（%d 张图），数据集输出到 %s" % (INPUT_DIR, total, DATASET_DIR))
    print("train %d 张 / val %d 张（VAL_RATIO=%.2f，SEED=%d）"
          % (train_count, val_count, VAL_RATIO, SEED))

    # 种子统一在这里设置：paste 候选选择、PlateSynth 内部随机（噪声/分块/方向）
    set_seed(SEED)
    synth = PlateSynth(paste_content, seed=SEED)
    writer = YoloPoseWriter(DATASET_DIR)

    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            print("跳过（读不到）：%s" % path.name)
            continue
        result, objects = synth.process(image)   # 传入 cv 图像，返回（处理后图像, 物体列表）
        split = "train" if index in train_indices else "val"
        writer.add(result, objects, split)
        print("%s -> %s，%d 个物体" % (path.name, split, len(objects)))

    yaml_path = writer.finish()
    print("完成：%s（train %d 张 / val %d 张，dataset.yaml: %s）"
          % (DATASET_DIR, writer.counts["train"], writer.counts["val"], yaml_path))


if __name__ == "__main__":
    main()
