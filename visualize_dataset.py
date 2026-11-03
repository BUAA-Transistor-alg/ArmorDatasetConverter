#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化 output/ 里的 YOLO-Pose 数据集（无命令行参数）。

读取 output/dataset.yaml，逐张画出物体框、关键点和骨架。
改 split 就把下面的 SPLIT 改成 "val"。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "output"       # 固定看 output 数据集
SPLIT = "train"                     # 想换 val 就改成 "val"

# 关键点骨架：4 个角点连成四边形，中心和每个角点相连（按 5 个关键点的约定画）
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]

BOX_COLOR = (0, 255, 0)       # 绿
KPT_COLOR = (0, 0, 255)       # 红
LINE_COLOR = (255, 128, 0)    # 蓝


def load_dataset():
    """读 dataset.yaml，返回 (类别名 dict, 关键点个数, [(图片路径, 标签路径), ...])。"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = config.get("names", [])
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    else:
        names = {i: name for i, name in enumerate(names)}
    keypoint_count = int(config.get("kpt_shape", [0, 3])[0])

    base = Path(config.get("path", DATASET_DIR))
    if not base.is_absolute():
        base = DATASET_DIR / base
    image_dir = base / config.get(SPLIT, "images/" + SPLIT)
    label_dir = base / "labels" / SPLIT
    pairs = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        label_path = label_dir / (image_path.stem + ".txt")
        pairs.append((image_path, label_path if label_path.exists() else None))
    return names, keypoint_count, pairs


def draw_object(image, class_id, class_name, bbox, keypoints):
    """在图上画一个物体的框、关键点和骨架（坐标均为归一化）。"""
    height, width = image.shape[:2]
    cx, cy, bw, bh = bbox
    x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
    x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
    cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.putText(image, "%s(%d)" % (class_name, class_id), (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)

    points = []
    for x, y, visible in keypoints:
        if visible >= 1:
            point = (int(x * width), int(y * height))
            cv2.circle(image, point, 3, KPT_COLOR, -1, lineType=cv2.LINE_AA)
            points.append(point)
        else:
            points.append(None)

    for start, end in SKELETON:
        if start < len(points) and end < len(points):
            if points[start] is not None and points[end] is not None:
                cv2.line(image, points[start], points[end], LINE_COLOR, 2, lineType=cv2.LINE_AA)


def parse_label_line(line, keypoint_count):
    """一行标签 -> (类别 id, bbox, 关键点列表)；格式不对返回 None。"""
    parts = [float(v) for v in line.split()]
    if len(parts) < 5 + 3 * keypoint_count:
        return None
    keypoints = []
    for i in range(keypoint_count):
        index = 5 + 3 * i
        keypoints.append((parts[index], parts[index + 1], int(parts[index + 2])))
    return int(parts[0]), parts[1:5], keypoints


def _has_gui():
    """试探当前 opencv 是否能开窗口（headless 版本会在 imshow/namedWindow 里报错）。"""
    try:
        cv2.namedWindow("probe")
        cv2.destroyWindow("probe")
        return True
    except cv2.error:
        return False


def main():
    yaml_path = DATASET_DIR / "dataset.yaml"
    if not yaml_path.is_file():
        print("找不到数据集：%s（先运行 main.py）" % yaml_path)
        return
    names, keypoint_count, pairs = load_dataset()
    if not pairs:
        print("%s 里没有图片" % SPLIT)
        return
    print("数据集：%s（%s 共 %d 张图，%d 个关键点/物体）" % (DATASET_DIR, SPLIT, len(pairs), keypoint_count))

    class_counts = {}
    can_show = _has_gui()
    if can_show:
        print("按任意键看下一张，q 退出。")
    else:
        print("当前 opencv 没有窗口支持")

    for index, (image_path, label_path) in enumerate(pairs):
        image = cv2.imread(str(image_path))
        if image is None:
            print("读不到图片：%s" % image_path)
            continue
        if label_path is not None:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parsed = parse_label_line(line, keypoint_count)
                if parsed is None:
                    continue
                class_id, bbox, keypoints = parsed
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                draw_object(image, class_id, names.get(class_id, "class%d" % class_id), bbox, keypoints)

        print("显示 %d/%d: %s" % (index + 1, len(pairs), image_path.name))
        key = ord("n")
        if can_show:
            try:
                cv2.imshow("YOLO-Pose", image)
                key = cv2.waitKey(0) & 0xFF
            except cv2.error:
                can_show = False
                print("当前 opencv 不支持窗口。")
        if key == ord("q"):
            break
    if can_show:
        cv2.destroyAllWindows()

    print("\n类别统计：")
    for class_id in sorted(class_counts):
        print("  %d (%s): %d 个" % (class_id, names.get(class_id, "?"), class_counts[class_id]))


if __name__ == "__main__":
    main()
