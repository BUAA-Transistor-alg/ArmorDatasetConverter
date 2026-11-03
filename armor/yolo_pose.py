# -*- coding: utf-8 -*-
"""YOLO-Pose 数据集写入器。

目录格式（与 RM_Power_Rune/generated_dataset/ 一致）：

    <输出文件夹>/
    ├── dataset.yaml
    ├── images/train/*.jpg
    ├── images/val/*.jpg
    ├── labels/train/*.txt
    └── labels/val/*.txt

每行标签格式：  class_id cx cy w h  kx1 ky1 v1  kx2 ky2 v2 ...
坐标都是相对图像宽高的归一化值；关键点可见性 v 用 2（可见且无遮挡）。

用法：
    writer = YoloPoseWriter(数据集文件夹)
    writer.add(处理后的图片, 物体列表, "train")   # 每张图调用一次
    writer.finish()                              # 全部写完后调用，生成 dataset.yaml
"""
import shutil
from pathlib import Path

import cv2
import yaml

SPLITS = ("train", "val")


class YoloPoseWriter:
    """把（图片, 物体列表）写成 YOLO-Pose 数据集。"""

    def __init__(self, dataset_dir):
        """构造时指定输出文件夹；旧的 images/ labels/ dataset.yaml 会被清掉，避免新旧数据混在一起。"""
        self.dataset_dir = Path(dataset_dir)
        self.counts = {split: 0 for split in SPLITS}
        for name in ("images", "labels"):
            shutil.rmtree(self.dataset_dir / name, ignore_errors=True)
        (self.dataset_dir / "dataset.yaml").unlink(missing_ok=True)
        for split in SPLITS:
            (self.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def add(self, image, objects, split):
        """保存一条数据：图片 + 物体列表（type/bbox/keypoints，均为归一化）。"""
        if split not in SPLITS:
            raise ValueError("split 只能是 %s，实际为 %r" % (SPLITS, split))
        index = self.counts[split]
        name = "%06d" % index
        image_path = self.dataset_dir / "images" / split / (name + ".jpg")
        label_path = self.dataset_dir / "labels" / split / (name + ".txt")
        if not cv2.imwrite(str(image_path), image):
            raise IOError("写图片失败：%s" % image_path)
        lines = [self._format_object(obj) for obj in objects]
        label_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        self.counts[split] += 1
        return image_path

    @staticmethod
    def _format_object(obj):
        """一个物体 -> 一行标签文本。"""
        cx, cy, width, height = (float(v) for v in obj["bbox"])
        parts = [str(int(obj["type"])),
                 "%.6f" % cx, "%.6f" % cy, "%.6f" % width, "%.6f" % height]
        for x, y in obj["keypoints"]:
            parts.append("%.6f" % float(x))
            parts.append("%.6f" % float(y))
            parts.append("2")  # 关键点可见性：2 = 可见且无遮挡
        return " ".join(parts)

    def finish(self):
        """遍历已写入的标签，统计类别数和关键点个数，生成 dataset.yaml。"""
        class_ids = set()
        keypoint_count = None
        for label_path in sorted((self.dataset_dir / "labels").glob("*/*.txt")):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_ids.add(int(float(parts[0])))
                count = (len(parts) - 5) // 3
                if keypoint_count is None:
                    keypoint_count = count
                elif count != keypoint_count:
                    raise ValueError("标签里的关键点个数不一致：%s" % label_path)

        num_classes = (max(class_ids) + 1) if class_ids else 0
        config = {
            "path": str(self.dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": num_classes,
            "names": ["class%d" % i for i in range(num_classes)],
            "kpt_shape": [keypoint_count or 0, 3],
            "flip_idx": [],
        }
        yaml_path = self.dataset_dir / "dataset.yaml"
        yaml_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                            encoding="utf-8")
        return yaml_path
