# ArmorDatasetConverter

把图片里检测到的装甲板替换成合成内容，并直接生成 **YOLO-Pose 数据集**：
检测 → 装甲板平面内取三个同心矩形 → 最暗 30% 采样 → 边界外像素按随机方向拉伸填充 + 同方差噪声 →
贴图区由 paste 函数提供尺寸/内容/关键点/类型（随机从配置的候选目标里选）→
贴图区再叠一层同样的噪声 → 投影出归一化物体框和关键点 → 写标签。
推理只用 onnxruntime，不需要 torch / ultralytics。

## 目录结构

```
ArmorDatasetConverter/
├── main.py                     # 主入口：input/ 全部图片 -> output/ YOLO-Pose 数据集（无命令行参数）
├── visualize_dataset.py        # 独立脚本：可视化 output/ 里的数据集（无命令行参数）
├── armor/                      # 全部 python 代码
│   ├── __init__.py
│   ├── config.yaml             # 大/小装甲板几何配置 + 贴图区候选目标列表
│   ├── detector.py             # 0726.onnx 推理（onnxruntime）：letterbox / 解码 / NMS
│   ├── homography.py           # 四点 DLT、纹理坐标 <-> 装甲板平面
│   ├── synth.py                # PlateSynth：三个矩形 + 采样/填充/贴图 + 物体标注
│   ├── paste_providers.py      # 从配置的候选目标里随机选一个，返回尺寸/内容/关键点/类型
│   └── yolo_pose.py            # YoloPoseWriter：YOLO-Pose 数据集写入器
├── assets/                      # 三张预先生成的占位图（深色底 + 红/蓝/绿边框），程序直接 imread
│   ├── placeholder_t0.png
│   ├── placeholder_t1.png
│   └── placeholder_t2.png
├── models/0726.onnx
├── input/                      # 输入图片（4 张样本 + 20 张 InputVideo 随机帧）
├── output/                     # 输出数据集（YoloPoseWriter 生成，可随时重跑覆盖）
├── requirements.txt
└── rect_homography_notes/      # 平面矩形单应重投影笔记（数学来源，独立）
```

## 环境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

依赖：`onnxruntime`（CPU）、`numpy`、`opencv-python-headless`、`pyyaml`。

## 用法

```bash
.venv/bin/python main.py               # 生成数据集
.venv/bin/python visualize_dataset.py  # 可视化生成的数据集
```

两个脚本都没有命令行参数，路径/比例写死在脚本顶部常量里：

| 脚本 | 常量 | 含义 |
| --- | --- | --- |
| `main.py` | `INPUT_DIR` / `DATASET_DIR` | 输入图片文件夹（`input/`）/ 输出数据集文件夹（`output/`） |
| `main.py` | `VAL_RATIO` / `SEED` | 验证集比例（默认 0.2）/ **统一随机种子**（默认 0） |
| `visualize_dataset.py` | `DATASET_DIR` / `SPLIT` | 固定看 `output/` 数据集 / 看哪个 split |

### train/val 划分

`val 数量 = round(总数 * VAL_RATIO)`，`train 数量 = 总数 - val 数量`，
然后用 `random.Random(SEED).sample(...)` 从**全部图片**里随机抽出 train，剩下的作 val。

所有随机性都由 `main.py` 里的 `SEED` 统一控制：train/val 划分、paste 候选选择、
(2) 的随机分块与拉伸方向、噪声；改这一个常量就能整体换一套随机结果。

## 配置：armor/config.yaml

尺寸单位米，`width` 是长边、`height` 是短边。大/小装甲板分类与真实尺寸取自
`UnifiedAutoAimPipeline`（`ArmorModel.h` 的 0.225 / 0.133 / 0.050，
`ArmorPipeline.cpp` 里 `label==1 || label==8` 用大装甲板，其余用小装甲板）。

| 类型 | 类别 label | 原矩形 | (1) 采样区 | (2) 填充区 |
| --- | --- | --- | --- | --- |
| 大装甲板 | 1, 8 | 0.225 x 0.050 | 0.1125 x 0.100 | 0.2925 x 0.127 |
| 小装甲板 | 0,2,3,4,5,6,7 | 0.133 x 0.050 | 0.0665 x 0.100 | 0.1729 x 0.125 |

其余项：`sampling.dark_ratio`、`fill.noise_gain`、`fill.clip`。

**(3) 贴图区**由 `paste_candidates` 列表配置，每项包括：

```yaml
paste_candidates:
  - width: 0.05
    height: 0.05
    type: 0
    image: assets/placeholder_t0.png   # 红
    keypoints: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
  - width: 0.10
    height: 0.10
    type: 1
    image: assets/placeholder_t1.png   # 蓝
    keypoints: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
  - width: 0.10
    height: 0.05
    type: 2
    image: assets/placeholder_t2.png   # 绿
    keypoints: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
```

| 字段 | 含义 |
| --- | --- |
| `width, height` | 矩形 (3) 的真实尺寸（米，width 为长边） |
| `type` | 写进 YOLO 标签的类别 id（int） |
| `image` | 该候选占位图的文件路径（相对项目根目录）；程序直接 `cv2.imread` 加载，不做任何修改 |
| `keypoints` | 内容图内的归一化关键点，左上 (0,0)、右下 (1,1) |

所有候选的关键点个数必须一致（YOLO-Pose 的 `kpt_shape` 是统一的），否则 `finish()` 会报错。

## 矩形 (3)：paste 函数

`armor/paste_providers.py`：

```python
def paste_content(armor_type: int, color: int):
    return width_m, height_m, image, keypoints, obj_type
```

| 参数/返回 | 含义 |
| --- | --- |
| `armor_type` | 检测到的类别 label（0~8：0~5 哨兵 / 1~5 号机器人，6 装甲板，7 基地，8 基地大装甲） |
| `color` | 颜色，**0 = 红，1 = 蓝**（与 C++ `Object::color` 一致） |
| `width_m, height_m` | 矩形 (3) 的真实尺寸（米） |
| `image` | 要贴进去的 BGR 图像（`HxWx3 uint8`），会缩放到该矩形纹理尺寸后单应映射 |
| `keypoints` | 该图像内的归一化关键点 `[(x, y), ...]` |
| `obj_type` | 该物体的类型，写进 YOLO 标签的类别 id |

当前实现：**从 `config.yaml` 的 `paste_candidates` 里随机挑一个**（随机数由 `main.py` 的 `SEED`
统一设置，所以整个数据集可复现）。占位图是三张**预先生成好的图片文件**（`assets/placeholder_t0/t1/t2.png`，
红/蓝/绿），这里只按配置的 `image` 路径直接 `cv2.imread` 加载，不在程序里画图/改色。
换真实图片时只改候选列表里的 `image` 路径（或用同签名函数替换 paste_content）。

## 处理流程（每张图里的所有装甲板）

1. **检测**：`0726.onnx` 输入 `[1,3,480,640]`（RGB/255，letterbox 灰边 114），输出
   `[1,21,6300]`：通道 0~3 颜色（图内已 sigmoid）、4~12 类别、13~20 四个关键点；
   关键点顺序左上/左下/右下/右上，没有 box 分支。解码、置信度过滤（0.65）、NMS（0.45）
   与原 C++ 后处理一致。
2. **求单应**：四个关键点 + 装甲板真实尺寸做四点 DLT，得到「板平面 -> 图像」的 `H`；
   三个矩形与板同心，共用 `H`。
3. **(1) 采样**：把采样区反投影成纹理，取区内全部像素（落在图外的剔除），
   按亮度升序取最暗的 30%，统计 B/G/R 各通道均值与方差。
4. **(2) 填充**：把 (2) 矩形往外扩一圈反投影，拿到边界外的像素；在 (2) 内部随机切成 3~6 块
   （Voronoi），每块随机一个拉伸方向，块内像素沿该方向射到 (2) 边界、再往外偏几个像素取样，
   等价于把边界外的像素沿相反方向拉伸进来，铺满整个 (2)；最后叠加与 (1) 采样方差相同的高斯噪声
   （`noise_gain` 可缩放），截断到 `[0,255]`。
5. **(3) 贴图**：调用 paste 函数拿到 `(宽, 高, 图, 关键点, 类型)`，图缩放后先**叠加上与 (2) 区域
   相同的噪声**（同样的逐通道标准差），再单应映射进贴图区。
6. **出标注**：把关键点（内容图归一化坐标）经同一个 `W` 投影到输出图像、归一化；
   物体框 = 该物体所有关键点的最小包围框（归一化 `cx,cy,w,h`，裁剪到图像内）。

两块板的区域可能互相覆盖，所以先从**原图**把所有板采样完，再统一合成；
关键点退化或有效像素太少的板会被跳过。

## 输出数据集（YOLO-Pose）

格式与 `RM_Power_Rune/generated_dataset/` 一致：

```
output/
├── dataset.yaml
├── images/train/000000.jpg ...
├── images/val/000000.jpg ...
├── labels/train/000000.txt ...
└── labels/val/000000.txt ...
```

每行标签：`class_id cx cy w h kx1 ky1 v1 kx2 ky2 v2 ...`（全部归一化，关键点可见性 v=2）。

`dataset.yaml` 由 `YoloPoseWriter.finish()` 扫描已写入的标签生成；当前 3 个候选类型都出现过，所以：

```yaml
path: /home/huhu233/rm2027/ArmorDatasetConverter/output
train: images/train
val: images/val
nc: 3
names:
- class0
- class1
- class2
kpt_shape:
- 5
- 3
flip_idx: []
```

类别数和关键点个数都是从实际写入的标签统计的；改候选列表后 `dataset.yaml` 会自动跟着变。
重跑 `main.py` 会清掉旧的 `images/`、`labels/`、`dataset.yaml`，不会新旧混在一起。

## 可视化

`visualize_dataset.py` 固定看 `output/dataset.yaml`：逐张画框 + 关键点 + 骨架并显示
（需要带窗口支持的 opencv：n 下一张，q 退出）；headless 版只打印类别统计，不写文件。

## 已验证

- `input/` 里有 24 张图（4 张原先的样本 + 从 `/home/huhu233/rm2026/transistor_rm2026_algorithm_visual_ws/InputVideo` 的 12 个 mp4 里随机抽帧、只保留有检测目标的 20 帧，`random.Random(1)`，来自 8 个不同视频；文件在 `input/video_<视频名>_f<帧号>.jpg`。
  抽取时按保存后的 jpg 复检过一遍检测结果（有一帧因 JPEG 重编码掉到阈值以下，已换成同视频的另一帧），20 帧都确实带目标）。
- `main.py` 跑通：`VAL_RATIO=0.2` → val 5 张、train 19 张，随机划分（seed 0）。
- 随机候选生效：28 个物体里 type0 8 个 / type1 11 个 / type2 9 个，`dataset.yaml` 自动 `nc: 3, names: class0..2`。
- 占位图：三张图片文件 `assets/placeholder_t0/t1/t2.png` 预先保存，**深色底 + 彩色边框**（t0 红 / t1 蓝 / t2 绿，中间仍是深灰底 + 十字 + PLACEHOLDER 文字），程序直接加载；
  对贴图区宽度 ≥ 40 px 的 11 个目标按边框环采样，主通道全部与类型一致（R / B / G）。
- 标签每行 5 + 5x3 = 20 个数值，`kpt_shape: [5, 3]`；物体框与关键点最小包围框一致。
- 新的 (2) 填充：把 paste 目标临时改小后单独看 (2)，能确认边界外的像素被随机方向拉伸铺满、
  3~6 块方向不同，并且叠加了噪声；重复跑两次所有输出图片 md5 一致（确定性保持）。
- 贴图区噪声：把 `noise_gain` 设为 0 与默认 1.0 对比，贴图区内像素平均差 4.35 / 最大 21，
  说明 (3) 区域确实叠上了与 (2) 相同的噪声。
- 同一份输入重复跑两次，所有图片、标签 md5 完全一致（`SEED` 统一控制划分/候选/分块方向/噪声）。
- 可视化脚本在有窗口的 opencv 下逐张显示；当前 venv 是 headless opencv，只打印类别统计。

