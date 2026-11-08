# assets/dot_matrix —— 点阵化流程（自包含、批量、无中间文件）

把 `input/` 里的图片批量渲染成 LED 点阵屏效果的独立小流程。**没有命令行参数**，
参数全部是 [`dot_matrix.py`](dot_matrix.py) 顶部的常量；每个通道的「单个 cell 蒙版」从
[`masks/`](masks/) 目录读取 PNG 文件，**不在运行时生成**，换蒙版就是换文件。

```
assets/dot_matrix/
├── dot_matrix.py         # 唯一入口：批量读 input/，只把最终图写到 output/
├── input/                # 待点阵化的图（png / jpg / jpeg / bmp，只看这一层，不递归）
├── masks/                # 单个 cell 蒙版，一个通道一张 8-bit 灰度 PNG
│   ├── circle_B.png   circle_G.png   circle_R.png     # 圆点布局
│   └── stripe_B.png   stripe_G.png   stripe_R.png     # RGB 竖条布局（当前默认）
└── output/               # 每张输入图 -> 一张同名 PNG（最终结果），没有任何中间文件
```

## 运行

```bash
python3 assets/dot_matrix/dot_matrix.py     # 或 .venv/bin/python
```

不接受任何参数。行为：

- 扫描 `input/` 里所有图片（`IMAGE_EXTS` 里的扩展名），按文件名排序逐张处理；
- 每张图输出 **`output/<原文件名去扩展名>.png`**（统一存 PNG，无损），同名直接覆盖；
- **不写任何中间文件**：下采样结果、蒙版、填充图、提亮图只在内存里；
- 读不到的图会打印「跳过」，不影响其他图；
- `output/` 里已有的其它文件不会被自动清理（想彻底重来先手动删除）。
  注意：`a.png` 和 `a.jpg` 会输出成同一个 `output/a.png`，后处理的覆盖先处理的。

## 参数常量（dot_matrix.py 顶部）

| 常量 | 默认 | 含义 |
| --- | --- | --- |
| `INPUT_DIR` | `input/` | 批量输入目录，处理这里面所有图片 |
| `MASK_DIR` | `masks/` | 蒙版目录 |
| `OUT_DIR` | `output/` | 只放最终结果的输出目录 |
| `IMAGE_EXTS` | `.png/.jpg/.jpeg/.bmp` | 认哪些扩展名（大小写不敏感） |
| `LAYOUT` | `stripe` | 用哪套蒙版：`circle`（圆点）/ `stripe`（RGB 竖条） |
| `ROWS` / `COLS` | 64 / 64 | 点阵屏分辨率 |
| `OUT_SIZE` | 2048 | 输出分辨率（取整为 cell 的整数倍） |
| `GAIN` | 1.0 | 提亮增益，越大越容易过曝成白 |
| `SIGMA` | 0.0 | 高斯模糊强度，**单位是 cell 边长**（cell 长短边不一致时取短边）；0 = 不模糊。实际像素 sigma = `SIGMA * min(cell_w, cell_h)` |

## 蒙版格式

- 文件名 `<LAYOUT>_<通道>.png`，通道后缀是 `B` / `G` / `R`（与 OpenCV 的 BGR 顺序一致）；
  `LAYOUT` 指到没有对应文件的布局会直接报错并提示缺哪几张。
- 8-bit 灰度，0 = 全暗（权重 0），255 = 全亮（权重 1.0），读进来 `/255` 当浮点权重。
- **只描述一个 cell**（一个点阵像素）内部：脚本用 `np.tile` 把它平铺到整屏
  `ROWS x COLS` 个像素上，所以全屏长得一样。
- 蒙版文件是 **16x16**；当前 `OUT_SIZE / COLS = 2048 / 64 = 32`，脚本会把蒙版放大到
  32x32 的 cell。改 `ROWS/COLS/OUT_SIZE` 后 cell 尺寸变化时脚本会自动缩放蒙版
  （缩小 `INTER_AREA`、放大 `INTER_LINEAR`），因此不用重画；按新 cell 尺寸重画蒙版
  能得到更清楚的细节。

### 两套蒙版的来源

蒙版用 SDF 公式离线生成（只生成一次，之后以 PNG 为准）：

| 布局 | 生成参数 | 形状 |
| --- | --- | --- |
| `circle` | `dot_w=0.78`，三通道圆各偏移 `±0.035cell_w / ±0.02cell_h` | 近圆亮点，边缘带红/青描边（模拟子像素错位） |
| `stripe` | `dot_w=0.78, bar_h=0.62, gap=0.05` | 三根竖直圆角条并排（RGB 子像素），每通道约 1/3 面积 |

蒙版边缘是 `clip(0.5 - d, 0, 1)` 的距离场，自带 1px 抗锯齿。
两种布局的通道都能单独换（比如只改 `circle_R.png`），互不影响。

## 处理链（每张图独立）

| 步 | 做什么 | 函数 |
| --- | --- | --- |
| 1 | 原图 `INTER_AREA` 下采样到 `ROWS x COLS` | `downsample_to_matrix()` |
| 2 | 建 `COLS*cell_w x ROWS*cell_h` 纯黑画布 | `render_dot_matrix()` |
| 3 | 每通道读一张单个 cell 蒙版 PNG 并缩放 | `load_cell_masks()` |
| 4 | `np.tile` 平铺到每个点阵像素，按通道乘颜色 | `tile_masks()` / `fill_by_channel_masks()` |
| 5-6 | uint8 → float32，乘 `GAIN` | `brighten()` |
| 7 | 把 `SIGMA`（cell 倍数）换算成像素后高斯模糊；换算结果为 0 时跳过 | `gaussian_glow()` |
| 8 | clip 回 [0,255] 转 uint8 | `clip_to_uint8()` |

## 输出

`output/<输入文件名去扩展名>.png`，尺寸 `COLS*cell_w x ROWS*cell_h`（默认 2048x2048），
BGR uint8。**只有这一张最终图，没有中间文件。**

## 依赖

`opencv-python`（或项目 `.venv` 里的 `opencv-python-headless`）+ `numpy`，与主项目一致。

## 已知取舍

- 蒙版存成 8-bit PNG 有 1/255 的量化误差，最终图最多差 1 个灰阶。
- `SIGMA = 0` 表示不模糊（硬边点阵、没有光晕）。
- `SIGMA` 以 **cell 边长为单位**（`sigma_px = SIGMA * min(cell_w, cell_h)`），
  换 `ROWS/COLS/OUT_SIZE` 后光晕和点阵的相对比例保持不变。
- `stripe` 每个通道只占 cell 的约 1/3 面积，亮度较低，通常需要调大 `GAIN`。
- 输出尺寸按 cell 取整（`OUT_SIZE` 只是目标值），例如 `COLS=80` 时结果会偏离 2048。
