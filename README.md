# 皮肤纹理方向分析系统

一个基于深度学习与计算机视觉的皮肤纹理方向分析工具，支持从单病例到全量批处理，自动生成分割、纹理、方向、严重度与报告结果。

## 快速开始（运行方法）

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 网页 Demo（上传 + 调参 + 重算）

```bash
streamlit run web_demo.py
```

### 3) 单病例完整流程（推荐）

```bash
python src/run_case_to_results.py 66
```

### 4) 全量批处理（推荐）

```bash
python src/run_all_cases_to_results.py --skip-existing
```

### 5) 单图全流程（路径或病例ID）

```bash
python src/run_one_full_pipeline.py 66
```

或：

```bash
python src/run_one_full_pipeline.py dataset/final_labeled/66.jpg
```

## 项目概述

当前流程包含以下阶段：

1. 皮肤分割：使用 U-Net 三分类分割（默认使用 Trans-UNet 权重）
2. 纹理提取：在目标区域内提取纹理线条
3. 方向分析：结构张量 + 8 扇区统计（密度、一致性、综合分数）
4. 局部严重度：像素级局部评分，输出 severity_map 与 presence_map
5. 最严重框分析：在有效区域内筛选最严重框并计算框内主方向
6. 报告与汇总：输出标准化目录 results 和精简目录 final_results

## 项目结构（已更新）

```text
best_unet_model/
├── dataset/
│   ├── final_labeled/                     # 主数据集，默认输入目录（{id}.jpg）
│   └── 入选动态图/                          # 支持中文路径的样例目录
├── best_unet_model.pth                    # 旧 U-Net 权重
├── best_trans_unet_model_20250614_122913.pth  # 当前默认模型权重
├── main.py                                # 传统单例主流程入口
├── Unet.py                                # U-Net 模型定义/训练脚本
├── skin.py                                # 纹理提取模块
├── predict.py                             # 方向与扇区分析模块
├── report.py                              # 报告生成模块
├── test.py                                # 单图分割可视化（含 line-only 叠加）
├── web_demo.py                            # Streamlit 网页 Demo（上传、全流程中间结果、参数调节重算）
├── src/
│   ├── run_all_overlays.py                # 批量分割可视化到 results/<id>/01_segmentation
│   ├── run_one_full_pipeline.py           # 单图全流程（支持中文路径）
│   ├── local_score_heatmap.py             # severity/presence 局部评分输出
│   ├── worst_box_direction.py             # 最严重框与框内方向分析
│   ├── build_final_results.py             # results -> final_results 精简映射
│   ├── run_case_to_results.py             # 单病例一键生成完整目录 + 精简目录
│   └── run_all_cases_to_results.py        # 全量批处理入口（支持断点跳过）
├── run_all_overlays.py                    # 兼容入口（转调 src）
├── run_one_full_pipeline.py               # 兼容入口（转调 src）
├── local_score_heatmap.py                 # 兼容入口（转调 src）
├── worst_box_direction.py                 # 兼容入口（转调 src）
├── build_final_results.py                 # 兼容入口（转调 src）
├── run_case_to_results.py                 # 兼容入口（转调 src）
├── run_all_cases_to_results.py            # 兼容入口（转调 src）
├── skin_output/                           # 中间输出：纹理图
├── predict_output/                        # 中间输出：方向图、扇区数据
├── final_output/                          # 中间输出：最终叠加图
├── heatmap_output/                        # 中间输出：severity/presence/worst 框
├── report/                                # 中间输出：分析报告
├── results/                               # 标准完整结果目录（按病例）
│   └── <id>/
│       ├── 01_segmentation/
│       ├── 02_texture/
│       ├── 03_orientation/
│       ├── 04_final_overlay/
│       ├── 05_report/
│       ├── 06_heatmap_r40/
│       └── 07_worst_boxes/
└── final_results/                         # 精简交付目录（按病例）
    └── <id>/
        ├── 01_segment.png
        ├── 02_texture.png
        ├── 03_orientation.png
        ├── 04_orientation_overlay.png
        ├── 05_sector_details/
        ├── 06_severity.png
        ├── 07_worst.png
        └── 08_presence_overlay.png
```

## 核心脚本说明

### 1) 分析主链路

- main.py
  - 传统入口，执行 skin -> predict -> final -> report
- src/run_case_to_results.py
  - 推荐单病例入口，统一输出到 results/<id> 与 final_results/<id>
- src/run_all_cases_to_results.py
  - 推荐全量入口，批量遍历 dataset/final_labeled/*.jpg

### 2) 局部严重度与最严重框

- src/local_score_heatmap.py
  - 输出：
    - <id>_severity_map.png
    - <id>_severity_overlay.png
    - <id>_presence_map.png
    - <id>_presence_overlay.png
    - <id>_top10_points.txt
  - 说明：
    - class1 默认使用 effective mask（close -> dilate -> erode）
    - presence_map 采用 effective mask 内四分位分级配色（蓝 -> 青 -> 黄 -> 红）
- src/worst_box_direction.py
  - 在有效区域内选择最严重框（支持 1~5 个不重叠框）
  - 支持半径和框大小按图像尺度自动缩放（也可手动指定）
  - 箭头方向约束：相对瘢痕主体外接矩形中心 A 与框中心 B，最终方向与 A->B 不成钝角

### 3) 精简目录构建

- src/build_final_results.py
  - 将 results/<id> 关键结果统一转为 PNG 并整理到 final_results/<id>

## 模型与关键规则

- 默认模型：best_trans_unet_model_20250614_122913.pth
- 方向箭头：使用连续主方向（不做 8 方向量化）
- severity 命名：已替代旧 heatmap 命名

## 环境要求

```bash
python >= 3.8
torch >= 1.8.0
torchvision >= 0.9.0
opencv-python >= 4.5.0
numpy >= 1.19.0
matplotlib >= 3.3.0
scikit-learn >= 0.24.0
Pillow >= 8.0.0
tqdm >= 4.60.0
streamlit >= 1.30.0   # 网页 Demo 需要
```

## 使用方式

### 1) 单病例完整流程（推荐）

```bash
python src/run_case_to_results.py 66
```

可选参数：

```bash
python src/run_case_to_results.py 66 --model best_trans_unet_model_20250614_122913.pth --radius 40 --box-size 80
```

### 2) 全量批处理（推荐）

```bash
python src/run_all_cases_to_results.py --skip-existing
```

常用参数：

```bash
python src/run_all_cases_to_results.py --only-cases 30 66 100
python src/run_all_cases_to_results.py --data-dir dataset/final_labeled --radius 40 --box-size 80
```

### 3) 仅做局部严重度图

```bash
python src/local_score_heatmap.py 66 --target-class 1 --radius 40 --output-subdir r40
```

### 4) 仅做最严重框

```bash
python src/worst_box_direction.py 66 --radius 40 --box-size 80 --num-boxes 1 --output-subdir r40
```

支持范围：

- `--num-boxes` 可设置 `1~5`

### 5) 从已有 results 生成精简目录

```bash
python src/build_final_results.py --cases 66
```

### 6) 网页 Demo（上传 + 可视化 + 调参重算）

```bash
streamlit run web_demo.py
```

能力说明：

- 上传任意图片后，一键生成主要中间结果并展示：分割、纹理、方向、扇区、最终叠加。
- 可在侧边栏手动调整参数后重复生成：
  - Heatmap radius（动态/固定）
  - Presence 分级数与每条分界线（支持 threshold 模式: 0~1，quantile 模式: 0~100 分位）
  - 纹理阈值、密度/一致性权重、热图叠加透明度
  - 最严重框大小（动态/固定）、框数量（1~5）、最小覆盖率
- 支持“仅重算热图/最严重框”，用于快速调参到满意效果。
- 运行按钮固定在侧边栏，滚动页面时无需回到顶部。
- `web_demo_output/` 下会按 `原图名__case_id/` 创建目录，并自动保存原图 PNG 副本：`00_original__*.png`。

### 7) 单图全流程（路径或病例ID）

```bash
python src/run_one_full_pipeline.py 66
```

或直接传图片路径（支持中文路径）：

```bash
python src/run_one_full_pipeline.py dataset/final_labeled/66.jpg
```

可选模型参数：

```bash
python src/run_one_full_pipeline.py 66 --model best_trans_unet_model_20250614_122913.pth
```

## 输出说明

### 中间目录

- skin_output/
  - texture_line_<id>.png
  - only_texture_line_<id>.png
- predict_output/
  - orientation_texture_line_<id>.png
  - orientation_only_texture_line_<id>.png
  - spatial_sector_directions_<id>.png
  - sector_info_<id>.json / .pkl
- heatmap_output/<id>/<subdir>/
  - <id>_severity_map.png
  - <id>_severity_overlay.png
  - <id>_presence_map.png
  - <id>_presence_overlay.png
  - <id>_worst*_box.png / <id>_worst*_direction.png / <id>_worst*_info.txt

### 标准结果目录

- results/<id>/01~07：完整链路结果，便于排错和科研分析
- final_results/<id>/01~08：交付向精简结果，均为可直接查看的图片

## 备注

- 输入建议使用 dataset/final_labeled/<id>.jpg。
- 若批处理失败，src/run_all_cases_to_results.py 会写入 results/batch_run_case_failures.txt。
- 根目录同名脚本仍可运行，但仅作为兼容入口，内部已统一转调 src。
