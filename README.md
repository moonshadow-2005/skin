# KeTAS 皮肤纹理分析项目

本项目使用语义分割、Gabor 纹理提取与结构张量方向估计，对皮肤图像生成三类主要结果：

- **Severity Map Overlay**：局部严重程度叠加图
- **Presence Level Overlay**：纹理存在等级叠加图
- **Direction Map**：区域平均方向图与局部方向图

当前推荐入口是 Streamlit Demo。历史批处理和报告脚本仍保留在 `src/` 与 `scripts/legacy/`，但不再与模型、文档和运行产物混放在项目根目录。

## 快速开始

建议在项目根目录执行：

```bash
pip install -r requirements.txt
streamlit run web_demo.py
```

默认模型为：

```text
models/best_trans_unet_model_20250614_122913.pth
```

Demo 的显示设置保存在 `runtime/config/web_demo_display_settings.json`，重启后仍然有效。

## 目录结构

```text
skin/
├── web_demo.py                 # Streamlit 主入口
├── requirements.txt
├── README.md
├── src/                        # 可复用算法模块与命令行工具
│   ├── project_paths.py        # 全项目统一路径定义
│   ├── unet.py                 # U-Net / TransUNet 模型与训练代码
│   ├── texture_extraction.py   # 分割与纹理线提取
│   ├── orientation_analysis.py # 结构张量与方向分析
│   ├── local_score_heatmap.py  # Severity / Presence 计算
│   ├── worst_box_direction.py  # Presence 区域及方向箭头
│   └── ...
├── scripts/
│   └── legacy/                 # 兼容旧流程的入口
├── models/                     # 模型权重，不提交 Git
├── dataset/                    # 原始数据，不提交 Git
├── docs/
│   ├── paper/                  # 论文、翻译与订正记录
│   ├── project/                # 项目原理与面试问答
│   └── notes/                  # 独立技术笔记
└── runtime/                    # 所有可再生成的运行产物，不提交 Git
    ├── config/
    ├── texture/
    ├── orientation/
    ├── heatmaps/
    ├── web_inputs/
    ├── web_outputs/
    ├── reports/
    ├── results/
    └── final_results/
```

所有内部路径均由 `src/project_paths.py` 根据该文件位置推导，不依赖启动命令所在的当前工作目录。模型参数既可传绝对路径，也可传项目相对路径或 `models/` 中的文件名。

## 核心流程

1. **Segmentation**：U-Net/TransUNet 将图像分为背景、目标皮肤区域及其他类别。
2. **Texture Extraction**：在有效区域内使用多尺度、多方向 Gabor 响应提取纹理线。
3. **Direction Analysis**：由图像梯度构造结构张量，估计无向轴方向，并计算方向一致性。
4. **Severity Analysis**：结合局部纹理密度与方向一致性生成 Severity Map。
5. **Presence Analysis**：按可调百分位阈值形成闭合连通区域，并按严重程度着色。
6. **Direction Presentation**：输出区域平均方向和指定局部范围内的方向箭头。

Demo 的 Analysis 页面和 Process Overview 页面只展示 Settings 中启用的参数及中间图片。

## 命令行工具

### 单病例归档流程

```bash
python src/run_case_to_results.py 66 \
  --model models/best_trans_unet_model_20250614_122913.pth
```

结果写入：

- `runtime/results/66/`：完整阶段产物
- `runtime/final_results/66/`：精简结果

### 批量处理

```bash
python src/run_all_cases_to_results.py --skip-existing
```

指定病例：

```bash
python src/run_all_cases_to_results.py --only-cases 30 66 100
```

### 局部严重度

```bash
python src/local_score_heatmap.py 66 --target-class 1 --radius 40
```

输出位于 `runtime/heatmaps/<病例>/<子目录>/`。

### Presence 区域与方向

```bash
python src/worst_box_direction.py 66 \
  --radius 40 \
  --box-size 80 \
  --area-percentile 80
```

### 单图历史完整流程

```bash
python src/run_one_full_pipeline.py dataset/final_labeled/66.jpg
```

该入口保留旧报告链路。当前论文和 Demo 已弃用的 8 扇区展示仅可能出现在历史报告工具中，不属于当前 Demo 的最终输出。

## 路径迁移

旧目录名仍保留在 `.gitignore` 中，避免旧版本生成的文件被误提交；新代码只写入下列位置：

| 旧位置 | 当前统一位置 |
| --- | --- |
| 根目录模型文件 | `models/` |
| `skin_output/` | `runtime/texture/` |
| `predict_output/` | `runtime/orientation/` |
| `heatmap_output/` | `runtime/heatmaps/` |
| `web_demo_inputs/` | `runtime/web_inputs/` |
| `web_demo_output/` | `runtime/web_outputs/` |
| `report/` | `runtime/reports/` |
| `results/` | `runtime/results/` |
| `final_results/` | `runtime/final_results/` |

## 文档

- [项目技术说明](docs/project/PROJECT_TECH_INTRO.md)
- [项目面试问答](docs/project/项目面试问答.md)
- [论文方法订正清单](docs/paper/KeTAS_方法订正清单.md)
- [结构张量中的 Omega(i)](docs/notes/结构张量Omega说明.md)

## 注意事项

- `models/`、`dataset/` 和 `runtime/` 默认不提交 Git。
- 不要在代码中拼接新的根目录相对路径；新增目录应统一定义在 `src/project_paths.py`。
- `scripts/legacy/main.py` 是旧 `main.py` 的兼容入口，新功能应放在 `src/` 或 `web_demo.py`。
