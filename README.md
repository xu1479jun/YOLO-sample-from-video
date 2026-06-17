# 视频 YOLO 数据集制作工具

从视频中自动抽帧、手动标注、导出 YOLO 训练格式的数据集。**图形界面一键操作，适合 Python 新手。**

## 功能

1. **抽帧** — 每隔 10 秒（可自定义）从视频抓取一帧图片
2. **标注** — 在 GUI 中鼠标画框，选择类别，自动保存 YOLO 格式标签
3. **导出** — 自动划分训练集/验证集，生成 `data.yaml` 配置文件
4. **类别设置** — 在界面中直接编辑检测类别

## 快速开始

### 第 1 步：安装 Python

确保已安装 [Python 3.8+](https://www.python.org/downloads/)。安装时勾选 **"Add Python to PATH"**。

### 第 2 步：安装依赖

```bash
cd d:\YOLO\VideoPostion
pip install -r requirements.txt
```

### 第 3 步：启动 GUI

```bash
python main.py
```

或直接运行：

```bash
python gui.py
```

## GUI 使用流程

| 标签页 | 操作 |
|--------|------|
| **① 视频抽帧** | 选择视频 → 设置间隔（默认 10 秒）→ 点击「开始抽帧」 |
| **② 样本标注** | 选择类别 → 鼠标拖拽画框 → 「下一张」自动保存 |
| **③ 导出数据集** | 设置验证集比例 → 点击「整理 YOLO 数据集」 |
| **④ 类别设置** | 编辑类别名称 → 点击「保存类别」 |

底部 **运行日志** 区域会显示抽帧和导出进度。

### 标注操作

| 操作 | 说明 |
|------|------|
| **选择类别** | 点击类别按钮（如 `0: person`） |
| **鼠标拖拽** | 在图片上画标注框 |
| **下一张 / 上一张** | 切换图片（自动保存） |
| **保存** | 手动保存当前标签 |
| **撤销框** | 删除最后一个框 |
| **删除图片** | 跳过坏样本 |
| **左侧列表** | 点击跳转到指定图片 |

## 命令行用法（可选）

```bash
python main.py --cli                       # 命令行菜单
python main.py extract 你的视频.mp4           # 每 10 秒抽一帧
python main.py extract 你的视频.mp4 -i 5      # 每 5 秒抽一帧
python main.py annotate                    # OpenCV 标注窗口
python main.py build                       # 整理数据集
```

## 输出目录结构

所有中间文件和最终数据集都在 `workspace/` 目录下：

```
workspace/
├── frames/          ← 抽帧图片
├── labels/          ← YOLO 格式标签（与图片同名 .txt）
└── dataset/         ← 最终训练数据集
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml    ← YOLO 训练配置文件
```

### YOLO 标签格式

每个 `.txt` 文件对应一张图片，每行一个目标：

```
类别ID 中心x 中心y 宽度 高度
```

数值均为 **0~1 的归一化坐标**（相对图片宽高）。

示例（类别 0，目标在图片正中央，占宽高各 50%）：

```
0 0.5 0.5 0.5 0.5
```

## 开始 YOLO 训练

整理完成后，可用 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 训练：

```bash
pip install ultralytics
yolo detect train data=workspace/dataset/data.yaml model=yolov8n.pt epochs=100
```

## 常见问题

**Q: 提示找不到视频？**  
A: 路径中有空格时用引号包裹，例如 `"D:\Videos\my video.mp4"`。

**Q: 启动报错 NumPy X86_V2？**  
A: 您的 CPU 较旧或使用了 32 位 Python。本项目 GUI 已改用 ffmpeg 抽帧，**不再需要 OpenCV/NumPy**。请执行：
```bash
pip install -r requirements.txt
python main.py
```
若仍有问题，建议安装 **64 位 Python 3.10~3.12**（[python.org](https://www.python.org/downloads/)）。

**Q: build 时提示没有已标注样本？**  
A: 标注时至少画一个框并按 N 保存，空标签不会被计入数据集。

**Q: 想修改类别？**  
A: 编辑 `classes.txt` 后重新标注（已有标签的类别 ID 可能与新区不一致，建议重新标注）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `gui.py` | **图形界面（主程序）** |
| `main.py` | 程序入口（默认启动 GUI） |
| `extract.py` | 视频抽帧 |
| `annotate.py` | OpenCV 标注（命令行模式用） |
| `dataset.py` | 数据集整理 |
| `utils.py` | 公共工具 |
| `classes.txt` | 类别名称列表 |
