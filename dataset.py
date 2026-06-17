"""整理 YOLO 训练数据集目录，并生成 data.yaml。"""

from __future__ import annotations

import random
import shutil
from collections.abc import Callable
from pathlib import Path

import yaml

from utils import ensure_dir, list_images, load_classes


def organize_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    classes_file: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    log: Callable[[str], None] | None = None,
) -> tuple[Path, dict]:
    """
    将标注好的图片和标签整理为 YOLO 标准目录结构:

        dataset/
          images/train/
          images/val/
          labels/train/
          labels/val/
          data.yaml

    只复制「有对应标签文件」的图片；无标签的图片会被跳过。
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = ensure_dir(output_dir)
    classes = load_classes(classes_file)

    images = list_images(images_dir)
    paired: list[tuple[Path, Path]] = []
    skipped = 0

    for img in images:
        label = labels_dir / f"{img.stem}.txt"
        if label.exists() and label.read_text(encoding="utf-8").strip():
            paired.append((img, label))
        else:
            skipped += 1

    if not paired:
        raise ValueError(
            "没有找到已标注的样本！请先完成标注（labels 目录下需有非空 .txt 文件）"
        )

    random.seed(seed)
    random.shuffle(paired)
    val_count = max(1, int(len(paired) * val_ratio)) if len(paired) > 1 else 0
    val_set = set(range(val_count))

    dirs = {
        "train_img": ensure_dir(output_dir / "images" / "train"),
        "val_img": ensure_dir(output_dir / "images" / "val"),
        "train_lbl": ensure_dir(output_dir / "labels" / "train"),
        "val_lbl": ensure_dir(output_dir / "labels" / "val"),
    }

    train_n, val_n = 0, 0
    for i, (img, lbl) in enumerate(paired):
        if i in val_set:
            shutil.copy2(img, dirs["val_img"] / img.name)
            shutil.copy2(lbl, dirs["val_lbl"] / lbl.name)
            val_n += 1
        else:
            shutil.copy2(img, dirs["train_img"] / img.name)
            shutil.copy2(lbl, dirs["train_lbl"] / lbl.name)
            train_n += 1

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }

    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    stats = {
        "train": train_n,
        "val": val_n,
        "skipped": skipped,
        "yaml_path": yaml_path,
        "output_dir": output_dir.resolve(),
    }

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    _log("=" * 50)
    _log("数据集整理完成！")
    _log(f"输出目录: {output_dir.resolve()}")
    _log(f"训练集: {train_n} 张")
    _log(f"验证集: {val_n} 张")
    _log(f"跳过（无标签）: {skipped} 张")
    _log(f"配置文件: {yaml_path}")
    _log("=" * 50)
    _log("\nYOLO 训练示例命令（Ultralytics）:")
    _log(f'  yolo detect train data="{yaml_path.resolve()}" model=yolov8n.pt epochs=100')
    _log("")

    return yaml_path, stats
