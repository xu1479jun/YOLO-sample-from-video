"""公共工具函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Box:
    """YOLO 标注框（像素坐标）。"""

    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def to_yolo_line(self, img_w: int, img_h: int) -> str:
        """转换为 YOLO 格式: class cx cy w h（归一化 0~1）。"""
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def load_classes(classes_file: str | Path) -> list[str]:
    """从 classes.txt 读取类别，每行一个类别名。"""
    path = Path(classes_file)
    if not path.exists():
        raise FileNotFoundError(f"找不到类别文件: {path}")

    classes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            classes.append(name)

    if not classes:
        raise ValueError(f"类别文件为空: {path}")
    return classes


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则创建。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: str | Path) -> list[Path]:
    """列出文件夹中所有常见图片格式。"""
    folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name.lower())
