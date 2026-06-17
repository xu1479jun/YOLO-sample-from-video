"""交互式 YOLO 格式标注工具（OpenCV 窗口）。"""

from __future__ import annotations

from pathlib import Path

from utils import Box, ensure_dir, list_images, load_classes


def _import_cv2():
    try:
        import cv2
    except RuntimeError as exc:
        raise RuntimeError(
            "OpenCV/NumPy 无法加载。您的 CPU 较旧，请执行：pip install \"numpy<2.0\" --force-reinstall"
        ) from exc
    return cv2


class YoloAnnotator:
    """用鼠标画框，键盘切换图片和类别。"""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        classes: list[str],
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = ensure_dir(labels_dir)
        self.classes = classes

        self.images = list_images(self.images_dir)
        if not self.images:
            raise FileNotFoundError(f"图片目录为空: {self.images_dir}")

        self.index = 0
        self.boxes: list[Box] = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_class = 0
        self.img = None
        self.display = None
        self.window = "YOLO 标注工具 - 按 H 查看帮助"

    def _label_path(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def _load_boxes(self, image_path: Path) -> list[Box]:
        """从已有 txt 标签加载标注。"""
        label_path = self._label_path(image_path)
        if not label_path.exists():
            return []

        boxes: list[Box] = []
        h, w = self.img.shape[:2] if self.img is not None else (1, 1)

        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            boxes.append(Box(cid, x1, y1, x2, y2))
        return boxes

    def _save_labels(self, image_path: Path) -> None:
        """保存当前图片的 YOLO 标签。"""
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        lines = [b.to_yolo_line(w, h) for b in self.boxes]
        self._label_path(image_path).write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    def _delete_image_and_label(self, image_path: Path) -> None:
        """删除当前图片及对应标签（跳过无用样本）。"""
        label_path = self._label_path(image_path)
        if label_path.exists():
            label_path.unlink()
        image_path.unlink()

    def _draw_ui(self) -> None:
        """重绘画面和标注框。"""
        if self.img is None:
            return

        self.display = self.img.copy()
        colors = [
            (0, 255, 0),
            (255, 128, 0),
            (0, 128, 255),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 255),
            (255, 255, 0),
        ]

        for i, box in enumerate(self.boxes):
            color = colors[box.class_id % len(colors)]
            cv2.rectangle(self.display, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            name = self.classes[box.class_id] if box.class_id < len(self.classes) else "?"
            cv2.putText(
                self.display,
                f"{box.class_id}:{name}",
                (box.x1, max(box.y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # 顶部状态栏
        cur_name = self.classes[self.current_class]
        info = (
            f"[{self.index + 1}/{len(self.images)}] {self.images[self.index].name}  "
            f"类别:{self.current_class}({cur_name})  框数:{len(self.boxes)}"
        )
        cv2.rectangle(self.display, (0, 0), (self.display.shape[1], 30), (40, 40, 40), -1)
        cv2.putText(
            self.display,
            info,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if self.img is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._draw_ui()
            cv2.rectangle(
                self.display,
                (self.start_x, self.start_y),
                (x, y),
                (0, 255, 255),
                2,
            )
            cv2.imshow(self.window, self.display)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if abs(x - self.start_x) > 5 and abs(y - self.start_y) > 5:
                self.boxes.append(
                    Box(self.current_class, self.start_x, self.start_y, x, y)
                )
            self._draw_ui()

    def _load_current(self) -> None:
        path = self.images[self.index]
        self.img = cv2.imread(str(path))
        if self.img is None:
            raise RuntimeError(f"无法读取图片: {path}")
        self.boxes = self._load_boxes(path)
        self._draw_ui()

    def _print_help(self) -> None:
        print("\n" + "=" * 50)
        print("标注快捷键")
        print("=" * 50)
        print("  鼠标左键拖拽     画标注框")
        print("  0-9             切换类别（对应 classes.txt 顺序）")
        print("  S               保存当前标签")
        print("  N / 空格         保存并下一张")
        print("  P               保存并上一张")
        print("  Z               撤销最后一个框")
        print("  D               删除当前图片（跳过坏样本）")
        print("  H               显示帮助")
        print("  Q / ESC         退出")
        print("=" * 50 + "\n")

    def run(self) -> None:
        """启动标注窗口。"""
        cv2 = _import_cv2()
        self._print_help()

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

        self._load_current()

        while True:
            if self.display is not None:
                cv2.imshow(self.window, self.display)

            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):  # Q 或 ESC
                path = self.images[self.index]
                self._save_labels(path)
                print("已保存并退出")
                break

            if key in (ord("n"), ord(" ")):  # N 或 空格：下一张
                path = self.images[self.index]
                self._save_labels(path)
                self.index = min(self.index + 1, len(self.images) - 1)
                self._load_current()
                continue

            if key == ord("p"):  # P：上一张
                path = self.images[self.index]
                self._save_labels(path)
                self.index = max(self.index - 1, 0)
                self._load_current()
                continue

            if key == ord("s"):
                self._save_labels(self.images[self.index])
                print(f"已保存: {self._label_path(self.images[self.index]).name}")
                continue

            if key == ord("z") and self.boxes:
                self.boxes.pop()
                self._draw_ui()
                continue

            if key == ord("d"):
                path = self.images[self.index]
                self._delete_image_and_label(path)
                print(f"已删除: {path.name}")
                self.images.pop(self.index)
                if not self.images:
                    print("没有更多图片了")
                    break
                self.index = min(self.index, len(self.images) - 1)
                self._load_current()
                continue

            if key == ord("h"):
                self._print_help()
                continue

            if ord("0") <= key <= ord("9"):
                cid = key - ord("0")
                if cid < len(self.classes):
                    self.current_class = cid
                    print(f"当前类别: {cid} - {self.classes[cid]}")
                    self._draw_ui()

        cv2.destroyAllWindows()


def run_annotation(
    images_dir: str | Path,
    labels_dir: str | Path,
    classes_file: str | Path,
) -> None:
    """启动标注工具的便捷入口。"""
    classes = load_classes(classes_file)
    print(f"共 {len(classes)} 个类别: {classes}")
    annotator = YoloAnnotator(images_dir, labels_dir, classes)
    annotator.run()
