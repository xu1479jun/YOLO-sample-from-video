"""统一 GUI：视频抽帧、YOLO 标注、数据集导出。"""

from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from utils import Box
from dataset import organize_dataset
from extract import extract_frames
from utils import ensure_dir, list_images, load_classes

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "workspace"
FRAMES_DIR = WORK_DIR / "frames"
LABELS_DIR = WORK_DIR / "labels"
DATASET_DIR = WORK_DIR / "dataset"
CLASSES_FILE = ROOT / "classes.txt"

BOX_COLORS = [
    "#00FF00",
    "#FF8000",
    "#0080FF",
    "#FF00FF",
    "#00FFFF",
    "#8000FF",
    "#FFFF00",
]


@dataclass
class ViewTransform:
    """画布坐标与原始图片坐标的映射。"""

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    disp_w: int = 0
    disp_h: int = 0

    def to_image(self, x: float, y: float) -> tuple[int, int]:
        ix = int((x - self.offset_x) / self.scale)
        iy = int((y - self.offset_y) / self.scale)
        return ix, iy

    def to_canvas(self, x: int, y: int) -> tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale


class YoloDatasetApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("视频 YOLO 数据集制作工具")
        self.geometry("1100x720")
        self.minsize(960, 640)

        ensure_dir(WORK_DIR)
        ensure_dir(FRAMES_DIR)
        ensure_dir(LABELS_DIR)

        self.classes: list[str] = []
        self.images: list[Path] = []
        self.index = 0
        self.boxes: list[Box] = []
        self.current_class = tk.IntVar(value=0)
        self.pil_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.transform = ViewTransform()
        self.drawing = False
        self.start_x = 0.0
        self.start_y = 0.0
        self.temp_rect: int | None = None
        self._busy = False

        self._build_ui()
        self._reload_classes()
        self._refresh_image_list()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        header = ttk.Frame(self, padding=(12, 10))
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="视频 YOLO 数据集制作工具",
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Label(
            header,
            text=f"工作目录: {WORK_DIR}",
            foreground="#666",
        ).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self, padding=8)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._build_extract_tab()
        self._build_annotate_tab()
        self._build_dataset_tab()
        self._build_settings_tab()

        log_frame = ttk.LabelFrame(self, text="运行日志", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=7, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_extract_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="① 视频抽帧")

        row = ttk.Frame(tab)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="视频文件:", width=12).pack(side=tk.LEFT)
        self.video_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.video_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row, text="浏览...", command=self._browse_video).pack(side=tk.LEFT)

        row2 = ttk.Frame(tab)
        row2.pack(fill=tk.X, pady=4)
        ttk.Label(row2, text="抽帧间隔(秒):", width=12).pack(side=tk.LEFT)
        self.interval_var = tk.StringVar(value="10")
        ttk.Entry(row2, textvariable=self.interval_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="（默认每 10 秒抓取一帧）", foreground="#666").pack(side=tk.LEFT)

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill=tk.X, pady=12)
        self.extract_btn = ttk.Button(btn_row, text="开始抽帧", command=self._run_extract)
        self.extract_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="打开 frames 文件夹", command=lambda: self._open_folder(FRAMES_DIR)).pack(
            side=tk.LEFT, padx=8
        )

        tip = (
            "说明：抽帧后的图片保存在 workspace/frames/ 目录。\n"
            "完成后请切换到「② 样本标注」标签页进行画框标注。"
        )
        ttk.Label(tab, text=tip, foreground="#444", justify=tk.LEFT).pack(anchor=tk.W, pady=8)

    def _build_annotate_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="② 样本标注")

        toolbar = ttk.Frame(tab)
        toolbar.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(toolbar, text="◀ 上一张", command=self._prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="下一张 ▶", command=self._next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="保存", command=self._save_current).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="撤销框", command=self._undo_box).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="删除图片", command=self._delete_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="刷新列表", command=self._refresh_image_list).pack(side=tk.LEFT, padx=8)

        self.status_var = tk.StringVar(value="暂无图片")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        body = ttk.Frame(tab)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(body, text="图片列表", padding=6, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        self.image_listbox = tk.Listbox(left, width=28, exportselection=False)
        self.image_listbox.pack(fill=tk.BOTH, expand=True)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        class_frame = ttk.LabelFrame(right, text="选择类别（画框前请先选类别）", padding=6)
        class_frame.pack(fill=tk.X, pady=(0, 6))
        self.class_btn_frame = ttk.Frame(class_frame)
        self.class_btn_frame.pack(fill=tk.X)

        canvas_frame = ttk.Frame(right, relief=tk.SUNKEN, borderwidth=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        ttk.Label(
            right,
            text="操作：在图片上按住鼠标左键拖拽画框 | 切换类别后画不同目标",
            foreground="#666",
        ).pack(anchor=tk.W, pady=4)

    def _build_dataset_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="③ 导出数据集")

        row = ttk.Frame(tab)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="验证集比例:", width=12).pack(side=tk.LEFT)
        self.val_ratio_var = tk.StringVar(value="0.2")
        ttk.Entry(row, textvariable=self.val_ratio_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="（0~1，例如 0.2 表示 20% 做验证集）", foreground="#666").pack(side=tk.LEFT)

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill=tk.X, pady=12)
        self.build_btn = ttk.Button(btn_row, text="整理 YOLO 数据集", command=self._run_build)
        self.build_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="打开 dataset 文件夹", command=lambda: self._open_folder(DATASET_DIR)).pack(
            side=tk.LEFT, padx=8
        )

        self.result_var = tk.StringVar(value="尚未导出")
        ttk.Label(tab, textvariable=self.result_var, justify=tk.LEFT).pack(anchor=tk.W, pady=8)

        tip = (
            "导出结构：workspace/dataset/\n"
            "  ├── images/train/  images/val/\n"
            "  ├── labels/train/  labels/val/\n"
            "  └── data.yaml\n\n"
            "训练命令示例：\n"
            '  yolo detect train data="workspace/dataset/data.yaml" model=yolov8n.pt epochs=100'
        )
        ttk.Label(tab, text=tip, foreground="#444", justify=tk.LEFT).pack(anchor=tk.W)

    def _build_settings_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="④ 类别设置")

        ttk.Label(
            tab,
            text="每行一个类别名称，第一行对应类别 0，第二行对应类别 1，以此类推。",
            foreground="#444",
        ).pack(anchor=tk.W, pady=(0, 8))

        self.classes_text = scrolledtext.ScrolledText(tab, height=12)
        self.classes_text.pack(fill=tk.BOTH, expand=True, pady=4)

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill=tk.X, pady=8)
        ttk.Button(btn_row, text="保存类别", command=self._save_classes).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="重新加载", command=self._reload_classes).pack(side=tk.LEFT, padx=8)

    # ------------------------------------------------------------------ 日志
    def _log(self, msg: str) -> None:
        def append() -> None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        self.after(0, append)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.extract_btn.configure(state=state)
        self.build_btn.configure(state=state)

    # ------------------------------------------------------------------ 抽帧
    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.video_var.set(path)

    def _run_extract(self) -> None:
        if self._busy:
            return

        video = self.video_var.get().strip().strip('"')
        if not video:
            messagebox.showwarning("提示", "请先选择视频文件")
            return

        try:
            interval = float(self.interval_var.get().strip())
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "抽帧间隔必须是大于 0 的数字")
            return

        def task() -> None:
            self._set_busy(True)
            try:
                saved = extract_frames(video, FRAMES_DIR, interval_sec=interval, log=self._log)
                self.after(0, lambda: messagebox.showinfo("完成", f"抽帧完成，共 {len(saved)} 张图片"))
                self.after(0, self._refresh_image_list)
                self.after(0, lambda: self.notebook.select(1))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("抽帧失败", str(exc)))
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------ 类别
    def _reload_classes(self) -> None:
        try:
            if CLASSES_FILE.exists():
                content = CLASSES_FILE.read_text(encoding="utf-8")
            else:
                content = "person\ncar\n"
                CLASSES_FILE.write_text(content, encoding="utf-8")

            self.classes_text.delete("1.0", tk.END)
            self.classes_text.insert("1.0", content)
            self.classes = load_classes(CLASSES_FILE)
            self._rebuild_class_buttons()
        except Exception as exc:
            messagebox.showerror("错误", f"加载类别失败: {exc}")

    def _save_classes(self) -> None:
        content = self.classes_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("提示", "至少填写一个类别")
            return
        CLASSES_FILE.write_text(content + "\n", encoding="utf-8")
        self._reload_classes()
        messagebox.showinfo("成功", "类别已保存")

    def _rebuild_class_buttons(self) -> None:
        for w in self.class_btn_frame.winfo_children():
            w.destroy()

        for i, name in enumerate(self.classes):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            rb = ttk.Radiobutton(
                self.class_btn_frame,
                text=f"{i}: {name}",
                variable=self.current_class,
                value=i,
                command=self._redraw_canvas,
            )
            rb.pack(side=tk.LEFT, padx=4, pady=2)

    # ------------------------------------------------------------------ 标注
    def _refresh_image_list(self) -> None:
        self.images = list_images(FRAMES_DIR) if FRAMES_DIR.exists() else []
        self.image_listbox.delete(0, tk.END)
        for p in self.images:
            self.image_listbox.insert(tk.END, p.name)

        if self.images:
            self.index = min(self.index, len(self.images) - 1)
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.index)
            self.image_listbox.see(self.index)
            self._load_current_image()
        else:
            self.index = 0
            self.boxes = []
            self.pil_image = None
            self.canvas.delete("all")
            self.status_var.set("暂无图片，请先在「① 视频抽帧」中抽取帧")

    def _on_list_select(self, _event: tk.Event) -> None:
        sel = self.image_listbox.curselection()
        if not sel:
            return
        new_index = sel[0]
        if new_index != self.index:
            self._save_current(silent=True)
            self.index = new_index
            self._load_current_image()

    def _label_path(self, image_path: Path) -> Path:
        return LABELS_DIR / f"{image_path.stem}.txt"

    def _load_boxes(self, image_path: Path, w: int, h: int) -> list[Box]:
        label_path = self._label_path(image_path)
        if not label_path.exists():
            return []

        boxes: list[Box] = []
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

    def _load_current_image(self) -> None:
        if not self.images:
            return

        path = self.images[self.index]
        try:
            self.pil_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("错误", f"无法打开图片: {exc}")
            return

        w, h = self.pil_image.size
        self.boxes = self._load_boxes(path, w, h)
        self.status_var.set(f"[{self.index + 1}/{len(self.images)}] {path.name}  |  框数: {len(self.boxes)}")
        self._redraw_canvas()

    def _compute_transform(self, cw: int, ch: int) -> ViewTransform:
        if not self.pil_image:
            return ViewTransform()

        iw, ih = self.pil_image.size
        scale = min(cw / iw, ch / ih, 1.0)
        disp_w = int(iw * scale)
        disp_h = int(ih * scale)
        offset_x = (cw - disp_w) / 2
        offset_y = (ch - disp_h) / 2
        return ViewTransform(scale, offset_x, offset_y, disp_w, disp_h)

    def _redraw_canvas(self) -> None:
        self.canvas.delete("all")
        if not self.pil_image:
            return

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        self.transform = self._compute_transform(cw, ch)

        rw = max(1, int(self.transform.disp_w))
        rh = max(1, int(self.transform.disp_h))

        disp = self.pil_image.resize((rw, rh), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(disp)
        self.canvas.create_image(
            self.transform.offset_x,
            self.transform.offset_y,
            anchor=tk.NW,
            image=self.tk_image,
        )

        for box in self.boxes:
            self._draw_box_on_canvas(box)

    def _draw_box_on_canvas(self, box: Box, dash: tuple[int, ...] | None = None) -> None:
        x1, y1 = self.transform.to_canvas(box.x1, box.y1)
        x2, y2 = self.transform.to_canvas(box.x2, box.y2)
        color = BOX_COLORS[box.class_id % len(BOX_COLORS)]
        name = self.classes[box.class_id] if box.class_id < len(self.classes) else "?"
        kw: dict = {"outline": color, "width": 2}
        if dash:
            kw["dash"] = dash
        self.canvas.create_rectangle(x1, y1, x2, y2, **kw)
        self.canvas.create_text(x1 + 4, max(y1 - 2, 12), text=f"{box.class_id}:{name}", fill=color, anchor=tk.W)

    def _on_press(self, event: tk.Event) -> None:
        if not self.pil_image:
            return
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

    def _on_drag(self, event: tk.Event) -> None:
        if not self.drawing or not self.pil_image:
            return
        if self.temp_rect is not None:
            self.canvas.delete(self.temp_rect)
        self.temp_rect = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            event.x,
            event.y,
            outline="#FFFF00",
            width=2,
            dash=(4, 2),
        )

    def _on_release(self, event: tk.Event) -> None:
        if not self.drawing or not self.pil_image:
            return
        self.drawing = False
        if self.temp_rect is not None:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

        if abs(event.x - self.start_x) < 5 or abs(event.y - self.start_y) < 5:
            return

        ix1, iy1 = self.transform.to_image(self.start_x, self.start_y)
        ix2, iy2 = self.transform.to_image(event.x, event.y)
        iw, ih = self.pil_image.size
        ix1 = max(0, min(ix1, iw - 1))
        ix2 = max(0, min(ix2, iw - 1))
        iy1 = max(0, min(iy1, ih - 1))
        iy2 = max(0, min(iy2, ih - 1))

        if abs(ix2 - ix1) > 5 and abs(iy2 - iy1) > 5:
            cid = self.current_class.get()
            self.boxes.append(Box(cid, ix1, iy1, ix2, iy2))
            self._redraw_canvas()
            self.status_var.set(
                f"[{self.index + 1}/{len(self.images)}] {self.images[self.index].name}  |  框数: {len(self.boxes)}"
            )

    def _save_current(self, silent: bool = False) -> None:
        if not self.images or not self.pil_image:
            return

        path = self.images[self.index]
        w, h = self.pil_image.size
        lines = [b.to_yolo_line(w, h) for b in self.boxes]
        self._label_path(path).write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        if not silent:
            self._log(f"已保存标签: {path.stem}.txt")
            messagebox.showinfo("保存", f"已保存: {path.stem}.txt")

    def _prev_image(self) -> None:
        if not self.images:
            return
        self._save_current(silent=True)
        self.index = max(0, self.index - 1)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.index)
        self.image_listbox.see(self.index)
        self._load_current_image()

    def _next_image(self) -> None:
        if not self.images:
            return
        self._save_current(silent=True)
        self.index = min(len(self.images) - 1, self.index + 1)
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.index)
        self.image_listbox.see(self.index)
        self._load_current_image()

    def _undo_box(self) -> None:
        if self.boxes:
            self.boxes.pop()
            self._redraw_canvas()

    def _delete_current(self) -> None:
        if not self.images:
            return
        path = self.images[self.index]
        if not messagebox.askyesno("确认", f"确定删除图片及标签？\n{path.name}"):
            return

        label = self._label_path(path)
        if label.exists():
            label.unlink()
        path.unlink()
        self._log(f"已删除: {path.name}")
        self._refresh_image_list()

    # ------------------------------------------------------------------ 数据集
    def _run_build(self) -> None:
        if self._busy:
            return

        if self.images:
            self._save_current(silent=True)

        try:
            val_ratio = float(self.val_ratio_var.get().strip())
            if not 0 <= val_ratio < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "验证集比例必须是 0~1 之间的小数（如 0.2）")
            return

        def task() -> None:
            self._set_busy(True)
            try:
                _, stats = organize_dataset(
                    FRAMES_DIR,
                    LABELS_DIR,
                    DATASET_DIR,
                    CLASSES_FILE,
                    val_ratio=val_ratio,
                    log=self._log,
                )
                msg = (
                    f"训练集: {stats['train']} 张\n"
                    f"验证集: {stats['val']} 张\n"
                    f"跳过: {stats['skipped']} 张\n"
                    f"配置: {stats['yaml_path']}"
                )
                self.after(0, lambda: self.result_var.set(msg))
                self.after(0, lambda: messagebox.showinfo("完成", "数据集整理完成！"))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("失败", str(exc)))
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------ 杂项
    def _open_folder(self, folder: Path) -> None:
        ensure_dir(folder)
        import os
        import subprocess
        import sys

        path = str(folder.resolve())
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)

    def _on_close(self) -> None:
        if self.images and self.pil_image:
            self._save_current(silent=True)
        self.destroy()


def run_gui() -> None:
    app = YoloDatasetApp()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
