"""
视频 → YOLO 数据集 制作工具

用法:
    python main.py                    打开 GUI 界面（推荐）
    python main.py --cli              命令行交互菜单
    python main.py extract 视频路径   抽帧
    python main.py annotate           标注（OpenCV 窗口）
    python main.py build              整理数据集
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根目录（本文件所在目录）
ROOT = Path(__file__).resolve().parent

# 默认工作目录
WORK_DIR = ROOT / "workspace"
FRAMES_DIR = WORK_DIR / "frames"       # 抽帧图片
LABELS_DIR = WORK_DIR / "labels"       # 标注标签（与 frames 同名 .txt）
DATASET_DIR = WORK_DIR / "dataset"     # 最终 YOLO 数据集
CLASSES_FILE = ROOT / "classes.txt"


def cmd_extract(video: str, interval: float) -> None:
    from extract import extract_frames

    extract_frames(video, FRAMES_DIR, interval_sec=interval)


def cmd_annotate() -> None:
    try:
        from annotate import run_annotation
    except RuntimeError as exc:
        print(f"错误: {exc}")
        print("提示: GUI 模式（python main.py）不需要 OpenCV，请直接使用图形界面标注。")
        sys.exit(1)

    if not FRAMES_DIR.exists() or not any(FRAMES_DIR.iterdir()):
        print(f"错误: 请先抽帧！图片目录为空: {FRAMES_DIR}")
        print("  运行: python main.py extract 你的视频.mp4")
        sys.exit(1)

    run_annotation(FRAMES_DIR, LABELS_DIR, CLASSES_FILE)


def cmd_build(val_ratio: float) -> None:
    from dataset import organize_dataset

    organize_dataset(
        FRAMES_DIR,
        LABELS_DIR,
        DATASET_DIR,
        CLASSES_FILE,
        val_ratio=val_ratio,
    )[0]


def cmd_all(video: str, interval: float, val_ratio: float) -> None:
    """抽帧 → 标注 → 整理（标注步骤需手动操作）。"""
    cmd_extract(video, interval)
    print("\n>>> 接下来请进行标注（会弹出标注窗口）<<<\n")
    input("按 Enter 键开始标注...")
    cmd_annotate()
    print("\n>>> 标注完成，正在整理数据集...<<<\n")
    cmd_build(val_ratio)


def show_menu() -> None:
    """新手友好的交互式菜单。"""
    print("\n" + "=" * 50)
    print("  视频 YOLO 数据集制作工具")
    print("=" * 50)
    print(f"  工作目录: {WORK_DIR}")
    print(f"  类别文件: {CLASSES_FILE}")
    print("-" * 50)
    print("  1. 从视频抽帧（默认每 10 秒一帧）")
    print("  2. 标注图片（画框）")
    print("  3. 整理为 YOLO 训练数据集")
    print("  4. 一键全流程（抽帧→标注→整理）")
    print("  0. 退出")
    print("=" * 50)

    choice = input("请输入选项 [0-4]: ").strip()

    if choice == "1":
        video = input("请输入视频文件路径: ").strip().strip('"')
        interval_str = input("抽帧间隔（秒，直接回车默认 10）: ").strip()
        interval = float(interval_str) if interval_str else 10.0
        cmd_extract(video, interval)

    elif choice == "2":
        cmd_annotate()

    elif choice == "3":
        ratio_str = input("验证集比例（0~1，直接回车默认 0.2）: ").strip()
        val_ratio = float(ratio_str) if ratio_str else 0.2
        cmd_build(val_ratio)

    elif choice == "4":
        video = input("请输入视频文件路径: ").strip().strip('"')
        interval_str = input("抽帧间隔（秒，直接回车默认 10）: ").strip()
        interval = float(interval_str) if interval_str else 10.0
        ratio_str = input("验证集比例（直接回车默认 0.2）: ").strip()
        val_ratio = float(ratio_str) if ratio_str else 0.2
        cmd_all(video, interval, val_ratio)

    elif choice == "0":
        print("再见！")
        sys.exit(0)

    else:
        print("无效选项，请重新选择")

    show_menu()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="视频 YOLO 数据集制作工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          打开 GUI
  python main.py --cli                    命令行菜单
  python main.py extract demo.mp4         每10秒抽一帧
  python main.py annotate                 OpenCV 标注窗口
  python main.py build                    整理数据集
        """,
    )
    parser.add_argument("--cli", action="store_true", help="使用命令行菜单（不用 GUI）")

    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract", help="从视频抽帧")
    p_extract.add_argument("video", help="视频文件路径")
    p_extract.add_argument("-i", "--interval", type=float, default=10.0, help="抽帧间隔（秒）")

    sub.add_parser("annotate", help="标注图片")

    p_build = sub.add_parser("build", help="整理 YOLO 数据集")
    p_build.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")

    p_all = sub.add_parser("all", help="全流程")
    p_all.add_argument("video", help="视频文件路径")
    p_all.add_argument("-i", "--interval", type=float, default=10.0, help="抽帧间隔（秒）")
    p_all.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")

    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "extract":
        cmd_extract(args.video, args.interval)
    elif args.command == "annotate":
        cmd_annotate()
    elif args.command == "build":
        cmd_build(args.val_ratio)
    elif args.command == "all":
        cmd_all(args.video, args.interval, args.val_ratio)
    elif args.cli:
        show_menu()
    else:
        from gui import run_gui

        run_gui()


if __name__ == "__main__":
    main()
