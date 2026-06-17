"""从视频中按固定间隔抽取帧（使用 ffmpeg，无需 OpenCV/NumPy）。"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import imageio_ffmpeg

from utils import ensure_dir, list_images


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    interval_sec: float = 10.0,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    """
    每隔 interval_sec 秒从视频抓取一帧，保存为 JPG。

    依赖 imageio-ffmpeg 内置的 ffmpeg，不依赖 OpenCV。
    """
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    if interval_sec <= 0:
        raise ValueError("抽帧间隔必须大于 0")

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            print(msg)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_pattern = str(output_dir / f"{video_path.stem}_%04d.jpg")

    _log("=" * 50)
    _log(f"视频: {video_path.name}")
    _log(f"抽帧间隔: 每 {interval_sec} 秒")
    _log(f"输出目录: {output_dir.resolve()}")
    _log("=" * 50)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_sec}",
        "-qscale:v",
        "2",
        "-y",
        out_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "未知错误").strip()
        raise RuntimeError(f"视频抽帧失败: {err}")

    saved = [p for p in list_images(output_dir) if p.name.startswith(f"{video_path.stem}_")]

    for i, path in enumerate(saved, 1):
        _log(f"  [{i}] 已保存: {path.name}")

    _log("-" * 50)
    _log(f"完成！共抽取 {len(saved)} 张图片")
    return saved
