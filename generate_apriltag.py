#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import cv2

def get_aruco_dict(tag_family: str):
    # OpenCV 支持的 AprilTag 字典名称可能因版本不同略有差异
    # 常见：DICT_APRILTAG_36h11
    if tag_family.lower() in ["t36h11", "36h11", "apriltag_36h11"]:
        if hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
            return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        raise RuntimeError("Your OpenCV build does not include DICT_APRILTAG_36h11. "
                           "Try installing opencv-contrib-python.")
    raise ValueError(f"Unsupported tag_family: {tag_family}")

def main():
    ap = argparse.ArgumentParser("Generate a high-res AprilTag grid PNG using OpenCV aruco.")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tag_family", type=str, default="t36h11")
    ap.add_argument("--tag_size_m", type=float, default=0.055)
    ap.add_argument("--tag_spacing_m", type=float, default=0.0165)
    ap.add_argument("--tag_border_bits", type=int, default=1, help="Black border bits around tag (OpenCV draws this).")
    ap.add_argument("--canvas_px", type=int, default=9000, help="Output square canvas size in pixels.")
    ap.add_argument("--margin_px", type=int, default=400, help="White margin around the whole board.")
    ap.add_argument("--start_id", type=int, default=0, help="First tag id in the grid (row-major).")
    ap.add_argument("--out", type=str, default="aprilgrid_6x6_t36h11.png")
    args = ap.parse_args()

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not found. Install opencv-contrib-python, e.g.: pip install opencv-contrib-python")

    aruco_dict = get_aruco_dict(args.tag_family)

    rows, cols = args.rows, args.cols
    canvas_px = int(args.canvas_px)
    margin_px = int(args.margin_px)

    # spacing ratio in meters -> pixels ratio
    spacing_ratio = args.tag_spacing_m / args.tag_size_m  # 0.0165/0.055 = 0.3
    # Each cell pitch = tag + spacing
    pitch_ratio = 1.0 + spacing_ratio

    # Compute tag size in pixels so that the whole grid fits canvas (minus margins)
    usable = canvas_px - 2 * margin_px
    # total grid width in "tag units" = cols*1 + (cols-1)*spacing_ratio = cols + (cols-1)*spacing_ratio
    grid_units = cols * 1.0 + (cols - 1) * spacing_ratio
    tag_px = int(usable / grid_units)
    if tag_px <= 50:
        raise RuntimeError("Canvas too small for requested board; increase --canvas_px or reduce margins.")

    spacing_px = int(tag_px * spacing_ratio)

    # Recompute exact grid px (might be slightly smaller than usable)
    grid_w = cols * tag_px + (cols - 1) * spacing_px
    grid_h = rows * tag_px + (rows - 1) * spacing_px

    # Create white canvas
    canvas = np.full((canvas_px, canvas_px), 255, dtype=np.uint8)

    # Top-left corner of grid on canvas (center it)
    x0 = (canvas_px - grid_w) // 2
    y0 = (canvas_px - grid_h) // 2

    # Draw tags
    tag_id = args.start_id
    for r in range(rows):
        for c in range(cols):
            # drawMarker returns a tag image (black/white) with border
            tag_img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, tag_px)

            x = x0 + c * (tag_px + spacing_px)
            y = y0 + r * (tag_px + spacing_px)

            canvas[y:y+tag_px, x:x+tag_px] = tag_img
            tag_id += 1

    out_path = Path(args.out).resolve()
    cv2.imwrite(str(out_path), canvas)
    print("✅ Saved:", out_path)
    print(f"tag_px={tag_px}, spacing_px={spacing_px}, spacing_ratio={spacing_ratio:.6f}")
    print(f"Grid area (px): {grid_w} x {grid_h}")
    board_m = cols * args.tag_size_m + (cols - 1) * args.tag_spacing_m
    print(f"Board physical size (no extra margin): {board_m*100:.2f} cm x {board_m*100:.2f} cm")

if __name__ == "__main__":
    main()