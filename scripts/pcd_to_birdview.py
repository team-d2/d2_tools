#!/usr/bin/env python3
import argparse
import sys
import math
from pathlib import Path
import json
import yaml
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d が必要です: pip install open3d", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None


def load_points(p: Path) -> np.ndarray:
    pc = o3d.io.read_point_cloud(str(p))
    if pc.is_empty():
        raise ValueError("PCD が空です: " + str(p))
    return np.asarray(pc.points, dtype=np.float32)


def compute_bounds(pts: np.ndarray, margin: float):
    min_x = float(pts[:, 0].min()) - margin
    max_x = float(pts[:, 0].max()) + margin
    min_y = float(pts[:, 1].min()) - margin
    max_y = float(pts[:, 1].max()) + margin
    return min_x, max_x, min_y, max_y


def rasterize_height(pts, bounds, resolution, z_min, z_max):
    min_x, max_x, min_y, max_y = bounds
    w = int(math.ceil((max_x - min_x) / resolution))
    h = int(math.ceil((max_y - min_y) / resolution))
    # -inf 初期化で最大値更新
    grid = np.full((h, w), -np.inf, dtype=np.float32)

    # 高さクリップ
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    pts = pts[mask]
    if len(pts) == 0:
        return np.zeros((h, w), dtype=np.float32)

    ix = ((pts[:, 0] - min_x) / resolution).astype(int)
    iy = ((pts[:, 1] - min_y) / resolution).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix = ix[valid]; iy = iy[valid]
    z = pts[valid, 2]

    for x, y, vz in zip(ix, iy, z):
        if vz > grid[y, x]:
            grid[y, x] = vz

    # -inf → 0
    grid[grid == -np.inf] = z_min
    # 正規化 0..1
    if z_max == z_min:
        return np.zeros_like(grid)
    norm = (grid - z_min) / (z_max - z_min)
    norm = np.clip(norm, 0.0, 1.0)
    return norm


def rasterize_density(pts, bounds, resolution, z_min, z_max, log_scale=True):
    min_x, max_x, min_y, max_y = bounds
    w = int(math.ceil((max_x - min_x) / resolution))
    h = int(math.ceil((max_y - min_y) / resolution))
    grid = np.zeros((h, w), dtype=np.uint32)

    # 高さフィルタ
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    pts = pts[mask]
    if len(pts) == 0:
        return np.zeros((h, w), dtype=np.float32)

    ix = ((pts[:, 0] - min_x) / resolution).astype(int)
    iy = ((pts[:, 1] - min_y) / resolution).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    np.add.at(grid, (iy[valid], ix[valid]), 1)

    if grid.max() == 0:
        return np.zeros_like(grid, dtype=np.float32)

    if log_scale:
        g = np.log1p(grid.astype(np.float32))
        norm = g / g.max()
    else:
        norm = grid.astype(np.float32) / grid.max()
    return norm


def save_images(norm_grid: np.ndarray, bounds, args):
    # norm_grid: 0..1 (値大=高/密)
    # 画像: 値大ほど暗く (cost 風)
    img = (255 - (norm_grid * 255)).astype(np.uint8)
    if args.invert:
        img = 255 - img

    h, w = img.shape
    out_prefix = Path(args.out)
    # PNG
    if Image:
        Image.fromarray(img, mode="L").save(out_prefix.with_suffix(".png"))
    else:
        try:
            import matplotlib.pyplot as plt
            plt.imsave(out_prefix.with_suffix(".png"), img, cmap="gray", vmin=0, vmax=255)
        except ImportError:
            print("[WARN] PNG 出力不可 (PIL/matplotlib 無し)", file=sys.stderr)

    # PGM (P5)
    with open(out_prefix.with_suffix(".pgm"), "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(img.tobytes())

    # NPY
    np.save(out_prefix.with_suffix(".npy"), norm_grid.astype(np.float32))

    meta = dict(
        mode=args.mode,
        resolution=args.resolution,
        bounds=dict(min_x=bounds[0], max_x=bounds[1], min_y=bounds[2], max_y=bounds[3]),
        z_min=args.z_min,
        z_max=args.z_max,
        invert=args.invert,
        log_density=args.log_density if args.mode == "density" else None,
        shape=[int(h), int(w)]
    )
    with open(out_prefix.with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True)
    with open(out_prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser(description="PCD を鳥瞰図 (height / density) 画像へ変換")
    p.add_argument("--pcd", required=True, help="入力 PCD")
    p.add_argument("--out", required=True, help="出力プレフィックス (拡張子不要)")
    p.add_argument("--mode", choices=["height", "density"], default="height")
    p.add_argument("--resolution", type=float, default=0.1, help="[m/cell]")
    p.add_argument("--margin", type=float, default=0.0, help="バウンディング拡張[m]")
    p.add_argument("--z-min", type=float, default=-1e3, help="高さ下限 (clip)")
    p.add_argument("--z-max", type=float, default=1e3, help="高さ上限 (clip)")
    p.add_argument("--log-density", action="store_true", help="density モードで log1p 正規化")
    p.add_argument("--invert", action="store_true", help="輝度反転")
    return p.parse_args()


def main():
    args = parse_args()
    pts = load_points(Path(args.pcd))
    if args.z_max < args.z_min:
        print("[ERROR] z_max < z_min", file=sys.stderr)
        sys.exit(1)

    bounds = compute_bounds(pts, args.margin)

    if args.mode == "height":
        norm = rasterize_height(pts, bounds, args.resolution, args.z_min, args.z_max)
    else:
        norm = rasterize_density(pts, bounds, args.resolution,
                                 args.z_min, args.z_max, log_scale=args.log_density)

    save_images(norm, bounds, args)
    print("[DONE] 出力:", args.out)


if __name__ == "__main__":
    main()