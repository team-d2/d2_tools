#!/usr/bin/env python3
import argparse
import sys
import math
import json
import yaml
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d が必要です: pip install open3d", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class Config:
    crop_radius: float = 10.0          # [m] 軌跡からの最近傍水平距離フィルタ (負値で無効 = 距離制限なし)
    z_min: float = -1.0                # 高さ(絶対 or 相対) 下限
    z_max: float =  2.0                # 高さ(絶対 or 相対) 上限
    resolution: float = 0.1            # [m/cell]
    margin: float = 5.0                # 軌跡バウンディングボックス拡張
    count_scale: float = 5.0           # コスト = min(max_cost, count*scale) (--binary 時は無視)
    inflation_radius: float = 0.0      # [m] 膨張 (0 で無効)
    max_cost: int = 100                # 最大コスト
    min_obstacle_points_per_cell: int = 1  # セル確定に必要な最小点数
    binary: bool = False               # True: 点ありなら max_cost
    height_mode: str = "relative"      # "relative" | "absolute"
    verbose: bool = True               # ログ出力


def load_pcd_points(pcd_path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise ValueError(f"PCD が空: {pcd_path}")
    return np.asarray(pcd.points, dtype=np.float32)


def build_grid_bounds(traj_pts: np.ndarray, cfg: Config):
    min_x = float(traj_pts[:, 0].min()) - cfg.margin
    max_x = float(traj_pts[:, 0].max()) + cfg.margin
    min_y = float(traj_pts[:, 1].min()) - cfg.margin
    max_y = float(traj_pts[:, 1].max()) + cfg.margin
    return min_x, max_x, min_y, max_y


def filter_points(global_pts: np.ndarray,
                  traj_pts: np.ndarray,
                  bounds,
                  cfg: Config) -> np.ndarray:
    """
    マスク生成:
      1) 軌跡 bbox (+margin) 内
      2) 近傍距離 (crop_radius >=0 の場合)
      3) 高さ判定:
         height_mode == "absolute": z_min <= z <= z_max
         height_mode == "relative": z_min <= (z - z_traj_nearest) <= z_max
    戻り値: bool マスク
    """
    min_x, max_x, min_y, max_y = bounds
    in_box = (global_pts[:, 0] >= min_x) & (global_pts[:, 0] <= max_x) & \
             (global_pts[:, 1] >= min_y) & (global_pts[:, 1] <= max_y)

    # 最近傍軌跡点 (XY) を一度だけ求める
    if cKDTree is not None:
        tree = cKDTree(traj_pts[:, :2])
        dists, nn_idx = tree.query(global_pts[:, :2], k=1, workers=-1)
    else:
        # フォールバック (遅い)
        diff = global_pts[:, None, :2] - traj_pts[None, :, :2]
        d2 = np.sum(diff * diff, axis=2)
        nn_idx = np.argmin(d2, axis=1)
        dists = np.sqrt(d2[np.arange(len(global_pts)), nn_idx])

    if cfg.crop_radius >= 0.0:
        near = dists <= cfg.crop_radius
    else:
        near = np.ones(len(global_pts), dtype=bool)

    if cfg.height_mode == "relative":
        # 相対高さ = 点の z - 最近傍軌跡点 z
        rel_z = global_pts[:, 2] - traj_pts[nn_idx, 2]
        in_height = (rel_z >= cfg.z_min) & (rel_z <= cfg.z_max)
    else:
        # 絶対高さ
        in_height = (global_pts[:, 2] >= cfg.z_min) & (global_pts[:, 2] <= cfg.z_max)

    mask = in_box & near & in_height
    return mask


def points_to_costmap(global_pts: np.ndarray,
                      mask: np.ndarray,
                      bounds,
                      cfg: Config) -> np.ndarray:
    min_x, max_x, min_y, max_y = bounds
    width = int(math.ceil((max_x - min_x) / cfg.resolution))
    height = int(math.ceil((max_y - min_y) / cfg.resolution))
    counts = np.zeros((height, width), dtype=np.uint32)

    pts = global_pts[mask]
    if len(pts) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    ix = ((pts[:, 0] - min_x) / cfg.resolution).astype(int)
    iy = ((pts[:, 1] - min_y) / cfg.resolution).astype(int)
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    ix = ix[valid]
    iy = iy[valid]

    np.add.at(counts, (iy, ix), 1)

    if cfg.min_obstacle_points_per_cell > 1:
        counts[counts < cfg.min_obstacle_points_per_cell] = 0

    if cfg.binary:
        cost = np.zeros_like(counts, dtype=np.uint8)
        cost[counts > 0] = cfg.max_cost
    else:
        cost = np.minimum(cfg.max_cost,
                          (counts.astype(np.float32) * cfg.count_scale)).astype(np.uint8)
    return cost


def inflate(costmap: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.inflation_radius <= 0:
        return costmap
    r_cells = int(round(cfg.inflation_radius / cfg.resolution))
    if r_cells <= 0:
        return costmap
    h, w = costmap.shape
    src = costmap
    out = costmap.copy()
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            if dx*dx + dy*dy > r_cells*r_cells:
                continue
            if dx == 0 and dy == 0:
                continue
            sy0 = max(0, -dy)
            sy1 = min(h, h - dy)
            sx0 = max(0, -dx)
            sx1 = min(w, w - dx)
            dy0 = sy0 + dy
            dy1 = sy1 + dy
            dx0 = sx0 + dx
            dx1 = sx1 + dx
            out[dy0:dy1, dx0:dx1] = np.maximum(out[dy0:dy1, dx0:dx1], src[sy0:sy1, sx0:sx1])
    return out


def save_outputs(costmap: np.ndarray, bounds, cfg: Config, out_prefix: Path):
    min_x, max_x, min_y, max_y = bounds
    meta = dict(
        resolution=cfg.resolution,
        width=int(costmap.shape[1]),
        height=int(costmap.shape[0]),
        origin=[float(min_x), float(min_y), 0.0],
        range_x=[float(min_x), float(max_x)],
        range_y=[float(min_y), float(max_y)],
        max_cost=int(cfg.max_cost),
        parameters={
            "crop_radius": cfg.crop_radius,
            "z_min": cfg.z_min,
            "z_max": cfg.z_max,
            "height_mode": cfg.height_mode,
            "resolution": cfg.resolution,
            "margin": cfg.margin,
            "count_scale": cfg.count_scale,
            "inflation_radius": cfg.inflation_radius,
            "min_obstacle_points_per_cell": cfg.min_obstacle_points_per_cell,
            "binary": cfg.binary
        }
    )

    np.save(out_prefix.with_suffix(".npy"), costmap)
    with open(out_prefix.with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True)

    img_array = (255 - (costmap.astype(np.float32) / cfg.max_cost) * 255).astype(np.uint8)
    if Image:
        Image.fromarray(img_array, mode="L").save(out_prefix.with_suffix(".png"))
    else:
        try:
            import matplotlib.pyplot as plt
            plt.imsave(out_prefix.with_suffix(".png"), img_array, cmap="gray", vmin=0, vmax=255)
        except ImportError:
            print("[WARN] PNG 出力不可 (PIL/matplotlib 無し)", file=sys.stderr)

    with open(out_prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser(description="trajectory 基準の相対高さまたは絶対高さで 2D コストマップ生成")
    p.add_argument("--global", dest="global_pcd", required=True, help="globalmap.pcd")
    p.add_argument("--trajectory", dest="traj_pcd", required=True, help="trajectory.pcd")
    p.add_argument("--out", dest="out_prefix", required=True, help="出力プレフィックス (拡張子不要)")
    p.add_argument("--crop-radius", type=float, default=10.0, help="軌跡最近傍水平距離閾値 (負値で無効)")
    p.add_argument("--z-min", type=float, default=-0.2, help="高さ下限 (height-mode=relative では相対高さ)")
    p.add_argument("--z-max", type=float, default=1.0, help="高さ上限 (height-mode=relative では相対高さ)")
    p.add_argument("--height-mode", choices=["relative", "absolute"], default="relative",
                   help="relative: z - z_traj_nearest を判定 / absolute: 世界座標 z を判定")
    p.add_argument("--resolution", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=5.0)
    p.add_argument("--count-scale", type=float, default=5.0)
    p.add_argument("--inflation-radius", type=float, default=0.0)
    p.add_argument("--min-points", type=int, default=1)
    p.add_argument("--binary", action="store_true", help="セルに点があれば max_cost, 無ければ 0")
    p.add_argument("--no-verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        crop_radius=args.crop_radius,
        z_min=args.z_min,
        z_max=args.z_max,
        resolution=args.resolution,
        margin=args.margin,
        count_scale=args.count_scale,
        inflation_radius=args.inflation_radius,
        min_obstacle_points_per_cell=args.min_points,
        binary=args.binary,
        height_mode=args.height_mode,
        verbose=not args.no_verbose
    )

    if cfg.z_max < cfg.z_min:
        print("[ERROR] z_max < z_min", file=sys.stderr)
        sys.exit(1)

    global_pcd_path = Path(args.global_pcd)
    traj_pcd_path = Path(args.traj_pcd)
    out_prefix = Path(args.out_prefix)

    if cfg.verbose:
        print("[INFO] Loading global map:", global_pcd_path)
    global_pts = load_pcd_points(global_pcd_path)

    if cfg.verbose:
        print("[INFO] Loading trajectory:", traj_pcd_path)
    traj_pts = load_pcd_points(traj_pcd_path)

    if cfg.verbose:
        print("[INFO] Building bounds...")
    bounds = build_grid_bounds(traj_pts, cfg)

    if cfg.verbose:
        print(f"[INFO] Filtering points (mode={cfg.height_mode})...")
    mask = filter_points(global_pts, traj_pts, bounds, cfg)

    if cfg.verbose:
        print(f"[INFO] Selected points: {int(mask.sum())} / {len(global_pts)}")

    if cfg.verbose:
        print("[INFO] Rasterizing...")
    costmap = points_to_costmap(global_pts, mask, bounds, cfg)

    if cfg.inflation_radius > 0:
        if cfg.verbose:
            print("[INFO] Inflating costmap...")
        costmap = inflate(costmap, cfg)

    if cfg.verbose:
        print("[INFO] Saving outputs...")
    save_outputs(costmap, bounds, cfg, out_prefix)

    if cfg.verbose:
        print("[DONE] Output prefix:", out_prefix)


if __name__ == "__main__":
    main()