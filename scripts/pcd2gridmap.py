#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import rclpy.utilities
import argparse
import sys
import os
import math
import numpy as np
import open3d as o3d
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

class Pcd2Gridmap(Node):
    def __init__(self, args):
        super().__init__('pcd2gridmap_node')

        # パラメータを argparse から取得
        self.pcd_file = args.pcd_file
        self.resolution = args.resolution
        self.min_z = args.min_z
        self.max_z = args.max_z
        self.frame_id = args.frame_id
        self.map_topic = args.map_topic
        self.margin = args.margin
        # ★ invert オプションを取得
        self.invert = args.invert 
        publish_rate = args.publish_rate

        self.get_logger().info("--- pcd2gridmap 設定 (invert対応版) ---")
        self.get_logger().info(f"  PCDファイル: {self.pcd_file}")
        self.get_logger().info(f"  解像度 (m): {self.resolution}")
        self.get_logger().info(f"  Z占有範囲: [{self.min_z}, {self.max_z}] m")
        self.get_logger().info(f"  マージン (m): {self.margin}")
        # ★ ログ出力
        self.get_logger().info(f"  反転 (invert): {self.invert}") 
        self.get_logger().info(f"  Mapトピック: {self.map_topic}")
        self.get_logger().info(f"  Frame ID: {self.frame_id}")
        self.get_logger().info(f"  配信レート (Hz): {publish_rate}")
        self.get_logger().info("-------------------------------------")

        self.gridmap_msg = None

        # QoS設定 (map_saver互換)
        qos_profile_map = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.gridmap_pub = self.create_publisher(
            OccupancyGrid,
            self.map_topic,
            qos_profile=qos_profile_map
        )

        try:
            cloud = self.load_pcd(self.pcd_file)
            if cloud is not None:
                self.gridmap_msg = self.cloud_to_gridmap(cloud)
                if self.gridmap_msg:
                    self.get_logger().info(f"グリッドマップ生成完了。トピック '{self.map_topic}' で配信開始。")
                    self.timer = self.create_timer(1.0 / publish_rate, self.publish_gridmap)
                    self.publish_gridmap() # 起動直後に発行
                else:
                    self.get_logger().error("グリッドマップのデータが空です。ノードを終了します。")
                    rclpy.shutdown()
            else:
                self.get_logger().error("グリッドマップの生成に失敗しました。ノードを終了します。")
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"初期化中にエラーが発生しました: {e}")
            rclpy.shutdown()

    def load_pcd(self, file_path):
        if not os.path.exists(file_path):
            self.get_logger().error(f"PCDファイルが見つかりません: {file_path}")
            return None
        self.get_logger().info(f"{file_path} を読み込んでいます...")
        try:
            cloud = o3d.io.read_point_cloud(file_path)
            if not cloud.has_points():
                self.get_logger().warn(f"PCDファイルに点が含まれていません: {file_path}")
                return None
            return cloud
        except Exception as e:
            self.get_logger().error(f"PCDファイルの読み込みに失敗しました: {e}")
            return None

    def cloud_to_gridmap(self, cloud):
        points = np.asarray(cloud.points, dtype=np.float32)
        if points.shape[0] == 0:
            self.get_logger().warn("点群データが空です。マップを生成できません。")
            return None

        min_x = float(points[:, 0].min()) - self.margin
        max_x = float(points[:, 0].max()) + self.margin
        min_y = float(points[:, 1].min()) - self.margin
        max_y = float(points[:, 1].max()) + self.margin

        width = int(math.ceil((max_x - min_x) / self.resolution))
        height = int(math.ceil((max_y - min_y) / self.resolution))

        if width == 0 or height == 0:
            self.get_logger().warn(f"グリッドサイズが0です (W:{width}, H:{height})。マップを生成できません。")
            return None

        height_grid = np.full((height, width), -np.inf, dtype=np.float32)

        ix = ((points[:, 0] - min_x) / self.resolution).astype(int)
        iy = ((points[:, 1] - min_y) / self.resolution).astype(int)
        
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        ix = ix[valid]
        iy = iy[valid]
        z = points[valid, 2]

        np.maximum.at(height_grid, (iy, ix), z)

        gridmap = OccupancyGrid()
        gridmap.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id=self.frame_id
        )
        gridmap.info.resolution = self.resolution
        gridmap.info.width = width
        gridmap.info.height = height
        gridmap.info.origin = Pose()
        gridmap.info.origin.position.x = min_x 
        gridmap.info.origin.position.y = min_y
        gridmap.info.origin.orientation.w = 1.0

        # ★ 7. OccupancyGrid (data) へ変換 (invert オプション適用)
        
        data = np.full((height, width), -1, dtype=np.int8) # 不明で初期化
        
        # マスクの定義
        free_mask = (height_grid < self.min_z) # 空き (床) 領域
        occ_mask = (height_grid >= self.min_z) & (height_grid <= self.max_z) # 占有 (障害物) 領域
        
        if self.invert:
            # 反転ロジック
            data[free_mask] = 100 # 元の空き -> 占有
            data[occ_mask] = 0    # 元の占有 -> 空き
        else:
            # 通常ロジック
            data[free_mask] = 0    # 空き
            data[occ_mask] = 100   # 占有
        
        gridmap.data = data.ravel().tolist()
        
        self.get_logger().info(f"グリッドマップサイズ: {width} x {height} (ロジック: max_height)")
        return gridmap

    def publish_gridmap(self):
        if self.gridmap_msg:
            self.gridmap_msg.header.stamp = self.get_clock().now().to_msg()
            self.gridmap_pub.publish(self.gridmap_msg)
        else:
            self.get_logger().warn("マップデータをパブリッシュしようとしましたが、データがありません。", throttle_duration_sec=5.0)


def main(args=None):
    try:
        import open3d
        import numpy
    except ImportError as e:
        print(f"エラー: 必要なPythonライブラリが見つかりません: {e}", file=sys.stderr)
        print("pip install open3d numpy", file=sys.stderr)
        return

    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='PCDをOccupancyGridに変換するROS2ノード (MaxHeightロジック)')
    parser.add_argument('--pcd_file', type=str, default='test.pcd', help='入力PCDファイル')
    parser.add_argument('--resolution', type=float, default=0.1, help='解像度 (m)')
    parser.add_argument('--min_z', type=float, default=0.0, help='占有(100)とみなすZ座標最小値')
    parser.add_argument('--max_z', type=float, default=1.0, help='占有(100)とみなすZ座標最大値')
    parser.add_argument('--margin', type=float, default=0.0, help='バウンディングボックスの拡張マージン (m)')
    parser.add_argument('--frame_id', type=str, default='map', help='frame_id')
    parser.add_argument('--map_topic', type=str, default='gridmap', help='トピック名')
    parser.add_argument('--publish_rate', type=float, default=1.0, help='配信レート (Hz)')
    parser.add_argument(
        '--invert', 
        action='store_true', 
        help='占有(100)と空き(0)を反転する'
    )

    remaining_args = rclpy.utilities.remove_ros_args(args=sys.argv[1:])
    custom_args = parser.parse_args(remaining_args)

    pcd2gridmap_node = Pcd2Gridmap(args=custom_args)

    try:
        rclpy.spin(pcd2gridmap_node)
    except KeyboardInterrupt:
        pass
    finally:
        pcd2gridmap_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()