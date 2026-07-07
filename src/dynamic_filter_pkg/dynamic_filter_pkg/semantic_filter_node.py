#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np
import torch
import os
import tf2_ros
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

# 引入光流运动引擎
from .motion_detector import MotionDetector
# P7: 统一物体追踪器（行人+非语义物体，匈牙利+Kalman统一流水线）
from .object_tracker import ObjectTracker

class SemanticFilterNode(Node):
    def __init__(self):
        super().__init__('semantic_filter_node')
        
        self.cv_bridge = CvBridge()
        
        # 1. 加载 YOLOv8 模型
        pkg_share_dir = get_package_share_directory('dynamic_filter_pkg')
        weights_path = os.path.join(pkg_share_dir, 'weights', 'yolov8n-seg.pt')
        if not os.path.exists(weights_path):
            self.get_logger().error(
                f'YOLO 权重文件未找到: {weights_path}\n'
                f'  请确认 colcon build 成功执行，且 setup.py 正确安装 weights 到 share 目录。'
            )
            raise FileNotFoundError(weights_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        self.get_logger().info(f'YOLO 加载成功 | 设备: {self.device}')

        # 初始化运动检测引擎
        self.motion_detector = MotionDetector(max_corners=300, min_displacement=2.0, depth_tolerance=0.15)
        self.get_logger().info('LK光流与深度生长引擎 初始化成功')

        # P7: 初始化统一物体追踪器（行人+箱子，匈牙利+Kalman统一流水线）
        self.object_tracker = ObjectTracker(
            match_max_dist=0.5, min_confirm=3, max_lost=10)
        self.get_logger().info('P7 统一物体追踪器 初始化成功')

        # 里程计缓存 — 用于相机世界位姿（追踪坐标转换）
        self.latest_odom = None

        # P1: TF — 获取 camera_link_optical 在世界帧中的真实位姿
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.T_base_to_camera = self._init_camera_transform()
        self.get_logger().info(
            f'相机静态变换T_base→optical已加载: '
            f'translation=({self.T_base_to_camera[0,3]:.3f}, '
            f'{self.T_base_to_camera[1,3]:.3f}, {self.T_base_to_camera[2,3]:.3f})')

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # 2. 设置时间同步订阅器
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/image_raw', qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw', qos_profile=qos_profile_sensor_data)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/camera_info', qos_profile=qos_profile_sensor_data)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.info_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

        # 3. 设置发布器
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_filtered', 10)
        self.debug_pub = self.create_publisher(Image, '/camera/image_debug', 10)
        self.track_pub = self.create_publisher(MarkerArray, '/dynamic_tracks', 10)

        # 精简 debug topic: 仅保留关键掩码用于验证过滤效果
        self.final_debug_pub = self.create_publisher(Image, '/camera/final_mask', 10)

        # P4: RGB域动态过滤 — 动态区域涂黑后发给 RTAB-Map，阻止ORB特征提取
        self.rgb_filtered_pub = self.create_publisher(Image, '/camera/image_filtered', 10)

        # P7: 追踪物体可视化
        self.tracked_objects_pub = self.create_publisher(
            MarkerArray, '/tracked_objects', 10)

    # ========================================================================
    #  P1: 里程计回调 + 帧间位姿计算
    # ========================================================================

    def odom_callback(self, msg):
        """缓存最新的轮式里程计消息。"""
        self.latest_odom = msg

    def _init_camera_transform(self):
        """
        获取 base_footprint → camera_link_optical 的静态刚体变换 (4×4)。

        优先从 TF 树查询（适配不同机器人URDF），失败时回退到硬编码值。
        该变换在节点生命周期内保持不变（static joint）。
        """
        # 尝试从 TF 获取
        try:
            t = self.tf_buffer.lookup_transform(
                'base_footprint', 'camera_link_optical',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3.0))
            trans = t.transform.translation
            rot = t.transform.rotation
            R = self._quat_to_rotm(rot.x, rot.y, rot.z, rot.w)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [trans.x, trans.y, trans.z]
            self.get_logger().info('相机静态变换已从 TF 树获取')
            return T
        except Exception as e:
            self.get_logger().warn(
                f'TF 查询 camera 变换失败 ({e})，使用 URDF 硬编码值')

        # 回退：从 robot.xacro 硬编码
        # 变换链: base_footprint → base_link(0,0,0.15) → camera_link(0.2,0,0.05) → optical(rpy=-π/2,0,-π/2)
        T = np.eye(4)
        # optical frame 旋转矩阵: z-forward, x-right, y-down → base(x-forward, y-left, z-up)
        T[:3, :3] = [[0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0]]
        T[:3, 3] = [0.2, 0, 0.2]   # x=0.2 (相机前移), z=0.15+0.05=0.2 (底座+支架高度)
        return T

    @staticmethod
    def _quat_to_rotm(qx, qy, qz, qw):
        """四元数 → 3×3 旋转矩阵（手动实现，避免 scipy 依赖）"""
        return np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
        ])

    def _odom_to_pose_matrix(self, odom_msg):
        """Odometry 消息 → 4×4 齐次变换矩阵（世界坐标系下的机器人位姿）。"""
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        R = self._quat_to_rotm(q.x, q.y, q.z, q.w)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [p.x, p.y, p.z]
        return T

    def _compute_relative_transform(self, T_world_prev, T_world_curr):
        """
        计算帧间相机运动：T_prev→curr

        P_world = T_world_prev @ P_prev = T_world_curr @ P_curr
        ⇒ P_curr = inv(T_world_curr) @ T_world_prev @ P_prev
        ⇒ T_prev→curr = inv(T_world_curr) @ T_world_prev
        """
        return np.linalg.inv(T_world_curr) @ T_world_prev

    def _run_motion_detection(self, cv_img, cv_depth, semantic_mask_np, fx, fy, cx, cy,
                              T_world_cam=None):
        """
        P5: 深度聚类检测（世界坐标系 — 主检测器）。

        P5 作为主力几何检测器，P6 仅在 P5 无法工作的特殊场景下降级启用。
        不再使用 OR 融合——P6 的常驻运行是前两轮仿真大面积误检的来源。

        参考:
          DetectFusion (2019): 松耦合预处理层，世界坐标 ICP 聚类
          DynaSLAM (2018): 多视图重投影前消除相机自运动
          DS-SLAM (2018): 紧耦合 ORB-SLAM2 内做光流+极线（松耦合下无可比成功先例）

        :return: (motion_mask, flow_mask, cluster_mask)
        """
        camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

        # --- P5: 深度聚类检测（主检测器，世界坐标系）---
        try:
            cluster_mask = self.motion_detector.detect_clustering(
                cv_depth, camera_intrinsics, semantic_mask_np,
                T_world_cam=T_world_cam)
        except Exception as e:
            self.get_logger().error(
                f'P5 深度聚类检测异常，返回空掩码: {e}',
                throttle_duration_sec=5.0)
            cluster_mask = np.zeros(cv_depth.shape, dtype=np.uint8)

        # --- P6: 光流检测（降级回退，仅在 P5 不可用时启用）---
        # P6 设计为偶尔的降级检测器，不是常驻主检测器。
        # 常驻运行会导致: 纯旋转→全帧误检, 白墙少特征→RANSAC不稳定,
        # floodFill沿深度梯度蔓延→大面积假阳性。参考 DS-SLAM 的成功经验：
        # 光流+极线仅在紧耦合 ORB-SLAM2（BA优化位姿+500+ORB特征）下可靠。
        n_clusters = np.count_nonzero(cluster_mask)
        flow_mask = np.zeros(cv_depth.shape, dtype=np.uint8)
        if n_clusters < 300:  # P5 簇覆盖像素数不足（无足够深度簇时启用P6）
            try:
                flow_mask = self.motion_detector.detect(
                    cv_img, cv_depth, semantic_mask_np)
            except Exception as e:
                self.get_logger().error(
                    f'P6 光流检测异常: {e}',
                    throttle_duration_sec=5.0)

        # --- 融合: P5 主力 + P6 补充 ---
        motion_mask = np.maximum(cluster_mask, flow_mask)

        # --- 最小面积过滤 ---
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            motion_mask, connectivity=4)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < 200:  # <200px ≈ 非真实物体
                motion_mask[labels == lbl] = 0

        # 保持帧缓冲同步（光流检测的 prev_gray 需要连续帧）
        try:
            self.motion_detector._store_frame(cv_img, cv_depth)
        except Exception as e:
            self.get_logger().error(
                f'帧缓冲更新异常，重置运动检测器状态: {e}',
                throttle_duration_sec=5.0)
            self.motion_detector.prev_gray = None
            self.motion_detector.prev_pts = None
            self.motion_detector.has_prev_frame = False

        return motion_mask, flow_mask, cluster_mask

    def sync_callback(self, img_msg, depth_msg, info_msg):
        try:
            K = info_msg.k
            fx, fy, cx, cy = K[0], K[4], K[2], K[5]

            cv_img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # 预先计算相机→世界变换（一帧内不变，避免重复计算）
            if self.latest_odom is not None:
                T_wb = self._odom_to_pose_matrix(self.latest_odom)
                T_world_cam = T_wb @ self.T_base_to_camera
            else:
                T_world_cam = np.eye(4, dtype=np.float64)

            # 方案A: 统一检测 → 不依赖 Bot-SORT 追踪ID, 由 ObjectTracker 的匈牙利算法统一关联
            results = self.model(
                cv_img, classes=[0], conf=0.10,
                verbose=False, retina_masks=True
            )

            h, w = cv_depth.shape
            semantic_mask_np = np.zeros((h, w), dtype=bool)
            external_detections = []

            if results and len(results[0].boxes) > 0:
                result = results[0]

                # 1. 语义掩码提取（用于深度过滤 + P3-②膨胀保护带）
                if result.masks is not None:
                    raw_mask = torch.any(result.masks.data, dim=0).cpu().numpy()
                    if raw_mask.shape != (h, w):
                        semantic_mask_np = cv2.resize(
                            raw_mask.astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    else:
                        semantic_mask_np = raw_mask

                # 2. YOLO检测 → 统一追踪器外部检测格式（纯几何匈牙利关联，无Bot-SORT）
                boxes = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w, x2), min(h, y2)

                    u = (x1 + x2) / 2.0
                    v = (y1 + y2) / 2.0

                    roi = cv_depth[y1c:y2c, x1c:x2c]
                    if roi.size == 0 or np.all(np.isnan(roi)):
                        continue

                    Z = np.nanmedian(roi)
                    if np.isnan(Z) or Z <= 0.1:
                        continue

                    # 相机坐标 → 世界坐标（复用预先计算的 T_world_cam）
                    X_cam = (u - cx) * Z / fx
                    Y_cam = (v - cy) * Z / fy
                    P_cam = np.array([X_cam, Y_cam, Z, 1.0])
                    P_world = T_world_cam @ P_cam

                    # 实例掩码: bbox ∩ YOLO精细分割
                    inst_mask = np.zeros((h, w), dtype=bool)
                    inst_mask[y1c:y2c, x1c:x2c] = True
                    if result.masks is not None:
                        sm = result.masks.data[i].cpu().numpy()
                        if sm.shape != (h, w):
                            sm = cv2.resize(
                                sm.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        inst_mask = inst_mask & sm.astype(bool)

                    external_detections.append({
                        'centroid_world': P_world[:3],
                        'centroid_2d': (u, v),
                        'bbox': (x1c, y1c, x2c - x1c, y2c - y1c),
                        'area': (x2 - x1) * (y2 - y1),
                        'depth_median': float(Z),
                        'mask': inst_mask,
                        'class_label': 'person',
                    })

            current_time = self.get_clock().now()
            marker_array = MarkerArray()

            # 3. 运动目标检测（P5 深度聚类 主检测器 + P6 光流 降级）
            motion_mask_np, flow_mask_np, cluster_mask_np = self._run_motion_detection(
                cv_img, cv_depth, semantic_mask_np, fx, fy, cx, cy,
                T_world_cam=T_world_cam)

            # 4. 掩码并集融合
            final_mask = semantic_mask_np | (motion_mask_np > 0)

            # ================================================================
            #  Debug 可视化 + 发布（在 P7 之前，确保 P7 异常时也能看到检测结果）
            #  四色: 红=语义 蓝=聚类(P5) 绿=光流(P6) 青=聚类∩光流 品红=语义∩运动
            # ================================================================
            cluster_bool = cluster_mask_np > 0
            flow_bool = flow_mask_np > 0
            debug_img = cv_img.copy()
            debug_img[cluster_bool & ~flow_bool & ~semantic_mask_np] = [255, 0, 0]       # 仅聚类: 蓝
            debug_img[flow_bool & ~cluster_bool & ~semantic_mask_np] = [0, 255, 0]       # 仅光流: 绿
            debug_img[cluster_bool & flow_bool & ~semantic_mask_np] = [255, 255, 0]      # 聚类∩光流: 青
            debug_img[semantic_mask_np & ~cluster_bool & ~flow_bool] = [0, 0, 255]       # 仅语义: 红
            debug_img[semantic_mask_np & (cluster_bool | flow_bool)] = [255, 0, 255]     # 语义∩运动: 品红

            debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = img_msg.header
            self.debug_pub.publish(debug_msg)

            final_mask_uint8 = final_mask.astype(np.uint8) * 255
            final_msg = self.cv_bridge.cv2_to_imgmsg(final_mask_uint8, encoding='mono8')
            final_msg.header = img_msg.header
            self.final_debug_pub.publish(final_msg)

            # ================================================================
            #  P7: 统一物体追踪（匈牙利+Kalman，行人+箱子统一流水线）
            #  追踪失败不应该阻断过滤输出——RTAB-Map依赖数据连续
            # ================================================================
            camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

            feedback_mask = np.zeros((h, w), dtype=np.uint8)
            tracked_objects = []
            obj_markers = []
            try:
                feedback_mask, tracked_objects, obj_markers = self.object_tracker.update(
                    motion_mask_np, cv_depth, camera_intrinsics,
                    img_msg.header, current_time, T_world_cam=T_world_cam,
                    external_detections=external_detections)
            except Exception as e:
                self.get_logger().error(
                    f'P7 追踪异常，跳过本帧追踪（不影响过滤输出）: {e}',
                    throttle_duration_sec=2.0)

            if feedback_mask.any():
                final_mask = final_mask | (feedback_mask > 0)

            # ================================================================
            #  深度域 + RGB 域动态过滤 → RTAB-Map 松耦合输入
            # ================================================================
            marker_array.markers.extend(obj_markers)
            if tracked_objects:
                tracked_msg = MarkerArray()
                tracked_msg.markers = obj_markers
                self.tracked_objects_pub.publish(tracked_msg)

            cv_depth[final_mask] = np.nan
            filtered_depth_msg = self.cv_bridge.cv2_to_imgmsg(cv_depth, encoding=depth_msg.encoding)
            filtered_depth_msg.header = depth_msg.header
            self.depth_pub.publish(filtered_depth_msg)

            rgb_filtered = cv_img.copy()
            rgb_filtered[final_mask] = 0
            rgb_filtered_msg = self.cv_bridge.cv2_to_imgmsg(rgb_filtered, encoding='bgr8')
            rgb_filtered_msg.header = img_msg.header
            self.rgb_filtered_pub.publish(rgb_filtered_msg)

            self.track_pub.publish(marker_array)

        except Exception as e:
            import traceback
            self.get_logger().error(f'Processing Error: {e}')
            self.get_logger().error(f'Traceback:\n{traceback.format_exc()}')

def main(args=None):
    rclpy.init(args=args)
    node = SemanticFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()