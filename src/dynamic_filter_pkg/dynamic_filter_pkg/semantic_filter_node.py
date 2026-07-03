#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
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
# P2: Kalman 追踪器
from .kalman_tracker import KalmanFilter3D, TrackState

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

        # [新增] 轨迹记忆字典与最大长度限制
        self.track_history = {}   # 格式: {track_id: [(X, Y, Z), ...]}
        self.max_history_len = 50 # 贪吃蛇的长度，保存过去 50 个点

        # P2: Kalman Filter 池 — 每个 track_id 对应一个 KF 实例
        self.kf_pool: dict[int, KalmanFilter3D] = {}

        # P1: 里程计订阅 — 用于多视图几何一致性的帧间位姿
        self.latest_odom = None
        self.latest_odom_time = None
        self.last_camera_pose = None   # 上一帧的相机位姿 (4,4) 齐次矩阵
        self.pose_timeout = 0.1        # 里程计超时阈值（秒），30Hz odom 容忍~3帧丢失

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
        self.track_pub = self.create_publisher(MarkerArray, '/pedestrian_tracks', 10)

    # ========================================================================
    #  P1: 里程计回调 + 帧间位姿计算
    # ========================================================================

    def odom_callback(self, msg):
        """缓存最新的轮式里程计消息。"""
        self.latest_odom = msg
        self.latest_odom_time = self.get_clock().now()

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

    def _run_motion_detection(self, cv_img, cv_depth, semantic_mask_np, fx, fy, cx, cy):
        """
        运动检测调度：多视图几何（主路径）→ LK+RANSAC（降级备份）。

        返回: motion_mask_np (uint8, H×W)
        """
        current_time = self.get_clock().now()

        # --- 判断里程计是否新鲜可用 ---
        pose_stale = (
            self.latest_odom is None or
            self.latest_odom_time is None or
            abs((current_time - self.latest_odom_time).nanoseconds * 1e-9) > self.pose_timeout
        )

        if pose_stale:
            # 降级路径：里程计超时或不可用
            if self.latest_odom is None:
                self.get_logger().warn(
                    '未收到 /odom 里程计数据，使用 LK+RANSAC 降级路径',
                    throttle_duration_sec=10.0)
            else:
                self.get_logger().info(
                    '里程计超时 (>100ms)，使用 LK+RANSAC 降级路径',
                    throttle_duration_sec=10.0)
            # 降级时也保持帧缓冲更新，避免多视图恢复时 prev_depth 过旧
            mask = self.motion_detector.detect(cv_img, cv_depth, semantic_mask_np)
            self.motion_detector._store_frame(cv_img, cv_depth)
            self.last_camera_pose = None  # 位姿不连续，重置多视图状态
            return mask

        # --- 主路径：多视图深度重投影 ---
        # 将 /odom (base_footprint→odom) 与静态变换组合，得到相机真实位姿
        T_world_base = self._odom_to_pose_matrix(self.latest_odom)
        T_world_curr = T_world_base @ self.T_base_to_camera  # camera_link_optical 在世界帧的位姿

        if self.last_camera_pose is None:
            # 首帧：仅有里程计但无上一帧位姿 → 先缓冲帧，用 LK 出结果后再持久化位姿
            motion_mask = self.motion_detector.detect(cv_img, cv_depth, semantic_mask_np)
            self.motion_detector._store_frame(cv_img, cv_depth)
            self.last_camera_pose = T_world_curr
            return motion_mask

        # 计算帧间相机运动
        T_prev_to_curr = self._compute_relative_transform(
            self.last_camera_pose, T_world_curr)

        # 防御性检查：位姿矩阵形状必须为 (4,4)
        if T_prev_to_curr.shape != (4, 4):
            self.get_logger().error(
                f'位姿矩阵形状异常: T_prev_to_curr.shape={T_prev_to_curr.shape}，'
                f'降级到 LK+RANSAC', throttle_duration_sec=5.0)
            mask = self.motion_detector.detect(cv_img, cv_depth, semantic_mask_np)
            self.motion_detector._store_frame(cv_img, cv_depth)
            self.last_camera_pose = None
            return mask

        camera_intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        try:
            motion_mask, used_multiview = self.motion_detector.detect_multiview(
                cv_img, cv_depth, semantic_mask_np, camera_intrinsics, T_prev_to_curr)
        except Exception as e:
            self.get_logger().error(
                f'多视图检测异常，降级到 LK+RANSAC: {e}',
                throttle_duration_sec=5.0)
            mask = self.motion_detector.detect(cv_img, cv_depth, semantic_mask_np)
            self.motion_detector._store_frame(cv_img, cv_depth)
            self.last_camera_pose = None  # 多视图状态不一致，重置
            return mask

        # 持久化当前帧位姿供下一帧使用
        self.last_camera_pose = T_world_curr

        if not used_multiview:
            # detect_multiview 因首帧/无有效像素等原因未产出有效结果，降级到 LK
            try:
                mask = self.motion_detector.detect(cv_img, cv_depth, semantic_mask_np)
            except Exception as e:
                self.get_logger().error(
                    f'LK+RANSAC 降级路径异常，返回空掩码: {e}',
                    throttle_duration_sec=5.0)
                mask = np.zeros(cv_depth.shape, dtype=np.uint8)
            self.motion_detector._store_frame(cv_img, cv_depth)
            return mask

        # 多视图成功：同步 LK 灰度缓冲（detect_multiview 不再更新 prev_gray）
        self.motion_detector._store_frame(cv_img, cv_depth)
        return motion_mask

    def sync_callback(self, img_msg, depth_msg, info_msg):
        try:
            K = info_msg.k
            fx, fy, cx, cy = K[0], K[4], K[2], K[5]

            cv_img = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            debug_img = cv_img.copy()

            results = self.model.track(
                cv_img, classes=[0], conf=0.10, persist=True, 
                tracker="botsort.yaml", verbose=False, retina_masks=True
            )

            marker_array = MarkerArray()
            
            h, w = cv_depth.shape
            semantic_mask_np = np.zeros((h, w), dtype=bool)
            
            # [新增] 获取当前帧存活的 track_ids，用于清理失效的历史轨迹
            current_active_ids = set()

            if results and len(results[0].boxes) > 0:
                result = results[0]
                
                # 1. 动态深度剔除
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

                # 2. 3D 目标追踪映射（Kalman Filter 平滑）
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy()
                    current_time = self.get_clock().now()

                    for box, track_id in zip(boxes, track_ids):
                        current_active_ids.add(track_id)
                        x1, y1, x2, y2 = map(int, box)

                        u = (x1 + x2) / 2.0
                        v = (y1 + y2) / 2.0

                        roi = cv_depth[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if roi.size == 0 or np.all(np.isnan(roi)):
                            continue

                        Z = np.nanmedian(roi)
                        if np.isnan(Z) or Z <= 0.1:
                            continue

                        X = (u - cx) * Z / fx
                        Y = (v - cy) * Z / fy
                        bbox_area = (x2 - x1) * (y2 - y1)

                        # --- KF 预测 + 更新 ---
                        if track_id not in self.kf_pool:
                            self.kf_pool[track_id] = KalmanFilter3D(
                                track_id, (X, Y, Z), current_time)

                        kf = self.kf_pool[track_id]
                        kf.predict(current_time=current_time)
                        kf_smoothed = kf.update(
                            (X, Y, Z), bbox_area=bbox_area, current_time=current_time)
                        kf_x, kf_y, kf_z = float(kf_smoothed[0]), float(kf_smoothed[1]), float(kf_smoothed[2])

                        # 轨迹历史：喂入 KF 平滑位置（用于 RViz 路径可视化）
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append((kf_x, kf_y, kf_z))
                        if len(self.track_history[track_id]) > self.max_history_len:
                            self.track_history[track_id].pop(0)

                        # 构建 Marker（使用 KF 平滑位置替代原始观测）
                        bbox_color = (0.0, 1.0, 0.0, 0.5)  # 绿色：正常
                        if kf.state == TrackState.OCCLUDED:
                            bbox_color = (1.0, 1.0, 0.0, 0.5)  # 黄色：遮挡

                        bbox_marker = self.create_marker(
                            header=img_msg.header, marker_type=Marker.CUBE,
                            m_id=int(track_id), color=bbox_color, scale=(0.6, 0.6, 1.6),
                            x=kf_x, y=kf_y, z=kf_z
                        )
                        text_marker = self.create_marker(
                            header=img_msg.header, marker_type=Marker.TEXT_VIEW_FACING,
                            m_id=int(track_id) + 1000, color=(1.0, 1.0, 1.0, 1.0),
                            scale=(0.0, 0.0, 0.4),
                            x=kf_x, y=kf_y, z=kf_z - 1.0, text=f"ID: {track_id}"
                        )
                        path_marker = self.create_path_marker(
                            header=img_msg.header, track_id=int(track_id),
                            history=self.track_history[track_id]
                        )

                        marker_array.markers.extend([bbox_marker, text_marker, path_marker])

            # P2: 对当前帧未观测到的轨迹执行预测（遮挡处理）
            current_time = self.get_clock().now()
            for track_id in list(self.kf_pool.keys()):
                if track_id not in current_active_ids:
                    kf = self.kf_pool[track_id]
                    kf.predict(current_time=current_time)
                    if kf.is_lost:
                        # 长期丢失：清理 KF 和轨迹历史
                        del self.kf_pool[track_id]
                        if track_id in self.track_history:
                            del self.track_history[track_id]

            # 3. 运动目标检测（多视图几何为主，LK+RANSAC 降级备份）
            motion_mask_np = self._run_motion_detection(cv_img, cv_depth, semantic_mask_np,
                                                         fx, fy, cx, cy)

            # 4. 掩码并集融合
            final_mask = semantic_mask_np | (motion_mask_np > 0)
            
            cv_depth[final_mask] = np.nan

            debug_img[semantic_mask_np] = [0, 0, 255]       
            debug_img[motion_mask_np > 0] = [255, 0, 0]      

            filtered_depth_msg = self.cv_bridge.cv2_to_imgmsg(cv_depth, encoding=depth_msg.encoding)
            filtered_depth_msg.header = depth_msg.header
            self.depth_pub.publish(filtered_depth_msg)

            debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = img_msg.header
            self.debug_pub.publish(debug_msg)
            
            self.track_pub.publish(marker_array)

        except Exception as e:
            import traceback
            self.get_logger().error(f'Processing Error: {e}')
            self.get_logger().error(f'Traceback:\n{traceback.format_exc()}')

    def create_marker(self, header, marker_type, m_id, color, scale, x, y, z, text=""):
        """Marker 辅助生成函数"""
        m = Marker()
        m.header = header
        m.ns = "pedestrians"
        m.id = m_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0 
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        if text: m.text = text
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200000000 
        return m

    # [新增] 专门用于绘制轨迹的辅助函数
    def create_path_marker(self, header, track_id, history):
        m = Marker()
        m.header = header
        m.ns = "pedestrian_paths"
        m.id = track_id + 2000  # 偏移 ID 空间避免与 Bbox 和 Text 冲突
        m.type = Marker.LINE_STRIP # 设定为连续线段格式
        m.action = Marker.ADD
        
        m.scale.x = 0.05 # 轨迹线条的粗细
        
        # 轨迹颜色设定：黄色半透明
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8 
        
        # 将历史坐标点组装进线条数组
        for pt in history:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2] - 0.7) # Z减去0.7是让轨迹线贴在行人的脚底位置，更符合视觉直觉
            m.points.append(p)
            
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200000000
        return m

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