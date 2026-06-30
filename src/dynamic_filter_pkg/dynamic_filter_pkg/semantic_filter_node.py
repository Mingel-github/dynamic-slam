#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point  # [新增] 用于构建轨迹点
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np
import torch
import os
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

# 引入光流运动引擎
from .motion_detector import MotionDetector

class SemanticFilterNode(Node):
    def __init__(self):
        super().__init__('semantic_filter_node')
        
        self.cv_bridge = CvBridge()
        
        # 1. 加载 YOLOv8 模型
        pkg_share_dir = get_package_share_directory('dynamic_filter_pkg')
        weights_path = os.path.join(pkg_share_dir, 'weights', 'yolov8n-seg.pt') 
        if not os.path.exists(weights_path):
             weights_path = os.path.join(os.getcwd(), 'src/dynamic_filter_pkg/dynamic_filter_pkg/weights/yolov8n-seg.pt')

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

                # 2. 3D 目标追踪映射
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy()

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

                        # [新增] 记录历史轨迹坐标
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append((X, Y, Z))
                        
                        # 限制轨迹长度，丢弃最早的点
                        if len(self.track_history[track_id]) > self.max_history_len:
                            self.track_history[track_id].pop(0)

                        # 构建 3D Bbox Marker
                        bbox_marker = self.create_marker(
                            header=img_msg.header, marker_type=Marker.CUBE, 
                            m_id=int(track_id), color=(0.0, 1.0, 0.0, 0.5), scale=(0.6, 0.6, 1.6),
                            x=X, y=Y, z=Z
                        )
                        # 构建 ID 文本 Marker
                        text_marker = self.create_marker(
                            header=img_msg.header, marker_type=Marker.TEXT_VIEW_FACING, 
                            m_id=int(track_id) + 1000, color=(1.0, 1.0, 1.0, 1.0), scale=(0.0, 0.0, 0.4),
                            x=X, y=Y, z=Z - 1.0, text=f"ID: {track_id}"
                        )
                        # [新增] 构建 轨迹线条 Marker
                        path_marker = self.create_path_marker(
                            header=img_msg.header, track_id=int(track_id), history=self.track_history[track_id]
                        )
                        
                        marker_array.markers.extend([bbox_marker, text_marker, path_marker])

            # [新增] 内存管理：清理已经消失超过一定时间的 ID 的轨迹
            expired_ids = [tid for tid in self.track_history.keys() if tid not in current_active_ids]
            for tid in expired_ids:
                # 为了防止刚离开视线轨迹就消失，这里可以做一个稍微延迟的删除逻辑。
                # 目前简单起见，如果从画面丢失，就清空该 ID 记录释放内存。
                del self.track_history[tid]

            # 3. 执行运动目标检测
            motion_mask_np = self.motion_detector.detect(cv_img, cv_depth, semantic_mask_np)

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
            self.get_logger().error(f'Processing Error: {e}')

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