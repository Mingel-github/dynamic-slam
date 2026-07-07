"""
P7: 非语义动态物体追踪与状态估计。

流水线:
  ① 实例提取: final_mask → 连通域 → 3D 反投影 → 世界坐标质心
  ② 匈牙利数据关联: M×N 距离矩阵 → 匹配/新生/丢失
  ③ Kalman 追踪: 状态 [x,y,z,vx,vy,vz], 预测→更新→平滑 → 速度+轨迹
  ④ RViz2 可视化: /tracked_objects (MarkerArray: 位置框+速度箭头+轨迹线+标签)

参考: DynaSLAM II (紧耦合 Joint BA 追踪)、DetectFusion (松耦合预处理层追踪)
"""

import numpy as np
import cv2
from collections import OrderedDict
from enum import Enum

# 复用 P2 的 Kalman Filter 实现
from .kalman_tracker import KalmanFilter3D, TrackState

# 匈牙利算法: scipy 快，不可用时回退贪心
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    linear_sum_assignment = None


class ObjectState(Enum):
    TENTATIVE = 0    # 候选（连续检测不足 N 帧）
    ACTIVE = 1       # 确认追踪中
    LOST = 2         # 丢失（准备删除）


class TrackedObject:
    """
    单个被追踪的非语义动态物体。

    封装 KalmanFilter3D 并附加追踪元数据:
      - 确认/丢失状态机
      - 历史轨迹（世界坐标）
      - 上次已知的 bbox 尺寸（用于预测区域投影）
    """

    __slots__ = ('id', 'kf', 'state', 'frames_since_seen',
                 'confirm_count', 'total_frames',
                 'position_history', 'last_bbox', 'last_mask_area',
                 'class_label', 'min_confirm', 'max_lost')

    def __init__(self, obj_id, initial_pos_3d, current_time,
                 bbox=None, mask_area=0, class_label='unknown',
                 min_confirm=3, max_lost=10):
        self.id = obj_id
        self.kf = KalmanFilter3D(
            obj_id, initial_pos_3d, current_time,
            dt=0.05, base_process_noise=0.5, base_measurement_noise=0.1)
        self.state = ObjectState.TENTATIVE
        self.frames_since_seen = 0
        self.confirm_count = 1
        self.total_frames = 1
        self.position_history = [tuple(initial_pos_3d)]
        self.last_bbox = bbox  # (x, y, w, h) 图像坐标
        self.last_mask_area = mask_area
        self.class_label = class_label
        self.min_confirm = min_confirm
        self.max_lost = max_lost

    def predict(self, current_time=None):
        """Kalman 预测，返回预测位置。"""
        return self.kf.predict(current_time=current_time)

    def update(self, measurement_3d, current_time=None,
               bbox=None, mask_area=0, class_label=None):
        """Kalman 更新 + 状态机推进。"""
        smoothed = self.kf.update(measurement_3d, current_time=current_time)

        self.frames_since_seen = 0
        self.confirm_count += 1
        self.total_frames += 1
        if self.confirm_count >= self.min_confirm:
            self.state = ObjectState.ACTIVE

        if bbox is not None:
            self.last_bbox = bbox
        self.last_mask_area = mask_area
        if class_label is not None:
            self.class_label = class_label

        self.position_history.append(tuple(smoothed.tolist()))
        if len(self.position_history) > 100:
            self.position_history.pop(0)

        return smoothed

    def mark_unseen(self):
        """标记本帧未观测到。"""
        self.frames_since_seen += 1
        if self.frames_since_seen >= self.max_lost:
            self.state = ObjectState.LOST

    @property
    def is_active(self):
        return self.state in (ObjectState.TENTATIVE, ObjectState.ACTIVE)

    @property
    def is_lost(self):
        return self.state == ObjectState.LOST

    @property
    def position(self):
        return self.kf.position

    @property
    def velocity(self):
        return self.kf.velocity

    @property
    def speed(self):
        v = self.velocity
        return float(np.linalg.norm(v))


class ObjectTracker:
    """
    P7: 统一动态物体追踪器（方案A）。

    同时处理语义检测（YOLO person）和几何检测（P5 box），
    统一经匈牙利数据关联 + Kalman 平滑 → 统一 RViz2 可视化。

    - 不依赖 SLAM 后端
    - 纯 3D 质心距离匈牙利关联（非密集场景下区分度 >100×）
    - 统一速度箭头 + 轨迹渐变 + 类别标签
    """

    def __init__(self, match_max_dist=0.5, min_confirm=3, max_lost=10):
        """
        :param match_max_dist: 匈牙利匹配最大 3D 距离（米）
        :param min_confirm:   TENTATIVE → ACTIVE 所需连续检测帧数
        :param max_lost:      ACTIVE → LOST 所需连续丢失帧数
        """
        self.tracks = OrderedDict()  # track_id → TrackedObject
        self.next_id = 1
        self.match_max_dist = match_max_dist
        self.min_confirm = min_confirm
        self.max_lost = max_lost

    # ================================================================
    #  主入口
    # ================================================================

    def update(self, motion_mask, depth_image, camera_intrinsics,
               header, current_time, T_world_cam=None,
               external_detections=None):
        """
        处理一帧。

        :param motion_mask:         (H,W) uint8，运动检测掩码（非语义动态区域）
        :param depth_image:         (H,W) float32，深度图
        :param camera_intrinsics:   dict {'fx','fy','cx','cy'}
        :param header:              ROS2 Header
        :param current_time:        rclpy Time 对象
        :param T_world_cam:         (4,4) 齐次矩阵，相机→世界变换，None 则用单位阵
        :param external_detections: list of dicts，外部检测（如 YOLO），格式:
                {'centroid_world': np.array(3,), 'centroid_2d': (u,v),
                 'bbox': (x,y,w,h), 'area': int, 'depth_median': float,
                 'mask': np.ndarray (H,W) bool, 'class_label': str}
        :return: (feedback_mask, tracked_objects_list, markers_list)
        """
        h, w = motion_mask.shape
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        if T_world_cam is None:
            T_world_cam = np.eye(4, dtype=np.float64)

        feedback_mask = np.zeros((h, w), dtype=np.uint8)

        # ---- ① 实例提取 ----
        # 仅追踪 YOLO 语义检测（行人）。P5 几何检测的假阳性残留会导致
        # 大量虚空 TrackedObject, 几何检测仅负责产生掩码（涂黑），不追踪。
        detections = []
        if external_detections:
            for ext_det in external_detections:
                ext_det['_source'] = 'semantic'
                detections.append(ext_det)

        # ---- ② 预测所有现有追踪到当前帧 ----
        for track in self.tracks.values():
            track.predict(current_time=current_time)
            track.mark_unseen()

        # ---- ③ 匈牙利数据关联 ----
        matches, unmatched_det, _unmatched_trk = self._associate(
            detections, list(self.tracks.values()))

        # ---- ④ Kalman 更新匹配对 ----
        for det_idx, trk_id in matches:
            det = detections[det_idx]
            track = self.tracks[trk_id]
            track.update(
                det['centroid_world'], current_time=current_time,
                bbox=det.get('bbox'), mask_area=det.get('area', 0),
                class_label=det.get('class_label'))

        # ---- ⑤ 处理未匹配检测（新生候选）----
        for det_idx in unmatched_det:
            det = detections[det_idx]
            new_id = self.next_id
            self.next_id += 1
            self.tracks[new_id] = TrackedObject(
                new_id, det['centroid_world'], current_time,
                bbox=det.get('bbox'), mask_area=det.get('area', 0),
                class_label=det.get('class_label', 'unknown'),
                min_confirm=self.min_confirm, max_lost=self.max_lost)

        # ---- ⑥ 清理丢失追踪 ----
        lost_ids = [tid for tid, t in self.tracks.items() if t.is_lost]
        for tid in lost_ids:
            del self.tracks[tid]

        # ---- 构建输出 ----
        tracked_objects = []
        markers = []

        for track in self.tracks.values():
            if not track.is_active:
                continue

            pos = track.position
            vel = track.velocity
            spd = track.speed
            state_str = 'MOVING' if spd > 0.05 else 'STATIONARY'
            label = getattr(track, 'class_label', 'unknown') or 'unknown'

            tracked_objects.append({
                'id': track.id,
                'position': pos,
                'velocity': vel,
                'speed': spd,
                'state': state_str,
                'class_label': label,
                'trajectory': list(track.position_history),
                'confirm_count': track.confirm_count,
            })

            # RViz2 Marker: 位置框 (CUBE) — 颜色区分类别
            markers.append(self._make_cube_marker(
                header, track.id, pos, spd, label))

            # RViz2 Marker: ID 标签
            markers.append(self._make_text_marker(
                header, track.id, pos, spd, state_str, label))

            # RViz2 Marker: 速度箭头
            if spd > 0.02:
                markers.append(self._make_arrow_marker(
                    header, track.id, pos, vel, spd))

            # RViz2 Marker: 轨迹线
            if len(track.position_history) >= 2:
                markers.append(self._make_trajectory_marker(
                    header, track.id, track.position_history))

        return feedback_mask, tracked_objects, markers

    # ================================================================
    #  ① 实例提取
    # ================================================================

    def _extract_instances(self, motion_mask, depth_image,
                           fx, fy, cx, cy, T_world_cam):
        """
        从运动掩码中提取物体实例。

        连通域分析 → 3D 反投影 → 世界坐标质心。
        """
        if np.count_nonzero(motion_mask) < 10:
            return []

        # 连通域分析
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            motion_mask, connectivity=4)

        detections = []
        for lbl in range(1, n_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 20:  # 过滤极小碎片
                continue

            # 有效性掩码：过滤无效深度后再提取坐标（确保 depths 与 rs/cs 始终对齐）
            mask = (labels == lbl) & (depth_image > 0.1) & np.isfinite(depth_image)
            depths = depth_image[mask]
            if len(depths) < 10:
                continue

            # 像素坐标（与 depths 同步提取，保证索引一致）
            rs, cs_array = np.where(mask)

            # 深度中值 + MAD 过滤（inlier 基于未截断的 depths，长度 = N）
            d_median = np.median(depths)
            d_mad = np.median(np.abs(depths - d_median))
            inlier = np.abs(depths - d_median) < 3.0 * max(1.4826 * d_mad, 0.05)

            depths_in = depths[inlier]
            if len(depths_in) < 10:
                continue

            rs_in = rs[inlier]
            cs_in = cs_array[inlier]

            # 反投影到相机 3D
            X_cam = (cs_in - cx) * depths_in / fx
            Y_cam = (rs_in - cy) * depths_in / fy
            Z_cam = depths_in

            # 相机 → 世界
            ones = np.ones_like(X_cam)
            P_cam = np.stack([X_cam, Y_cam, Z_cam, ones], axis=0)  # (4, N)
            P_world = T_world_cam @ P_cam  # (4, N)
            centroid_world = np.mean(P_world[:3, :], axis=1)  # (3,)

            x, y, w_box, h_box = stats[lbl, cv2.CC_STAT_LEFT], stats[lbl, cv2.CC_STAT_TOP], \
                                  stats[lbl, cv2.CC_STAT_WIDTH], stats[lbl, cv2.CC_STAT_HEIGHT]

            detections.append({
                'id': None,  # 待关联
                'centroid_world': centroid_world,
                'centroid_2d': (centroids[lbl][0], centroids[lbl][1]),
                'bbox': (x, y, w_box, h_box),
                'area': area,
                'depth_median': float(d_median),
                'mask': mask,
            })

        return detections

    # ================================================================
    #  ② 匈牙利数据关联
    # ================================================================

    def _associate(self, detections, tracks):
        """
        M 个检测 × N 个追踪 → 匈牙利最优匹配。

        :param detections: list of detection dicts
        :param tracks:     list of TrackedObject
        :return: (matches, unmatched_det_indices, unmatched_track_indices)
        """
        M = len(detections)
        N = len(tracks)

        if M == 0:
            return [], [], list(range(N))
        if N == 0:
            return [], list(range(M)), []

        # 构建 M×N 距离矩阵
        cost = np.full((M, N), np.inf, dtype=np.float64)
        for i, det in enumerate(detections):
            c_det = det['centroid_world']
            if c_det is None or not np.all(np.isfinite(c_det)):
                continue  # 跳过质心无效的检测
            for j, track in enumerate(tracks):
                c_trk = track.position
                if not np.all(np.isfinite(c_trk)):
                    continue  # 跳过位置无效的追踪
                d = float(np.linalg.norm(c_det - c_trk))
                if np.isfinite(d):
                    cost[i, j] = d

        # 防御: 确保 cost 矩阵无 NaN/Inf 后再送匈牙利算法
        if not np.all(np.isfinite(cost)):
            cost[~np.isfinite(cost)] = 1e6  # 无效配对用大距离替代

        # 匈牙利匹配
        row_ind = np.array([], dtype=int)
        col_ind = np.array([], dtype=int)
        if _HAS_SCIPY and M > 0 and N > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
        elif M > 0 and N > 0:
            row_ind, col_ind = self._greedy_assign(cost)

        # 过滤距离过大的配对
        matches = []
        matched_det = set()
        matched_trk = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < self.match_max_dist:
                matches.append((int(i), tracks[j].id))
                matched_det.add(int(i))
                matched_trk.add(j)

        unmatched_det = [i for i in range(M) if i not in matched_det]
        unmatched_trk = [j for j in range(N) if j not in matched_trk]

        return matches, unmatched_det, unmatched_trk

    @staticmethod
    def _greedy_assign(cost_matrix):
        """贪心分配（scipy 不可用时的回退）。"""
        cost = cost_matrix.copy()
        M, N = cost.shape
        row_ind, col_ind = [], []

        for _ in range(min(M, N)):
            if np.all(np.isinf(cost)):
                break
            min_idx = np.unravel_index(np.argmin(cost), cost.shape)
            if np.isinf(cost[min_idx]):
                break
            row_ind.append(int(min_idx[0]))
            col_ind.append(int(min_idx[1]))
            cost[min_idx[0], :] = np.inf
            cost[:, min_idx[1]] = np.inf

        return np.array(row_ind, dtype=int), np.array(col_ind, dtype=int)

    # ================================================================
    #  ⑧ RViz2 Marker 构建
    # ================================================================

    @staticmethod
    def _make_cube_marker(header, track_id, position, speed, class_label='unknown'):
        """物体位置框（类别区分颜色: 行人=蓝, 箱子=橙, 其他=绿, 速度→红）。"""
        from visualization_msgs.msg import Marker
        m = Marker()
        m.header = header
        m.ns = "dynamic_objects"
        m.id = int(track_id)
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(position[0])
        m.pose.position.y = float(position[1])
        m.pose.position.z = float(position[2])
        m.pose.orientation.w = 1.0

        # 尺寸区分类别
        if class_label == 'person':
            m.scale.x, m.scale.y, m.scale.z = 0.6, 0.6, 1.7  # 人形比例
        else:
            m.scale.x = m.scale.y = m.scale.z = 0.6           # 方形（箱子等）

        # 基础颜色按类别: 行人=蓝, 箱子=橙, 其他=绿
        if class_label == 'person':
            base_r, base_g, base_b = 0.2, 0.4, 1.0   # 蓝色调
        elif class_label in ('box', 'crate', 'pallet'):
            base_r, base_g, base_b = 1.0, 0.5, 0.0   # 橙色调
        else:
            base_r, base_g, base_b = 0.0, 0.8, 0.2   # 绿色调（未知）

        # 速度叠加：越快越红（防护 NaN: isnan 时 ratio=0）
        ratio = min(speed / 0.5, 1.0) if np.isfinite(speed) else 0.0
        m.color.r = float(base_r + (1.0 - base_r) * ratio)
        m.color.g = float(base_g * (1.0 - ratio))
        m.color.b = float(base_b * (1.0 - ratio))
        m.color.a = 0.7
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200000000
        return m

    @staticmethod
    def _make_text_marker(header, track_id, position, speed, state_str, class_label='unknown'):
        """物体标签（含类别名）。"""
        from visualization_msgs.msg import Marker
        m = Marker()
        m.header = header
        m.ns = "dynamic_object_labels"
        m.id = int(track_id) + 10000
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(position[0])
        m.pose.position.y = float(position[1])
        m.pose.position.z = float(position[2]) + 0.9
        m.scale.z = 0.3
        label_display = class_label.capitalize() if class_label != 'unknown' else 'Obj'
        m.text = f"{label_display}#{track_id} | {speed:.2f} m/s | {state_str}"
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200000000
        return m

    @staticmethod
    def _make_arrow_marker(header, track_id, position, velocity, speed):
        """速度方向箭头（长度 ∝ 速率）。"""
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point
        m = Marker()
        m.header = header
        m.ns = "dynamic_object_arrows"
        m.id = int(track_id) + 20000
        m.type = Marker.ARROW
        m.action = Marker.ADD

        # 起点 = 物体位置
        start = Point()
        start.x = float(position[0])
        start.y = float(position[1])
        start.z = float(position[2])

        # 终点 = 位置 + 速度方向（缩放显示）
        vel_norm = float(np.linalg.norm(velocity))
        scale = 0.5 / max(vel_norm, 0.01)  # 箭头固定 0.5m 长
        end = Point()
        end.x = float(position[0] + velocity[0] * scale)
        end.y = float(position[1] + velocity[1] * scale)
        end.z = float(position[2] + velocity[2] * scale)

        m.points = [start, end]
        m.scale.x = 0.05  # 轴直径
        m.scale.y = 0.1   # 箭头头部直径
        m.scale.z = 0.0
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.8
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200000000
        return m

    @staticmethod
    def _make_trajectory_marker(header, track_id, history):
        """历史轨迹线（冷色→暖色渐变）。"""
        from visualization_msgs.msg import Marker
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        m = Marker()
        m.header = header
        m.ns = "dynamic_object_trails"
        m.id = int(track_id) + 30000
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04

        n = len(history)
        for i, pos in enumerate(history):
            p = Point()
            p.x = float(pos[0])
            p.y = float(pos[1])
            p.z = float(pos[2]) - 0.3  # 略低于物体中心
            m.points.append(p)

            # 颜色渐变: 旧→新 = 蓝→红
            c = ColorRGBA()
            ratio = i / max(n - 1, 1)
            c.r = float(ratio)
            c.b = float(1.0 - ratio)
            c.g = 0.2
            c.a = 0.7
            m.colors.append(c)

        m.lifetime.sec = 0
        m.lifetime.nanosec = 200000000
        return m
