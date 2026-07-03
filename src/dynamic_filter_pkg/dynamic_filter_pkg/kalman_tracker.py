"""
P2: 6D 匀速 Kalman Filter 3D 行人追踪器。

状态向量: [x, y, z, vx, vy, vz]^T
观测向量: [x, y, z]^T（来自 YOLO+深度中值估计）

特性:
  - 自适应观测噪声 R（根据 bbox 面积和深度值调整）
  - 轨迹生命周期状态机（ACTIVE → OCCLUDED → LOST）
  - 加速度异常检测（>5 m/s² 标记为可疑）
"""

import numpy as np
from enum import Enum


class TrackState(Enum):
    ACTIVE = 0       # 正常追踪中
    OCCLUDED = 1     # 遮挡/短暂丢失（仅预测）
    LOST = 2         # 长期丢失，待清理


class KalmanFilter3D:
    """
    6D 匀速运动模型 Kalman Filter。

    使用离散化连续白噪声加速度模型（Discretized Continuous White Noise
    Acceleration Model），通过 base_process_noise 控制过程噪声强度。
    """

    def __init__(self, track_id, initial_pos, initial_time,
                 dt=0.05,
                 base_process_noise=0.5,
                 base_measurement_noise=0.1):
        """
        :param track_id:               追踪 ID（来自 BoT-SORT）
        :param initial_pos:            (x, y, z) 初始位置
        :param initial_time:           rclpy Time 对象，首帧时间戳
        :param dt:                     名义帧间隔（秒），实际 dt 从时间戳计算
        :param base_process_noise:     基础过程噪声强度 q
        :param base_measurement_noise: 基础观测噪声 r
        """
        self.track_id = track_id
        self.dt = dt
        self.base_process_noise = base_process_noise
        self.base_measurement_noise = base_measurement_noise

        # ---- 状态向量 [x, y, z, vx, vy, vz]^T ----
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.x[0, 0] = initial_pos[0]
        self.x[1, 0] = initial_pos[1]
        self.x[2, 0] = initial_pos[2]
        # vx, vy, vz 初始化为 0

        # ---- 协方差矩阵（初始不确定性大）----
        self.P = np.eye(6, dtype=np.float64) * 100.0

        # ---- 生命周期 ----
        self.state = TrackState.ACTIVE
        self.frames_since_seen = 0
        self.total_frames = 1
        self.last_update_time = initial_time

        # ---- 异常检测 ----
        self._prev_velocity = np.zeros(3)
        self.last_acceleration = 0.0  # m/s²，供外部读取

        # ---- 平滑轨迹历史 ----
        self.position_history = [tuple(initial_pos)]

    # ================================================================
    #  矩阵属性（每次调用按当前 dt 构建）
    # ================================================================

    @property
    def F(self):
        """状态转移矩阵（匀速模型）。"""
        dt = self.dt
        return np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=np.float64)

    @property
    def H(self):
        """观测矩阵（仅观测位置，不直接观测速度）。"""
        return np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float64)

    @property
    def Q(self):
        """
        过程噪声协方差（离散化连续白噪声加速度模型）。

        噪声来源：行人不会完美匀速运动。
        Q 的大小由 base_process_noise 控制。
        """
        dt = self.dt
        q = self.base_process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        return q * np.array([
            [dt4/4, 0,     0,     dt3/2, 0,     0    ],
            [0,     dt4/4, 0,     0,     dt3/2, 0    ],
            [0,     0,     dt4/4, 0,     0,     dt3/2],
            [dt3/2, 0,     0,     dt2,   0,     0    ],
            [0,     dt3/2, 0,     0,     dt2,   0    ],
            [0,     0,     dt3/2, 0,     0,     dt2   ]
        ], dtype=np.float64)

    def compute_R(self, bbox_area=None, depth=None):
        """
        自适应观测噪声协方差。

        远距离（小 bbox / 深深度）→ R 增大 → 更信任预测
        近距离（大 bbox / 浅深度）→ R 减小 → 更信任观测

        :param bbox_area: bbox 面积 (px²)，None 则使用默认值
        :param depth:     目标深度 (m)，None 则使用默认值
        """
        scale = 1.0

        if bbox_area is not None and bbox_area < 500:
            scale *= 3.0   # 目标很小（远处）→ 观测不可靠，乘 3
        elif bbox_area is not None and bbox_area > 5000:
            scale *= 0.5   # 目标很大（近处）→ 观测较可靠，减半

        if depth is not None and depth > 5.0:
            scale *= 2.0   # 远距离深度噪声显著增大

        r = self.base_measurement_noise * scale
        return np.eye(3, dtype=np.float64) * r

    # ================================================================
    #  核心 Kalman 步骤
    # ================================================================

    def predict(self, current_time=None):
        """
        预测步骤。每帧必调（无论是否有观测）。

        :param current_time: rclpy Time 对象，用于计算实际 dt
        :return: 预测位置 (x, y, z)
        """
        # 更新时间步长（防御 None 输入）
        if current_time is not None and self.last_update_time is not None:
            dt_ns = (current_time - self.last_update_time).nanoseconds
            self.dt = dt_ns * 1e-9
            self.dt = max(0.01, min(self.dt, 0.5))  # 夹紧到合理范围

        # 标准 Kalman 预测
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.frames_since_seen += 1

        # ---- 生命周期转换 ----
        if self.frames_since_seen >= 30:
            self.state = TrackState.LOST
        elif self.frames_since_seen >= 3:
            self.state = TrackState.OCCLUDED

        return self.x[0:3, 0].copy()

    def update(self, measurement, bbox_area=None, current_time=None):
        """
        观测更新步骤。有检测结果时调用。

        :param measurement: (x, y, z) 观测向量
        :param bbox_area:   bbox 面积（用于自适应 R），可选
        :param current_time: rclpy Time 对象
        :return: 更新后的平滑位置 (x, y, z)
        """
        z = np.array(measurement, dtype=np.float64).reshape(3, 1)

        # 更新时间步长（防御 None 输入）
        if current_time is not None and self.last_update_time is not None:
            dt_ns = (current_time - self.last_update_time).nanoseconds
            self.dt = dt_ns * 1e-9
            self.dt = max(0.01, min(self.dt, 0.5))

        R = self.compute_R(bbox_area=bbox_area, depth=measurement[2])

        # ---- Kalman 更新 ----
        y = z - self.H @ self.x                      # 创新 (3,1)
        S = self.H @ self.P @ self.H.T + R           # 创新协方差 (3,3)
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman 增益 (6,3)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        # ---- 加速度异常检测 ----
        v_new = self.x[3:6, 0]
        if hasattr(self, '_prev_velocity'):
            dv = v_new - self._prev_velocity
            self.last_acceleration = float(np.linalg.norm(dv) / max(self.dt, 0.01))
        self._prev_velocity = v_new.copy()

        # ---- 生命周期恢复 ----
        self.frames_since_seen = 0
        self.state = TrackState.ACTIVE
        self.total_frames += 1
        self.last_update_time = current_time if current_time is not None else self.last_update_time

        # ---- 记录平滑轨迹 ----
        pos = tuple(self.x[0:3, 0].tolist())
        self.position_history.append(pos)
        if len(self.position_history) > 50:
            self.position_history.pop(0)

        return self.x[0:3, 0].copy()

    # ================================================================
    #  便捷方法
    # ================================================================

    @property
    def position(self):
        """当前平滑位置 (x, y, z)。"""
        return self.x[0:3, 0].copy()

    @property
    def velocity(self):
        """当前估计速度 (vx, vy, vz)。"""
        return self.x[3:6, 0].copy()

    @property
    def is_lost(self):
        return self.state == TrackState.LOST

    @property
    def is_occluded(self):
        return self.state == TrackState.OCCLUDED
