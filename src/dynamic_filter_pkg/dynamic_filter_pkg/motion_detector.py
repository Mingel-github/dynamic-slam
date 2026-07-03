import cv2
import numpy as np

class MotionDetector:
    def __init__(self, max_corners=300, min_displacement=2.0, depth_tolerance=0.15,
                 reprojection_threshold=0.1, min_valid_depth=0.1):
        """
        :param max_corners: LK光流追踪的最大角点数
        :param min_displacement: 判定为动态外点的最小像素位移（过滤背景微小抖动）
        :param depth_tolerance: FloodFill 深度生长的容忍度（米）
        :param reprojection_threshold: 多视图深度重投影的动态判定阈值（米）
        :param min_valid_depth: 参与多视图运算的最小有效深度（米）
        """
        self.max_corners = max_corners
        self.min_displacement = min_displacement
        self.depth_tolerance = depth_tolerance
        self.reprojection_threshold = reprojection_threshold
        self.min_valid_depth = min_valid_depth
        self.prev_gray = None
        self.prev_pts = None

        # 多视图几何帧缓冲
        self.prev_rgb = None
        self.prev_depth = None
        self.has_prev_frame = False

    def detect(self, current_bgr, current_depth, semantic_mask):
        current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        h, w = current_gray.shape
        motion_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. 屏蔽区生成：将已知语义目标（人）所在的区域设为 0，避免在行人身上提取运动特征点
        if semantic_mask is not None and semantic_mask.any():
            valid_bg_mask = np.where(semantic_mask, 0, 255).astype(np.uint8)
        else:
            valid_bg_mask = np.ones((h, w), dtype=np.uint8) * 255

        # 初始化或点数过少时，重新提取角点
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 20:
            self.prev_pts = cv2.goodFeaturesToTrack(
                current_gray, mask=valid_bg_mask, maxCorners=self.max_corners, 
                qualityLevel=0.01, minDistance=10
            )
            self.prev_gray = current_gray
            return motion_mask

        # 2. 稀疏 LK 光流追踪
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_pts, None)

        if curr_pts is not None and status is not None:
            # 过滤出追踪成功的点
            good_new = curr_pts[status.ravel() == 1]
            good_old = self.prev_pts[status.ravel() == 1]

            if len(good_new) >= 8: # RANSAC 计算基础矩阵至少需要 8 个点
                # 3. RANSAC 极线几何校验，拟合全局背景运动
                F, ransac_mask = cv2.findFundamentalMat(good_old, good_new, cv2.FM_RANSAC, 3.0, 0.99)
                
                if F is not None and ransac_mask is not None:
                    # RANSAC 返回为 0 的点即为外点（未遵循背景运动学规律的叛徒）
                    outliers = good_new[ransac_mask.ravel() == 0].reshape(-1, 2)
                    old_outliers = good_old[ransac_mask.ravel() == 0].reshape(-1, 2)
                    
                    # OpenCV floodFill 要求 mask 的长宽必须比原图大 2
                    floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                    
                    # 清洗深度图中的 NaN 噪点以防 floodFill C++ 底层崩溃
                    clean_depth = np.nan_to_num(current_depth, nan=0.0).astype(np.float32)

                    # 4. 基于种子的 32FC1 深度图区域生长
                    for i, pt in enumerate(outliers):
                        x, y = int(pt[0]), int(pt[1])

                        # 边界安全检查
                        if x < 0 or x >= w or y < 0 or y >= h:
                            continue

                        # 位移标量检查：剔除由于 RANSAC 矩阵解算误差导致的微小静止外点
                        dx = float(pt[0] - old_outliers[i][0])
                        dy = float(pt[1] - old_outliers[i][1])
                        disp = (dx*dx + dy*dy) ** 0.5
                        if disp < self.min_displacement:
                            continue

                        # 剔除无效深度种子点
                        z_val = clean_depth[y, x]
                        if z_val <= 0.1:
                            continue

                        # ★ 跳过已被之前种子生长过的区域，避免重复 floodFill
                        if floodfill_mask[y+1, x+1] > 0:
                            continue

                        # 执行区域生长。flags: 4邻域连通 | 掩码填充值为255 | 仅将结果输出至mask不改变原图
                        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
                        cv2.floodFill(
                            clean_depth,
                            floodfill_mask,
                            (x, y),
                            newVal=0,
                            loDiff=self.depth_tolerance,
                            upDiff=self.depth_tolerance,
                            flags=flags
                        )
                    
                    # 裁剪出与原图同尺寸的有效 Mask
                    motion_mask = floodfill_mask[1:-1, 1:-1]

        # 每帧强制刷新角点，避免特征点随时间漂移失效
        self.prev_gray = current_gray
        self.prev_pts = cv2.goodFeaturesToTrack(
            current_gray, mask=valid_bg_mask, maxCorners=self.max_corners,
            qualityLevel=0.01, minDistance=10
        )

        return motion_mask

    # ========================================================================
    #  P1: 多视图几何一致性检测（深度重投影法）
    # ========================================================================

    def detect_multiview(self, current_bgr, current_depth, semantic_mask,
                         camera_intrinsics, T_prev_to_curr):
        """
        多视图深度重投影运动检测（P1 主路径）。

        原理: 将前一帧背景像素投影到3D → 用帧间位姿变换到当前帧 → 重投影回2D
              → 比较投影深度与实测深度 → 差异大者为动态物体。

        :param current_bgr:      当前帧 BGR 图像 (H,W,3)
        :param current_depth:    当前帧深度图 (H,W) float32
        :param semantic_mask:    已知动态区域布尔掩码 (H,W)，这些像素不参与检测
        :param camera_intrinsics: dict {'fx','fy','cx','cy'}
        :param T_prev_to_curr:   (4,4) 齐次变换阵，前一帧相机坐标系→当前帧相机坐标系
        :return: (motion_mask_uint8, used_multiview_bool)
        """
        h, w = current_depth.shape
        motion_mask = np.zeros((h, w), dtype=np.uint8)

        # --- 首帧：仅缓冲，返回空掩码 ---
        if not self.has_prev_frame:
            self._store_frame(current_bgr, current_depth)
            return motion_mask, False

        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        # --- 背景像素掩码：排除已知语义动态区域 ---
        if semantic_mask is not None and semantic_mask.any():
            bg_mask = ~semantic_mask
        else:
            bg_mask = np.ones((h, w), dtype=bool)

        z_prev = self.prev_depth

        # --- 有效像素：背景 + 深度有效 + 有限值 ---
        valid = (bg_mask &
                 (z_prev > self.min_valid_depth) &
                 np.isfinite(z_prev))

        if np.count_nonzero(valid) == 0:
            self._store_frame(current_bgr, current_depth)
            return motion_mask, False  # 无有效背景像素 → 降级到 LK

        # --- 逐像素 3D 反投影（向量化） ---
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))  # (H,W)

        X_prev = np.zeros_like(z_prev)
        Y_prev = np.zeros_like(z_prev)

        X_prev[valid] = (u_grid[valid] - cx) * z_prev[valid] / fx
        Y_prev[valid] = (v_grid[valid] - cy) * z_prev[valid] / fy

        # --- 组装有效点的 (N,3) 点云 ---
        P_prev = np.stack([
            X_prev[valid], Y_prev[valid], z_prev[valid]
        ], axis=-1)  # (N,3)

        # 一次性形状诊断日志：仅首次成功时打印，用于远程排障
        if not hasattr(self, '_shapes_logged'):
            self._shapes_logged = True
            import sys
            print(
                f'[detect_multiview] First call OK: '
                f'z_prev.shape={z_prev.shape}, '
                f'valid_count={np.count_nonzero(valid)}, '
                f'P_prev.shape={P_prev.shape}, '
                f'T_prev_to_curr.shape={T_prev_to_curr.shape}',
                file=sys.stderr)

        # --- 帧间刚体变换 ---
        R = T_prev_to_curr[:3, :3]
        t = T_prev_to_curr[:3, 3]

        # 防御性形状检查：避免因上游数据异常导致 matmul 崩溃
        if P_prev.ndim != 2 or P_prev.shape[1] != 3:
            import sys
            print(
                f'[detect_multiview] SHAPE ERROR: '
                f'P_prev.shape={P_prev.shape}, P_prev.ndim={P_prev.ndim}, '
                f'R.shape={R.shape}, T_prev_to_curr.shape={T_prev_to_curr.shape}, '
                f'z_prev.shape={z_prev.shape}, current_depth.shape={current_depth.shape}, '
                f'bg_mask.shape={bg_mask.shape}, valid.shape={valid.shape}, '
                f'valid_count={np.count_nonzero(valid)}, '
                f'X_prev.shape={X_prev.shape}, u_grid.shape={u_grid.shape}',
                file=sys.stderr)
            self._store_frame(current_bgr, current_depth)
            return motion_mask, False

        P_curr = (R @ P_prev.T + t.reshape(3, 1)).T  # (N,3)

        # --- 重投影到当前帧像素坐标 ---
        u_proj = (fx * P_curr[:, 0] / P_curr[:, 2]) + cx
        v_proj = (fy * P_curr[:, 1] / P_curr[:, 2]) + cy
        z_proj = P_curr[:, 2]

        # --- 过滤投影越界或深度无效的点 ---
        reproj_valid = (
            (u_proj >= 0) & (u_proj < w) &
            (v_proj >= 0) & (v_proj < h) &
            (z_proj > self.min_valid_depth)
        )

        if np.count_nonzero(reproj_valid) == 0:
            self._store_frame(current_bgr, current_depth)
            return motion_mask, False  # 无有效重投影 → 降级到 LK

        # --- 深度差比较 → 动态种子 ---
        u_int = np.clip(np.round(u_proj[reproj_valid]).astype(int), 0, w - 1)
        v_int = np.clip(np.round(v_proj[reproj_valid]).astype(int), 0, h - 1)
        z_measured = current_depth[v_int, u_int]

        depth_diff = np.abs(z_proj[reproj_valid] - z_measured)
        dynamic_flags = depth_diff > self.reprojection_threshold

        dynamic_u = u_int[dynamic_flags]
        dynamic_v = v_int[dynamic_flags]

        # --- 深度一致性区域生长 ---
        n_seeds = len(dynamic_u)
        if n_seeds > 0:
            floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            clean_depth = np.nan_to_num(current_depth, nan=0.0).astype(np.float32)

            # 种子数量安全上限：过多时截断取前 N 个（避免 floodFill 耗时爆炸）
            max_seeds = 500
            if n_seeds > max_seeds:
                dynamic_u = dynamic_u[:max_seeds]
                dynamic_v = dynamic_v[:max_seeds]

            for i in range(len(dynamic_u)):
                x_seed = int(dynamic_u[i])
                y_seed = int(dynamic_v[i])

                if not (0 <= x_seed < w and 0 <= y_seed < h):
                    continue
                if clean_depth[y_seed, x_seed] <= self.min_valid_depth:
                    continue
                if floodfill_mask[y_seed + 1, x_seed + 1] > 0:
                    continue  # 已被之前的生长覆盖

                flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
                cv2.floodFill(
                    clean_depth, floodfill_mask, (x_seed, y_seed),
                    newVal=0,
                    loDiff=self.depth_tolerance,
                    upDiff=self.depth_tolerance,
                    flags=flags
                )

            motion_mask = floodfill_mask[1:-1, 1:-1]

        self._store_frame(current_bgr, current_depth)
        return motion_mask, True

    def _store_frame(self, bgr, depth):
        """缓存当前帧为下一帧的多视图参考帧，同时预热 LK 灰度图。"""
        self.prev_rgb = bgr.copy()
        self.prev_depth = depth.copy()
        self.prev_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.has_prev_frame = True