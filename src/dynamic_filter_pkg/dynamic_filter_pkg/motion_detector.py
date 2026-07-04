import cv2
import numpy as np

class MotionDetector:
    def __init__(self, max_corners=300, min_displacement=2.0, depth_tolerance=0.15,
                 reprojection_threshold=0.15, min_valid_depth=0.1):
        """
        :param max_corners: LK光流追踪的最大角点数
        :param min_displacement: 判定为动态外点的最小像素位移（过滤背景微小抖动）
        :param depth_tolerance: FloodFill 深度生长的容忍度（米）
        :param reprojection_threshold: 多视图深度重投影的基础判定阈值（米）。
               实际采用深度自适应: max(0.15, min(0.5, base + 0.02*Z))，参考 DynaSLAM τ_z=0.4m。
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

        # P3-①: 时序一致性投票状态（EMA 滑动窗口）
        self.dynamic_votes = None  # (H,W) float32，每像素动态票数

    def detect(self, current_bgr, current_depth, semantic_mask):
        current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        h, w = current_gray.shape
        motion_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. 屏蔽区生成：将已知语义目标（人）所在的区域设为 0，避免在行人身上提取运动特征点
        # P3-②: 膨胀 semantic_mask 15px 保护带，覆盖遮挡/去遮挡边界
        if semantic_mask is not None and semantic_mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            semantic_dilated = cv2.dilate(
                semantic_mask.astype(np.uint8), kernel).astype(bool)
            valid_bg_mask = np.where(semantic_dilated, 0, 255).astype(np.uint8)
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

        # --- 首帧：仅标记，返回空掩码（不调用 _store_frame，避免污染 prev_gray）---
        if not self.has_prev_frame:
            self.prev_depth = current_depth.copy()
            self.has_prev_frame = True
            return motion_mask, False

        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        # --- 背景像素掩码：排除已知语义动态区域（P3-② 膨胀保护带）---
        if semantic_mask is not None and semantic_mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            semantic_dilated = cv2.dilate(
                semantic_mask.astype(np.uint8), kernel).astype(bool)
            bg_mask = ~semantic_dilated
        else:
            bg_mask = np.ones((h, w), dtype=bool)

        z_prev = self.prev_depth

        # --- 有效像素：背景 + 深度有效 + 有限值 ---
        valid = (bg_mask &
                 (z_prev > self.min_valid_depth) &
                 np.isfinite(z_prev))

        if np.count_nonzero(valid) == 0:
            self.prev_depth = current_depth.copy()
            self.has_prev_frame = True
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
            self.prev_depth = current_depth.copy()
            self.has_prev_frame = True
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
            self.prev_depth = current_depth.copy()
            self.has_prev_frame = True
            return motion_mask, False  # 无有效重投影 → 降级到 LK

        # --- 视差角过滤（B2）：α > 30° 的匹配忽略 ---
        # 参考 DynaSLAM Section III-C：大角度观测时深度变化不可靠
        P_prev_reproj = P_prev[reproj_valid]   # (K,3) 前帧相机→3D点
        P_curr_reproj = P_curr[reproj_valid]   # (K,3) 当前帧相机→同一3D点

        norm_prev = np.linalg.norm(P_prev_reproj, axis=1)
        norm_curr = np.linalg.norm(P_curr_reproj, axis=1)
        cos_alpha = np.sum(P_prev_reproj * P_curr_reproj, axis=1) / (norm_prev * norm_curr + 1e-10)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        parallax_ok = cos_alpha > 0.866  # cos(30°) ≈ 0.866

        if np.count_nonzero(parallax_ok) == 0:
            self.prev_depth = current_depth.copy()
            self.has_prev_frame = True
            return motion_mask, False  # 视差角全部过大 → 降级到 LK

        # --- 深度差比较（B1：深度自适应阈值）---
        u_int = np.clip(np.round(u_proj[reproj_valid][parallax_ok]).astype(int), 0, w - 1)
        v_int = np.clip(np.round(v_proj[reproj_valid][parallax_ok]).astype(int), 0, h - 1)
        z_measured = current_depth[v_int, u_int]
        z_proj_filtered = z_proj[reproj_valid][parallax_ok]

        # F1: 过滤 z_measured 无效的匹配（超出量程=0 / NaN / inf）
        measured_valid = (z_measured > self.min_valid_depth) & np.isfinite(z_measured)
        if np.count_nonzero(measured_valid) == 0:
            self.prev_depth = current_depth.copy()
            self.has_prev_frame = True
            return motion_mask, False

        z_measured = z_measured[measured_valid]
        z_proj_filtered = z_proj_filtered[measured_valid]
        u_int = u_int[measured_valid]
        v_int = v_int[measured_valid]

        # B1: 深度自适应阈值 = 基准0.15 + Z的2%，上界0.5m
        # 近处(~1m): 0.17m, 中距离(~5m): 0.25m, 远处(≥17.5m): 0.50m
        # 参考 DynaSLAM 经验最优值 τ_z = 0.4m
        adaptive_threshold = self.reprojection_threshold + 0.02 * z_proj_filtered
        adaptive_threshold = np.clip(adaptive_threshold, 0.15, 0.50)

        depth_diff = np.abs(z_proj_filtered - z_measured)
        dynamic_flags = depth_diff > adaptive_threshold

        # --- P3-①: 时序一致性投票（EMA 滑动窗口）---
        # 原理：单帧 depth_diff 对均匀深度+横向运动的物体存在盲区（箱子内部 ~80% 面积 diff≈0），
        #      但物体边缘在部分帧中能被检测到。EMA 投票让边缘证据"延续"到后续帧，
        #      模拟 DynaSLAM 多关键帧投票效果，不增加计算开销。
        if self.dynamic_votes is None:
            self.dynamic_votes = np.zeros((h, w), dtype=np.float32)

        # 当前帧动态种子 → 投票 +1
        current_votes = np.zeros((h, w), dtype=np.float32)
        if np.count_nonzero(dynamic_flags) > 0:
            current_votes[v_int[dynamic_flags], u_int[dynamic_flags]] = 1.0

        # EMA: 新票 = 0.8×旧票 + 0.2×新票（时间常数 ≈ 5 帧）
        self.dynamic_votes = 0.8 * self.dynamic_votes + 0.2 * current_votes

        # 票数 > 0.15 → 动态（等价于最近 ~5 帧内 ≥2 帧检测到）
        vote_mask = self.dynamic_votes > 0.15

        # --- F2: 深度差异图连通域分析（种子改为投票掩码）---
        if np.count_nonzero(vote_mask) > 0:
            # 1. 构建二值种子图（vote_mask 已是全分辨率布尔数组）
            seed_img = np.zeros((h, w), dtype=np.uint8)
            seed_img[vote_mask] = 255

            # 2. 形态学闭运算：填充动态物体内部的小孔洞
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            seed_closed = cv2.morphologyEx(seed_img, cv2.MORPH_CLOSE, kernel_close)

            # 3. 形态学开运算：去除孤立的小噪点
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            seed_clean = cv2.morphologyEx(seed_closed, cv2.MORPH_OPEN, kernel_open)

            # 4. 连通域分析 + 面积过滤（去除 < 50px 的残存噪点）
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                seed_clean, connectivity=4)
            min_area = 50
            for lbl in range(1, n_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                    motion_mask[labels == lbl] = 255

        # 只更新多视图需要的深度缓冲，不碰 prev_gray（避免污染 LK 降级路径）
        self.prev_depth = current_depth.copy()
        self.has_prev_frame = True
        return motion_mask, True

    def _store_frame(self, bgr, depth):
        """缓存当前帧为下一帧的多视图参考帧，同时预热 LK 灰度图。"""
        self.prev_rgb = bgr.copy()
        self.prev_depth = depth.copy()
        self.prev_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.has_prev_frame = True