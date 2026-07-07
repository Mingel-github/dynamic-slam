import cv2
import numpy as np

class MotionDetector:
    def __init__(self, max_corners=300, min_displacement=2.0, depth_tolerance=0.15,
                 reprojection_threshold=0.15, min_valid_depth=0.1,
                 cluster_depth_threshold=0.05, cluster_min_size=100,
                 cluster_match_max_dist=0.3, cluster_motion_sigma=3.0):
        """
        :param max_corners: LK光流追踪的最大角点数
        :param min_displacement: 判定为动态外点的最小像素位移（过滤背景微小抖动）
        :param depth_tolerance: FloodFill 深度生长的容忍度（米）
        :param reprojection_threshold: 多视图深度重投影的基础判定阈值（米）。
               实际采用深度自适应: max(0.15, min(0.5, base + 0.02*Z))，参考 DynaSLAM τ_z=0.4m。
        :param min_valid_depth: 参与多视图运算的最小有效深度（米）
        :param cluster_depth_threshold: P5 深度聚类相邻像素最大深度差（米），默认5cm
        :param cluster_min_size: P5 聚类最小像素数，默认100
        :param cluster_match_max_dist: P5 帧间簇匹配最大3D距离（米），默认0.3
        :param cluster_motion_sigma: P5 簇运动判定的sigma倍数，默认3.0
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

        # P5: 深度聚类检测状态
        self.cluster_depth_threshold = cluster_depth_threshold
        self.cluster_min_size = cluster_min_size
        self.cluster_match_max_dist = cluster_match_max_dist
        self.cluster_motion_sigma = cluster_motion_sigma
        self.prev_clusters_3d = None  # 上一帧的聚类列表

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

    # ========================================================================
    #  P5: 深度聚类检测（√N 降噪，替代逐像素深度重投影）
    # ========================================================================

    def detect_clustering(self, current_depth, camera_intrinsics, semantic_mask=None,
                          T_world_cam=None):
        """
        P5: 深度聚类运动检测。

        原理: 将判断单元从"1个像素"提升为"1个物体簇"，利用 √N 降噪。
              逐像素 SNR ≈ 0.12（不可检测），逐聚类 SNR ≈ 21（高度可靠）。

        步骤:
          1. 深度图区域生长聚类（相邻像素深度差 < 5cm → 同一簇）
          2. 3D 质心计算（反投影到相机坐标 → 转到世界坐标）
          3. 帧间簇关联（匈牙利算法，世界坐标 3D 质心最近邻）
          4. 簇运动判断（|Δd_world| > 3σ_cluster → 动态）
          5. 动态簇 → 掩码输出

        参考: DetectFusion (2019) — 松耦合预处理层使用世界坐标 ICP 聚类
              DynaSLAM (2018) — 多视图重投影前先消除相机自运动

        :param current_depth:    当前帧深度图 (H,W) float32
        :param camera_intrinsics: dict {'fx','fy','cx','cy'}
        :param semantic_mask:    已知动态区域布尔掩码 (H,W)，这些像素不参与检测
        :param T_world_cam:      (4,4) 相机→世界齐次变换矩阵，None则用单位阵（首帧）
        :return: cluster_mask (H,W) uint8
        """
        h, w = current_depth.shape
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        if T_world_cam is None:
            T_world_cam = np.eye(4, dtype=np.float64)

        cluster_mask = np.zeros((h, w), dtype=np.uint8)

        # --- 构建有效深度掩码 ---
        valid_depth = (current_depth > self.min_valid_depth) & np.isfinite(current_depth)

        # 排除语义已知动态区域（膨胀保护带）
        if semantic_mask is not None and semantic_mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            semantic_dilated = cv2.dilate(
                semantic_mask.astype(np.uint8), kernel).astype(bool)
            valid_depth = valid_depth & ~semantic_dilated

        if np.count_nonzero(valid_depth) < self.cluster_min_size:
            self.prev_clusters_3d = None
            return cluster_mask

        # --- Step 1: 深度聚类提取 ---
        clusters, _ = self._extract_depth_clusters(current_depth, valid_depth)

        if len(clusters) == 0:
            self.prev_clusters_3d = None
            return cluster_mask

        # --- Step 2: 计算每个簇的 3D 质心（相机坐标 → 世界坐标）---
        # 参考: DetectFusion (2019) 松耦合预处理层使用世界坐标聚类
        #       DynaSLAM (2018) 多视图重投影前先消除相机自运动
        curr_clusters_3d = []
        for cluster in clusters:
            info_3d = self._compute_cluster_3d(
                cluster['mask'], current_depth, fx, fy, cx, cy)
            if info_3d is not None:
                # 相机坐标 → 世界坐标
                centroid_cam = info_3d['centroid_3d']
                centroid_homo = np.array([centroid_cam[0], centroid_cam[1],
                                          centroid_cam[2], 1.0], dtype=np.float64)
                centroid_world = (T_world_cam @ centroid_homo)[:3]
                info_3d['centroid_world'] = centroid_world
                info_3d['mask'] = cluster['mask']
                info_3d['bbox'] = cluster['bbox']
                curr_clusters_3d.append(info_3d)

        if len(curr_clusters_3d) == 0:
            self.prev_clusters_3d = None
            return cluster_mask

        # --- Step 3-4: 帧间匹配 + 运动判定 ---
        if self.prev_clusters_3d is not None and len(self.prev_clusters_3d) > 0:
            dynamic_indices = self._match_and_detect(
                curr_clusters_3d, self.prev_clusters_3d)

            for idx in dynamic_indices:
                if idx < len(curr_clusters_3d):
                    cluster_mask[curr_clusters_3d[idx]['mask']] = 255

        # 持久化当前帧聚类供下一帧匹配（不含mask，节省内存）
        self.prev_clusters_3d = [
            {k: v for k, v in c.items() if k != 'mask'}
            for c in curr_clusters_3d
        ]

        return cluster_mask

    def _extract_depth_clusters(self, depth, valid_mask):
        """
        使用深度梯度边界 + 连通域分析提取深度一致的区域。

        方法: Sobel梯度幅值 → 阈值化找到深度边界 → 反转得到内部区域
              → 膨胀回收边界像素 → 连通域标记 → 面积过滤。

        :return: (clusters_list, labels_image)
        """
        h, w = depth.shape
        depth_clean = np.where(valid_mask, depth, 0.0).astype(np.float32)

        # 计算深度梯度幅值
        grad_x = cv2.Sobel(depth_clean, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_clean, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # 深度边界: 梯度 > 聚类阈值（默认 5cm/px）
        edge_mask = grad_mag > self.cluster_depth_threshold

        # 内部区域: 有效深度 且 非边界
        interior = valid_mask & ~edge_mask

        # 膨胀回收边界像素（把被标记为边界的物体表面像素拉回簇内）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        interior_dilated = cv2.dilate(
            interior.astype(np.uint8), kernel, iterations=1)
        cluster_region = (interior_dilated > 0) & valid_mask

        # 连通域分析
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cluster_region.astype(np.uint8), connectivity=4)

        # 提取簇信息，过滤小碎片
        clusters = []
        for lbl in range(1, n_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= self.cluster_min_size:
                clusters.append({
                    'id': len(clusters),
                    'mask': labels == lbl,
                    'area': area,
                    'bbox': (stats[lbl, cv2.CC_STAT_LEFT],
                             stats[lbl, cv2.CC_STAT_TOP],
                             stats[lbl, cv2.CC_STAT_WIDTH],
                             stats[lbl, cv2.CC_STAT_HEIGHT]),
                    'centroid_2d': (centroids[lbl][0], centroids[lbl][1])
                })

        return clusters, labels

    def _compute_cluster_3d(self, cluster_mask, depth, fx, fy, cx, cy):
        """
        计算一个簇的 3D 质心和统计量。

        使用 MAD (Median Absolute Deviation) 过滤深度离群值，
        然后反投影到相机坐标系计算 3D 质心。

        :return: dict with centroid_3d, depth_mean, depth_std, pixel_count
                 或 None（如果有效像素不足）
        """
        depths = depth[cluster_mask]

        if len(depths) < 10:
            return None

        # MAD 离群值过滤（比 z-score 更鲁棒）
        d_median = np.median(depths)
        d_mad = np.median(np.abs(depths - d_median))
        d_std_est = 1.4826 * max(d_mad, 1e-4)  # MAD → σ 估计
        inlier = np.abs(depths - d_median) < 3.0 * d_std_est

        depths_in = depths[inlier]
        if len(depths_in) < 10:
            return None

        # 获取像素坐标
        rs, cs_array = np.where(cluster_mask)
        rs_in = rs[inlier]
        cs_in = cs_array[inlier]

        # 反投影到 3D 相机坐标
        X = (cs_in - cx) * depths_in / fx
        Y = (rs_in - cy) * depths_in / fy

        return {
            'centroid_3d': np.array([np.mean(X), np.mean(Y), np.mean(depths_in)]),
            'depth_mean': float(np.mean(depths_in)),
            'depth_std': float(np.std(depths_in)),
            'pixel_count': len(depths_in),
        }

    def _match_and_detect(self, curr_clusters, prev_clusters):
        """
        帧间簇匹配（匈牙利算法）+ 运动判定（世界坐标系）。

        世界坐标系下，相机自运动已消除，静态簇质心不变（|Δd| ≈ 0），
        仅真正在世界中运动的物体产生 |Δd| > 3σ_cluster。

        σ_cluster = depth_std / √N（聚类 √N 降噪）

        :return: set of indices into curr_clusters that are dynamic
        """
        M = len(curr_clusters)
        N = len(prev_clusters)

        if M == 0 or N == 0:
            return set()

        # 构建 M×N 距离矩阵（世界坐标 3D 质心距离）
        cost = np.full((M, N), np.inf, dtype=np.float64)
        for i in range(M):
            ci = curr_clusters[i].get('centroid_world')
            if ci is None or not np.all(np.isfinite(ci)):
                continue
            for j in range(N):
                pj = prev_clusters[j].get('centroid_world')
                if pj is None or not np.all(np.isfinite(pj)):
                    continue
                cost[i, j] = float(np.linalg.norm(ci - pj))

        # 匈牙利匹配
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except ImportError:
            row_ind, col_ind = self._greedy_assign(cost)

        # 逐对运动判定
        dynamic_indices = set()
        max_dist = self.cluster_match_max_dist

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= max_dist:
                continue

            # 世界坐标 3D 位移（静态物体≈0，仅传感器噪声）
            d3d = float(cost[i, j])

            # √N 降噪: σ_cluster = σ_depth / √N
            n_pixels = curr_clusters[i].get('pixel_count', 1)
            depth_std = curr_clusters[i].get('depth_std', 0.01)
            sigma_cluster = depth_std / np.sqrt(max(n_pixels, 1))
            threshold_3sigma = self.cluster_motion_sigma * sigma_cluster

            # 世界坐标下动态判定: 3D 位移超过 3σ_cluster
            if d3d > max(threshold_3sigma, 0.005):  # 至少 0.5cm
                dynamic_indices.add(i)

        return dynamic_indices

    @staticmethod
    def _greedy_assign(cost_matrix):
        """
        贪心分配算法（scipy 不可用时的回退方案）。

        每次取距离矩阵中最小值对应的配对，然后移除该行和列。
        适用于小矩阵（典型场景 M,N < 20）。
        """
        cost = cost_matrix.copy()
        M, N = cost.shape
        row_ind = []
        col_ind = []

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