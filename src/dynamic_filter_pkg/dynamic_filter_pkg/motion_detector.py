import cv2
import numpy as np

class MotionDetector:
    def __init__(self, max_corners=300, min_displacement=2.0, depth_tolerance=0.15):
        """
        :param max_corners: LK光流追踪的最大角点数
        :param min_displacement: 判定为动态外点的最小像素位移（过滤背景微小抖动）
        :param depth_tolerance: FloodFill 深度生长的容忍度（米）。0.15表示深度差在15cm内视为同一物体。
        """
        self.max_corners = max_corners
        self.min_displacement = min_displacement
        self.depth_tolerance = depth_tolerance
        self.prev_gray = None
        self.prev_pts = None

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