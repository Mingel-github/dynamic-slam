import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from rclpy.qos import qos_profile_sensor_data
import time
import math
import numpy as np

class DynamicsMonitor(Node):
    def __init__(self):
        super().__init__('dynamics_monitor')
        
        # 订阅器 (QoS: Best Effort)
        self.create_subscription(Odometry, '/lio_sam/mapping/odometry', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu_raw', self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/lio_sam/feature/cloud_surface', self.feature_cb, qos_profile_sensor_data)

        # 状态变量
        self.start_time = time.time()
        self.last_print_time = 0
        self.feature_count = 0
        self.imu_acc_z = 0.0
        
        # 速度缓存 (用于检测抽搐)
        self.vel_history = []
        
        print("\n" + "="*60)
        print("🚀 LIO-SAM 动态性能监控终端 (v3.0)")
        print("   监测重点: [三维位置] [三维速度] [Z轴抽搐]")
        print("   请启动算法，观察 15 秒后的数据变化...")
        print("="*60)
        print(f"{'时间':<8} | {'位置 (X, Y, Z)':<22} | {'速度 (Vx, Vy, Vz)':<24} | {'状态'}")
        print("-" * 75)

    def imu_cb(self, msg):
        # 记录瞬时垂直加速度 (用于判断是否是物理引擎炸了)
        self.imu_acc_z = msg.linear_acceleration.z

    def feature_cb(self, msg):
        self.feature_count += 1

    def odom_cb(self, msg):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 1. 提取位置
        pos = msg.pose.pose.position
        
        # 2. 提取速度 (这是你要求的核心指标)
        # 注意：Odometry 的 twist 是相对于 Child Frame (base_link) 的
        vel = msg.twist.twist.linear
        
        # 3. 抽搐检测 (Jitter Detection)
        # 如果 Z 轴速度在短时间内剧烈波动 (比如上一帧 +0.5, 这一帧 -0.5)
        is_twitching = False
        if len(self.vel_history) > 5:
            vz_std = np.std([v[2] for v in self.vel_history[-5:]])
            if vz_std > 0.2: # 标准差大于 0.2m/s 说明很不稳
                is_twitching = True
        
        self.vel_history.append((vel.x, vel.y, vel.z))
        if len(self.vel_history) > 10:
            self.vel_history.pop(0)

        # 4. 打印逻辑 (5Hz 刷新，不刷屏，只更新一行)
        if current_time - self.last_print_time > 0.2: # 0.2s = 5Hz
            self.last_print_time = current_time
            
            # 状态判定
            status_str = "✅ 稳定"
            status_color = "\033[92m" # Green
            
            # 异常 A: 高度漂移
            if abs(pos.z) > 0.5:
                status_str = f"⚠️ 漂移 ({pos.z:.2f}m)"
                status_color = "\033[93m" # Yellow
                
            # 异常 B: 速度过快 (飞车前兆)
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            if speed > 2.0: 
                 status_str = f"🚀 起飞 ({speed:.1f}m/s)"
                 status_color = "\033[91m" # Red
            
            # 异常 C: 抽搐 (IMU 与 雷达打架)
            if is_twitching:
                status_str = "⚡ 剧烈抽搐!"
                status_color = "\033[95m" # Purple

            # 异常 D: 特征丢失
            if self.feature_count == 0 and elapsed > 5:
                 status_str = "💀 特征丢失"
                 status_color = "\033[91m"
            self.feature_count = 0 # 重置计数器

            # 格式化输出
            # 重点看 Vz (Z轴速度)
            pos_str = f"[{pos.x:>5.1f}, {pos.y:>5.1f}, {pos.z:>5.2f}]"
            vel_str = f"[{vel.x:>5.2f}, {vel.y:>5.2f}, {vel.z:>5.2f}]"
            
            # 如果 Z 轴速度异常，高亮显示
            if abs(vel.z) > 0.5:
                vel_str = f"[{vel.x:>5.2f}, {vel.y:>5.2f}, \033[91m{vel.z:>5.2f}\033[0m]"

            print(f"T+{elapsed:04.1f}s | {pos_str} | {vel_str} | {status_color}{status_str}\033[0m")

            # 严重发散保护
            if abs(pos.z) > 5.0:
                print("\n❌ [FATAL] 积分完全发散，停止监控。")
                raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = DynamicsMonitor()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()