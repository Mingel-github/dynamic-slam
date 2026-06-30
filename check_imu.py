import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np
import time

class ImuInspector(Node):
    def __init__(self):
        super().__init__('imu_inspector')
        self.subscription = self.create_subscription(
            Imu,
            '/imu_raw',  # 确保这里和你的 params.yaml 里的 imuTopic 一致
            self.listener_callback,
            10)
        self.acc_z_buffer = []
        self.start_time = time.time()
        self.done = False
        print("正在收集 200 个 IMU 数据点 (约 1-2 秒)... 请保持小车静止！")

    def listener_callback(self, msg):
        # 只看 Z 轴加速度
        az = msg.linear_acceleration.z
        self.acc_z_buffer.append(az)

        if len(self.acc_z_buffer) >= 200:
            self.analyze_and_report()
            self.done = True

    def analyze_and_report(self):
        data = np.array(self.acc_z_buffer)
        mean_z = np.mean(data)
        std_z = np.std(data)
        max_z = np.max(data)
        min_z = np.min(data)
        
        print("\n" + "="*40)
        print("🔍 IMU 数据体检报告")
        print("="*40)
        print(f"1. 重力均值 (Mean Z): {mean_z:.4f} m/s^2")
        print(f"   -> 理想值应该是 9.8 左右")
        
        print(f"2. 震动噪声 (StdDev): {std_z:.5f}")
        print(f"   -> 你的 params.yaml 里 imuAccNoise 必须大于这个值！")
        print(f"   -> 建议设为: {std_z * 1.5:.5f}")

        print(f"3. 冲击检测 (Max/Min): [{min_z:.2f}, {max_z:.2f}]")
        if max_z > 15.0 or min_z < 5.0:
            print("   ⚠️ 警告：检测到巨大冲击！小车可能在弹跳！")
            print("   -> 绝对不要在此时启动 LIO-SAM！")
        else:
            print("   ✅ 状态平稳")
        print("="*40)

def main(args=None):
    rclpy.init(args=args)
    inspector = ImuInspector()
    try:
        while rclpy.ok() and not inspector.done:
            rclpy.spin_once(inspector, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        inspector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()