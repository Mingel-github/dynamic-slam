#!/usr/bin/env python3
#box_patrol_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class BoxPatrolNode(Node):
    def __init__(self):
        super().__init__('box_patrol_node')
        
        # 1. 订阅箱子的真实物理里程计 (闭环防撞墙的关键)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/box_odom',
            self.odom_callback,
            10
        )
        
        # 2. 发布速度指令到隐形马达
        self.cmd_pub = self.create_publisher(Twist, '/box_cmd_vel', 10)
        
        # 状态参数设定
        self.speed = 0.5  # 箱子移动速度 (m/s)
        self.direction = 1.0  # 1.0 为向前(北)，-1.0 为向后(南)
        self.upper_y_limit = 5.0
        self.lower_y_limit = -5.0
        
        self.get_logger().info('📦 箱子自动巡逻测试引擎 已启动！')
        self.get_logger().info(f'巡逻区间: Y轴 [{self.lower_y_limit}, {self.upper_y_limit}]')

    def odom_callback(self, msg):
        # 实时提取箱子在世界中的 Y 坐标
        current_y = msg.pose.pose.position.y
        
        # 触碰边界时反转方向 (模拟打乒乓球)
        if current_y > self.upper_y_limit and self.direction == 1.0:
            self.direction = -1.0
            self.get_logger().info(f'到达北界 ({current_y:.2f})，挂倒挡向南！')
        elif current_y < self.lower_y_limit and self.direction == -1.0:
            self.direction = 1.0
            self.get_logger().info(f'到达南界 ({current_y:.2f})，挂前进挡向北！')
            
        # 组装并发送速度指令
        # 因为箱子生成时面朝北 (yaw=1.57)，在它自己的局部坐标系里，Linear X 就是正前方(北)
        twist = Twist()
        twist.linear.x = self.speed * self.direction
        # 我们不需要它转身，直接通过正负号实现前进和倒退，这样光流测试最纯粹
        twist.angular.z = 0.0 
        
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = BoxPatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出前发个 0 速度，让箱子安全停下
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()