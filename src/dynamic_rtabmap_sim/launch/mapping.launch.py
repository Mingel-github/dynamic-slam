import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    
    # 获取功能包路径
    pkg_path = get_package_share_directory('dynamic_rtabmap_sim')
    
    # 1. 核心参数配置
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    qos = LaunchConfiguration('qos', default='2')
    
    # 2. 解析 Xacro 并启动机器人状态发布
    xacro_file = os.path.join(pkg_path, 'urdf', 'robot.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_desc = robot_description_config.toxml()

    # 机器人状态发布节点
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc, 
            'use_sim_time': True
        }]
    )

    # 关节状态发布节点
    node_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': True}]
    )

    # 3. 调用 RTAB-Map 算法
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'frame_id': 'base_footprint',           
            'subscribe_depth': 'true',              
            'subscribe_scan': 'false',
            'subscribe_scan_cloud': 'false',        # 纯RGB-D管线（学术公平对比）
            'visual_odometry': 'false',
            'odom_topic': '/odom',

            'rgb_topic': '/camera/image_filtered',  # 过滤后RGB，阻断动态物体ORB特征
            'depth_topic': '/camera/depth/image_filtered',
            'camera_info_topic': '/camera/camera_info',

            'qos': qos,
            'queue_size': '20',
            'approx_sync': 'true',

            # ==========================================================
            # 纯视觉 RGB-D SLAM（对标 DynaSLAM/DS-SLAM）
            # g2o优化器 + DetectionRate 5Hz + 严格视觉门槛(MinInliers 25)
            # ==========================================================
            'args': '--delete_db_on_start --Grid/RangeMax 5.0 --Grid/RayTracing true '
                    '--Optimizer/Strategy 1 --Optimizer/VarianceIgnored true '
                    '--Vis/MinInliers 25 --RGBD/ProximityBySpace true '
                    '--Rtabmap/DetectionRate 5.0',
            
            # 打破 Gazebo 的绝对自信
            'odom_tf_linear_variance': '0.001',
            'odom_tf_angular_variance': '0.001',
            
            'rtabmap_viz': 'false',                 
            'rviz': 'false',                        
        }.items()
    )

    return LaunchDescription([
        node_robot_state_publisher,   
        node_joint_state_publisher,   
        rtabmap_launch
    ])