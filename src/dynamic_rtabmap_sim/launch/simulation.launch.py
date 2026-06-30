import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro



def generate_launch_description():
    pkg_name = 'dynamic_rtabmap_sim'
    pkg_path = get_package_share_directory(pkg_name)
    
    # 1. 路径配置：锁定长廊世界
    # 假设你的世界文件在 dynamic_rtabmap_sim/worlds/loop_corridor.world
    world_path = os.path.join(pkg_path, 'worlds', 'loop_corridor.world')
    xacro_file = os.path.join(pkg_path, 'urdf', 'robot.xacro')
    
    # 2. 解析 Xacro
    robot_description_config = xacro.process_file(xacro_file)
    robot_desc = robot_description_config.toxml()

    # 3. Gazebo 启动 (只负责物理世界)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'verbose': 'true'
        }.items()
    )

    # 4. 机器人状态发布节点 (TF 树的核心)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc, 
            'use_sim_time': True
        }]
    )

    # 5. 生成机器人实体
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'geo_bot',
            '-x', '0.0', '-y', '-8.5', '-z', '0.3'  # <--- 修改这里
        ],
        output='screen'
    )


    # [新增] 6. 动态生成我们的遥控纸箱
    # 使用 os.path.expanduser 优雅地解析 ~/ 路径
    box_sdf_path = os.path.expanduser('~/ros2_ws/src/dynamic_rtabmap_sim/models/model_box.sdf')
    spawn_moving_box = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'dynamic_box',
            '-file', box_sdf_path,
            '-x', '-7.0',     # 【修正】放在西侧走廊，避开东侧的行人！
            '-y', '0.0', 
            '-z', '0.3',
            '-Y', '1.5708'    # 【新增】Yaw=90度，确保箱子的局部X轴朝北
        ],
        output='screen'
    )


    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        spawn_moving_box # <--- 将纸箱节点加入启动列表
        
    ])