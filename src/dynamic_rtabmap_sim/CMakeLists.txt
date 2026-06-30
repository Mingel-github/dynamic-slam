cmake_minimum_required(VERSION 3.8)
project(dynamic_rtabmap_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
# 依赖包
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
#find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(xacro REQUIRED)

# =========================================
# ★★★ 关键：安装资源文件到 Share 目录 ★★★
# =========================================
install(DIRECTORY
  launch
  worlds
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
