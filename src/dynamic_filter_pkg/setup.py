import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dynamic_filter_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ！！！关键：将 weights 目录下的模型文件安装到 share 目录下 ！！！
        (os.path.join('share', package_name, 'weights'), glob('dynamic_filter_pkg/weights/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhu',
    maintainer_email='zhu@todo.todo',
    description='YOLOv8 Semantic Depth Filter for Dynamic SLAM',
    license='Apache-2.0', # 顺手帮你把许可证警告修了
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'box_patrol = dynamic_filter_pkg.box_patrol_node:main',
            'semantic_filter_node = dynamic_filter_pkg.semantic_filter_node:main'
        ],
    },
)