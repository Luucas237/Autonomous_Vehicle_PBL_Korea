import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. Węzeł Podglądu Wizji na PC
    vision_pc_node = Node(
        package='mentorpi_vision',
        executable='lane_detector',
        name='vision_pc_node'
    )

    map_pc_node = Node(
        package='mentorpi_navigation',
        executable='map_publisher',
        name='map_pc_node'
    )

    # 2. RViz z zapisaną konfiguracją
    rviz_config_dir = os.path.join(
        get_package_share_directory('mentorpi_navigation'),
        'rviz',
        'digital_twin.rviz'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir]
    )

    return LaunchDescription([
        vision_pc_node,
        map_pc_node,
        rviz_node
    ])