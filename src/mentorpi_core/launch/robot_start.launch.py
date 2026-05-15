import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    vision_node = Node(
        package='mentorpi_vision',
        executable='lane_detector_robot',
        name='vision_node'
    )
    core_node = Node(
        package='mentorpi_core',
        executable='simple_drive',
        name='simple_drive_node'
    )
    avoider_node = Node(
        package='mentorpi_navigation',
        executable='obstacle_avoider',
        name='obstacle_avoider_node'
    )

    # =================================================================
    # lidar_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(['/opt/ros/humble/share/launch.py'])
    # )

    return LaunchDescription([
        vision_node,
        core_node,
        avoider_node
        # lidar_launch
    ])
