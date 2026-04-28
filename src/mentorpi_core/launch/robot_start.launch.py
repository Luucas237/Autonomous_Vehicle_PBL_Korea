import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    # 1. Węzeł Wizji (Przetwarzanie obrazu na robocie)
    vision_node = Node(
        package='mentorpi_vision',
        executable='lane_detector_robot',
        name='vision_node'
    )

    # # 2. Węzeł Silników (Regulator PID)
    # core_node = Node(
    #     package='mentorpi_core',
    #     executable='simple_drive',
    #     name='simple_drive_node'
    # )

    # 3. Węzeł Omijania Przeszkód (Maszyna Stanów)
    avoider_node = Node(
        package='mentorpi_navigation',
        executable='obstacle_avoider',
        name='obstacle_avoider_node'
    )

    # =================================================================
    # MIEJSCE NA LIDAR (Odkomentuj i wpisz ścieżkę, gdy ją znajdziesz)
    # =================================================================
    # lidar_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(['/opt/ros/humble/share/SCIEZKA_DO_LIDARA/launch.py'])
    # )

    return LaunchDescription([
        vision_node,
        # core_node,
        avoider_node
        # lidar_launch
    ])