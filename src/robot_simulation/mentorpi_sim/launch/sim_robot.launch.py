import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('mentorpi_sim')

    set_model_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=os.path.join(pkg_share, '..')
    )

    # Konwersja XACRO -> URDF w Pythonie
    xacro_file = os.path.join(pkg_share, 'urdf', 'ack.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_urdf = robot_description_config.toxml()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_urdf}]
    )

    # Uruchomienie Gazebo
    world_file = os.path.join(pkg_share, 'worlds', 'test_track.sdf')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'hiwonder',
            '-allow_renaming', 'true',
            '-z', '0.2'
        ],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist'
        ],
        output='screen'
    )

    return LaunchDescription([
        set_model_path,
        gazebo_launch,
        robot_state_publisher,
        spawn_robot,
        bridge
    ])