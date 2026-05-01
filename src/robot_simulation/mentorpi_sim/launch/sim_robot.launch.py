import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
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

    # Uruchomienie Gazebo (flaga -r od razu startuje symulację)
    world_file = os.path.join(pkg_share, 'worlds', 'test_track.sdf')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    # Zespawnowanie robota w Gazebo
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

    # MOST (BRIDGE) - Przesyłanie obrazu z Gazebo i sterowania do Gazebo
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Znak '[' wymusza kierunek Gazebo -> ROS 2 (Zmniejsza lagi!)
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            
            # Znak ']' wymusza kierunek ROS 2 -> Gazebo (Komendy jazdy)
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist'
        ],
        output='screen'
    )

    # FOXGLOVE BRIDGE - Dodany z Twojej magisterki
    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        parameters=[{'port': 8765}],
        output='screen'
    )

    # Węzeł śledzenia pasa (Twój nowy kod z ROI)
    lane_detector_node = Node(
        package='mentorpi_vision', # ZMIEŃ na nazwę paczki, w której trzymasz ten skrypt
        executable='lane_detector_sim', # ZMIEŃ na nazwę Twojego skryptu
        name='sim_lane_tracker',
        output='screen'
    )

    return LaunchDescription([
        set_model_path,
        gazebo_launch,
        robot_state_publisher,
        bridge,
        foxglove,
        lane_detector_node, # <--- DODANO TUTAJ
        TimerAction(
            period=3.0,
            actions=[spawn_robot]
        )
    ])