"""Start the motor node and the HW2 navigator together.

Start the camera first, in another terminal:
    ros2 launch robot_vision_camera robot_vision_camera.launch.py
Then:
    ros2 launch robot_control hw2_launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = LaunchConfiguration('params')
    declare_params = DeclareLaunchArgument(
        'params',
        default_value=PathJoinSubstitution(
            [FindPackageShare('robot_control'), 'config', 'hw2_params.yaml']))

    motor = Node(package='robot_control', executable='motor_control',
                 name='motor_controller_node', output='screen', emulate_tty=True)
    navigator = Node(package='robot_control', executable='hw2_navigator',
                     name='hw2_navigator', output='screen', emulate_tty=True,
                     parameters=[params])

    # When the run finishes (or crashes), shut everything down.
    done = RegisterEventHandler(OnProcessExit(
        target_action=navigator, on_exit=[Shutdown(reason='navigator finished')]))

    return LaunchDescription([declare_params, motor, navigator, done])
