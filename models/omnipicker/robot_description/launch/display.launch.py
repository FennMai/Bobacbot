import os
from launch.substitutions import Command, PathJoinSubstitution
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    xacro_path = os.path.join(get_package_share_directory('robot_description'), 'urdf', 'robot.xacro')
    params_file = PathJoinSubstitution([get_package_share_directory('robot_description'), 'config', 'state_publisher.yaml'])

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', xacro_path])}]
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        parameters=[params_file]
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', [os.path.join(get_package_share_directory('robot_description'), 'rviz', 'urdf.rviz')]]
    )

    return LaunchDescription([
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz2_node,
    ])
