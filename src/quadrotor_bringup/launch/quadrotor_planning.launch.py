from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():

    reference_publisher_node = Node(
        package='quadrotor_trajectory_generation',
        executable='quadrotor_reference_publisher',
        output='screen'
    )

    trajectory_poly_optimizer_node = Node(
        package='quadrotor_trajectory_generation',
        executable='quadrotor_poly_optimizer',
        output='screen'
    )

    mapping_node = Node(
        package='quadrotor_mapping',
        executable='quadrotor_default_map',
        output='screen'
    )
    path_finding_node = Node(
        package='quadrotor_path_finding',
        executable='quadrotor_rrt',
        output='screen'
    )

    path_visualizer_node = Node(
        package='quadrotor_dashboard',
        executable='quadrotor_path_visualizer',
        output='screen'
    )

    return LaunchDescription([
        reference_publisher_node,
        trajectory_poly_optimizer_node,
        mapping_node,
        path_finding_node,
        path_visualizer_node
    ])
