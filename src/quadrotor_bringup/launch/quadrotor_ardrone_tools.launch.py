from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression

from ament_index_python.packages import get_package_share_directory
import yaml
import os


def generate_launch_description():
    config_folder = os.path.join('src', 'quadrotor_bringup', 'config')
    config_file = os.path.join(config_folder, 'ardrone.yaml')
    with open(config_file, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            parameters = dict()
    poly_traj_optimizer_parameters = parameters['PolyTrajOptimizer']
    path_visualizer_parameters = parameters['QuadrotorPathVisualizer']

    trajectory_poly_optimizer_node = Node(
        package='quadrotor_trajectory_generation',
        executable='quadrotor_poly_optimizer',
        output='screen',
        parameters=[poly_traj_optimizer_parameters]
    )
    reference_publisher_node = Node(
        package='quadrotor_control',
        executable='quadrotor_reference_publisher',
        output='screen'
    )
    mapping_node = Node(
        package='quadrotor_mapping',
        executable='quadrotor_501_map',
        output='screen'
    )
    path_visualizer_node = Node(
        package='quadrotor_dashboard',
        executable='quadrotor_path_visualizer',
        parameters=[
            {'refresh_rate': path_visualizer_parameters['refresh_rate']}],
        output='screen'
    )
    return LaunchDescription([mapping_node,
                              reference_publisher_node,
                              trajectory_poly_optimizer_node,
                              path_visualizer_node,
                              ])
