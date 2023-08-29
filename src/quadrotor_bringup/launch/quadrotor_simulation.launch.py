from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression

from ament_index_python.packages import get_package_share_directory
import yaml
import os


def generate_launch_description():

    decalred_arguments = []

    decalred_arguments.append(
        DeclareLaunchArgument(
            name='controller',
            default_value='quadrotor_pid',
            description='Controller to use (default: quadrotor_pid)'
        )
    )
    config_folder = os.path.join('src', 'quadrotor_bringup', 'config')
    config_file = os.path.join(config_folder, 'simulation.yaml')
    with open(config_file, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            parameters = dict()
    path_visualizer_parameters = parameters['QuadrotorPathVisualizer']
    pybullet_simulation_parameters = parameters['PybulletSimulation']
    pid_controller_parameters = parameters['PIDController']

    simulation_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_pybullet',
        output='screen',
        parameters=[pybullet_simulation_parameters]
    )

    controller = LaunchConfiguration("controller")

    controller_node = Node(
        package='quadrotor_control',
        executable=controller,
        output='screen',
        parameters=[pid_controller_parameters]
    )

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
        parameters=[{'refresh_rate': path_visualizer_parameters['refresh_rate']}],
        output='screen'
    )

    return LaunchDescription(
        decalred_arguments+[
            simulation_node,
            controller_node,
            reference_publisher_node,
            trajectory_poly_optimizer_node,
            mapping_node,
            path_finding_node,
            path_visualizer_node
        ])
