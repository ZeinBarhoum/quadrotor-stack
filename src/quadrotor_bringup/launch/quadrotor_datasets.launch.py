from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression

from ament_index_python.packages import get_package_share_directory
import yaml
import os


def generate_launch_description():

    config_folder = os.path.join('src', 'quadrotor_bringup', 'config')
    config_file = os.path.join(config_folder, 'dataset_tests.yaml')
    with open(config_file, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            parameters = dict()
    pybullet_physics_simulation_parameters = parameters['PybulletPhysicsSimulation']
    pybullet_camera_simulation_parameters = parameters['PybulletCameraSimulation']
    dataset_controller_parameters = parameters['Dataset']
    model_error_vis_parameters = parameters['ModelErrorVis']

    simulation_physics_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_pybullet_physics',
        output='screen',
        parameters=[pybullet_physics_simulation_parameters]
    )
    simulation_camera_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_pybullet_camera',
        output='screen',
        parameters=[pybullet_camera_simulation_parameters]
    )
    dataset_controller_node = Node(
        package='quadrotor_control',
        executable='quadrotor_dataset',
        output='screen',
        parameters=[dataset_controller_parameters]
    )
    model_error_visualizer_node = Node(
        package='quadrotor_dashboard',
        executable='quadrotor_model_errors_visualizer',
        output='screen',
        parameters=[model_error_vis_parameters],
    )

    return LaunchDescription([
        simulation_physics_node,
        simulation_camera_node,
        dataset_controller_node,
        model_error_visualizer_node
    ])
