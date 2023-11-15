from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression

from ament_index_python.packages import get_package_share_directory
import yaml
import os


def generate_launch_description():
    config_folder = os.path.join('src', 'quadrotor_bringup', 'config')
    config_file = os.path.join(config_folder, 'simulation.yaml')
    with open(config_file, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            parameters = dict()
    path_visualizer_parameters = parameters['QuadrotorPathVisualizer']
    pybullet_physics_simulation_parameters = parameters['PybulletPhysicsSimulation']
    pybullet_camera_simulation_parameters = parameters['PybulletCameraSimulation']
    pid_controller_parameters = parameters['PIDController']
    dfbc_controller_parameters = parameters['DFBCController']
    poly_traj_optimizer_parameters = parameters['PolyTrajOptimizer']

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

    dfbc_controller_node = Node(
        package='quadrotor_control',
        executable='quadrotor_dfbc',
        output='screen',
        parameters=[dfbc_controller_parameters]
    )

    reference_publisher_node = Node(
        package='quadrotor_trajectory_generation',
        executable='quadrotor_reference_publisher',
        output='screen'
    )

    trajectory_poly_optimizer_node = Node(
        package='quadrotor_trajectory_generation',
        executable='quadrotor_poly_optimizer',
        output='screen',
        parameters=[poly_traj_optimizer_parameters]
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
        parameters=[
            {'refresh_rate': path_visualizer_parameters['refresh_rate']}],
        output='screen'
    )
    image_visualizer_node = Node(
        package='quadrotor_dashboard',
        executable='quadrotor_image_visualizer',
        output='screen'
    )
    rqt_gui_node = Node(
        package='rqt_gui',
        executable='rqt_gui',
        output='screen',
    )

    return LaunchDescription([simulation_physics_node,
                              simulation_camera_node,
                              dfbc_controller_node,
                              reference_publisher_node,
                              trajectory_poly_optimizer_node,
                              mapping_node,
                              path_finding_node,
                              path_visualizer_node,
                              image_visualizer_node,
                              #   rqt_gui_node,
                              ])
