from launch import LaunchDescription
from launch_ros.actions import Node
import yaml
import os


def generate_launch_description():
    config_folder = os.path.join('src', 'quadrotor_bringup', 'config')
    config_file = os.path.join(config_folder, 'config.yaml')
    with open(config_file, "r") as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            parameters = dict()
    parameters = parameters['QuadrotorSimulation']
    pybullet_physics_parameters = parameters['PybulletPhysics']
    pybullet_camera_parameters = parameters['PybulletCamera']
    imu_parameters = parameters['IMU']

    simulation_physics_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_pybullet_physics',
        output='screen',
        parameters=[pybullet_physics_parameters]
    )
    simulation_camera_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_pybullet_camera',
        output='screen',
        parameters=[pybullet_camera_parameters]
    )
    simulation_imu_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_imu',
        output='screen',
        parameters=[imu_parameters]
    )

    return LaunchDescription([simulation_physics_node,
                              simulation_camera_node,
                              simulation_imu_node
                              ])
