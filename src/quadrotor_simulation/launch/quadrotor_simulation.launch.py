from launch import LaunchDescription
from launch_ros.actions import Node
import yaml
import os


def generate_launch_description():

    simulation_physics_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_pybullet_physics',
        output='screen',
        parameters=[{'physics_server': 'DIRECT'}]
    )
    simulation_camera_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_pybullet_camera',
        output='screen',
    )
    simulation_imu_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_imu',
        output='screen',
    )

    return LaunchDescription([simulation_physics_node,
                              simulation_camera_node,
                              simulation_imu_node
                              ])
