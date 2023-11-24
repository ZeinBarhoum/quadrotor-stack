from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    dfbc_controller_node = Node(
        package='quadrotor_control',
        executable='quadrotor_dfbc',
        output='screen',
    )

    reference_publisher_node = Node(
        package='quadrotor_control',
        executable='quadrotor_reference_publisher',
        output='screen'
    )

    return LaunchDescription([dfbc_controller_node,
                              reference_publisher_node,
                              ])
