from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression


def generate_launch_description():

    decalred_arguments = []

    decalred_arguments.append(
        DeclareLaunchArgument(
            name='controller',
            default_value='quadrotor_pid',
            description='Controller to use (default: pid)'
        )
    )
    simulation_node = Node(
        package='quadrotor_simulation',
        executable='quadrotor_pybullet',
        output='screen'
    )

    controller = LaunchConfiguration("controller")

    controller_node = Node(
        package='quadrotor_control',
        executable=controller,
        output='screen'
    )

    return LaunchDescription(
        decalred_arguments+[
            simulation_node,
            controller_node,
        ])
