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
            description='Controller to use (default: quadrotor_pid)'
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

    reference_publisher_node = Node(
        package= 'quadrotor_trajectory_generation',
        executable= 'quadrotor_reference_publisher',
        output= 'screen'
    )

    trajectory_poly_optimizer_node = Node(
        package = 'quadrotor_trajectory_generation',
        executable = 'quadrotor_poly_optimizer',
        output = 'screen'
    )

    mapping_node = Node(
        package = 'quadrotor_mapping',
        executable = 'quadrotor_default_map',
        output = 'screen'
    )
    path_finding_node = Node(
        package = 'quadrotor_path_finding',
        executable= 'quadrotor_rrt',
        output = 'screen'
    )

    return LaunchDescription(
        decalred_arguments+[
            simulation_node,
            controller_node,
            reference_publisher_node,
            trajectory_poly_optimizer_node,
            mapping_node,
            path_finding_node,
        ])
