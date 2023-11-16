import os
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
import pybullet as p
import pybullet_data

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from quadrotor_interfaces.msg import RotorCommand, State, ModelError
from geometry_msgs.msg import Vector3Stamped
import xacro


try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys
    sys.excepthook = IPython.core.ultratb.ColorTB()


DEFAULT_FREQUENCY = 240  # Hz
DEFAULT_QOS_PROFILE = 10


class QuadrotorPybulletPhysics(Node):

    def __init__(self):
        """ Initializes the node."""
        super().__init__('quadrotor_pybullet_physics_node')

        # Declare the parameters
        self.declare_parameters(namespace='', parameters=[('physics_server', 'DIRECT'),  # GUI, DIRECT
                                                          ('quadrotor_description', 'cf2x'),
                                                          ('obstacles_description', ['NONE']),
                                                          ('obstacles_poses', [0.0]),
                                                          ('render_ground', True),
                                                          ('simulation_step_frequency', DEFAULT_FREQUENCY),
                                                          ('state_topic', 'quadrotor_state'),
                                                          ('ff_state_topic', 'quadrotor_ff_state'),
                                                          ('rotor_speeds_topic', 'quadrotor_rotor_speeds'),
                                                          ('wind_speed_topic', 'quadrotor_wind_speed'),
                                                          ('model_error_topic', 'quadrotor_model_error'),
                                                          ('calculate_linear_drag', True),
                                                          ('calculate_quadratic_drag', True),
                                                          ('calculate_residuals', False),
                                                          ('residuals_model', 'NONE'),
                                                          ('residuals_device', 'cuda'),
                                                          ('use_rotor_dynamics', True),
                                                          ('use_wind_speed', True),
                                                          ('use_ff_state', False),
                                                          ('manual_tau_xy_calculation', False),
                                                          ('publish_model_errors', False),
                                                          ('sequential_mode', False),])
        # Get the parameters
        self.physics_server = self.get_parameter('physics_server').get_parameter_value().string_value
        self.quadrotor_description_file_name = self.get_parameter('quadrotor_description').get_parameter_value().string_value
        self.obstacles_description_file_names = self.get_parameter('obstacles_description').get_parameter_value().string_array_value
        self.obstacles_poses = self.get_parameter('obstacles_poses').get_parameter_value().double_array_value
        self.render_ground = self.get_parameter('render_ground').get_parameter_value().bool_value
        self.simulation_step_frequency = self.get_parameter('simulation_step_frequency').get_parameter_value().integer_value
        self.state_topic = self.get_parameter('state_topic').get_parameter_value().string_value
        self.ff_state_topic = self.get_parameter('ff_state_topic').get_parameter_value().string_value
        self.rotor_speeds_topic = self.get_parameter('rotor_speeds_topic').get_parameter_value().string_value
        self.wind_speed_topic = self.get_parameter('wind_speed_topic').get_parameter_value().string_value
        self.model_error_topic = self.get_parameter('model_error_topic').get_parameter_value().string_value
        self.use_rotor_dynamics = self.get_parameter('use_rotor_dynamics').get_parameter_value().bool_value
        self.calculate_linear_drag = self.get_parameter('calculate_linear_drag').get_parameter_value().bool_value
        self.calculate_quadratic_drag = self.get_parameter('calculate_quadratic_drag').get_parameter_value().bool_value
        self.calculate_residuals = self.get_parameter('calculate_residuals').get_parameter_value().bool_value
        self.residuals_model = self.get_parameter('residuals_model').get_parameter_value().string_value
        self.residuals_device = self.get_parameter('residuals_device').get_parameter_value().string_value
        self.use_wind_speed = self.get_parameter('use_wind_speed').get_parameter_value().bool_value
        self.use_ff_state = self.get_parameter('use_ff_state').get_parameter_value().bool_value
        self.manual_tau_xy_calculation = self.get_parameter('manual_tau_xy_calculation').get_parameter_value().bool_value
        self.publish_model_errors = self.get_parameter('publish_model_errors').get_parameter_value().bool_value
        self.sequential_mode = self.get_parameter('sequential_mode').get_parameter_value().bool_value

        # Subscribers and Publishers
        self.rotor_speeds_subscriber = self.create_subscription(msg_type=RotorCommand,
                                                                topic=self.rotor_speeds_topic,
                                                                callback=self.receive_commands_callback,
                                                                qos_profile=DEFAULT_QOS_PROFILE)
        if (self.use_wind_speed):
            self.wind_speed_subscriber = self.create_subscription(msg_type=Vector3Stamped,
                                                                  topic=self.wind_speed_topic,
                                                                  callback=self.receive_wind_speed_callback,
                                                                  qos_profile=DEFAULT_QOS_PROFILE)
        if (self.use_ff_state):
            self.ff_state_subscriber = self.create_subscription(msg_type=State,
                                                                topic=self.ff_state_topic,
                                                                callback=self.receive_ff_state_callback,
                                                                qos_profile=DEFAULT_QOS_PROFILE)
        self.state_publisher = self.create_publisher(msg_type=State,
                                                     topic=self.state_topic,
                                                     qos_profile=DEFAULT_QOS_PROFILE)

        if (self.publish_model_errors):
            self.model_error_publisher = self.create_publisher(msg_type=ModelError,
                                                               topic=self.model_error_topic,
                                                               qos_profile=DEFAULT_QOS_PROFILE)

        # Control the frequencies of simulation
        self.simulation_step_period = 1.0 / self.simulation_step_frequency  # seconds

        # initialize the constants, the urdf file and the pybullet client
        self.initialize_urdf()
        self.initialize_constants()
        self.initialize_pybullet()

        # Initialize the published and received data
        self.initialize_data()

        # initialize timers
        if not self.sequential_mode:
            self.simulation_step_timer = self.create_timer(self.simulation_step_period, self.simulation_step_callback)

        # Announce that the node is initialized
        self.start_time = self.get_clock().now()  # For logging purposes
        self.get_logger().info(
            f'QuadrotorPybulletPhysics node initialized at {self.start_time.seconds_nanoseconds()}. Frequency: {self.simulation_step_frequency} Hz')

    def initialize_constants(self):
        config_folder = os.path.join(get_package_share_directory('quadrotor_description'), 'config')
        config_file = os.path.join(config_folder, self.quadrotor_description_file_name+'_params.yaml')
        with open(config_file, "r") as stream:
            try:
                parameters = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                self.get_logger().error(
                    f"Cofiguration File {config_file} Couldn't Be Loaded, Raised Error {exc}")
                parameters = dict()

        quadrotor_params = parameters[f'{self.quadrotor_description_file_name.upper()}_PARAMS']
        self.G = 9.81
        self.KF = quadrotor_params['KF']
        self.KM = quadrotor_params['KM']
        self.M = quadrotor_params['M']
        self.W = self.M*self.G
        self.ROT_HOVER_VEL = np.sqrt(self.W/(4*self.KF))
        self.T2W = quadrotor_params['T2W']
        self.ROT_MAX_VEL = np.sqrt(self.T2W*self.W/(4*self.KF))
        self.ROT_MAX_ACC = quadrotor_params['ROT_MAX_ACC']
        self.ROT_TIME_STEP = quadrotor_params['ROT_TIME_STEP']
        self.DRAG_MAT_LIN = np.array(quadrotor_params['DRAG_MAT_LIN'])
        self.DRAG_MAT_QUAD = np.array(quadrotor_params['DRAG_MAT_QUAD'])
        self.ROTOR_DIRS = quadrotor_params['ROTOR_DIRS']
        self.ARM_X = quadrotor_params['ARM_X']
        self.ARM_Y = quadrotor_params['ARM_Y']
        self.ARM_Z = quadrotor_params['ARM_Z']
        self.J = np.array(quadrotor_params['J'])
        if self.calculate_residuals:
            from quadrotor_simulation.quadrotor_residuals import prepare_residuals_model
            self.RES_NET, self.RES_PARAMS = prepare_residuals_model(self.residuals_model)
            self.RES_DEVICE = self.residuals_device
            self.RES_NET.to(self.RES_DEVICE)

    def initialize_urdf(self):
        quadrotor_description_folder = os.path.join(get_package_share_directory('quadrotor_description'), 'description')
        quadrotor_description_file = os.path.join(quadrotor_description_folder, self.quadrotor_description_file_name+'.urdf.xacro')
        quadrotor_description_content = xacro.process_file(quadrotor_description_file).toxml()
        new_file = os.path.join(quadrotor_description_folder, self.quadrotor_description_file_name+'.urdf')
        with open(new_file, 'w+') as f:
            f.write(quadrotor_description_content)
        self.quadrotor_urdf_file = new_file

        # Retreive the obstacle urdf file and save it for pybullet to read
        obstacles_description_folder = os.path.join(get_package_share_directory('quadrotor_simulation'), 'world')
        self.obstacle_urdf_files = []
        for name in self.obstacles_description_file_names:
            if (name == 'NONE'):
                break
            self.obstacle_description_file_name = name
            obstacle_description_file = os.path.join(obstacles_description_folder, self.obstacle_description_file_name+'.urdf.xacro')
            obstacle_description_content = xacro.process_file(obstacle_description_file).toxml()
            new_file = os.path.join(obstacles_description_folder, name + '.urdf')
            with open(new_file, 'w+') as f:
                f.write(obstacle_description_content)
            self.obstacle_urdf_files.append(new_file)

    def initialize_pybullet(self):
        if (self.physics_server == 'DIRECT'):
            self.physicsClient = p.connect(p.DIRECT)
        else:
            self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.simulation_step_period)
        p.setGravity(0, 0, -self.G)
        if (self.render_ground):
            self.planeId = p.loadURDF("plane.urdf")
        self.obstacleIds = []
        for (i, obstacle_urdf_file) in enumerate(self.obstacle_urdf_files):
            self.obstacleIds.append(p.loadURDF(obstacle_urdf_file, self.obstacles_poses[i*7: i*7+3], self.obstacles_poses[i*7+3: i*7+7], useFixedBase=1))
        self.quadrotor_id = p.loadURDF(self.quadrotor_urdf_file, [0, 0, 0.25], flags=p.URDF_USE_INERTIA_FROM_FILE)
        # Disable default damping of pybullet!
        p.changeDynamics(self.quadrotor_id, -1, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.quadrotor_id, 0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.quadrotor_id, 1, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.quadrotor_id, 2, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.quadrotor_id, 3, linearDamping=0, angularDamping=0)

        self.get_logger().info(f"Loaded quadrotor with Dynamics {p.getDynamicsInfo(self.quadrotor_id, -1)}")

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

    def initialize_data(self):
        self.rotor_speeds = np.array([self.ROT_HOVER_VEL] * 4)
        self.wind_speed = np.array([0, 0, 0])
        self.ff_state = State()
        self.state = State()
        self.model_error = ModelError()
        if self.use_rotor_dynamics:
            self.current_time = self.get_clock().now()  # for rotor dynamics integration

    def receive_ff_state_callback(self, msg: State):
        self.ff_state = msg

    def receive_commands_callback(self, msg: RotorCommand):
        if not self.use_rotor_dynamics:
            self.rotor_speeds = np.array(msg.rotor_speeds)
        else:
            dt = (self.get_clock().now() - self.current_time).nanoseconds / 1e9
            self.current_time = self.get_clock().now()
            command_rotor_speeds = np.array(msg.rotor_speeds)
            rotor_acceleration = (command_rotor_speeds - self.rotor_speeds) / self.ROT_TIME_STEP
            rotor_acceleration = np.clip(rotor_acceleration, -self.ROT_MAX_ACC, self.ROT_MAX_ACC)
            self.rotor_speeds += rotor_acceleration * dt
            self.rotor_speeds = np.clip(self.rotor_speeds, 0, self.ROT_MAX_VEL)
        self.simulation_step_callback()

    def receive_wind_speed_callback(self, msg: Vector3Stamped):
        self.wind_speed = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def apply_ff_state(self):
        pos = np.array([self.ff_state.state.pose.position.x, self.ff_state.state.pose.position.y, self.ff_state.state.pose.position.z])
        quat = np.array([self.ff_state.state.pose.orientation.x, self.ff_state.state.pose.orientation.y,
                        self.ff_state.state.pose.orientation.z, self.ff_state.state.pose.orientation.w])
        v = np.array([self.ff_state.state.twist.linear.x, self.ff_state.state.twist.linear.y, self.ff_state.state.twist.linear.z])
        w = np.array([self.ff_state.state.twist.angular.x, self.ff_state.state.twist.angular.y, self.ff_state.state.twist.angular.z])
        w_W = Rotation.from_quat(quat).apply(w)
        p.resetBasePositionAndOrientation(self.quadrotor_id, pos, quat)
        p.resetBaseVelocity(self.quadrotor_id, v, w_W)

    def get_F_T(self):  # not used
        F = np.array(self.rotor_speeds**2)*self.KF
        T = np.array(self.rotor_speeds**2)*self.KM
        Tz = (-T[0] + T[1] - T[2] + T[3])
        return F, np.array([0, 0, Tz])

    def calculate_drag(self):
        # calculate both linear (rotor) and quadratic (fuselage) drag
        quat = np.array([self.state.state.pose.orientation.x, self.state.state.pose.orientation.y,
                        self.state.state.pose.orientation.z, self.state.state.pose.orientation.w])
        vel_W = np.array([self.state.state.twist.linear.x, self.state.state.twist.linear.y, self.state.state.twist.linear.z])
        v_rel_B = Rotation.from_quat(quat).inv().apply(vel_W - self.wind_speed)
        v_rel_norm = np.linalg.norm(v_rel_B)
        drag = np.zeros(3)
        if (self.calculate_linear_drag):
            drag -= (self.DRAG_MAT_LIN @ v_rel_B.reshape(-1, 1)).flatten()
        if (self.calculate_quadratic_drag):
            drag -= (v_rel_norm * self.DRAG_MAT_QUAD @ v_rel_B.reshape(-1, 1)).flatten()
        return drag

    def calculate_nominal_thrust_torques(self):
        rotor_thrusts = np.array(self.rotor_speeds**2)*self.KF
        rotor_torques = np.array(self.rotor_speeds**2)*self.KM
        torque_z = -(self.ROTOR_DIRS[0]*rotor_torques[0] + self.ROTOR_DIRS[1]*rotor_torques[1] +
                     self.ROTOR_DIRS[2]*rotor_torques[2] + self.ROTOR_DIRS[3]*rotor_torques[3])
        torque_x = self.ARM_Y * (-rotor_thrusts[0] + rotor_thrusts[1] + rotor_thrusts[2] - rotor_thrusts[3])
        torque_y = self.ARM_X * (-rotor_thrusts[0] - rotor_thrusts[1] + rotor_thrusts[2] + rotor_thrusts[3])

        return rotor_thrusts, torque_x, torque_y, torque_z

    def calculate_residual_thrust_torques(self):
        residuals = np.zeros(6)
        if self.calculate_residuals:
            from quadrotor_simulation.quadrotor_residuals import calculate_residuals
            residuals = calculate_residuals(self.state, RotorCommand(rotor_speeds=self.rotor_speeds), self.RES_NET, self.RES_DEVICE, self.RES_PARAMS)
        return residuals

    def apply_forces_torques(self):
        rotor_thrusts, torque_x, torque_y, torque_z = self.calculate_nominal_thrust_torques()
        drag_force = self.calculate_drag()
        residuals = self.calculate_residual_thrust_torques()

        if (self.manual_tau_xy_calculation):
            for i in range(4):
                p.applyExternalForce(self.quadrotor_id, -1, forceObj=[0, 0, rotor_thrusts[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME)
            p.applyExternalTorque(self.quadrotor_id, -1, torqueObj=[torque_x, torque_y, torque_z], flags=p.LINK_FRAME)
        else:
            for i in range(4):
                p.applyExternalForce(self.quadrotor_id, i, forceObj=[0, 0, rotor_thrusts[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME)
            # applying Tz on the center of mass, the only one that depend on the drag and isn't simulated by the forces before
            p.applyExternalTorque(self.quadrotor_id, -1, torqueObj=[0, 0, torque_z], flags=p.LINK_FRAME)

        p.applyExternalForce(self.quadrotor_id, -1, forceObj=drag_force, posObj=[0, 0, 0], flags=p.LINK_FRAME)
        p.applyExternalForce(self.quadrotor_id, -1, forceObj=residuals[:3], posObj=[0, 0, 0], flags=p.LINK_FRAME)
        p.applyExternalTorque(self.quadrotor_id, -1, torqueObj=residuals[3:], flags=p.LINK_FRAME)

    def apply_simulation_step(self):
        pos0, quat0 = p.getBasePositionAndOrientation(self.quadrotor_id)
        pos0, quat0 = np.array(pos0), np.array(quat0)
        vel0, avel0_W = p.getBaseVelocity(self.quadrotor_id)
        vel0, avel0_B = np.array(vel0), Rotation.from_quat(quat0).inv().apply(np.array(avel0_W))

        p.stepSimulation()

        pos, quat = p.getBasePositionAndOrientation(self.quadrotor_id)
        pos, quat = np.array(pos), np.array(quat)
        vel, avel_W = p.getBaseVelocity(self.quadrotor_id)
        vel, avel_B = np.array(vel), Rotation.from_quat(quat).inv().apply(np.array(avel_W))
        accel, anaccel = (vel-vel0)/self.simulation_step_period, (avel_B-avel0_B)/self.simulation_step_period

        self.state.header.stamp = self.get_clock().now().to_msg()
        self.state.state.pose.position.x = pos0[0]
        self.state.state.pose.position.y = pos0[1]
        self.state.state.pose.position.z = pos0[2]
        self.state.state.pose.orientation.x = quat0[0]
        self.state.state.pose.orientation.y = quat0[1]
        self.state.state.pose.orientation.z = quat0[2]
        self.state.state.pose.orientation.w = quat0[3]
        self.state.state.twist.linear.x = vel0[0]
        self.state.state.twist.linear.y = vel0[1]
        self.state.state.twist.linear.z = vel0[2]
        self.state.state.twist.angular.x = avel0_B[0]
        self.state.state.twist.angular.y = avel0_B[1]
        self.state.state.twist.angular.z = avel0_B[2]
        self.state.state.accel.linear.x = accel[0]
        self.state.state.accel.linear.y = accel[1]
        self.state.state.accel.linear.z = accel[2]
        self.state.state.accel.angular.x = anaccel[0]
        self.state.state.accel.angular.y = anaccel[1]
        self.state.state.accel.angular.z = anaccel[2]
        self.state.quadrotor_id = self.quadrotor_id

    def inverse_rigid_body_dynamics(self, m, g, J, pos, quat, vel, anvel, accel, anaccel, force_body_frame=True):
        R = Rotation.from_quat(quat)
        F_world = m*(accel) - m*np.array([0, 0, -self.G])
        F_body = R.inv().apply(F_world)
        tau_body = J@anaccel + np.cross(anvel, J@anvel)
        if (force_body_frame):
            return F_body, tau_body
        return F_world, tau_body

    def forward_rigid_body_dynamics(self, m, g, J, pos, quat, vel, anvel, F, tau, force_body_frame=True):
        R = Rotation.from_quat(quat)
        if (force_body_frame):
            F_world = R.apply(F)
        else:
            F_world = F
        accel = F_world/m + np.array([0, 0, -self.G])
        anaccel = np.linalg.inv(J)@(tau - np.cross(anvel, J@anvel))
        return accel, anaccel

    def fill_model_error(self):
        quad_pos = np.array([self.state.state.pose.position.x, self.state.state.pose.position.y, self.state.state.pose.position.z])
        quad_quat = np.array([self.state.state.pose.orientation.x, self.state.state.pose.orientation.y,
                              self.state.state.pose.orientation.z, self.state.state.pose.orientation.w])
        quad_vel = np.array([self.state.state.twist.linear.x, self.state.state.twist.linear.y, self.state.state.twist.linear.z])
        quad_avel = np.array([self.state.state.twist.angular.x, self.state.state.twist.angular.y, self.state.state.twist.angular.z])
        quad_accel = np.array([self.state.state.accel.linear.x, self.state.state.accel.linear.y, self.state.state.accel.linear.z])
        quad_anaccel = np.array([self.state.state.accel.angular.x, self.state.state.accel.angular.y, self.state.state.accel.angular.z])

        ff_pos = np.array([self.ff_state.state.pose.position.x, self.ff_state.state.pose.position.y, self.ff_state.state.pose.position.z])
        ff_quat = np.array([self.ff_state.state.pose.orientation.x, self.ff_state.state.pose.orientation.y,
                            self.ff_state.state.pose.orientation.z, self.ff_state.state.pose.orientation.w])
        ff_vel = np.array([self.ff_state.state.twist.linear.x, self.ff_state.state.twist.linear.y, self.ff_state.state.twist.linear.z])
        ff_avel = np.array([self.ff_state.state.twist.angular.x, self.ff_state.state.twist.angular.y, self.ff_state.state.twist.angular.z])
        ff_accel = np.array([self.ff_state.state.accel.linear.x, self.ff_state.state.accel.linear.y, self.ff_state.state.accel.linear.z])
        ff_anaccel = np.array([self.ff_state.state.accel.angular.x, self.ff_state.state.accel.angular.y, self.ff_state.state.accel.angular.z])

        F_model_body, tau_model = self.inverse_rigid_body_dynamics(self.M, self.G, self.J, quad_pos,
                                                                   quad_quat, quad_vel, quad_avel,
                                                                   quad_accel, quad_anaccel, force_body_frame=True)
        F_ff_body, tau_ff = self.inverse_rigid_body_dynamics(self.M, self.G, self.J, quad_pos,
                                                             quad_quat, quad_vel, quad_avel,
                                                             ff_accel, ff_anaccel, force_body_frame=True)
        F_model_world, tau_model = self.inverse_rigid_body_dynamics(self.M, self.G, self.J, quad_pos,
                                                                    quad_quat, quad_vel, quad_avel,
                                                                    quad_accel, quad_anaccel, force_body_frame=False)
        F_ff_world, tau_ff = self.inverse_rigid_body_dynamics(self.M, self.G, self.J, quad_pos,
                                                              quad_quat, quad_vel, quad_avel,
                                                              ff_accel, ff_anaccel, force_body_frame=False)

        rot = Rotation.from_quat(quad_quat)

        self.model_error.dataset.force_world = np.array(F_ff_world, dtype=np.float32)
        self.model_error.dataset.force_body = np.array(F_ff_body, dtype=np.float32)
        self.model_error.dataset.accel_world = np.array(ff_accel, dtype=np.float32)
        self.model_error.dataset.accel_body = np.array(rot.inv().apply(ff_accel), dtype=np.float32)
        self.model_error.dataset.torque_body = np.array(tau_ff, dtype=np.float32)
        self.model_error.dataset.anaccel_body = np.array(ff_anaccel, dtype=np.float32)
        self.model_error.dataset.position = np.array(quad_pos, dtype=np.float32)

        self.model_error.actual.force_world = np.array(F_model_world, dtype=np.float32)
        self.model_error.actual.force_body = np.array(F_model_body, dtype=np.float32)
        self.model_error.actual.accel_world = np.array(quad_accel, dtype=np.float32)
        self.model_error.actual.accel_body = np.array(rot.inv().apply(quad_accel), dtype=np.float32)
        self.model_error.actual.torque_body = np.array(tau_model, dtype=np.float32)
        self.model_error.actual.anaccel_body = np.array(quad_anaccel, dtype=np.float32)

        self.model_error.error.force_world = np.array(F_model_world - F_ff_world, dtype=np.float32)
        self.model_error.error.force_body = np.array(F_model_body - F_ff_body, dtype=np.float32)
        self.model_error.error.accel_world = np.array(quad_accel - ff_accel, dtype=np.float32)
        self.model_error.error.accel_body = np.array(rot.inv().apply(quad_accel) - rot.inv().apply(ff_accel), dtype=np.float32)
        self.model_error.error.torque_body = np.array(tau_model - tau_ff, dtype=np.float32)
        self.model_error.error.anaccel_body = np.array(quad_anaccel, dtype=np.float32)

    def simulation_step_callback(self):
        # first: apply feed-forward state (if it's enabled)
        # second: calculate forces and apply them
        # third: apply simulation step and calculate state
        # fourth: publish the state
        if (self.use_ff_state):
            self.apply_ff_state()

        self.apply_forces_torques()

        self.apply_simulation_step()

        self.state_publisher.publish(self.state)

        if (self.publish_model_errors and self.use_ff_state):
            self.fill_model_error()
            self.model_error_publisher.publish(self.model_error)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrotorPybulletPhysics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
