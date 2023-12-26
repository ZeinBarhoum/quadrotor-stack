from typing import Union, Tuple

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from pytransform3d.plot_utils import Frame
import scipy as sp
from scipy.spatial.transform import Rotation as R


class QuadrotorPhysics:
    """a class for physics simulation of the quadrotor (without physics engine)"""

    def __init__(self,
                 pos: ArrayLike = (0, 0, 0),
                 quat: ArrayLike = (0, 0, 0, 1),
                 vel: ArrayLike = (0, 0, 0),
                 ang_vel: ArrayLike = (0, 0, 0),
                 rotor_speeds=(0, 0, 0, 0),
                 params: Union[dict, None] = None,
                 config: Union[dict, None] = None) -> None:
        """Initializes the quadrotor object
        Args:
            pos (ArrayLike, optional): The initial position of the quadrotor. Defaults to (0, 0, 0).
            quat (ArrayLike, optional): The initial quaternion of the quadrotor. Defaults to (0, 0, 0, 1).
            vel (ArrayLike, optional): The initial velocity of the quadrotor. Defaults to (0, 0, 0).
            ang_vel (ArrayLike, optional): The initial angular velocity of the quadrotor. Defaults to (0, 0, 0).
            rotor_speeds (ArrayLike, optional): The initial rotor speeds of the quadrotor. Defaults to (0, 0, 0, 0).
            params (Union[dict, None], optional): The parameters of the quadrotor. Defaults to None (default parameters). The parameters include the following: G, M, J, ARM_X, ARM_Y, ARM_Z, KF, KM, ROT_DIRS, ROT_TIME_CONST, ROT_MAX_VEL, ROT_MAX_ACC, DRAG_MAT_ROT, DRAG_MAT_FUS_LIN, DRAG_MAT_FUS_ANG. Any parameters that's not defined will be set to the default, see QuadrotorPhysics().get_default_params().
            config (Union[dict, None], optional): The configuration of the quadrotor. Defaults to None (default configuration). The configuration include the following: enable_rotor_dynamics, enable_rotor_drag, enable_fuselage_drag, TIME_STEP. Any configuration that's not defined will be set to the default, see QuadrotorPhysics().get_default_config().

        Returns:
            None
        """
        self.reset(pos=pos,
                   quat=quat,
                   vel=vel,
                   ang_vel=ang_vel,
                   rotor_speeds=rotor_speeds,
                   )
        self.reset_param_config(params=params,
                                config=config)

    def reset_param_config(self,
                           params: Union[dict, None] = None,
                           config: Union[dict, None] = None):
        """Resets the parameters and configuration of the quadrotor.

        Args:
            params (Union[dict, None], optional): The parameters of the quadrotor. Defaults to None (default parameters). The parameters include the following: G, M, J, ARM_X, ARM_Y, ARM_Z, KF, KM, ROT_DIRS, ROT_TIME_CONST, ROT_MAX_VEL, ROT_MAX_ACC, DRAG_MAT_ROT, DRAG_MAT_FUS_LIN, DRAG_MAT_FUS_ANG. Any parameters that's not defined will be set to the default, see QuadrotorPhysics().get_default_params().
            config (Union[dict, None], optional): The configuration of the quadrotor. Defaults to None (default configuration). The configuration include the following: enable_rotor_dynamics, enable_rotor_drag, enable_fuselage_drag, TIME_STEP. Any configuration that's not defined will be set to the default, see QuadrotorPhysics().get_default_config().
        """
        if not params:
            self.params = self.get_default_params()
            print(f'Using default parameters, please specify parameters for your quadrotor, the loaded parameters are {self.params}')
        else:
            self.params = self.get_default_params()
            self.params.update(params)
        if not config:
            self.config = self.get_default_config()
            print(f'Using default configuration, please specify configuration for your quadrotor, the loaded configuration is {self.config}')
        else:
            self.config = self.get_default_config()
            self.config.update(config)

    def reset(self,
              pos: ArrayLike = (0, 0, 0),
              quat: ArrayLike = (0, 0, 0, 1),
              vel: ArrayLike = (0, 0, 0),
              ang_vel: ArrayLike = (0, 0, 0),
              rotor_speeds: ArrayLike = (0, 0, 0, 0),
              ) -> None:
        """Resets the state (pose+twist) and input (rotor_speeds) of the quadrotor. also resets time to 0

        Args:
            pos (ArrayLike, optional): The initial position of the quadrotor. Defaults to (0, 0, 0).
            quat (ArrayLike, optional): The initial quaternion of the quadrotor. Defaults to (0, 0, 0, 1).
            vel (ArrayLike, optional): The initial velocity of the quadrotor. Defaults to (0, 0, 0).
            ang_vel (ArrayLike, optional): The initial angular velocity of the quadrotor. Defaults to (0, 0, 0).
            rotor_speeds (ArrayLike, optional): The initial rotor speeds of the quadrotor. Defaults to (0, 0, 0, 0).
        """
        self.pos = np.array(pos, dtype=np.float32).reshape(3, 1)
        self.quat = np.array(quat, dtype=np.float32).reshape(4, 1)
        self.vel = np.array(vel, dtype=np.float32).reshape(3, 1)
        self.ang_vel = np.array(ang_vel, dtype=np.float32).reshape(3, 1)
        self.rotor_speeds = np.array(rotor_speeds, dtype=np.float32).reshape(4, 1)
        self.forces = np.zeros((3, 1), dtype=np.float32)
        self.torques = np.zeros((3, 1), dtype=np.float32)
        self.acc = np.zeros((3, 1), dtype=np.float32)
        self.ang_acc = np.zeros((3, 1), dtype=np.float32)
        self.wind_speed = np.zeros((3, 1), dtype=np.float32)
        self.time = 0

    def get_default_params(self) -> dict:
        """Returns a dictionary of default parameters for the quadrotor. These are used if no parameters are specified."""
        params = {
            'G': 9.81,
            'M': 1.0,
            'J': np.eye(3),
            'ARM_X': 1.0,
            'ARM_Y': 1.0,
            'ARM_Z': 1.0,
            'KF': 1.0,
            'KM': 1.0,
            'ROT_DIRS': [1, -1, 1, -1],
            'ROT_TIME_CONST': 1.0,
            'ROT_MAX_VEL': 1.0,
            'ROT_MAX_ACC': 1.0,
            'DRAG_MAT_ROT': None,
            'DRAG_MAT_FUS_LIN': None,
            'DRAG_MAT_FUS_ANG': None, }
        return params

    def get_default_config(self) -> dict:
        """Returns a dictionary of default configuration for the quadrotor. These are used if no configuration is specified."""
        config = {
            'enable_rotor_dynamics': False,
            'enable_rotor_drag': False,
            'enable_fuselage_drag': False,
            'TIME_STEP': 0.01,
        }
        return config

    def update_rotor_speeds(self, desired_rotor_speeds: ArrayLike):
        """Updates the rotor speeds of the quadrotor. Uses rotor dynamics if enabled in the configuration.

        Args:
            desired_rotor_speeds (ArrayLike): The desired rotor speeds. Must be of length 4.
        """
        desired_rotor_speeds = np.array(desired_rotor_speeds).reshape(4)
        if self.config.get('enable_rotor_dynamics'):
            for i in range(4):
                des_rot_accel = (desired_rotor_speeds[i] - self.rotor_speeds[i])/self.params['ROT_TIME_CONST']
                if (self.rotor_speeds[i] <= 0 or self.rotor_speeds[i] >= self.params['ROT_MAX_VEL']):
                    des_rot_accel = 0
                des_rot_accel = np.clip(des_rot_accel, -self.params['ROT_MAX_ACC'], self.params['ROT_MAX_ACC'])
                self.rotor_speeds[i] += des_rot_accel*self.config['TIME_STEP']
        else:
            self.rotor_speeds = desired_rotor_speeds

    def apply_forward_rigid_body_dynamics(self,
                                          residuals_accelerations: ArrayLike = (0, 0, 0, 0, 0, 0),
                                          ) -> None:
        """Calculates the accelerations and angular accelerations of the quadrotor using the forces and torques applied.
        Parameters[M, J] must be set in the params dictionary.

        Args:
            residuals_accelerations(ArrayLike, optional): The residuals of accelerations(world frame) and angular accelerations(body frame). Defaults to(0, 0, 0, 0, 0, 0).

        Returns:
            Does not return anything, but updates the accelerations and angular accelerations of the quadrotor.
        """
        self.acc = np.array([0, 0, -self.params['G']]).reshape(3, 1) + self.forces/self.params['M']
        self.acc += np.array(residuals_accelerations)[:3].reshape(3, 1)

        J = np.array(self.params['J']).reshape(3, 3)
        self.ang_acc = np.linalg.inv(J) @ (self.torques - np.cross(self.ang_vel, J @ self.ang_vel, axis=0))
        self.ang_acc += np.array(residuals_accelerations)[3:].reshape(3, 1)

    def apply_forward_wrench_dynamics(self,
                                      residuals_forces_torques: ArrayLike = (0, 0, 0, 0, 0, 0),
                                      ):
        """Calculates the forces and torques applied on the quadrotor from the rotor speeds.

        Args:
            residuals_forces_torques(ArrayLike, optional): The residuals of forces(world frame) and torques(body frame). Defaults to(0, 0, 0, 0, 0, 0).
        Returns:
            Does not return anything, but updates the forces and torques on the quadrotor.
        """
        forces_world, torques_body = self.cal_rot_eff_quadratic_model()
        if self.config.get('enable_rotor_drag'):
            rotor_drag_forces_world, rotor_drag_torques_body = self.calc_aero_rotor_drag()
            forces_world += rotor_drag_forces_world
            torques_body += rotor_drag_torques_body
        if self.config.get('enable_fuselage_drag'):
            fus_drag_forces_world, fus_drag_torques_body = self.calc_aero_fus_drag()
            forces_world += fus_drag_forces_world
            torques_body += fus_drag_torques_body

        forces_world += np.array(residuals_forces_torques)[:3].reshape(3, 1)
        torques_body += np.array(residuals_forces_torques)[3:].reshape(3, 1)

        self.forces = forces_world
        self.torques = torques_body

    def cal_rot_eff_quadratic_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the forces and torques generated by the rotors using the quadratic model.
        Assumes X-configuration of the rotors, rotor 1 is the front right, others follow in counter-clockwise order.
        Requires the rotor speeds to be set before calling this function.
        Parameters[KF, KM, ROT_DIRS, ARM_X, ARM_Y] must be set in the params dictionary.

        Returns:
            forces_world(np.ndarray): The forces generated by the rotors in world frame.
            torques_body(np.ndarray): The torques generated by the rotors in body frame.
        """
        def calculate_rotor_thrusts():
            """Calculates the thrust generated by all four rotors."""
            return np.array(self.params['KF']*self.rotor_speeds**2).reshape(4)

        def calculate_rotor_torques():
            """Calculates the torques generated by all four rotors."""
            return np.array(-self.params['ROT_DIRS']*self.params['KM']*self.rotor_speeds**2).reshape(4)

        rotor_thrusts = calculate_rotor_thrusts()
        rotor_torques = calculate_rotor_torques()

        torque_x = self.params['ARM_Y'] * (-rotor_thrusts[0] + rotor_thrusts[1] + rotor_thrusts[2] - rotor_thrusts[3])
        torque_y = self.params['ARM_X'] * (-rotor_thrusts[0] - rotor_thrusts[1] + rotor_thrusts[2] + rotor_thrusts[3])
        torque_z = np.sum(rotor_torques)
        torques_body = np.array([torque_x, torque_y, torque_z]).reshape(3, 1)

        forces_x = 0
        forces_y = 0
        forces_z = np.sum(rotor_thrusts)
        forces_body = np.array([forces_x, forces_y, forces_z]).reshape(3, 1)
        forces_world = R.from_quat(self.quat).apply(forces_body)

        return forces_world, torques_body

    def calc_aero_rotor_drag(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the forces and torques due to the rotor drag effect.
        Parameter DRAG_MAT_ROT must be set in the params dictionary.

        Returns:
            forces_world(np.ndarray): The forces generated by the rotors in world frame.
            torques_body(np.ndarray): The torques generated by the rotors in body frame.
        """
        D = np.array(self.params['DRAG_MAT_ROT']).reshape(3, 3)
        wind_speed = np.array(self.wind_speed).reshape(3, 1)
        wind_vel_body = R.from_quat(self.quat).inv().apply(wind_speed)
        vel_body = R.from_quat(self.quat).inv().apply(self.vel)
        rel_vel_body = vel_body - wind_vel_body
        drag_body = -D @ rel_vel_body

        forces_world = R.from_quat(self.quat).apply(drag_body)
        torques_body = np.zeros((3, 1))
        return forces_world, torques_body

    def calc_aero_fus_drag(self, wind_speed: ArrayLike = (0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the forces and torques due to the fuselage drag effect.
        Parameters DRAG_MAT_FUS_LIN and DRAG_MAT_FUS_ANG must be set in the params dictionary.

        Args:
            wind_speed(ArrayLike): The wind speed vector in world frame. Defaults to(0, 0, 0).

        Returns:
            forces_world(np.ndarray): The forces generated by the rotors in world frame.
            torques_body(np.ndarray): The torques generated by the rotors in body frame.
        """
        D_lin = np.array(self.params['DRAG_MAT_FUS_LIN']).reshape(3, 3)
        D_ang = np.array(self.params['DRAG_MAT_FUS_ANG']).reshape(3, 3)
        wind_speed = np.array(wind_speed).reshape(3, 1)
        wind_vel_body = R.from_quat(self.quat).inv().apply(wind_speed)
        vel_body = R.from_quat(self.quat).inv().apply(self.vel)
        rel_vel_body = vel_body - wind_vel_body
        drag_lin_body = -D_lin @ rel_vel_body * np.linalg.norm(rel_vel_body)
        drag_ang_body = -D_ang @ self.ang_vel * np.linalg.norm(self.ang_vel)

        forces_world = R.from_quat(self.quat).apply(drag_lin_body)
        torques_body = drag_ang_body
        return forces_world, torques_body

    def set_wind_speed(self,
                       wind_speed: ArrayLike = (0, 0, 0),
                       ) -> None:
        """Sets the wind speed of the quadrotor.
        Args:
            wind_speed(ArrayLike, optional): The wind speed vector in world frame. Defaults to(0, 0, 0).

        Returns:
            None

        """
        self.wind_speed = np.array(wind_speed).reshape(3, 1)

    def set_wrench(self,
                   forces: ArrayLike = (0, 0, 0),
                   torques: ArrayLike = (0, 0, 0),
                   ) -> None:
        """Sets the forces and torques applied on the quadrotor.
        Args:
            forces(ArrayLike, optional): The forces applied on the quadrotor. Defaults to(0, 0, 0).
            torques(ArrayLike, optional): The torques applied on the quadrotor. Defaults to(0, 0, 0).

        Returns:
            None
        """
        self.forces = np.array(forces).reshape(3, 1)
        self.torques = np.array(torques).reshape(3, 1)

    def set_accelerations(self,
                          acc: ArrayLike = (0, 0, 0),
                          ang_acc: ArrayLike = (0, 0, 0),
                          ) -> None:
        """Sets the accelerations and angular accelerations of the quadrotor.
        Args:
            acc(ArrayLike, optional): The accelerations of the quadrotor. Defaults to(0, 0, 0).
            ang_acc(ArrayLike, optional): The angular accelerations of the quadrotor. Defaults to(0, 0, 0).

        Returns:
            None
        """
        self.acc = np.array(acc).reshape(3, 1)
        self.ang_acc = np.array(ang_acc).reshape(3, 1)

    def set_rotor_speeds(self,
                         rotor_speeds: ArrayLike = (0, 0, 0, 0),
                         ) -> None:
        """Sets the rotor speeds of the quadrotor.
        Args:
            rotor_speeds(ArrayLike, optional): The rotor speeds of the quadrotor. Defaults to(0, 0, 0, 0).
        """
        self.rotor_speeds = np.array(rotor_speeds).reshape(4, 1)

    def set_state(self,
                  pos: Union[ArrayLike, None] = None,
                  quat: Union[ArrayLike, None] = None,
                  vel: Union[ArrayLike, None] = None,
                  ang_vel: Union[ArrayLike, None] = None,
                  ) -> None:
        """set the state (or any part of it) of the quadrotor.
        Args:
            pos(Union[ArrayLike, None], optional): The position of the quadrotor. Defaults to None.
            quat(Union[ArrayLike, None], optional): The quaternion of the quadrotor. Defaults to None.
            vel(Union[ArrayLike, None], optional): The velocity of the quadrotor. Defaults to None.
            ang_vel(Union[ArrayLike, None], optional): The angular velocity of the quadrotor. Defaults to None.

        Returns:
            None
        """
        if pos is not None:
            self.pos = np.array(pos).reshape(3, 1)
        if quat is not None:
            self.quat = np.array(quat).reshape(4, 1)
        if vel is not None:
            self.vel = np.array(vel).reshape(3, 1)
        if ang_vel is not None:
            self.ang_vel = np.array(ang_vel).reshape(3, 1)

    def update_state_euler_integration(self) -> None:
        """Updates the state and derivative of the state of the quadrotor using Euler integration.

        Returns:
            None
        """

        def quat_dot_Omega(ang_vel: ArrayLike):
            """Calculates the Omega matrix used to calculate derivative of quaternion.
            return Omega where dot(quat) = 0.5*Omega@quat

            Args:
                ang_vel(ArrayLike): The angular velocity of the quadrotor.
            """
            ang_vel = np.array(ang_vel).reshape(3)
            skew_ang_vel = np.array([[0, -ang_vel[2], ang_vel[1]],
                                     [ang_vel[2], 0, -ang_vel[0]],
                                     [-ang_vel[1], ang_vel[0], 0]])
            return np.block([[-skew_ang_vel, ang_vel], [-ang_vel.T, 0]]).reshape(4, 4)

        def quat_dot_Omega_efficient(ang_vel: ArrayLike):
            """Calculates the Omega matrix used to calculate derivative of quaternion.
            return Omega where dot(quat) = 0.5*Omega@quat

            Args:
                ang_vel(ArrayLike): The angular velocity of the quadrotor.
            """
            ang_vel = np.array(ang_vel).reshape(3)
            return np.array([[0, ang_vel[2], -ang_vel[1], ang_vel[0]],
                             [-ang_vel[2], 0, ang_vel[0], ang_vel[1]],
                             [ang_vel[1], -ang_vel[0], 0, ang_vel[2]],
                             [-ang_vel[0], -ang_vel[1], -ang_vel[2], 0]])

        def integrate_quat_method1(quat, ang_vel, dt):
            return sp.linalg.expm(0.5*quat_dot_Omega_efficient(ang_vel)*dt)@quat

        def integrate_quat_method2(quat, ang_vel, dt):
            w = ang_vel
            w_norm = np.linalg.norm(w)
            w = np.array(w).reshape(3)
            qv = w*np.sin(w_norm*dt/2)/w_norm
            qw = np.cos(w_norm*dt/2)
            qr = np.array([qv[0], qv[1], qv[2], qw])
            q = quat
            q = np.array([qr[3]*q[0] + qr[0]*q[3] + qr[1]*q[2] - qr[2]*q[1],
                          qr[3]*q[1] - qr[0]*q[2] + qr[1]*q[3] + qr[2]*q[0],
                          qr[3]*q[2] + qr[0]*q[1] - qr[1]*q[0] + qr[2]*q[3],
                          qr[3]*q[3] - qr[0]*q[0] - qr[1]*q[1] - qr[2]*q[2]])
            q = q/np.linalg.norm(q)
            return q
        pos_prev = self.pos
        vel_prev = self.vel
        quat_prev = self.quat
        ang_vel_prev = self.ang_vel

        self.pos += self.vel*self.config['TIME_STEP']
        self.vel += self.acc*self.config['TIME_STEP']
        self.quat = integrate_quat_method1(self.quat, self.ang_vel, self.config['TIME_STEP'])
        self.ang_vel += self.ang_acc*self.config['TIME_STEP']

        self.pos = np.array(self.pos).reshape(3, 1)
        self.vel = np.array(self.vel).reshape(3, 1)
        self.quat = np.array(self.quat).reshape(4, 1)
        self.ang_vel = np.array(self.ang_vel).reshape(3, 1)

        self.time += self.config['TIME_STEP']

        dpos = self.pos - pos_prev
        dvel = self.vel - vel_prev
        dquat = self.quat - quat_prev
        dang_vel = self.ang_vel - ang_vel_prev

        self.pos_dot = dpos/self.config['TIME_STEP']
        self.vel_dot = dvel/self.config['TIME_STEP']
        self.quat_dot = dquat/self.config['TIME_STEP']
        self.ang_vel_dot = dang_vel/self.config['TIME_STEP']

    def get_time(self):
        return self.time

    def get_state(self):
        return np.vstack([self.pos, self.quat, self.vel, self.ang_vel])

    def get_wrench(self):
        return np.vstack([self.forces, self.torques])

    def get_accelerations(self):
        return np.vstack([self.acc, self.ang_acc])

    def pretty_repr(self):
        def vector_to_string(v):
            def sign_str(a):
                if (a >= 0):
                    return ' '
                else:
                    return '-'
            s = ''
            for i in np.array(v).flatten():
                s = f'{s} {sign_str(i)}{abs(i):.2f}'
            return s
        s = "----------------\n"
        s = s + f"Time :   {self.get_time():.3f}\n"
        s = s + f"Pos  : {vector_to_string(self.pos)}\n"
        s = s + f"Quat : {vector_to_string(self.quat)}\n"
        s = s + f"Vel  : {vector_to_string(self.vel)}\n"
        s = s + f"AVel : {vector_to_string(self.ang_vel)}\n"
        s = s + f"ACC  : {vector_to_string(self.acc)}\n"
        s = s + f"AACC : {vector_to_string(self.ang_acc)}\n"
        s = s + f"ROT  : {vector_to_string(self.rotor_speeds)}\n"
        s = s + f"FOR  : {vector_to_string(self.forces)}\n"
        s = s + f"TOR  : {vector_to_string(self.torques)}\n"
        s = s + f"DPOS : {vector_to_string(self.pos_dot)}\n"
        s = s + f"DQUAT: {vector_to_string(self.quat_dot)}\n"
        s = s + f"DVEL : {vector_to_string(self.vel_dot)}\n"
        s = s + f"DAVEL: {vector_to_string(self.ang_vel_dot)}\n"
        return s

    def get_hover_rotor_speed(self):
        return np.sqrt(self.params['M']*self.params['G']/(4*self.params['KF']))

    def get_T_matrix(self):
        return np.block([[R.from_quat(self.quat.flatten()).as_matrix(), self.pos], [np.zeros((1, 3)), 1]])


def main():
    plt.ion()
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    quad = QuadrotorPhysics(params={'M': 2.0, 'J': [[1, 0, 0], [0, 2, 0], [0, 0, 3]]}, config={'TIME_STEP': 0.1})

    tau = [0, 0, 2]
    force = [0, 0, quad.params['M']*quad.params['G']]
    for _ in range(10):
        quad.set_wrench(force, tau)
        quad.apply_forward_rigid_body_dynamics()
        quad.update_state_euler_integration()
        frame = Frame(quad.get_T_matrix(), label="rotating frame", s=0.5)
        frame.add_frame(ax)
        print(quad.pretty_repr())

    input('Press Enter To End')


if __name__ == '__main__':
    main()
