import numpy as np
from scipy.spatial.transform import Rotation as R
from .core import State, Accelerations, Rotors, ExternalEffects


def rbd_model(state: State, params):
    J = params['J']
    G = params['G']

    acc_world = np.array([0, 0, -G])
    ang_acc_body = np.linalg.inv(J) @ (-np.cross(state.ang_vel, J @ state.ang_vel))

    return Accelerations(acc_world, ang_acc_body)


def nominal_external_effects_model(state: State, inp: Rotors, params):
    """ Compute external effects for the nominal model

    Args:
        state (State): current state
        inp (Rotors): current input
        params (dict): parameters of the model

    Returns:
        ExternalEffects: external effects of the model

    """
    def nominal_rotor_model(inp: Rotors, params):
        KF = params['KF']
        KM = params['KM']

        rotor_thrusts = KF*(inp.mot**2)
        rotor_torques = KM*(inp.mot**2)

        return rotor_thrusts, rotor_torques

    rotor_thrusts, rotor_torques = nominal_rotor_model(inp, params)

    ARM_X = params['ARM_X']
    ARM_Y = params['ARM_Y']
    DIRS = params['DIRS']

    torque_z = -(DIRS[0]*rotor_torques[0] + DIRS[1]*rotor_torques[1] +
                 DIRS[2]*rotor_torques[2] + DIRS[3]*rotor_torques[3])
    torque_x = ARM_Y * (-rotor_thrusts[0] + rotor_thrusts[1] + rotor_thrusts[2] - rotor_thrusts[3])
    torque_y = ARM_X * (-rotor_thrusts[0] - rotor_thrusts[1] + rotor_thrusts[2] + rotor_thrusts[3])

    forces_body = np.array([0, 0, sum(rotor_thrusts)])

    forces_world = R.from_quat(state.quat).apply(forces_body)

    torques_body = np.array([torque_x, torque_y, torque_z])

    ext_eff = ExternalEffects(forces_world, torques_body)

    return ext_eff


def nominal_model(state: State, inp: Rotors, params, return_B=False):
    """ Compute accelerations using the nominal model
    Uses the formula: f(x,u) = f_rbd(x) + Bf_W(x,u)
    Args:
        state (State): current state
        inp (Rotors): current input
        params (dict): parameters of the model

    Returns:
        Accelerations: accelerations of the model
        ExternalEffects: external effects of the model
    """
    M = params['M']
    J = params['J']

    acc_rbd = rbd_model(state, params)

    B = np.block([[np.eye(3)/M, np.zeros((3, 3))],
                  [np.zeros((3, 3)), np.linalg.inv(J)]])

    ext_eff = nominal_external_effects_model(state, inp, params)

    acc_ext = B @ ext_eff.as_vector()
    acc_ext = Accelerations(acc_ext[:3].flatten(), acc_ext[3:].flatten())

    acc = acc_rbd + acc_ext
    if (return_B):
        return acc, ext_eff, B
    return acc, ext_eff


def inverse_nominal_model(state: State, acc: Accelerations, params):
    M = params['M']
    J = params['J']

    acc_rbd = rbd_model(state, params)

    B = np.block([[np.eye(3)/M, np.zeros((3, 3))],
                  [np.zeros((3, 3)), np.linalg.inv(J)]])

    acc_ext = acc - acc_rbd

    ext_eff = np.linalg.inv(B)@acc_ext.as_vector()
    ext_eff = ExternalEffects(ext_eff[:3].flatten(), ext_eff[3:].flatten())
    # TODO: return input as well

    return ext_eff, Rotors()
