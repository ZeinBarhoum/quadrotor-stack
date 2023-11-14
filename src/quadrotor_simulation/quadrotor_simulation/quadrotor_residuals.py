import numpy as np
from quadrotor_interfaces.msg import RotorCommand, State
import torch
from torch import nn
from quadrotor_simulation.modeling.nets import get_MLP
from quadrotor_simulation.modeling.core import State as CoreState
from quadrotor_simulation.modeling.core import Rotors as CoreRotorCommand
from quadrotor_simulation.modeling.nominalNN import residualNN_model

from glob import glob
from ament_index_python.packages import get_package_share_directory


WEIGHTS_PATH = get_package_share_directory('quadrotor_simulation')+'/models/'


def prepare_residuals_model(model):

    net_params = torch.load(WEIGHTS_PATH + model + '_params.pt')
    hyper_params = net_params['hyper_params']
    net = get_MLP(hyper_params)
    net.load_state_dict(torch.load(WEIGHTS_PATH + model + '_dict.pt'))

    return net, net_params


def calculate_residuals(state: State, rotor_command: RotorCommand, net, device, params):
    state_core = CoreState(pos=np.array([state.state.pose.position.x, state.state.pose.position.y, state.state.pose.position.z]),
                           quat=np.array([state.state.pose.orientation.x, state.state.pose.orientation.y,
                                         state.state.pose.orientation.z, state.state.pose.orientation.w]),
                           vel=np.array([state.state.twist.linear.x, state.state.twist.linear.y, state.state.twist.linear.z]),
                           ang_vel=np.array([state.state.twist.angular.x, state.state.twist.angular.y, state.state.twist.angular.z]))
    rotor_command_core = CoreRotorCommand(mot=np.array(
        [rotor_command.rotor_speeds[0], rotor_command.rotor_speeds[1], rotor_command.rotor_speeds[2], rotor_command.rotor_speeds[3]]))
    residuals = residualNN_model(state_core, rotor_command_core, net, device, params).as_vector()
    return residuals.flatten()
