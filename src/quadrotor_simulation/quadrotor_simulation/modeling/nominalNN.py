from scipy.spatial.transform import Rotation as R
from .core import State, Accelerations, Rotors, ExternalEffects
from .nominal import nominal_model
from .nets import prepare_tensor, normalize_tensor, denormalize_tensor

import sys


def residualNN_model(state: State, inp: Rotors, net, device, params):

    state_vec = state.as_vector()
    inp_vec = inp.as_vector()[0:4]
    net_inp = prepare_tensor([state_vec.T, inp_vec.T])

    input_mean = params['mean_inputs']
    input_std = params['std_inputs']

    target_mean = params['mean_targets']
    target_std = params['std_targets']

    net_inp = normalize_tensor(net_inp.to(device), input_mean.to(device), input_std.to(device))[0]
    net_inp = net_inp.to(device)

    net_out = net(net_inp)
    net_out = denormalize_tensor(net_out.to(device), target_mean.to(device), target_std.to(device))
    net_out = net_out.detach().cpu().numpy().reshape(6, 1).flatten()

    return ExternalEffects(net_out[0:3], net_out[3:])


def nominalNN_model(state: State, inp: Rotors, net, device,  params):
    acc, ext_eff, B = nominal_model(state, inp, params, return_B=True)

    residuals = residualNN_model(state, inp, net, device, params)

    residuals.force = R.from_quat(state.quat).apply(residuals.force).flatten()

    acc_res = B @ residuals.as_vector()
    acc_res = Accelerations(acc_res[0:3].flatten(), acc_res[3:].flatten())

    acc = acc + acc_res
    ext_eff = ext_eff + residuals

    return acc, ext_eff
