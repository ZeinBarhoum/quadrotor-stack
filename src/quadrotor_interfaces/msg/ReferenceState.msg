# Message Type for the desired quadrotor state, supports receding horizon controllers by providing a list of future states

std_msgs/Header header

quadrotor_interfaces/StateData current_state

uint32 n # number of future states (for receding horizon controllers)
float32 dt # time between future states
quadrotor_interfaces/StateData[] future_states