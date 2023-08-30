# Message Type for the desired quadrotor state, supports receding horizon controllers by providing a list of future states

std_msgs/Header header
quadrotor_interfaces/State current_state
uint32 n # number of future states (for receding horizon controllers)
quadrotor_interfaces/State[] future_states