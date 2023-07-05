# Header for timestamp and coordinate frame information
std_msgs/Header header

# Dimensions of the grid (number of cells along each axis)
uint32 width
uint32 height
uint32 depth

# Cell size (distance between adjacent cells in each axis)
# the same as float32 resolution
float32 cell_size

# Origin of the grid (position of the (0, 0, 0) cell in space)
geometry_msgs/Point origin

# Occupancy status of each cell in the grid (row-major order)
# 0 means unoccupied, 1 means occupied, -1 means unknown
uint8[] data
