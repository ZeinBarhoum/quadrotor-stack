import rclpy
from rclpy.clock import Clock
from rclpy.time import Time
from rclpy.duration import Duration


class CustomClock(Clock):
    def __init__(self):
        super().__init__()
        self._time_factor = 1
        self._start_time = None  # Initialize _start_time to None

    def now(self):
        if self._start_time is None:
            # If _start_time is not initialized, set it to the current time
            self._start_time = super().now()

        current_time = super().now()
        elapsed_time = (current_time - self._start_time).nanoseconds
        simulated_elapsed_time = int(elapsed_time / self._time_factor)
        simulated_time = Time(nanoseconds=simulated_elapsed_time)
        return simulated_time

    def set_time_factor(self, time_factor):
        self._time_factor = time_factor

    def get_time_factor(self):
        return self._time_factor

    def reset(self):
        self._start_time = self.now()

    def advance_by(self, nanoseconds):
        self._start_time += Time(nanoseconds=nanoseconds)
