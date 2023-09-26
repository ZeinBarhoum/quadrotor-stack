from quadrotor_trajectory_generation.trajectory_optimizer import PolynomialTrajectoryOptimizer
import numpy as np
import pytest
import time
from timeit import timeit

successes = 0
total = 0


def test_solve_one_segment(poly_order=10):
    global successes, total

    rng = np.random.default_rng()

    high_order_constraints = rng.integers(0, min(4, (poly_order+1)//2))
    n = poly_order + 1 - 2 * high_order_constraints

    waypoints = rng.random(n)*10
    times = rng.random(n)*10
    times.sort()

    optimizer = PolynomialTrajectoryOptimizer(one_segment_optimization=True,
                                              high_order_constraints=high_order_constraints,
                                              added_poly_order=0,  # no optimization then, just solving Ax=b
                                              )
    try:
        poly = optimizer._solve_one_segment_2(waypoints, times)

        # check positions
        for (i, t) in enumerate(times):
            assert np.polyval(poly, t) == pytest.approx(waypoints[i], abs=0.5)
        # check higher order derivatives at start and end
        for der_order in range(1, optimizer.high_order_constraints+1):
            # check derivatives
            assert np.polyval(np.polyder(poly, der_order), times[0]) == pytest.approx(0, abs=0.1)
            assert np.polyval(np.polyder(poly, der_order), times[-1]) == pytest.approx(0, abs=0.1)
        successes += 1
    except Exception as e:
        pass

    total += 1


for order in range(4, 20):
    number = 1000
    time = timeit(lambda: test_solve_one_segment(order), number=number)
    print(f"Testing one segment optimization with {order}th order polynomial")
    print(f"Time for {number} runs: {time} seconds, {time/number*1000:.2f} ms per run")
    print(f"{successes}/{total} tests passed with success rate {successes/total*100:.2f}%")
    print()
    successes = 0
    total = 0

# times = [0, 1, 2, 3]
# waypoints = [1, 2, 3, 4]
# optimizer = PolynomialTrajectoryOptimizer(added_poly_order=0, high_order_constraints=2, one_segment_optimization=True)
# optimizer._solve_one_segment_2(waypoints, times)
