import sys
import math
from quadrotor_trajectory_generation.trajectory_optimizer import PolynomialTrajectoryOptimizer
import numpy as np
import pytest
import time
from timeit import timeit
from pprint import pprint
import sympy as sym
from sympy import init_printing
import matplotlib.pyplot as plt

init_printing()
successes = 0
total = 0

example = {}


def test_one_segment(waypoints, times, poly, high_order_constraints):
    # check positions
    for (i, t) in enumerate(times):
        assert np.polyval(poly, t) == pytest.approx(waypoints[i], abs=0.5)
    # check higher order derivatives at start and end
    for der_order in range(1, high_order_constraints+1):
        # check derivatives
        assert np.polyval(np.polyder(poly, der_order), times[0]) == pytest.approx(0, abs=0.1)
        assert np.polyval(np.polyder(poly, der_order), times[-1]) == pytest.approx(0, abs=0.1)


def test_multiple_segment(waypoints, times, segments, high_order_constraints):
    # check positions
    for (i, t) in enumerate(times):
        if i != 0:
            assert np.polyval(segments[i-1], t) == pytest.approx(waypoints[i], abs=0.5)
        if i != len(times) - 1:
            assert np.polyval(segments[i], t) == pytest.approx(waypoints[i], abs=0.5)

        for der_order in range(1, high_order_constraints+1):
            # check derivatives
            if (i == 0):
                assert np.polyval(np.polyder(segments[i], der_order), times[0]) == pytest.approx(0, abs=0.1)
            if (i == len(times)-1):
                assert np.polyval(np.polyder(segments[i-1], der_order), times[-1]) == pytest.approx(0, abs=0.1)
            if (i != 0 and i != len(times)-1):
                assert np.polyval(np.polyder(segments[i-1], der_order), times[i]) == pytest.approx(
                    np.polyval(np.polyder(segments[i], der_order), times[i]), abs=0.1)


def test_solve_multi_segment_variable_waypoints(n=10, tries=1, added_poly_order=0):
    global successes, total

    rng = np.random.default_rng()

    # high_order_constraints = rng.choice([2, 4])
    high_order_constraints = 4
    waypoints = rng.random(n)*10
    times = rng.random(n-1)*10
    times.sort()
    times = np.insert(times, 0, 0)
    # if (min(np.diff(times)) < 0.1):
    #     times = rng.random(n)*10
    #     times.sort()

    optimizer = PolynomialTrajectoryOptimizer(one_segment_optimization=False,
                                              high_order_constraints=high_order_constraints,
                                              added_poly_order=added_poly_order,  # no optimization then, just solving Ax=b
                                              kl=[0, 0, 1],
                                              )
    for i in range(tries):
        try:
            poly = optimizer.optimize(waypoints, times)
            test_multiple_segment(waypoints, times, poly, optimizer.high_order_constraints)
            successes += 1
            break
        except (AssertionError, np.linalg.LinAlgError, RuntimeError) as e:
            example['waypoints'] = waypoints
            example['times'] = times
            continue

    total += 1


def test_solve_one_segment_variable_waypoints(n=10, tries=1, added_poly_order=0):
    global successes, total

    rng = np.random.default_rng()

    high_order_constraints = rng.integers(0, 4, endpoint=True)

    waypoints = rng.random(n)*10
    times = rng.random(n)*10
    times.sort()

    optimizer = PolynomialTrajectoryOptimizer(one_segment_optimization=True,
                                              high_order_constraints=high_order_constraints,
                                              added_poly_order=added_poly_order,  # no optimization then, just solving Ax=b
                                              )
    for i in range(tries):
        try:
            poly = optimizer.optimize(waypoints, times)
            test_one_segment(waypoints, times, poly, optimizer.high_order_constraints)
            successes += 1
            break
        except (AssertionError, np.linalg.LinAlgError) as e:
            continue

    total += 1


def test_solve_one_segment_variable_order(poly_order=10, tries=1, added_poly_order=0):
    global successes, total

    rng = np.random.default_rng()

    high_order_constraints = rng.integers(0, min(4, (poly_order-1)/2), endpoint=True)
    n = poly_order + 1 - 2 * high_order_constraints
    if (n < 2):
        print('YA')
    waypoints = rng.random(n)*10
    times = rng.random(n)*10
    times.sort()

    optimizer = PolynomialTrajectoryOptimizer(one_segment_optimization=True,
                                              high_order_constraints=high_order_constraints,
                                              added_poly_order=added_poly_order,  # no optimization then, just solving Ax=b
                                              )
    for i in range(tries):
        try:
            poly = optimizer.optimize(waypoints, times)
            test_one_segment(waypoints, times, poly, optimizer.high_order_constraints)
            successes += 1
            break
        except (AssertionError, np.linalg.LinAlgError) as e:
            continue

    total += 1


def _test_time_sr(fun, start, end, name, number=1000, tries=1, added_poly_order=0):
    global total, successes
    for value in range(start, end):
        stdout = sys.stdout
        # with open(r'/home/zein/temp/test.log', 'w') as sys.stdout:
        time = timeit(lambda: fun(value, tries, added_poly_order), number=number)
        sys.stdout = stdout
        print(f"{name} : {value}")
        print(f"Time for {number} runs: {time} seconds, {time/number*1000:.2f} ms per run")
        print(f"{successes}/{total} tests passed with success rate {successes/total*100:.2f}%")
        print()
        successes = 0
        total = 0


# _test_time_sr(test_solve_one_segment_variable_order, 4, 20, "poly_order")
# _test_time_sr(test_solve_one_segment_variable_waypoints, 2, 20, "num_waypoints")
# _test_time_sr(test_solve_multi_segment_variable_waypoints, 4,  5, "num_waypoints", added_poly_order=1, number=1, tries=1)
rng = np.random.default_rng()

# high_order_constraints = rng.choice([2, 4])
n = 15
high_order_constraints = 4
waypoints = rng.random(n)*10
times = rng.random(n-1)*10
waypoints.sort()
times.sort()
times = np.insert(times, 0, 0)

print(f"{waypoints=}")
print(f"{times=}")
# waypoints = [1, 4, -5, 10]
# times = [0, 3, 5, 10]
optimzer = PolynomialTrajectoryOptimizer(one_segment_optimization=False, high_order_constraints=high_order_constraints, added_poly_order=1, kl=[0, 0, 1])
segments = optimzer.optimize(waypoints, times)


# waypoints = [-0.7, 0.1]
# times = [1, 2]

# optimizer = PolynomialTrajectoryOptimizer(one_segment_optimization=False, high_order_constraints=2, added_poly_order=1)
# segments = optimizer.optimize(waypoints, times)
# print(segments)
# test_multiple_segment(waypoints, times, segments, 2)
# pprint(params)
# H1 = optimizer._optimize_multiple_segments(waypoints, times)
# kl = [1, 0.1, 4, 1, 2]
# r = len(kl) - 1
# g = np.zeros(params['n_coef_per_poly'][0])
# for j in range(len(g)):
#     val = 0
#     for l in range(min(j, r)+1):
#         P_l_j = math.perm(j, l)
#         val += kl[l] * P_l_j * (times[1]**(j-l+1) - times[0]**(j-l+1))/(j-l+1) if j != l-1 else 0.0
#     g[j] = val
# H = (g.reshape(-1, 1)@g.reshape(1, -1))
# pprint(g)
# pprint(H1)
# segments = optimizer._solve_multiple_segments(waypoints, times)
# print(segments)

# test_multiple_segment(waypoints, times, segments, optimizer.high_order_constraints)
# print(example)
