""" Module for trajectory optimization. specific to quadrotor trajectory generation but can be used for other applications as well."""
import numpy as np
import scipy as sp
import sympy as sym
import casadi as ca
import math
import cvxpy as cp
import numpy.typing as npt
from typing import List, Tuple, Union, Optional

from pprint import pprint

DEFAULT_HIGH_ORDER_CONSTRAINTS = 2
DEFAULT_ADDED_POLY_ORDER = 2


class PolynomialTrajectoryOptimizer:
    """ Class for polynomial trajectory optimization """

    def __init__(self,
                 high_order_constraints=DEFAULT_HIGH_ORDER_CONSTRAINTS,
                 added_poly_order=DEFAULT_ADDED_POLY_ORDER,
                 one_segment_optimization=False,
                 ):
        """ Constructor 
        Args:
            high_order_constraints: Highest orders for constraints on terminal(fixed) and middle waypoints(continuity) (1 for velocity constraints, 2 for acc. and so on) to optimize for
            added_poly_order: Order of polynomial to add to the trajectory in addition to the ammount necessery to satisfy the number of constraints
            one_segment_optimization: If true, the resultant trajectory will be a single polynomial segment. If false, the trajectory will be a concatenation of multiple segments.
        Raises:
            ValueError: If high_order_constraints or added_poly_order is negative
        """
        if high_order_constraints < 0:
            raise ValueError("high_order_constraintsmust be non-negative")
        if added_poly_order < 0:
            raise ValueError("added_poly_order must be non-negative")
        self.high_order_constraints = high_order_constraints
        self.added_poly_order = added_poly_order
        self.one_segment_optimization = one_segment_optimization

    def _convert_waypoint_times_to_np(self, waypoints: npt.ArrayLike, times: npt.ArrayLike) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        waypoints = np.array(waypoints, dtype=np.float64)
        times = np.array(times, dtype=np.float64)
        if waypoints.ndim != 1:
            raise NotImplementedError("Only 1D waypoints are supported")
        if len(waypoints) != len(times):
            raise ValueError("Number of waypoints and times must be equal")
        if len(waypoints) < 2:
            raise ValueError("Need at least two waypoints")
        return waypoints, times

    def optimize(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)
        if (self.added_poly_order == 0):  # no optimization then, just solving Ax=b
            if (self.one_segment_optimization):
                return self._solve_one_segment(waypoints, times)
            else:
                return self._solve_multiple_segments(waypoints, times)
        else:
            if (self.one_segment_optimization):
                return self._optimize_one_segment(waypoints, times)
            else:
                return self._optimize_multiple_segments(waypoints, times)

    def _solve_one_segment(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)

        num_waypoints = len(waypoints)
        num_constraints = num_waypoints + 2 * self.high_order_constraints
        num_params = num_constraints  # no optimization

        A = np.zeros((num_constraints, num_params), dtype=np.float64)

        for i in range(num_waypoints):  # filling the position constraints
            A[i, :] = np.array([times[i]**j for j in range(num_params)])

        # filling the higher order constraints (velocity, acc, etc.) for the first and last waypoints
        for der_order in range(1, self.high_order_constraints+1):
            A[num_waypoints+der_order-1, :] = np.array([math.perm(j, der_order)*times[0]**(j-der_order) if j >= der_order else 0 for j in range(num_params)])
            A[-der_order, :] = np.array([math.perm(j, der_order)*times[-1]**(j-der_order) if j >= der_order else 0 for j in range(num_params)])

        b = np.zeros(num_constraints)
        b[:num_waypoints] = waypoints

        poly = np.linalg.solve(A, b)

        return list(reversed(poly))

    def _process_constraint(self,
                            poly_times: npt.ArrayLike,
                            T: float,
                            direction: str,
                            n_coef_per_poly: npt.ArrayLike = [0],
                            der_order: int = 0,
                            b_value: int = 0,
                            ) -> Tuple[np.ndarray, float]:
        r: Union[Tuple[int], Tuplep[int, int]]  # the index of the specific polynomial in case of exact constraint or two indices for continuity constraints
        for i in range(len(poly_times)):
            if (T == poly_times[i]):
                if (direction == '-'):
                    r = (i-1,)
                elif (direction == '+'):
                    r = (i,)
                elif (direction == '+-'):
                    if (i-1 < 0 or i >= len(poly_times)-1):
                        raise ValueError('continuity constraints only applicable on middle waypoints')
                    r = (i-1, i)
                else:
                    raise ValueError('direction should be +, - or +-')
            else:
                if (i != len(poly_times) - 1):
                    if (T > poly_times[i] and T < poly_times[i+1]):
                        if (direction in ['+', '-']):
                            r = (i,)
                        elif (direction == '+-'):
                            raise ValueError('continuity constraints only applicable on middle waypoints')
                        else:
                            raise ValueError('direction should be +, - or +-')
        a = np.zeros(sum(n_coef_per_poly))  # total number of coefficients
        for i in r:
            a_i = np.array([math.perm(j, der_order)*T**(j-der_order) if j >= der_order else 0 for j in range(n_coef_per_poly[i])])
            a[sum(n_coef_per_poly[:i]): sum(n_coef_per_poly[:i+1])] = a_i

        b = b_value
        if (direction == '+-'):
            b = 0
        return a, b

    def _solve_one_segment_2(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)

        num_waypoints = len(waypoints)
        num_constraints = num_waypoints + 2 * self.high_order_constraints
        num_params = num_constraints  # no optimization

        A = np.zeros((num_constraints, num_params))
        b = np.zeros(num_constraints)
        n_coef_per_poly = [num_params]
        pc = 0  # processed constraints
        for i in range(len(waypoints)):
            direction = '-'
            if (i == 0):
                direction = '+'
            A[pc, :], b[pc] = self._process_constraint(poly_times=[times[0], times[-1]],
                                                       T=times[i],
                                                       direction=direction,
                                                       n_coef_per_poly=n_coef_per_poly,
                                                       der_order=0,
                                                       b_value=waypoints[i])
            pc += 1
        for der_order in range(1, self.high_order_constraints+1):
            A[pc, :], b[pc] = self._process_constraint(poly_times=[times[0], times[-1]],
                                                       T=times[0],
                                                       direction='+',
                                                       n_coef_per_poly=n_coef_per_poly,
                                                       der_order=der_order,
                                                       b_value=0)
            pc += 1
            A[pc, :], b[pc] = self._process_constraint(poly_times=[times[0], times[-1]],
                                                       T=times[-1],
                                                       direction='-',
                                                       n_coef_per_poly=n_coef_per_poly,
                                                       der_order=der_order,
                                                       b_value=0)
            pc += 1

        poly = np.linalg.solve(A, b)

        return list(reversed(poly))

    def _solve_one_segment_sym(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)

        num_waypoints = len(waypoints)
        num_constraints = num_waypoints + 2 * self.high_order_constraints
        num_params = num_constraints  # no optimization

        coeffs = sym.symbols(f'c:{num_params}')
        t = sym.symbols('t')
        poly = 0
        for i in range(num_params):
            poly += coeffs[i] * t**i

        A = np.zeros((num_constraints, num_params))
        current_row = 0
        for i in range(num_waypoints):
            constraint = poly.subs(t, times[i])
            A[current_row, :] = sym.derive_by_array(constraint, coeffs)
            current_row += 1

        for der_order in range(1, self.high_order_constraints+1):
            constraint = sym.diff(poly, t, der_order).subs(t, times[0])
            A[current_row, :] = sym.derive_by_array(constraint, coeffs)
            current_row += 1
            constraint = sym.diff(poly, t, der_order).subs(t, times[-1])
            A[current_row, :] = sym.derive_by_array(constraint, coeffs)
            current_row += 1

        b = np.zeros(num_constraints)
        b[:num_waypoints] = waypoints

        poly = np.linalg.solve(A, b)

        return list(reversed(poly))
