""" Module for trajectory optimization. specific to quadrotor trajectory generation but can be used for other applications as well."""
import numpy as np
import scipy as sp
import sympy as sym
import casadi as ca
import math
import cvxpy as cp
import numpy.typing as npt
from typing import List, Tuple, Union, Optional
from qpsolvers import solve_qp
import time
from pprint import pprint

DEFAULT_HIGH_ORDER_CONSTRAINTS = 2
DEFAULT_ADDED_POLY_ORDER = 2


class PolynomialTrajectoryOptimizer:
    """ Class for polynomial trajectory optimization """

    def __init__(self,
                 high_order_constraints=DEFAULT_HIGH_ORDER_CONSTRAINTS,
                 added_poly_order=DEFAULT_ADDED_POLY_ORDER,
                 one_segment_optimization=False,
                 qp_solver='cvxopt',
                 kl: npt.ArrayLike = (0, 0, 0, 0, 1),
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
        self.qp_solver = qp_solver
        self.kl = kl

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

    def _analyze_multi_segment_problem(self, num_waypoints):
        num_waypoints_mid = num_waypoints - 2
        num_waypoints_terminal = 2
        # types of polynomials: terminal, middle
        num_polys = num_waypoints - 1
        num_polys_mid = max(num_waypoints - 3, 0)
        num_polys_terminal = num_polys - num_polys_mid

        # check issue #36 for explanation on polynomial orders
        poly_order_mid = self.high_order_constraints + 1
        poly_order_terminal = (poly_order_mid * 3 - 1) // 2
        if (num_waypoints == 2):
            poly_order_terminal = 2 * poly_order_mid - 1
        # number of parameters
        num_params_per_poly_mid = poly_order_mid + 1 + self.added_poly_order
        num_params_per_poly_terminal = poly_order_terminal + 1 + self.added_poly_order

        # redundant for now
        num_constraints_per_waypoint_mid = 2 + self.high_order_constraints
        num_constraints_per_waypoint_terminal = 1 + self.high_order_constraints

        num_params = num_params_per_poly_mid * num_polys_mid + num_params_per_poly_terminal * num_polys_terminal
        num_constraints = num_constraints_per_waypoint_mid * num_waypoints_mid + num_constraints_per_waypoint_terminal * num_waypoints_terminal
        # print(f"{num_params = }, {num_constraints = }")
        n_coef_per_poly = [num_params_per_poly_terminal if (i == 0 or i == num_polys-1) else num_params_per_poly_mid for i in range(num_polys)]

        result = {'num_waypoints': num_waypoints,
                  'num_waypoints_mid': num_waypoints_mid,
                  'num_waypoints_terminal': num_waypoints_terminal,
                  'num_polys': num_polys,
                  'num_polys_mid': num_polys_mid,
                  'num_polys_terminal': num_polys_terminal,
                  'poly_order_mid': poly_order_mid,
                  'poly_order_terminal': poly_order_terminal,
                  'num_params_per_poly_mid': num_params_per_poly_mid,
                  'num_params_per_poly_terminal': num_params_per_poly_terminal,
                  'num_constraints_per_waypoint_mid': num_constraints_per_waypoint_mid,
                  'num_constraints_per_waypoint_terminal': num_constraints_per_waypoint_terminal,
                  'num_params': num_params,
                  'num_constraints': num_constraints,
                  'n_coef_per_poly': n_coef_per_poly,
                  }
        return result

    def _constraints_multi_segment_problem(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)
        parameters = self._analyze_multi_segment_problem(len(waypoints))
        A = np.zeros((parameters['num_constraints'], parameters['num_params']))
        b = np.zeros(parameters['num_constraints'])
        # print(f"{n_coef_per_poly = }")
        pc = 0  # processed constraints
        for i in range(len(waypoints)):
            # print(i)
            if (i != 0):
                A[pc, :], b[pc] = self._process_constraint(poly_times=times,
                                                           T=times[i],
                                                           direction='-',
                                                           n_coef_per_poly=parameters['n_coef_per_poly'],
                                                           der_order=0,
                                                           b_value=waypoints[i])
                pc += 1
            if (i != len(waypoints) - 1):
                A[pc, :], b[pc] = self._process_constraint(poly_times=times,
                                                           T=times[i],
                                                           direction='+',
                                                           n_coef_per_poly=parameters['n_coef_per_poly'],
                                                           der_order=0,
                                                           b_value=waypoints[i])
                pc += 1

            for der_order in range(1, self.high_order_constraints+1):
                direction = '+-'
                if (i == 0):
                    direction = '+'
                if (i == parameters['num_waypoints']-1):
                    direction = '-'
                A[pc, :], b[pc] = self._process_constraint(poly_times=times,
                                                           T=times[i],
                                                           direction=direction,
                                                           n_coef_per_poly=parameters['n_coef_per_poly'],
                                                           der_order=der_order,
                                                           b_value=0)
                pc += 1
        return A, b

    def _objective_multi_segment_problem_sym(self, times: npt.ArrayLike, kl: npt.ArrayLike, parameters: dict):
        num_polys = parameters['num_polys']
        num_params_per_poly_mid = parameters['num_params_per_poly_mid']
        num_params_per_poly_terminal = parameters['num_params_per_poly_terminal']
        t = sym.symbols('t')
        coeffs = []
        coeffs_flat = []
        polys = []
        # create symbols
        for i in range(num_polys):
            if (i == 0 or i == num_polys-1):
                coeffs.append(sym.symbols(f'c{i}_:{num_params_per_poly_terminal}'))
            else:
                coeffs.append(sym.symbols(f'c{i}_:{num_params_per_poly_mid}'))
            coeffs_flat.extend(coeffs[i])
        # create polynomials
        for i in range(num_polys):
            poly = 0
            for j in range(len(coeffs[i])):
                poly += coeffs[i][j] * t**j
            polys.append(poly)
        # create objective function
        obj = 0
        for i in range(num_polys):
            for (l, k_l) in enumerate(kl):
                if (k_l == 0):
                    continue
                obj += k_l * sym.integrate(sym.diff(polys[i], t, l) ** 2, (t, times[i], times[i+1]))
        # calculate the matrix H
        H = sym.hessian(obj, coeffs_flat)
        H = np.array(H).astype(np.float64)

        g = np.zeros(len(coeffs_flat))
        return H, g

    def _process_constraint(self,
                            poly_times: npt.ArrayLike,
                            T: float,
                            direction: str,
                            n_coef_per_poly: npt.ArrayLike = [0],
                            der_order: int = 0,
                            b_value: int = 0,
                            ) -> Tuple[np.ndarray, float]:
        """Processes a single constraints of the forms:
        1. der((traj(t), t->T+),der_order) [=,>,<] b_value / for exact conditions like fixed position (waypoints) and zero derivatives (terminal). Applies on the trajectory segment that starts from T (or passes through it)
        2. der((traj(t), t->T-),der_order) [=,>,<] b_value / the same as above but applies on the trajectory segment that ends at T (or passes through it)
        3. der((traj(t), t->T-),der_order) - der((traj(t), t->T+),der_order) [=,>,<] b_value / for continuity constraints. Applies on the trajectory segment that ends at T and the one that starts from T

        Args:
            poly_times (npt.ArrayLike): The times of the start and end of the polynomial segment. for example [1,2,3] means that there are two polynomial segment, one from 1s to 2s and the other from 2s to 3s.
            T (float): The time at which the constraint is applied
            direction (str): The direction of the constraint. Can be '+', '-' or '+-' for above cases 1,2 and 3 respectively.
            n_coef_per_poly (npt.ArrayLike, optional): Number of coefficients for each polynomial segment in the tajectory. Defaults to [0].
            der_order (int, optional): the derivative order of the constraint (0 for position, 1 for velocity etc.). Defaults to 0.
            b_value (int, optional): the right hand side of the constraint. Defaults to 0.

        Raises:
            ValueError: If direction is not one of '+', '-' or '+-'
            ValueError: If direction is '+-' and the constraint can not be a continuity constraint (for example if T is the first or last waypoint or T lies on the middle of a polynomial segment)

        Returns:
            Tuple[np.ndarray, float]: The vector a and scalar b that satisfies dot(a,c) = b where c is the flattened vector of coefficients of all polynomial segment
        """
        r: Union[Tuple[int], Tuplep[int, int]]  # the index of the specific polynomial in case of exact constraint or two indices for continuity constraints
        for i in range(len(poly_times)):
            if (T == poly_times[i]):
                if (direction == '-'):
                    if (i == 0):
                        raise ValueError("exact constraints from the negative side doesn't apply for the first waypoint")
                    r = (i-1,)
                elif (direction == '+'):
                    if (i == len(poly_times) - 1):
                        raise ValueError("exact constraints from the positive side doesn't apply for the last waypoint")
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
        # print(r)
        a = np.zeros(sum(n_coef_per_poly))  # total number of coefficients
        for (i, r_i) in enumerate(r):
            a_i = np.array([math.perm(j, der_order)*T**(j-der_order) if j >= der_order else 0 for j in range(n_coef_per_poly[r_i])])
            a[sum(n_coef_per_poly[:r_i]): sum(n_coef_per_poly[:r_i+1])] = a_i if i == 0 else -a_i

        b = b_value
        return a, b

    def _solve_one_segment_old(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        """(old) finds a one-segment trajectory that passes through the given waypoints at the given times."""
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

    def _solve_one_segment(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        """(new, fast) implementation of _solve_one_segment using the _process_constraint method to find A, b"""
        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)

        num_waypoints = len(waypoints)
        num_constraints = num_waypoints + 2 * self.high_order_constraints
        num_params = num_constraints  # no optimization

        A = np.zeros((num_constraints, num_params))
        b = np.zeros(num_constraints)

        n_coef_per_poly = [num_params]  # only one polynomial segment with num_params coefficients

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
        """(new, slow) implementation of _solve_one_segment using sympy to derive A, b"""
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

    def _solve_multiple_segments(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)
        parameters = self._analyze_multi_segment_problem(len(waypoints))
        A, b = self._constraints_multi_segment_problem(waypoints, times)
        polys = np.linalg.solve(A, b)
        segments = []
        for i in range(parameters['num_polys'],):
            segments.append(list(reversed(polys[sum(parameters['n_coef_per_poly'][:i]): sum(parameters['n_coef_per_poly'][:i+1])])))
        return segments

    def _solve_multiple_segments_old(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        # this function construct the matrices A,B that satisfy the equation Ax = B where x is the vector of unknown parameters
        # the vector x is the concatenation of the parameters of all the polynomials in order exept for the first polynomial which is
        # forms the last parameters and not the first (for easier construction of A)

        def get_poly_param_indices(poly_index: int, num_params_per_poly_mid: int, num_params_per_poly_terminal: int, num_polys: int) -> slice:
            if (poly_index == 0):
                return slice(-num_params_per_poly_terminal, None)
            elif (poly_index == num_polys-1):
                return slice(-2*num_params_per_poly_terminal, -num_params_per_poly_terminal)
            else:
                return slice((poly_index-1)*num_params_per_poly_mid, poly_index*num_params_per_poly_mid)

        def get_poly_der_values(poly_index: int, num_params_per_poly_mid: int, num_params_per_poly_terminal: int, der_order: int, t: float) -> np.ndarray:
            num_params = num_params_per_poly_mid if poly_index not in [0, num_polys - 1] else num_params_per_poly_terminal

            return np.array([math.perm(j, der_order) * (t**(j-der_order)) if j >= der_order else 0 for j in range(num_params)])

        def add_constraint_to_A_B(A: np.ndarray,
                                  B: np.ndarray,
                                  t: float,
                                  b_value: float,
                                  constraint_index: int,
                                  constraint_der: int,
                                  poly_index: int,
                                  num_params_per_poly_mid: int,
                                  num_params_per_poly_terminal: int,
                                  num_polys: int,
                                  sign_a_values: int = 1,
                                  ) -> Tuple[np.ndarray, np.ndarray]:
            indices = get_poly_param_indices(poly_index, num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
            # self.get_logger().info(f'{indices=}')
            A_values = get_poly_der_values(poly_index, num_params_per_poly_mid, num_params_per_poly_terminal, constraint_der, t)
            # self.get_logger().info(f'{A_values=}')
            A[constraint_index, indices] = A_values * sign_a_values
            B[constraint_index] = b_value
            return A, B

        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)
        num_waypoints = len(waypoints)
        # types of polynomials: terminal, middle
        num_polys = num_waypoints - 1
        num_polys_mid = max(len(waypoints) - 3, 0)
        num_polys_terminal = num_polys - num_polys_mid

        # check issue #36 for explanation on polynomial orders
        poly_order_mid = self.high_order_constraints + 1
        poly_order_terminal = (poly_order_mid * 3 - 1) // 2

        # number of parameters
        num_params_per_poly_mid = poly_order_mid + 1
        num_params_per_poly_terminal = poly_order_terminal + 1
        # self.get_logger().info(f"{num_params_per_poly_mid=}, {num_params_per_poly_terminal=}")

        # redundant for now
        num_constraints_per_poly_mid = poly_order_mid + 1
        num_constraints_per_poly_terminal = poly_order_terminal + 1

        num_params = num_params_per_poly_mid * num_polys_mid + num_params_per_poly_terminal * num_polys_terminal
        num_constraints = num_constraints_per_poly_mid * num_polys_mid + num_constraints_per_poly_terminal * num_polys_terminal

        A = np.zeros((num_constraints, num_params))
        B = np.zeros(num_constraints)

        done_constraints = 0

        # add constraints of the first waypoint
        for der_order in range(self.high_order_constraints + 1):
            b_value = 0
            if (der_order == 0):
                b_value = waypoints[0]
            A, B = add_constraint_to_A_B(A, B, times[0], b_value, done_constraints, der_order, 0,
                                         num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
            done_constraints += 1
        if not done_constraints == num_constraints:  # to check one segment trajectories (2 waypoints)
            # add constraints of the last waypoint
            for der_order in range(self.high_order_constraints + 1):
                b_value = 0
                if (der_order == 0):
                    b_value = waypoints[-1]
                A, B = add_constraint_to_A_B(A, B, times[-1], b_value, done_constraints, der_order, num_polys -
                                             1, num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
                done_constraints += 1

        # add constraints of the middle waypoints
        for i in range(1, num_waypoints - 1):
            # add position constraint for two polynomials
            A, B = add_constraint_to_A_B(A, B, times[i], waypoints[i], done_constraints, 0, i - 1,
                                         num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
            done_constraints += 1
            A, B = add_constraint_to_A_B(A, B, times[i], waypoints[i], done_constraints, 0, i,
                                         num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
            done_constraints += 1
            for der_order in range(1, self.high_order_constraints+1):
                # add higher derivatives constraints
                A, B = add_constraint_to_A_B(A, B, times[i], 0, done_constraints, der_order, i - 1,
                                             num_params_per_poly_mid, num_params_per_poly_terminal, num_polys)
                A, B = add_constraint_to_A_B(A, B, times[i], 0, done_constraints, der_order, i,
                                             num_params_per_poly_mid, num_params_per_poly_terminal, num_polys, sign_a_values=-1)
                done_constraints += 1

        # self.get_logger().info(f'{A=}')
        # self.get_logger().info(f'{B=}')
        sol_prams = np.linalg.lstsq(A, B, rcond=None)[0]  # not accurate
        # self.get_logger().info(f'{sol_prams=}')
        segments = []
        for i in range(num_polys):
            if (i == 0):
                segments.append(list(reversed(sol_prams[-num_params_per_poly_terminal:])))
            elif (i == num_polys-1):
                segments.append(list(reversed(sol_prams[-2*num_params_per_poly_terminal:-num_params_per_poly_terminal])))
            else:
                segments.append(list(reversed(sol_prams[(i-1)*num_params_per_poly_mid:i*num_params_per_poly_mid])))
        # self.get_logger().info(f'{segments=}')
        return segments

    def _optimize_multiple_segments(self, waypoints: npt.ArrayLike, times: npt.ArrayLike):
        ts = []
        ts.append(time.time())
        waypoints, times = self._convert_waypoint_times_to_np(waypoints, times)
        ts.append(time.time())  # 0
        parameters = self._analyze_multi_segment_problem(len(waypoints))
        ts.append(time.time())  # 1
        A, b = self._constraints_multi_segment_problem(waypoints, times)
        ts.append(time.time())  # 2
        # P, q = self._objective_multi_segment_problem_sym(times, kl, parameters)
        H, g = self._objective_multi_segment_problem_sym(times, self.kl, parameters)
        ts.append(time.time())  # 3
        # polys = solve_qp(P=P, q=q, A=A, b=b, solver=self.qp_solver)

        A = ca.DM(A)
        b = ca.DM(b)
        H = ca.DM(H)

        qp = {}

        qp['h'] = H.sparsity()
        qp['a'] = A.sparsity()
        opts = {}
        opts['printLevel'] = 'none'
        # opts['error_on_fail'] = False
        S = ca.conic('S', 'qpoases', qp, opts)

        r = S(h=H, a=A, lba=b, uba=b)

        sol_params = r['x']
        polys = np.array(sol_params).reshape(-1)
        ts.append(time.time())  # 4

        segments = []
        for i in range(parameters['num_polys'],):
            segments.append(list(reversed(polys[sum(parameters['n_coef_per_poly'][:i]): sum(parameters['n_coef_per_poly'][:i+1])])))
        ts.append(time.time())  # 5
        print(f"{dict(enumerate(np.diff(ts)))}")
        return segments
