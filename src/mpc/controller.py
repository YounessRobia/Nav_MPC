"""Main MPC controller implementation."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np

from ..dynamics.uav_state import UAVState, UAVControl
from ..dynamics.point_mass_model import PointMassModel
from ..obstacles.base import Obstacle
from ..obstacles.dynamic import DynamicObstacle
from ..utils.config_loader import MPCConfig, UAVConfig
from .constraints import (
    StateBounds,
    ControlBounds,
    ObstacleConstraints,
    create_state_bounds_from_config,
    create_control_bounds_from_config,
)
from .solver_interface import create_ipopt_options


class SolverStatus(Enum):
    """MPC solver status."""

    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"
    INFEASIBLE = "infeasible"
    MAX_ITER = "max_iterations"
    ERROR = "error"


@dataclass
class MPCSolution:
    """Container for MPC solution data."""

    x_trajectory: np.ndarray  # (N+1, nx) predicted states
    u_trajectory: np.ndarray  # (N, nu) optimal controls
    cost: float  # Optimal cost value
    solve_time: float  # Computation time in seconds
    status: SolverStatus  # Solver status
    iterations: int  # Solver iterations

    @property
    def success(self) -> bool:
        """Check if solution is usable."""
        return self.status in [SolverStatus.OPTIMAL, SolverStatus.SUBOPTIMAL]


class MPCController:
    """
    Model Predictive Controller for UAV path planning.

    Uses CasADi + IPOPT for nonlinear optimization with:
    - Triple integrator dynamics (jerk control)
    - State and control constraints
    - Obstacle avoidance (soft constraints)
    - Reference tracking cost
    """

    # Dimensions
    NX = 10  # State: [x, y, z, vx, vy, vz, ax, ay, az, yaw]
    NU = 4  # Control: [jx, jy, jz, yaw_rate]

    def __init__(
        self,
        mpc_config: MPCConfig,
        uav_config: UAVConfig,
    ):
        """
        Initialize MPC controller.

        Args:
            mpc_config: MPC parameters (horizon, weights, etc.)
            uav_config: UAV constraints (velocity limits, etc.)
        """
        self.config = mpc_config
        self.uav_config = uav_config

        # Horizon parameters
        self.N = mpc_config.horizon
        self.dt = mpc_config.dt

        # Dynamics model
        self.dynamics = PointMassModel(dt=self.dt)
        self.Ad, self.Bd = self.dynamics.get_discrete_matrices()

        # Bounds
        self.state_bounds = create_state_bounds_from_config(uav_config)
        self.control_bounds = create_control_bounds_from_config(uav_config)

        # Cost matrices
        self.Q = mpc_config.get_Q_matrix()
        self.R = mpc_config.get_R_matrix()
        self.Qf = mpc_config.get_Qf_matrix()

        # Obstacle constraints
        self.obstacle_constraints = ObstacleConstraints(
            weight=mpc_config.obstacle_weight,
            use_soft_constraints=True,
        )

        # Setup optimization problem
        self._setup_problem()

        # Warm start storage
        self._prev_solution: Optional[MPCSolution] = None

    def _setup_problem(self):
        """Set up the CasADi optimization problem."""
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(self.NX, self.N + 1)  # States
        self.U = self.opti.variable(self.NU, self.N)  # Controls

        # Parameters
        self.x0_param = self.opti.parameter(self.NX)  # Initial state
        self.x_ref_param = self.opti.parameter(self.NX, self.N + 1)  # Reference

        # Obstacle parameters (positions over horizon)
        # We'll handle these dynamically in solve()

        # Dynamics constraints
        for k in range(self.N):
            x_next = self.Ad @ self.X[:, k] + self.Bd @ self.U[:, k]
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # Initial state constraint
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        # State bounds
        x_min, x_max = self.state_bounds.to_arrays()
        for k in range(self.N + 1):
            self.opti.subject_to(self.opti.bounded(x_min, self.X[:, k], x_max))

        # Control bounds
        u_min, u_max = self.control_bounds.to_arrays()
        for k in range(self.N):
            self.opti.subject_to(self.opti.bounded(u_min, self.U[:, k], u_max))

        # Base cost function (without obstacles - added in solve())
        self._setup_base_cost()

        # Configure solver
        solver_opts = create_ipopt_options(
            max_iter=self.config.solver.max_iter,
            tolerance=self.config.solver.tolerance,
            print_level=self.config.solver.print_level,
            warm_start=self.config.solver.warm_start,
        )
        self.opti.solver("ipopt", solver_opts)

    def _setup_base_cost(self):
        """Set up base cost function (tracking + control effort)."""
        cost = 0

        # Stage costs
        for k in range(self.N):
            # Tracking cost
            x_err = self.X[:, k] - self.x_ref_param[:, k]
            cost += ca.mtimes([x_err.T, self.Q, x_err])

            # Control effort cost
            cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])

        # Terminal cost
        x_err_N = self.X[:, self.N] - self.x_ref_param[:, self.N]
        cost += ca.mtimes([x_err_N.T, self.Qf, x_err_N])

        self.base_cost = cost

    def solve(
        self,
        state: UAVState,
        reference: np.ndarray,
        static_obstacles: List[Obstacle] = None,
        dynamic_obstacles: List[DynamicObstacle] = None,
        t_current: float = 0.0,
    ) -> MPCSolution:
        """
        Solve MPC optimization problem.

        Args:
            state: Current UAV state
            reference: Reference trajectory (N+1, nx) or goal state (nx,)
            static_obstacles: List of static obstacles
            dynamic_obstacles: List of dynamic obstacles with known trajectories
            t_current: Current simulation time

        Returns:
            MPCSolution with optimal trajectory and controls
        """
        static_obstacles = static_obstacles or []
        dynamic_obstacles = dynamic_obstacles or []

        start_time = time.time()

        # Set initial state
        x0 = state.to_mpc_state()
        self.opti.set_value(self.x0_param, x0)

        # Set reference trajectory
        if reference.ndim == 1:
            # Constant reference (goal state)
            x_ref = np.tile(reference.reshape(-1, 1), (1, self.N + 1))
        else:
            x_ref = reference.T  # Transpose to (nx, N+1)
        self.opti.set_value(self.x_ref_param, x_ref)

        # Compute obstacle cost
        obstacle_cost = self._compute_obstacle_cost(
            static_obstacles, dynamic_obstacles, t_current
        )

        # Total cost
        total_cost = self.base_cost + obstacle_cost
        self.opti.minimize(total_cost)

        # Warm start from previous solution
        self._apply_warm_start(x0)

        # Solve
        try:
            sol = self.opti.solve()
            status = SolverStatus.OPTIMAL
            iterations = sol.stats()["iter_count"]
        except RuntimeError as e:
            sol = self.opti.debug
            if "Maximum_Iterations_Exceeded" in str(e):
                status = SolverStatus.MAX_ITER
            elif "Infeasible" in str(e):
                status = SolverStatus.INFEASIBLE
            else:
                status = SolverStatus.ERROR
            iterations = -1

        solve_time = time.time() - start_time

        # Extract solution
        try:
            X_opt = sol.value(self.X).T  # (N+1, nx)
            U_opt = sol.value(self.U).T  # (N, nu)
            cost_opt = float(sol.value(total_cost))
        except Exception:
            # Return previous solution or zero
            if self._prev_solution is not None:
                return self._prev_solution
            X_opt = np.tile(x0, (self.N + 1, 1))
            U_opt = np.zeros((self.N, self.NU))
            cost_opt = float("inf")

        solution = MPCSolution(
            x_trajectory=X_opt,
            u_trajectory=U_opt,
            cost=cost_opt,
            solve_time=solve_time,
            status=status,
            iterations=iterations,
        )

        # Store for warm start
        if solution.success:
            self._prev_solution = solution

        return solution

    def _compute_obstacle_cost(
        self,
        static_obstacles: List[Obstacle],
        dynamic_obstacles: List[DynamicObstacle],
        t0: float,
    ) -> ca.SX:
        """
        Compute obstacle avoidance cost over horizon.

        Args:
            static_obstacles: Static obstacles
            dynamic_obstacles: Dynamic obstacles
            t0: Current time

        Returns:
            CasADi expression for obstacle cost
        """
        cost = 0  # Will be promoted to MX when combined with Opti variables
        weight = self.config.obstacle_weight

        for k in range(self.N + 1):
            position = self.X[:3, k]  # [x, y, z]
            t = t0 + k * self.dt

            # Static obstacles
            for obs in static_obstacles:
                # Use proper signed distance for the obstacle type
                try:
                    signed_dist = obs.signed_distance_casadi(position)
                    # Soft-plus barrier on negative signed distance
                    # signed_dist < 0 means inside obstacle
                    scale = 1.0  # Activation scaling
                    violation = -signed_dist  # Positive when inside
                    soft_violation = scale * ca.log(1 + ca.exp(violation / scale))
                    cost += weight * soft_violation
                except (NotImplementedError, AttributeError):
                    # Fallback to sphere approximation
                    center = obs.get_center()
                    safe_radius = obs.get_bounding_radius()
                    diff = position - center
                    dist_sq = ca.dot(diff, diff)
                    scale = safe_radius**2 * 0.5
                    violation = safe_radius**2 - dist_sq
                    soft_violation = scale * ca.log(1 + ca.exp(violation / scale))
                    cost += weight * soft_violation

            # Dynamic obstacles (evaluate at predicted time)
            for obs in dynamic_obstacles:
                obs_pos = obs.get_position_at_time(t)
                safe_radius = obs.get_bounding_radius()

                diff = position - obs_pos
                dist_sq = ca.dot(diff, diff)

                scale = safe_radius**2 * 0.5
                violation = safe_radius**2 - dist_sq
                soft_violation = scale * ca.log(1 + ca.exp(violation / scale))
                cost += weight * soft_violation

        return cost

    def _apply_warm_start(self, x0: np.ndarray):
        """Apply warm start from previous solution."""
        if self._prev_solution is not None and self._prev_solution.success:
            # Shift previous trajectory
            X_init = np.zeros((self.NX, self.N + 1))
            X_init[:, :-1] = self._prev_solution.x_trajectory[1:].T
            X_init[:, -1] = self._prev_solution.x_trajectory[-1]

            U_init = np.zeros((self.NU, self.N))
            U_init[:, :-1] = self._prev_solution.u_trajectory[1:].T
            U_init[:, -1] = self._prev_solution.u_trajectory[-1]

            # Override first state with actual current state
            X_init[:, 0] = x0

            self.opti.set_initial(self.X, X_init)
            self.opti.set_initial(self.U, U_init)
        else:
            # Initialize with constant state/zero control
            self.opti.set_initial(self.X, np.tile(x0.reshape(-1, 1), (1, self.N + 1)))
            self.opti.set_initial(self.U, np.zeros((self.NU, self.N)))

    def get_control(self, solution: MPCSolution) -> UAVControl:
        """
        Extract first control from MPC solution.

        Args:
            solution: MPC solution

        Returns:
            UAVControl for current timestep
        """
        if solution.success:
            u0 = solution.u_trajectory[0]
        else:
            u0 = np.zeros(self.NU)
        return UAVControl.from_array(u0)

    def get_predicted_trajectory(self, solution: MPCSolution) -> np.ndarray:
        """
        Get predicted position trajectory from solution.

        Args:
            solution: MPC solution

        Returns:
            (N+1, 3) array of predicted positions
        """
        return solution.x_trajectory[:, :3]

    def reset_warm_start(self):
        """Reset warm start (e.g., after large disturbance)."""
        self._prev_solution = None


def create_mpc_controller(
    mpc_config_path: str = None,
    uav_config_path: str = None,
) -> MPCController:
    """
    Factory function to create MPC controller from config files.

    Args:
        mpc_config_path: Path to MPC config YAML
        uav_config_path: Path to UAV config YAML

    Returns:
        Configured MPCController instance
    """
    from ..utils.config_loader import load_config, get_default_config_paths

    if mpc_config_path is None or uav_config_path is None:
        default_uav, default_mpc = get_default_config_paths()
        uav_config_path = uav_config_path or default_uav
        mpc_config_path = mpc_config_path or default_mpc

    uav_config, mpc_config, _ = load_config(uav_config_path, mpc_config_path)
    return MPCController(mpc_config, uav_config)
