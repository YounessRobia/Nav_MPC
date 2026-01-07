"""Constraint handling for MPC."""

import casadi as ca
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..obstacles.base import Obstacle
from ..obstacles.dynamic import DynamicObstacle


@dataclass
class StateBounds:
    """State bound constraints."""

    # Position bounds
    x_min: float = -np.inf
    x_max: float = np.inf
    y_min: float = -np.inf
    y_max: float = np.inf
    z_min: float = 5.0   # Minimum altitude [m]
    z_max: float = 25.0  # Maximum altitude [m]

    # Velocity bounds
    v_max: float = 10.0  # Max speed per axis

    # Acceleration bounds
    a_max: float = 5.0  # Max acceleration per axis

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to min/max arrays for 10D state.

        State: [x, y, z, vx, vy, vz, ax, ay, az, yaw]

        Returns:
            Tuple of (x_min, x_max) arrays
        """
        x_min = np.array(
            [
                self.x_min,
                self.y_min,
                self.z_min,
                -self.v_max,
                -self.v_max,
                -self.v_max,
                -self.a_max,
                -self.a_max,
                -self.a_max,
                -np.pi,
            ]
        )
        x_max = np.array(
            [
                self.x_max,
                self.y_max,
                self.z_max,
                self.v_max,
                self.v_max,
                self.v_max,
                self.a_max,
                self.a_max,
                self.a_max,
                np.pi,
            ]
        )
        return x_min, x_max


@dataclass
class ControlBounds:
    """Control bound constraints."""

    # Jerk bounds
    j_max: float = 30.0  # Max jerk per axis [m/s^3]

    # Yaw rate bounds
    yaw_rate_max: float = 1.5  # Max yaw rate [rad/s]

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to min/max arrays for 4D control.

        Control: [jx, jy, jz, yaw_rate]

        Returns:
            Tuple of (u_min, u_max) arrays
        """
        u_min = np.array(
            [-self.j_max, -self.j_max, -self.j_max, -self.yaw_rate_max]
        )
        u_max = np.array(
            [self.j_max, self.j_max, self.j_max, self.yaw_rate_max]
        )
        return u_min, u_max


class ObstacleConstraints:
    """
    Obstacle avoidance constraints for MPC.

    Supports both hard constraints and soft constraints (barrier functions).
    """

    def __init__(
        self,
        weight: float = 1000.0,
        use_soft_constraints: bool = True,
    ):
        """
        Initialize obstacle constraints.

        Args:
            weight: Penalty weight for soft constraints
            use_soft_constraints: If True, use barrier functions instead of hard constraints
        """
        self.weight = weight
        self.use_soft_constraints = use_soft_constraints

    def add_static_obstacle_cost(
        self,
        position: ca.SX,
        obstacle: Obstacle,
    ) -> ca.SX:
        """
        Add soft constraint cost for static obstacle.

        Args:
            position: 3D position (symbolic) [x, y, z]
            obstacle: Obstacle instance

        Returns:
            CasADi cost expression
        """
        center = obstacle.get_center()
        safe_radius = obstacle.get_bounding_radius()

        # Squared distance
        diff = position - center
        dist_sq = ca.dot(diff, diff)
        safe_radius_sq = safe_radius**2

        # Quadratic barrier: penalize when distance < safe_radius
        violation = ca.fmax(0, safe_radius_sq - dist_sq)
        return self.weight * violation**2

    def add_dynamic_obstacle_cost(
        self,
        position: ca.SX,
        obstacle: DynamicObstacle,
        t: float,
    ) -> ca.SX:
        """
        Add soft constraint cost for dynamic obstacle at time t.

        Args:
            position: 3D position (symbolic)
            obstacle: Dynamic obstacle instance
            t: Time at which to evaluate obstacle position

        Returns:
            CasADi cost expression
        """
        # Get obstacle position at time t
        obs_position = obstacle.get_position_at_time(t)
        safe_radius = obstacle.get_bounding_radius()

        # Squared distance
        diff = position - obs_position
        dist_sq = ca.dot(diff, diff)
        safe_radius_sq = safe_radius**2

        # Quadratic barrier
        violation = ca.fmax(0, safe_radius_sq - dist_sq)
        return self.weight * violation**2

    def add_obstacle_hard_constraint(
        self,
        opti: ca.Opti,
        position: ca.SX,
        obstacle: Obstacle,
    ):
        """
        Add hard constraint for obstacle avoidance.

        ||p - p_obs||^2 >= r_safe^2

        Args:
            opti: CasADi Opti instance
            position: 3D position (symbolic)
            obstacle: Obstacle instance
        """
        center = obstacle.get_center()
        safe_radius = obstacle.get_bounding_radius()

        diff = position - center
        dist_sq = ca.dot(diff, diff)
        safe_radius_sq = safe_radius**2

        opti.subject_to(dist_sq >= safe_radius_sq)

    def compute_obstacle_costs(
        self,
        X: ca.SX,
        static_obstacles: List[Obstacle],
        dynamic_obstacles: List[DynamicObstacle],
        t0: float,
        dt: float,
        N: int,
    ) -> ca.SX:
        """
        Compute total obstacle avoidance cost over horizon.

        Args:
            X: State trajectory (nx, N+1)
            static_obstacles: List of static obstacles
            dynamic_obstacles: List of dynamic obstacles
            t0: Current time
            dt: Time step
            N: Horizon length

        Returns:
            Total obstacle cost
        """
        cost = ca.SX(0)

        for k in range(N + 1):
            position = X[:3, k]  # Extract [x, y, z]
            t = t0 + k * dt

            # Static obstacles
            for obs in static_obstacles:
                cost += self.add_static_obstacle_cost(position, obs)

            # Dynamic obstacles
            for obs in dynamic_obstacles:
                cost += self.add_dynamic_obstacle_cost(position, obs, t)

        return cost


def create_state_bounds_from_config(config) -> StateBounds:
    """Create StateBounds from UAVConfig."""
    bounds = StateBounds()

    if hasattr(config, "position_bounds"):
        pb = config.position_bounds
        bounds.x_min = pb.x_min
        bounds.x_max = pb.x_max
        bounds.y_min = pb.y_min
        bounds.y_max = pb.y_max
        bounds.z_min = pb.z_min
        bounds.z_max = pb.z_max

    if hasattr(config, "max_velocity"):
        bounds.v_max = config.max_velocity
    if hasattr(config, "max_acceleration"):
        bounds.a_max = config.max_acceleration

    return bounds


def create_control_bounds_from_config(config) -> ControlBounds:
    """Create ControlBounds from UAVConfig."""
    bounds = ControlBounds()

    if hasattr(config, "max_jerk"):
        bounds.j_max = config.max_jerk
    if hasattr(config, "max_yaw_rate"):
        bounds.yaw_rate_max = config.max_yaw_rate

    return bounds
