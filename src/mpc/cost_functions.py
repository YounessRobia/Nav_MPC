"""Cost function components for MPC."""

import casadi as ca
import numpy as np
from typing import Optional


def quadratic_cost(
    x: ca.SX,
    x_ref: ca.SX,
    Q: np.ndarray,
) -> ca.SX:
    """
    Quadratic tracking cost.

    J = (x - x_ref)^T Q (x - x_ref)

    Args:
        x: State (symbolic)
        x_ref: Reference state (symbolic)
        Q: Weight matrix (nx, nx)

    Returns:
        CasADi expression for cost
    """
    error = x - x_ref
    return ca.mtimes([error.T, Q, error])


def control_effort_cost(
    u: ca.SX,
    R: np.ndarray,
) -> ca.SX:
    """
    Control effort cost.

    J = u^T R u

    Args:
        u: Control input (symbolic)
        R: Weight matrix (nu, nu)

    Returns:
        CasADi expression for cost
    """
    return ca.mtimes([u.T, R, u])


def control_smoothness_cost(
    u_current: ca.SX,
    u_previous: ca.SX,
    S: np.ndarray,
) -> ca.SX:
    """
    Control smoothness (rate) cost.

    J = (u_k - u_{k-1})^T S (u_k - u_{k-1})

    Args:
        u_current: Current control (symbolic)
        u_previous: Previous control (symbolic)
        S: Weight matrix (nu, nu)

    Returns:
        CasADi expression for cost
    """
    delta_u = u_current - u_previous
    return ca.mtimes([delta_u.T, S, delta_u])


def terminal_cost(
    x_N: ca.SX,
    x_goal: ca.SX,
    Qf: np.ndarray,
) -> ca.SX:
    """
    Terminal cost at end of horizon.

    J = (x_N - x_goal)^T Qf (x_N - x_goal)

    Args:
        x_N: Terminal state (symbolic)
        x_goal: Goal state (symbolic)
        Qf: Terminal weight matrix (nx, nx)

    Returns:
        CasADi expression for cost
    """
    error = x_N - x_goal
    return ca.mtimes([error.T, Qf, error])


def obstacle_barrier_cost(
    position: ca.SX,
    obstacle_center: np.ndarray,
    safe_radius: float,
    weight: float = 1000.0,
) -> ca.SX:
    """
    Soft barrier cost for obstacle avoidance.

    Uses quadratic penalty when inside safe distance:
    J = weight * max(0, r_safe^2 - ||p - p_obs||^2)^2

    Args:
        position: 3D position (symbolic)
        obstacle_center: Obstacle center position
        safe_radius: Safe distance (obstacle radius + safety margin)
        weight: Penalty weight

    Returns:
        CasADi expression for barrier cost
    """
    diff = position - obstacle_center
    dist_sq = ca.dot(diff, diff)
    safe_radius_sq = safe_radius**2

    # Quadratic penalty when dist_sq < safe_radius_sq
    violation = ca.fmax(0, safe_radius_sq - dist_sq)
    return weight * violation**2


def obstacle_log_barrier_cost(
    position: ca.SX,
    obstacle_center: np.ndarray,
    safe_radius: float,
    weight: float = 10.0,
    epsilon: float = 0.1,
) -> ca.SX:
    """
    Log barrier cost for obstacle avoidance.

    J = -weight * log(||p - p_obs||^2 - r_safe^2 + epsilon)

    This provides smooth gradient that increases sharply near obstacles.

    Args:
        position: 3D position (symbolic)
        obstacle_center: Obstacle center position
        safe_radius: Safe distance
        weight: Barrier weight
        epsilon: Small constant for numerical stability

    Returns:
        CasADi expression for log barrier cost
    """
    diff = position - obstacle_center
    dist_sq = ca.dot(diff, diff)
    safe_radius_sq = safe_radius**2

    # Log barrier (only penalizes when getting close)
    margin = dist_sq - safe_radius_sq + epsilon
    return -weight * ca.log(ca.fmax(margin, 1e-6))


class CostBuilder:
    """
    Builder class for constructing MPC cost functions.

    Provides a fluent interface for adding cost terms.
    """

    def __init__(self, nx: int, nu: int):
        """
        Initialize cost builder.

        Args:
            nx: State dimension
            nu: Control dimension
        """
        self.nx = nx
        self.nu = nu
        self.stage_costs = []
        self.terminal_costs = []

    def add_tracking_cost(self, Q: np.ndarray) -> "CostBuilder":
        """Add state tracking cost."""
        Q = np.asarray(Q)
        if Q.ndim == 1:
            Q = np.diag(Q)

        def cost_func(x, x_ref, u, k):
            return quadratic_cost(x, x_ref, Q)

        self.stage_costs.append(cost_func)
        return self

    def add_control_cost(self, R: np.ndarray) -> "CostBuilder":
        """Add control effort cost."""
        R = np.asarray(R)
        if R.ndim == 1:
            R = np.diag(R)

        def cost_func(x, x_ref, u, k):
            return control_effort_cost(u, R)

        self.stage_costs.append(cost_func)
        return self

    def add_terminal_cost(self, Qf: np.ndarray) -> "CostBuilder":
        """Add terminal cost."""
        Qf = np.asarray(Qf)
        if Qf.ndim == 1:
            Qf = np.diag(Qf)

        def cost_func(x_N, x_goal):
            return terminal_cost(x_N, x_goal, Qf)

        self.terminal_costs.append(cost_func)
        return self

    def compute_stage_cost(
        self,
        x: ca.SX,
        x_ref: ca.SX,
        u: ca.SX,
        k: int,
    ) -> ca.SX:
        """
        Compute total stage cost.

        Args:
            x: State (symbolic)
            x_ref: Reference (symbolic)
            u: Control (symbolic)
            k: Time step index

        Returns:
            Total stage cost
        """
        cost = ca.SX(0)
        for cost_func in self.stage_costs:
            cost += cost_func(x, x_ref, u, k)
        return cost

    def compute_terminal_cost(
        self,
        x_N: ca.SX,
        x_goal: ca.SX,
    ) -> ca.SX:
        """
        Compute total terminal cost.

        Args:
            x_N: Terminal state (symbolic)
            x_goal: Goal state (symbolic)

        Returns:
            Total terminal cost
        """
        cost = ca.SX(0)
        for cost_func in self.terminal_costs:
            cost += cost_func(x_N, x_goal)
        return cost
