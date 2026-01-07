"""Point mass dynamics model with jerk input (triple integrator)."""

import numpy as np
from scipy.linalg import expm
from typing import Tuple

from .uav_state import UAVState, UAVControl


class PointMassModel:
    """
    Point mass model with triple integrator dynamics and yaw.

    State: x = [x, y, z, vx, vy, vz, ax, ay, az, yaw]^T  (10D)
    Control: u = [jx, jy, jz, yaw_rate]^T  (4D)

    Continuous dynamics:
        ṗ = v         (position derivative = velocity)
        v̇ = a         (velocity derivative = acceleration)
        ȧ = j         (acceleration derivative = jerk, i.e., control input)
        ψ̇ = yaw_rate  (yaw derivative = yaw rate control)

    The model uses zero-order hold (ZOH) discretization for MPC.
    """

    # State dimensions
    NX = 10  # State dimension
    NU = 4  # Control dimension

    # State indices
    IDX_POS = slice(0, 3)  # Position indices
    IDX_VEL = slice(3, 6)  # Velocity indices
    IDX_ACC = slice(6, 9)  # Acceleration indices
    IDX_YAW = 9  # Yaw index

    def __init__(self, dt: float = 0.05):
        """
        Initialize the point mass model.

        Args:
            dt: Discretization timestep in seconds
        """
        self.dt = dt

        # Build continuous-time system matrices
        self._build_continuous_matrices()

        # Discretize using ZOH
        self._discretize()

    def _build_continuous_matrices(self):
        """Build continuous-time state-space matrices Ac, Bc."""
        # For triple integrator (position subsystem):
        # State per axis: [p, v, a]^T, Control: j (jerk)
        # ṗ = v, v̇ = a, ȧ = j
        #
        # Ac_axis = [[0, 1, 0],
        #            [0, 0, 1],
        #            [0, 0, 0]]
        # Bc_axis = [[0], [0], [1]]

        # Full 10D system (decoupled xyz + yaw)
        self.Ac = np.zeros((self.NX, self.NX))
        self.Bc = np.zeros((self.NX, self.NU))

        # Position -> Velocity coupling (for each axis)
        for i in range(3):
            self.Ac[i, i + 3] = 1.0  # ṗ = v

        # Velocity -> Acceleration coupling (for each axis)
        for i in range(3):
            self.Ac[i + 3, i + 6] = 1.0  # v̇ = a

        # Jerk input to acceleration (for each axis)
        for i in range(3):
            self.Bc[i + 6, i] = 1.0  # ȧ = j

        # Yaw rate input to yaw
        self.Bc[self.IDX_YAW, 3] = 1.0  # ψ̇ = yaw_rate

    def _discretize(self):
        """
        Discretize continuous-time system using zero-order hold.

        Uses matrix exponential for exact discretization of the position
        subsystem and analytical solution for the triple integrator.
        """
        dt = self.dt

        # For the triple integrator subsystem (decoupled per axis),
        # the analytical ZOH discretization is:
        # Ad_axis = [[1, dt, dt^2/2],
        #            [0, 1,  dt    ],
        #            [0, 0,  1     ]]
        # Bd_axis = [[dt^3/6], [dt^2/2], [dt]]

        self.Ad = np.eye(self.NX)
        self.Bd = np.zeros((self.NX, self.NU))

        # Fill in each axis (x, y, z)
        for i in range(3):
            # Position update
            self.Ad[i, i + 3] = dt  # p += v*dt
            self.Ad[i, i + 6] = 0.5 * dt**2  # p += 0.5*a*dt^2

            # Velocity update
            self.Ad[i + 3, i + 6] = dt  # v += a*dt

            # Jerk input effects
            self.Bd[i, i] = (dt**3) / 6  # p contribution from jerk
            self.Bd[i + 3, i] = 0.5 * dt**2  # v contribution from jerk
            self.Bd[i + 6, i] = dt  # a contribution from jerk

        # Yaw (simple integrator)
        self.Bd[self.IDX_YAW, 3] = dt  # yaw += yaw_rate * dt

    def continuous_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute continuous-time state derivative ẋ = f(x, u).

        Args:
            x: State vector (10D)
            u: Control vector (4D)

        Returns:
            State derivative ẋ (10D)
        """
        return self.Ac @ x + self.Bc @ u

    def discrete_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute discrete-time state update x[k+1] = Ad*x[k] + Bd*u[k].

        Args:
            x: Current state vector (10D)
            u: Control vector (4D)

        Returns:
            Next state x[k+1] (10D)
        """
        x_next = self.Ad @ x + self.Bd @ u
        # Wrap yaw angle to [-pi, pi]
        x_next[self.IDX_YAW] = self._wrap_angle(x_next[self.IDX_YAW])
        return x_next

    def step(self, state: UAVState, control: UAVControl) -> UAVState:
        """
        Propagate UAV state forward by one timestep.

        Args:
            state: Current UAV state
            control: Control input

        Returns:
            New UAV state after dt seconds
        """
        x = state.to_mpc_state()
        u = control.to_array()
        x_next = self.discrete_dynamics(x, u)
        return UAVState.from_mpc_state(x_next)

    def get_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get system matrices for MPC formulation.

        Returns:
            Tuple of (Ac, Bc, Ad, Bd)
        """
        return self.Ac.copy(), self.Bc.copy(), self.Ad.copy(), self.Bd.copy()

    def get_discrete_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get discrete-time matrices for MPC.

        Returns:
            Tuple of (Ad, Bd)
        """
        return self.Ad.copy(), self.Bd.copy()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def simulate_trajectory(
        self, x0: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """
        Simulate trajectory given initial state and control sequence.

        Args:
            x0: Initial state (10D)
            controls: Control sequence (N, 4)

        Returns:
            State trajectory (N+1, 10) including initial state
        """
        N = controls.shape[0]
        trajectory = np.zeros((N + 1, self.NX))
        trajectory[0] = x0

        x = x0.copy()
        for k in range(N):
            x = self.discrete_dynamics(x, controls[k])
            trajectory[k + 1] = x

        return trajectory

    def linearize_at(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get linearization at a given state-control pair.

        For this linear system, the Jacobians are constant (Ad, Bd).

        Args:
            x: State (10D) - unused for linear system
            u: Control (4D) - unused for linear system

        Returns:
            Tuple of (A, B) Jacobian matrices
        """
        # System is already linear, Jacobians are constant
        return self.Ad.copy(), self.Bd.copy()


def create_triple_integrator_casadi(dt: float):
    """
    Create CasADi symbolic dynamics for the triple integrator model.

    Args:
        dt: Discretization timestep

    Returns:
        CasADi function f(x, u) -> x_next
    """
    import casadi as ca

    # Symbolic state and control
    x = ca.SX.sym("x", PointMassModel.NX)
    u = ca.SX.sym("u", PointMassModel.NU)

    # Build discrete dynamics symbolically
    # Position update: p_next = p + v*dt + 0.5*a*dt^2 + (1/6)*j*dt^3
    p = x[0:3]
    v = x[3:6]
    a = x[6:9]
    yaw = x[9]
    j = u[0:3]
    yaw_rate = u[3]

    p_next = p + v * dt + 0.5 * a * dt**2 + (1.0 / 6.0) * j * dt**3
    v_next = v + a * dt + 0.5 * j * dt**2
    a_next = a + j * dt
    yaw_next = yaw + yaw_rate * dt

    # Assemble next state
    x_next = ca.vertcat(p_next, v_next, a_next, yaw_next)

    # Create CasADi function
    dynamics_func = ca.Function("dynamics", [x, u], [x_next], ["x", "u"], ["x_next"])

    return dynamics_func
