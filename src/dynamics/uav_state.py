"""UAV state representation for MPC path planning."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..utils.transforms import wrap_angle, euler_to_rotation_matrix


@dataclass
class UAVState:
    """
    Complete UAV state for MPC path planning.

    The state vector for MPC is 10-dimensional:
    x = [x, y, z, vx, vy, vz, ax, ay, az, yaw]^T

    Attributes:
        position: 3D position [x, y, z] in meters (inertial frame)
        velocity: 3D velocity [vx, vy, vz] in m/s (inertial frame)
        acceleration: 3D acceleration [ax, ay, az] in m/s^2 (inertial frame)
        yaw: Yaw angle (psi) in radians [-pi, pi]
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0

    def __post_init__(self):
        """Ensure arrays are numpy arrays with correct shape."""
        self.position = np.asarray(self.position, dtype=float).flatten()
        self.velocity = np.asarray(self.velocity, dtype=float).flatten()
        self.acceleration = np.asarray(self.acceleration, dtype=float).flatten()
        self.yaw = wrap_angle(float(self.yaw))

        assert self.position.shape == (3,), "Position must be 3D"
        assert self.velocity.shape == (3,), "Velocity must be 3D"
        assert self.acceleration.shape == (3,), "Acceleration must be 3D"

    @classmethod
    def from_mpc_state(cls, x: np.ndarray) -> "UAVState":
        """
        Create UAVState from MPC state vector.

        Args:
            x: 10D state vector [x, y, z, vx, vy, vz, ax, ay, az, yaw]

        Returns:
            UAVState instance
        """
        x = np.asarray(x).flatten()
        assert x.shape == (10,), f"Expected 10D state vector, got {x.shape}"
        return cls(
            position=x[0:3].copy(),
            velocity=x[3:6].copy(),
            acceleration=x[6:9].copy(),
            yaw=x[9],
        )

    def to_mpc_state(self) -> np.ndarray:
        """
        Convert to MPC state vector.

        Returns:
            10D numpy array [x, y, z, vx, vy, vz, ax, ay, az, yaw]
        """
        return np.concatenate(
            [self.position, self.velocity, self.acceleration, [self.yaw]]
        )

    def copy(self) -> "UAVState":
        """Create a deep copy of the state."""
        return UAVState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            yaw=self.yaw,
        )

    @property
    def x(self) -> float:
        """X position."""
        return self.position[0]

    @property
    def y(self) -> float:
        """Y position."""
        return self.position[1]

    @property
    def z(self) -> float:
        """Z position (altitude)."""
        return self.position[2]

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector."""
        return float(np.linalg.norm(self.velocity))

    @property
    def horizontal_speed(self) -> float:
        """Horizontal (xy-plane) speed."""
        return float(np.linalg.norm(self.velocity[:2]))

    def distance_to(self, other: "UAVState") -> float:
        """Euclidean distance to another state."""
        return float(np.linalg.norm(self.position - other.position))

    def distance_to_point(self, point: np.ndarray) -> float:
        """Euclidean distance to a point."""
        return float(np.linalg.norm(self.position - np.asarray(point)))

    def __repr__(self) -> str:
        return (
            f"UAVState(pos=[{self.x:.2f}, {self.y:.2f}, {self.z:.2f}], "
            f"vel=[{self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f}], "
            f"yaw={np.degrees(self.yaw):.1f}°)"
        )


@dataclass
class UAVControl:
    """
    Control input for the UAV (jerk-based).

    The control vector for MPC is 4-dimensional:
    u = [jx, jy, jz, yaw_rate]^T

    Attributes:
        jerk: 3D jerk [jx, jy, jz] in m/s^3 (acceleration derivative)
        yaw_rate: Yaw rate (psi_dot) in rad/s
    """

    jerk: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rate: float = 0.0

    def __post_init__(self):
        """Ensure arrays are numpy arrays with correct shape."""
        self.jerk = np.asarray(self.jerk, dtype=float).flatten()
        self.yaw_rate = float(self.yaw_rate)
        assert self.jerk.shape == (3,), "Jerk must be 3D"

    @classmethod
    def from_array(cls, u: np.ndarray) -> "UAVControl":
        """
        Create UAVControl from control vector.

        Args:
            u: 4D control vector [jx, jy, jz, yaw_rate]

        Returns:
            UAVControl instance
        """
        u = np.asarray(u).flatten()
        assert u.shape == (4,), f"Expected 4D control vector, got {u.shape}"
        return cls(jerk=u[0:3].copy(), yaw_rate=u[3])

    def to_array(self) -> np.ndarray:
        """
        Convert to control vector.

        Returns:
            4D numpy array [jx, jy, jz, yaw_rate]
        """
        return np.concatenate([self.jerk, [self.yaw_rate]])

    def copy(self) -> "UAVControl":
        """Create a deep copy of the control."""
        return UAVControl(jerk=self.jerk.copy(), yaw_rate=self.yaw_rate)

    @classmethod
    def zero(cls) -> "UAVControl":
        """Create zero control input."""
        return cls(jerk=np.zeros(3), yaw_rate=0.0)

    def __repr__(self) -> str:
        return (
            f"UAVControl(jerk=[{self.jerk[0]:.2f}, {self.jerk[1]:.2f}, {self.jerk[2]:.2f}], "
            f"yaw_rate={np.degrees(self.yaw_rate):.1f}°/s)"
        )


@dataclass
class SimulationLog:
    """
    Container for simulation trajectory data.

    Stores the complete history of states, controls, and timing information.
    """

    times: list = field(default_factory=list)
    states: list = field(default_factory=list)
    controls: list = field(default_factory=list)
    mpc_trajectories: list = field(default_factory=list)
    solve_times: list = field(default_factory=list)
    costs: list = field(default_factory=list)

    def append(
        self,
        t: float,
        state: UAVState,
        control: Optional[UAVControl] = None,
        mpc_trajectory: Optional[np.ndarray] = None,
        solve_time: Optional[float] = None,
        cost: Optional[float] = None,
    ):
        """Add a timestep to the log."""
        self.times.append(t)
        self.states.append(state.copy())
        if control is not None:
            self.controls.append(control.copy())
        if mpc_trajectory is not None:
            self.mpc_trajectories.append(mpc_trajectory.copy())
        if solve_time is not None:
            self.solve_times.append(solve_time)
        if cost is not None:
            self.costs.append(cost)

    def get_position_trajectory(self) -> np.ndarray:
        """Get Nx3 array of positions."""
        return np.array([s.position for s in self.states])

    def get_velocity_trajectory(self) -> np.ndarray:
        """Get Nx3 array of velocities."""
        return np.array([s.velocity for s in self.states])

    def get_state_trajectory(self) -> np.ndarray:
        """Get Nx10 array of full MPC states."""
        return np.array([s.to_mpc_state() for s in self.states])

    def get_control_trajectory(self) -> np.ndarray:
        """Get Nx4 array of controls."""
        return np.array([c.to_array() for c in self.controls])

    def __len__(self) -> int:
        return len(self.times)
