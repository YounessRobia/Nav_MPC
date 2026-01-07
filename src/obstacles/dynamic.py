"""Dynamic obstacles with known trajectories."""

import numpy as np
from typing import Callable, List, Optional, Dict, Any

from .base import Obstacle
from .static import SphereObstacle


class DynamicObstacle(Obstacle):
    """
    Dynamic obstacle with known trajectory.

    Wraps a base obstacle shape (sphere) and moves it according to
    a trajectory function that returns position at any given time.
    """

    def __init__(
        self,
        radius: float,
        trajectory_func: Callable[[float], np.ndarray],
        safety_margin: float = 0.5,
    ):
        """
        Initialize dynamic obstacle.

        Args:
            radius: Obstacle radius (spherical approximation)
            trajectory_func: Function t -> [x, y, z] returning position at time t
            safety_margin: Additional clearance
        """
        super().__init__(safety_margin)
        self.radius = float(radius)
        self.trajectory_func = trajectory_func
        self._current_time = 0.0
        self._current_position = trajectory_func(0.0)

    def update_time(self, t: float):
        """
        Update the obstacle's current time (and hence position).

        Args:
            t: Current simulation time
        """
        self._current_time = t
        self._current_position = self.trajectory_func(t)

    def get_position_at_time(self, t: float) -> np.ndarray:
        """
        Get obstacle position at specified time.

        Args:
            t: Time in seconds

        Returns:
            3D position [x, y, z]
        """
        return self.trajectory_func(t)

    def predict_over_horizon(
        self, t0: float, dt: float, N: int
    ) -> List[np.ndarray]:
        """
        Predict positions over MPC horizon.

        Args:
            t0: Start time
            dt: Time step
            N: Number of prediction steps

        Returns:
            List of N+1 positions from t0 to t0 + N*dt
        """
        return [self.trajectory_func(t0 + k * dt) for k in range(N + 1)]

    def signed_distance(self, point: np.ndarray) -> float:
        """
        Signed distance using current position.
        """
        point = np.asarray(point).flatten()
        dist_to_center = np.linalg.norm(point - self._current_position)
        return dist_to_center - (self.radius + self.safety_margin)

    def signed_distance_at_time(self, point: np.ndarray, t: float) -> float:
        """
        Signed distance at a specific time.

        Args:
            point: 3D position
            t: Time at which to evaluate

        Returns:
            Signed distance
        """
        pos = self.trajectory_func(t)
        dist_to_center = np.linalg.norm(point - pos)
        return dist_to_center - (self.radius + self.safety_margin)

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Gradient of signed distance at current position.
        """
        point = np.asarray(point).flatten()
        diff = point - self._current_position
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return diff / dist

    def get_center(self) -> np.ndarray:
        return self._current_position.copy()

    def get_bounding_radius(self) -> float:
        return self.radius + self.safety_margin

    @property
    def current_position(self) -> np.ndarray:
        """Current obstacle position."""
        return self._current_position.copy()

    @property
    def current_time(self) -> float:
        """Current time."""
        return self._current_time

    def __repr__(self) -> str:
        return (
            f"DynamicObstacle(radius={self.radius}, "
            f"pos={self._current_position}, t={self._current_time:.2f})"
        )


# =============================================================================
# Trajectory generator functions
# =============================================================================


def create_linear_trajectory(
    initial_position: np.ndarray, velocity: np.ndarray
) -> Callable[[float], np.ndarray]:
    """
    Create linear (constant velocity) trajectory function.

    Args:
        initial_position: Position at t=0
        velocity: Constant velocity vector [vx, vy, vz]

    Returns:
        Function t -> position(t)
    """
    p0 = np.asarray(initial_position, dtype=float).flatten()
    v = np.asarray(velocity, dtype=float).flatten()

    def trajectory(t: float) -> np.ndarray:
        return p0 + v * t

    return trajectory


def create_circular_trajectory(
    center: np.ndarray,
    radius: float,
    angular_velocity: float,
    initial_phase: float = 0.0,
    altitude: Optional[float] = None,
) -> Callable[[float], np.ndarray]:
    """
    Create circular trajectory in xy-plane.

    Args:
        center: Center of circular path [x, y, z]
        radius: Radius of circular path
        angular_velocity: Angular velocity in rad/s (positive = CCW)
        initial_phase: Initial angle in radians
        altitude: Override z-coordinate (uses center[2] if None)

    Returns:
        Function t -> position(t)
    """
    c = np.asarray(center, dtype=float).flatten()
    z = altitude if altitude is not None else c[2]

    def trajectory(t: float) -> np.ndarray:
        angle = initial_phase + angular_velocity * t
        x = c[0] + radius * np.cos(angle)
        y = c[1] + radius * np.sin(angle)
        return np.array([x, y, z])

    return trajectory


def create_sinusoidal_trajectory(
    center: np.ndarray,
    amplitude: np.ndarray,
    frequency: np.ndarray,
    phase: np.ndarray = None,
) -> Callable[[float], np.ndarray]:
    """
    Create sinusoidal trajectory.

    position(t) = center + amplitude * sin(2*pi*frequency*t + phase)

    Args:
        center: Mean position
        amplitude: Oscillation amplitude for each axis
        frequency: Oscillation frequency (Hz) for each axis
        phase: Phase offset for each axis

    Returns:
        Function t -> position(t)
    """
    c = np.asarray(center, dtype=float).flatten()
    a = np.asarray(amplitude, dtype=float).flatten()
    f = np.asarray(frequency, dtype=float).flatten()
    p = np.zeros(3) if phase is None else np.asarray(phase, dtype=float).flatten()

    def trajectory(t: float) -> np.ndarray:
        return c + a * np.sin(2 * np.pi * f * t + p)

    return trajectory


def create_lemniscate_trajectory(
    center: np.ndarray,
    scale: float,
    angular_velocity: float,
    altitude: Optional[float] = None,
) -> Callable[[float], np.ndarray]:
    """
    Create lemniscate (figure-8) trajectory in xy-plane.

    Args:
        center: Center of the figure-8 [x, y, z]
        scale: Size scaling factor
        angular_velocity: Angular parameter velocity
        altitude: Override z-coordinate

    Returns:
        Function t -> position(t)
    """
    c = np.asarray(center, dtype=float).flatten()
    z = altitude if altitude is not None else c[2]

    def trajectory(t: float) -> np.ndarray:
        theta = angular_velocity * t
        # Parametric lemniscate
        denom = 1 + np.sin(theta) ** 2
        x = c[0] + scale * np.cos(theta) / denom
        y = c[1] + scale * np.sin(theta) * np.cos(theta) / denom
        return np.array([x, y, z])

    return trajectory


def create_dynamic_obstacle_from_config(config: Dict[str, Any]) -> DynamicObstacle:
    """
    Factory function to create DynamicObstacle from configuration.

    Args:
        config: Dictionary with obstacle and trajectory parameters

    Returns:
        DynamicObstacle instance
    """
    radius = config.get("radius", 1.0)
    safety_margin = config.get("safety_margin", 0.5)
    traj_config = config.get("trajectory", {})
    traj_type = traj_config.get("type", "linear")

    if traj_type == "linear":
        initial_pos = np.array(traj_config["initial_position"])
        velocity = np.array(traj_config["velocity"])
        trajectory_func = create_linear_trajectory(initial_pos, velocity)

    elif traj_type == "circular":
        center = np.array(traj_config["center"])
        traj_radius = traj_config["radius"]
        omega = traj_config.get("angular_velocity", 1.0)
        phase = traj_config.get("initial_phase", 0.0)
        trajectory_func = create_circular_trajectory(
            center, traj_radius, omega, phase
        )

    elif traj_type == "sinusoidal":
        center = np.array(traj_config["center"])
        amplitude = np.array(traj_config["amplitude"])
        frequency = np.array(traj_config["frequency"])
        phase = traj_config.get("phase")
        trajectory_func = create_sinusoidal_trajectory(
            center, amplitude, frequency, phase
        )

    elif traj_type == "lemniscate":
        center = np.array(traj_config["center"])
        scale = traj_config["scale"]
        omega = traj_config.get("angular_velocity", 1.0)
        trajectory_func = create_lemniscate_trajectory(center, scale, omega)

    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return DynamicObstacle(
        radius=radius,
        trajectory_func=trajectory_func,
        safety_margin=safety_margin,
    )
