"""Tracked obstacle classes for sensor-based perception."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..obstacles.base import Obstacle
from ..obstacles.dynamic import DynamicObstacle
from ..obstacles.static import SphereObstacle
from .sensor_model import Detection


@dataclass
class TrackedObstacle:
    """
    A static obstacle that has been discovered through sensing.

    Once discovered, static obstacles are assumed to persist
    at their detected location with optional confidence decay.
    """

    obstacle: Obstacle
    """The detected obstacle (copy with measured position)."""

    obstacle_id: int
    """Unique identifier for tracking."""

    first_detected: float
    """Time first observed."""

    last_seen: float
    """Time last observed."""

    confidence: float = 1.0
    """Confidence level (1.0 = certain, decays when not observed)."""

    def update(self, detection: Detection):
        """
        Update from new detection.

        Args:
            detection: New detection of this obstacle
        """
        self.last_seen = detection.timestamp
        self.confidence = 1.0

        # Update obstacle position (in case of measurement refinement)
        if isinstance(self.obstacle, SphereObstacle):
            self.obstacle = SphereObstacle(
                center=detection.position,
                radius=detection.radius,
                safety_margin=detection.safety_margin,
            )

    def get_obstacle_for_mpc(self) -> Obstacle:
        """
        Get obstacle for MPC constraints.

        Returns:
            The tracked obstacle
        """
        return self.obstacle

    def decay_confidence(self, current_time: float, decay_rate: float):
        """
        Reduce confidence when not observed.

        Args:
            current_time: Current simulation time
            decay_rate: Confidence decay per second
        """
        dt = current_time - self.last_seen
        self.confidence = max(0.0, 1.0 - decay_rate * dt)

    def __repr__(self) -> str:
        return (
            f"TrackedObstacle(id={self.obstacle_id}, "
            f"pos={self.obstacle.get_center()}, conf={self.confidence:.2f})"
        )


class TrackedDynamicObstacle:
    """
    A dynamic obstacle discovered through sensing.

    Since we don't know the true trajectory, we estimate it from
    observations using a constant-velocity model.
    """

    def __init__(
        self,
        radius: float,
        safety_margin: float,
        initial_position: np.ndarray,
        initial_velocity: Optional[np.ndarray],
        t_observed: float,
        obstacle_id: int,
    ):
        """
        Initialize tracked dynamic obstacle.

        Args:
            radius: Obstacle radius
            safety_margin: Safety margin around obstacle
            initial_position: First observed position
            initial_velocity: Initial velocity estimate (None if stationary)
            t_observed: Time of first observation
            obstacle_id: Unique identifier for tracking
        """
        self.radius = float(radius)
        self.safety_margin = float(safety_margin)
        self.obstacle_id = obstacle_id

        # State estimation
        self.position = np.asarray(initial_position).flatten().copy()
        self.velocity = (
            np.asarray(initial_velocity).flatten().copy()
            if initial_velocity is not None
            else np.zeros(3)
        )
        self.last_update_time = t_observed

        # History for velocity estimation smoothing
        self.position_history: List[Tuple[float, np.ndarray]] = [
            (t_observed, self.position.copy())
        ]
        self._max_history = 10

        # Velocity smoothing factor (EMA)
        self._velocity_alpha = 0.7

        # Tracking metadata
        self.first_detected = t_observed
        self.last_seen = t_observed
        self.confidence = 1.0

    def update(self, detection: Detection):
        """
        Update state estimate from new detection.

        Uses exponential moving average for velocity smoothing.

        Args:
            detection: New detection of this obstacle
        """
        dt = detection.timestamp - self.last_update_time

        if dt > 0.001:  # Avoid division by near-zero
            # Compute observed velocity from position change
            observed_velocity = (detection.position - self.position) / dt

            # Use detection velocity if available, else use computed
            if detection.velocity is not None:
                # Average detection velocity with computed velocity
                observed_velocity = 0.5 * (observed_velocity + detection.velocity)

            # Exponential moving average for smoothing
            self.velocity = (
                self._velocity_alpha * observed_velocity
                + (1 - self._velocity_alpha) * self.velocity
            )

        # Update position
        self.position = detection.position.copy()
        self.last_update_time = detection.timestamp
        self.last_seen = detection.timestamp
        self.confidence = 1.0

        # Store in history
        self.position_history.append((detection.timestamp, detection.position.copy()))
        if len(self.position_history) > self._max_history:
            self.position_history.pop(0)

    def predict(self, t: float) -> np.ndarray:
        """
        Predict position at time t using constant velocity model.

        Args:
            t: Time to predict position at

        Returns:
            Predicted 3D position
        """
        dt = t - self.last_update_time
        return self.position + self.velocity * dt

    def get_trajectory_function(self) -> Callable[[float], np.ndarray]:
        """
        Return trajectory function for MPC.

        Uses constant velocity extrapolation from last known state.

        Returns:
            Function t -> position(t)
        """
        # Capture current state (avoid closure over mutable self)
        pos = self.position.copy()
        vel = self.velocity.copy()
        t0 = self.last_update_time

        def trajectory(t: float) -> np.ndarray:
            return pos + vel * (t - t0)

        return trajectory

    def to_dynamic_obstacle(self) -> DynamicObstacle:
        """
        Convert to DynamicObstacle for MPC.

        Returns:
            DynamicObstacle with estimated constant-velocity trajectory
        """
        return DynamicObstacle(
            radius=self.radius,
            trajectory_func=self.get_trajectory_function(),
            safety_margin=self.safety_margin,
        )

    def decay_confidence(self, current_time: float, decay_rate: float):
        """
        Reduce confidence when not observed.

        Args:
            current_time: Current simulation time
            decay_rate: Confidence decay per second
        """
        dt = current_time - self.last_seen
        self.confidence = max(0.0, 1.0 - decay_rate * dt)

    def get_estimated_velocity_magnitude(self) -> float:
        """Get magnitude of estimated velocity."""
        return float(np.linalg.norm(self.velocity))

    def time_since_last_seen(self, current_time: float) -> float:
        """Get time elapsed since last observation."""
        return current_time - self.last_seen

    def __repr__(self) -> str:
        vel_mag = self.get_estimated_velocity_magnitude()
        return (
            f"TrackedDynamicObstacle(id={self.obstacle_id}, "
            f"pos={self.position}, vel_mag={vel_mag:.2f}, conf={self.confidence:.2f})"
        )


def create_tracked_static_from_detection(detection: Detection) -> TrackedObstacle:
    """
    Create TrackedObstacle from detection.

    Args:
        detection: Detection of a static obstacle

    Returns:
        New TrackedObstacle
    """
    obstacle = SphereObstacle(
        center=detection.position,
        radius=detection.radius,
        safety_margin=detection.safety_margin,
    )
    return TrackedObstacle(
        obstacle=obstacle,
        obstacle_id=detection.obstacle_id,
        first_detected=detection.timestamp,
        last_seen=detection.timestamp,
        confidence=1.0,
    )


def create_tracked_dynamic_from_detection(
    detection: Detection,
) -> TrackedDynamicObstacle:
    """
    Create TrackedDynamicObstacle from detection.

    Args:
        detection: Detection of a dynamic obstacle

    Returns:
        New TrackedDynamicObstacle
    """
    return TrackedDynamicObstacle(
        radius=detection.radius,
        safety_margin=detection.safety_margin,
        initial_position=detection.position,
        initial_velocity=detection.velocity,
        t_observed=detection.timestamp,
        obstacle_id=detection.obstacle_id,
    )
