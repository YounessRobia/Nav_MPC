"""Sensor model for simulating obstacle detection."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..obstacles.base import Obstacle
from ..obstacles.dynamic import DynamicObstacle


@dataclass
class SensorConfig:
    """Configuration for sensor model."""

    detection_radius: float = 15.0
    """Maximum detection range in meters."""

    field_of_view: float = np.pi / 2
    """Half-angle of detection cone in radians (default: 90 degrees total FOV)."""

    noise_std_position: float = 0.1
    """Standard deviation of position measurement noise in meters."""

    noise_std_velocity: float = 0.05
    """Standard deviation of velocity measurement noise in m/s."""

    detection_probability: float = 1.0
    """Probability of detecting an obstacle in range (0-1)."""

    min_detection_range: float = 0.5
    """Minimum detection range (blind spot near robot) in meters."""

    detect_behind: bool = False
    """If True, also detect obstacles behind (omnidirectional mode)."""


@dataclass
class Detection:
    """Raw detection from sensor."""

    position: np.ndarray
    """Detected center position [x, y, z]."""

    radius: float
    """Detected obstacle radius."""

    velocity: Optional[np.ndarray]
    """Estimated velocity if motion detected, None otherwise."""

    obstacle_id: int
    """Unique identifier for tracking association."""

    timestamp: float
    """Time of detection."""

    is_static: bool
    """True if obstacle appears stationary."""

    safety_margin: float = 0.5
    """Safety margin of the detected obstacle."""


class SensorModel:
    """
    Simulates sensor-based obstacle perception.

    Models a forward-facing sensor with limited field of view and range.
    Can detect both static and dynamic obstacles, with optional
    measurement noise.
    """

    def __init__(self, config: SensorConfig = None):
        """
        Initialize sensor model.

        Args:
            config: Sensor configuration parameters
        """
        self.config = config or SensorConfig()
        self._rng = np.random.default_rng()
        self._obstacle_id_counter = 0
        self._obstacle_id_map = {}  # Maps obstacle object id to detection id

    def detect(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        static_obstacles: List[Obstacle],
        dynamic_obstacles: List[DynamicObstacle],
        t: float,
    ) -> List[Detection]:
        """
        Simulate detection of obstacles within sensor range and FOV.

        Args:
            robot_position: Robot position [x, y, z]
            robot_yaw: Robot heading in radians
            static_obstacles: List of static obstacles in world
            dynamic_obstacles: List of dynamic obstacles in world
            t: Current simulation time

        Returns:
            List of Detection objects for visible obstacles
        """
        robot_position = np.asarray(robot_position).flatten()
        detections = []

        # Detect static obstacles
        for obs in static_obstacles:
            detection = self._detect_static_obstacle(
                robot_position, robot_yaw, obs, t
            )
            if detection is not None:
                detections.append(detection)

        # Detect dynamic obstacles
        for obs in dynamic_obstacles:
            detection = self._detect_dynamic_obstacle(
                robot_position, robot_yaw, obs, t
            )
            if detection is not None:
                detections.append(detection)

        return detections

    def _detect_static_obstacle(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        obstacle: Obstacle,
        t: float,
    ) -> Optional[Detection]:
        """Attempt to detect a static obstacle."""
        obs_center = obstacle.get_center()

        if not self._is_in_fov(robot_position, robot_yaw, obs_center):
            return None

        # Check detection probability
        if self._rng.random() > self.config.detection_probability:
            return None

        # Get or assign obstacle ID
        obs_id = self._get_obstacle_id(obstacle)

        # Add measurement noise
        noisy_position = self._add_position_noise(obs_center)

        return Detection(
            position=noisy_position,
            radius=obstacle.get_bounding_radius() - obstacle.safety_margin,
            velocity=None,
            obstacle_id=obs_id,
            timestamp=t,
            is_static=True,
            safety_margin=obstacle.safety_margin,
        )

    def _detect_dynamic_obstacle(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        obstacle: DynamicObstacle,
        t: float,
    ) -> Optional[Detection]:
        """Attempt to detect a dynamic obstacle."""
        obs_center = obstacle.get_center()

        if not self._is_in_fov(robot_position, robot_yaw, obs_center):
            return None

        # Check detection probability
        if self._rng.random() > self.config.detection_probability:
            return None

        # Get or assign obstacle ID
        obs_id = self._get_obstacle_id(obstacle)

        # Estimate velocity from trajectory (small time step)
        dt = 0.05
        pos_prev = obstacle.get_position_at_time(t - dt)
        pos_curr = obstacle.get_position_at_time(t)
        true_velocity = (pos_curr - pos_prev) / dt

        # Add measurement noise
        noisy_position = self._add_position_noise(pos_curr)
        noisy_velocity = self._add_velocity_noise(true_velocity)

        # Determine if obstacle appears static (velocity below threshold)
        velocity_magnitude = np.linalg.norm(noisy_velocity)
        is_static = velocity_magnitude < 0.1  # Threshold for "stationary"

        return Detection(
            position=noisy_position,
            radius=obstacle.radius,
            velocity=noisy_velocity if not is_static else None,
            obstacle_id=obs_id,
            timestamp=t,
            is_static=is_static,
            safety_margin=obstacle.safety_margin,
        )

    def _is_in_fov(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        obstacle_position: np.ndarray,
    ) -> bool:
        """
        Check if obstacle is within sensor field of view.

        Uses a cone-shaped FOV in the xy-plane centered on robot_yaw.
        Vertical (z) is not restricted.

        Args:
            robot_position: Robot position [x, y, z]
            robot_yaw: Robot heading in radians
            obstacle_position: Obstacle center [x, y, z]

        Returns:
            True if obstacle is within FOV
        """
        # Vector to obstacle in xy-plane
        to_obstacle = obstacle_position[:2] - robot_position[:2]
        distance_xy = np.linalg.norm(to_obstacle)

        # Also consider 3D distance for range check
        distance_3d = np.linalg.norm(obstacle_position - robot_position)

        # Check minimum range (blind spot)
        if distance_3d < self.config.min_detection_range:
            return False

        # Check maximum range
        if distance_3d > self.config.detection_radius:
            return False

        # If omnidirectional mode, skip angle check
        if self.config.detect_behind:
            return True

        # Check angle in xy-plane
        if distance_xy < 1e-6:
            return True  # Directly above/below

        angle_to_obstacle = np.arctan2(to_obstacle[1], to_obstacle[0])
        angle_diff = self._wrap_angle(angle_to_obstacle - robot_yaw)

        return abs(angle_diff) <= self.config.field_of_view

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _add_position_noise(self, position: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to position measurement."""
        if self.config.noise_std_position > 0:
            noise = self._rng.normal(0, self.config.noise_std_position, 3)
            return position + noise
        return position.copy()

    def _add_velocity_noise(self, velocity: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to velocity measurement."""
        if self.config.noise_std_velocity > 0:
            noise = self._rng.normal(0, self.config.noise_std_velocity, 3)
            return velocity + noise
        return velocity.copy()

    def _get_obstacle_id(self, obstacle) -> int:
        """Get or assign a unique ID for an obstacle."""
        py_id = id(obstacle)
        if py_id not in self._obstacle_id_map:
            self._obstacle_id_map[py_id] = self._obstacle_id_counter
            self._obstacle_id_counter += 1
        return self._obstacle_id_map[py_id]

    def reset(self):
        """Reset sensor state (obstacle ID mappings)."""
        self._obstacle_id_counter = 0
        self._obstacle_id_map = {}
