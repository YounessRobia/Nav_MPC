"""Waypoint management for path planning."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Waypoint:
    """
    A waypoint in 3D space with optional velocity and heading.

    Attributes:
        position: 3D position [x, y, z]
        velocity: Optional desired velocity at waypoint [vx, vy, vz]
        yaw: Optional desired yaw angle at waypoint
        tolerance: Distance tolerance for considering waypoint reached
    """

    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    yaw: Optional[float] = None
    tolerance: float = 0.5

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).flatten()
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=float).flatten()

    def is_reached(self, current_position: np.ndarray) -> bool:
        """Check if waypoint is reached."""
        distance = np.linalg.norm(current_position - self.position)
        return distance < self.tolerance

    def distance_to(self, position: np.ndarray) -> float:
        """Compute distance from position to waypoint."""
        return float(np.linalg.norm(position - self.position))

    def to_state(self) -> np.ndarray:
        """
        Convert waypoint to 10D MPC state.

        Uses zero velocity/acceleration if not specified.
        """
        velocity = self.velocity if self.velocity is not None else np.zeros(3)
        yaw = self.yaw if self.yaw is not None else 0.0

        return np.concatenate([
            self.position,
            velocity,
            np.zeros(3),  # Zero acceleration
            [yaw]
        ])


class WaypointManager:
    """
    Manages a sequence of waypoints for navigation.

    Tracks current waypoint and handles transitions.
    """

    def __init__(
        self,
        waypoints: List[Waypoint] = None,
        loop: bool = False,
    ):
        """
        Initialize waypoint manager.

        Args:
            waypoints: List of waypoints to follow
            loop: If True, loop back to first waypoint after reaching last
        """
        self.waypoints = waypoints or []
        self.loop = loop
        self._current_index = 0

    def add_waypoint(self, waypoint: Waypoint):
        """Add a waypoint to the list."""
        self.waypoints.append(waypoint)

    def add_position(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        yaw: Optional[float] = None,
        tolerance: float = 0.5,
    ):
        """Add a waypoint by position."""
        self.waypoints.append(
            Waypoint(position, velocity, yaw, tolerance)
        )

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        """Get current target waypoint."""
        if not self.waypoints or self._current_index >= len(self.waypoints):
            return None
        return self.waypoints[self._current_index]

    @property
    def current_index(self) -> int:
        """Get current waypoint index."""
        return self._current_index

    @property
    def is_complete(self) -> bool:
        """Check if all waypoints have been visited."""
        return self._current_index >= len(self.waypoints)

    def update(self, current_position: np.ndarray) -> bool:
        """
        Update waypoint tracking based on current position.

        Args:
            current_position: Current UAV position

        Returns:
            True if waypoint was switched
        """
        waypoint = self.current_waypoint
        if waypoint is None:
            return False

        if waypoint.is_reached(current_position):
            self._current_index += 1
            if self.loop and self._current_index >= len(self.waypoints):
                self._current_index = 0
            return True

        return False

    def get_target_state(self) -> Optional[np.ndarray]:
        """Get current target as 10D MPC state."""
        waypoint = self.current_waypoint
        if waypoint is None:
            return None
        return waypoint.to_state()

    def reset(self):
        """Reset to first waypoint."""
        self._current_index = 0

    def get_remaining_waypoints(self) -> List[Waypoint]:
        """Get list of remaining waypoints."""
        return self.waypoints[self._current_index:]

    def get_all_positions(self) -> np.ndarray:
        """Get all waypoint positions as (N, 3) array."""
        if not self.waypoints:
            return np.zeros((0, 3))
        return np.array([wp.position for wp in self.waypoints])

    def __len__(self) -> int:
        return len(self.waypoints)


def create_waypoints_from_positions(
    positions: List[np.ndarray],
    tolerance: float = 0.5,
) -> List[Waypoint]:
    """
    Create waypoint list from position array.

    Args:
        positions: List of 3D positions
        tolerance: Arrival tolerance for all waypoints

    Returns:
        List of Waypoint instances
    """
    return [
        Waypoint(position=np.array(pos), tolerance=tolerance)
        for pos in positions
    ]
