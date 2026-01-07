"""Collision checking utilities."""

import numpy as np
from typing import List, Optional, Tuple

from .base import Obstacle
from .dynamic import DynamicObstacle
from ..dynamics.uav_state import UAVState


def check_collision(
    position: np.ndarray,
    obstacles: List[Obstacle],
) -> bool:
    """
    Check if position collides with any obstacle.

    Args:
        position: 3D position [x, y, z]
        obstacles: List of obstacles

    Returns:
        True if in collision with any obstacle
    """
    position = np.asarray(position).flatten()
    for obs in obstacles:
        if obs.contains(position):
            return True
    return False


def check_state_collision(
    state: UAVState,
    obstacles: List[Obstacle],
) -> bool:
    """
    Check if UAV state collides with any obstacle.

    Args:
        state: UAV state
        obstacles: List of obstacles

    Returns:
        True if in collision
    """
    return check_collision(state.position, obstacles)


def minimum_distance(
    position: np.ndarray,
    obstacles: List[Obstacle],
) -> float:
    """
    Compute minimum distance to any obstacle.

    Args:
        position: 3D position
        obstacles: List of obstacles

    Returns:
        Minimum signed distance (negative if in collision)
    """
    if not obstacles:
        return float("inf")

    position = np.asarray(position).flatten()
    return min(obs.signed_distance(position) for obs in obstacles)


def get_closest_obstacle(
    position: np.ndarray,
    obstacles: List[Obstacle],
) -> Optional[Tuple[Obstacle, float]]:
    """
    Find the closest obstacle and its distance.

    Args:
        position: 3D position
        obstacles: List of obstacles

    Returns:
        Tuple of (closest obstacle, signed distance) or None if no obstacles
    """
    if not obstacles:
        return None

    position = np.asarray(position).flatten()
    min_dist = float("inf")
    closest = None

    for obs in obstacles:
        dist = obs.signed_distance(position)
        if dist < min_dist:
            min_dist = dist
            closest = obs

    return (closest, min_dist) if closest is not None else None


def check_trajectory_collision(
    trajectory: np.ndarray,
    obstacles: List[Obstacle],
    times: Optional[np.ndarray] = None,
) -> Tuple[bool, int]:
    """
    Check if trajectory collides with obstacles.

    For dynamic obstacles, if times are provided, the obstacles are
    evaluated at the corresponding time for each trajectory point.

    Args:
        trajectory: (N, 3) or (N, 10) array of positions/states
        obstacles: List of obstacles
        times: Optional (N,) array of times for dynamic obstacles

    Returns:
        Tuple of (collision_occurred, first_collision_index)
        If no collision, returns (False, -1)
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)

    # Extract positions (first 3 components)
    if trajectory.shape[1] >= 3:
        positions = trajectory[:, :3]
    else:
        raise ValueError(f"Invalid trajectory shape: {trajectory.shape}")

    for k, pos in enumerate(positions):
        # Update dynamic obstacles if times provided
        if times is not None:
            for obs in obstacles:
                if isinstance(obs, DynamicObstacle):
                    obs.update_time(times[k])

        if check_collision(pos, obstacles):
            return True, k

    return False, -1


def compute_clearance_profile(
    trajectory: np.ndarray,
    obstacles: List[Obstacle],
    times: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute signed distance profile along trajectory.

    Args:
        trajectory: (N, 3) or (N, 10) array
        obstacles: List of obstacles
        times: Optional times for dynamic obstacles

    Returns:
        (N,) array of minimum signed distances at each point
    """
    trajectory = np.asarray(trajectory)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)

    positions = trajectory[:, :3] if trajectory.shape[1] >= 3 else trajectory
    clearances = np.zeros(len(positions))

    for k, pos in enumerate(positions):
        if times is not None:
            for obs in obstacles:
                if isinstance(obs, DynamicObstacle):
                    obs.update_time(times[k])

        clearances[k] = minimum_distance(pos, obstacles)

    return clearances


def compute_collision_gradient(
    position: np.ndarray,
    obstacles: List[Obstacle],
) -> np.ndarray:
    """
    Compute combined gradient pointing away from all close obstacles.

    Useful for gradient-based obstacle avoidance.

    Args:
        position: 3D position
        obstacles: List of obstacles

    Returns:
        3D gradient vector (normalized)
    """
    position = np.asarray(position).flatten()
    gradient = np.zeros(3)

    for obs in obstacles:
        dist = obs.signed_distance(position)
        if dist < 2.0:  # Only consider nearby obstacles
            # Weight by inverse distance (closer = stronger)
            weight = 1.0 / (dist + 0.1) ** 2
            gradient += weight * obs.gradient(position)

    norm = np.linalg.norm(gradient)
    if norm > 1e-10:
        gradient /= norm

    return gradient


class CollisionChecker:
    """
    Collision checker with caching and batch operations.

    Provides efficient collision checking for MPC with static
    obstacle lists.
    """

    def __init__(
        self,
        static_obstacles: List[Obstacle] = None,
        dynamic_obstacles: List[DynamicObstacle] = None,
    ):
        """
        Initialize collision checker.

        Args:
            static_obstacles: List of static obstacles
            dynamic_obstacles: List of dynamic obstacles
        """
        self.static_obstacles = static_obstacles or []
        self.dynamic_obstacles = dynamic_obstacles or []

    @property
    def all_obstacles(self) -> List[Obstacle]:
        """Get combined list of all obstacles."""
        return self.static_obstacles + self.dynamic_obstacles

    def update_time(self, t: float):
        """Update time for dynamic obstacles."""
        for obs in self.dynamic_obstacles:
            obs.update_time(t)

    def check_collision(self, position: np.ndarray) -> bool:
        """Check collision at current time."""
        return check_collision(position, self.all_obstacles)

    def minimum_distance(self, position: np.ndarray) -> float:
        """Get minimum distance at current time."""
        return minimum_distance(position, self.all_obstacles)

    def check_trajectory(
        self, trajectory: np.ndarray, t0: float, dt: float
    ) -> Tuple[bool, int]:
        """
        Check trajectory for collisions.

        Args:
            trajectory: (N, 3) or (N, 10) trajectory
            t0: Start time
            dt: Time step

        Returns:
            (collision_occurred, first_collision_index)
        """
        N = len(trajectory)
        times = t0 + dt * np.arange(N)
        return check_trajectory_collision(trajectory, self.all_obstacles, times)

    def get_obstacle_positions_over_horizon(
        self, t0: float, dt: float, N: int
    ) -> List[List[np.ndarray]]:
        """
        Get positions of dynamic obstacles over MPC horizon.

        Args:
            t0: Start time
            dt: Time step
            N: Number of steps

        Returns:
            List (per obstacle) of lists (per timestep) of positions
        """
        positions = []
        for obs in self.dynamic_obstacles:
            obs_positions = obs.predict_over_horizon(t0, dt, N)
            positions.append(obs_positions)
        return positions
