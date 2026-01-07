"""Abstract base class for obstacles."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Obstacle(ABC):
    """
    Abstract base class for all obstacle types.

    Obstacles define a region in 3D space that the UAV must avoid.
    Each obstacle provides methods for computing signed distance
    (negative inside, positive outside) and gradients for optimization.
    """

    def __init__(self, safety_margin: float = 0.5):
        """
        Initialize obstacle with safety margin.

        Args:
            safety_margin: Additional clearance around obstacle in meters
        """
        self.safety_margin = safety_margin

    @abstractmethod
    def signed_distance(self, point: np.ndarray) -> float:
        """
        Compute signed distance from point to obstacle surface.

        Positive: point is outside the obstacle
        Negative: point is inside the obstacle
        Zero: point is on the obstacle surface

        The signed distance INCLUDES the safety margin.

        Args:
            point: 3D position [x, y, z]

        Returns:
            Signed distance in meters
        """
        pass

    @abstractmethod
    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Compute gradient of signed distance function at point.

        The gradient points in the direction of maximum distance increase
        (away from the obstacle center).

        Args:
            point: 3D position [x, y, z]

        Returns:
            3D gradient vector
        """
        pass

    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is inside the obstacle (including safety margin).

        Args:
            point: 3D position [x, y, z]

        Returns:
            True if point is in collision
        """
        return self.signed_distance(point) < 0

    def distance(self, point: np.ndarray) -> float:
        """
        Compute unsigned distance from point to obstacle surface.

        Args:
            point: 3D position [x, y, z]

        Returns:
            Distance in meters (always >= 0)
        """
        return max(0.0, self.signed_distance(point))

    @abstractmethod
    def get_center(self) -> np.ndarray:
        """
        Get the center position of the obstacle.

        Returns:
            3D position of obstacle center
        """
        pass

    @abstractmethod
    def get_bounding_radius(self) -> float:
        """
        Get bounding sphere radius (including safety margin).

        Returns:
            Radius of smallest enclosing sphere
        """
        pass

    def fast_collision_check(self, point: np.ndarray) -> bool:
        """
        Fast preliminary collision check using bounding sphere.

        This is faster than full signed distance computation for
        points far from the obstacle.

        Args:
            point: 3D position

        Returns:
            True if point MAY be in collision (needs detailed check)
        """
        dist_to_center = np.linalg.norm(point - self.get_center())
        return dist_to_center < self.get_bounding_radius()

    def signed_distance_casadi(self, point):
        """
        CasADi-compatible signed distance (for MPC constraints).

        Override in subclasses for symbolic computation support.
        Default implementation raises NotImplementedError.

        Args:
            point: CasADi symbolic 3D position

        Returns:
            CasADi expression for signed distance
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement CasADi signed distance"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(safety_margin={self.safety_margin})"
