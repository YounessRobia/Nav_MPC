"""Static obstacle implementations (sphere, cylinder, box)."""

import numpy as np
from typing import Optional

from .base import Obstacle


class SphereObstacle(Obstacle):
    """
    Spherical obstacle.

    Constraint: ||p - center||^2 >= (radius + safety_margin)^2
    """

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        safety_margin: float = 0.5,
    ):
        """
        Initialize spherical obstacle.

        Args:
            center: 3D center position [x, y, z]
            radius: Sphere radius in meters
            safety_margin: Additional clearance
        """
        super().__init__(safety_margin)
        self.center = np.asarray(center, dtype=float).flatten()
        self.radius = float(radius)
        assert self.center.shape == (3,), "Center must be 3D"
        assert self.radius > 0, "Radius must be positive"

    def signed_distance(self, point: np.ndarray) -> float:
        """
        Signed distance to sphere surface (including safety margin).

        Positive outside, negative inside.
        """
        point = np.asarray(point).flatten()
        dist_to_center = np.linalg.norm(point - self.center)
        return dist_to_center - (self.radius + self.safety_margin)

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Gradient of signed distance function.

        Points radially outward from center.
        """
        point = np.asarray(point).flatten()
        diff = point - self.center
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            # At center, return arbitrary unit vector
            return np.array([1.0, 0.0, 0.0])
        return diff / dist

    def get_center(self) -> np.ndarray:
        return self.center.copy()

    def get_bounding_radius(self) -> float:
        return self.radius + self.safety_margin

    def signed_distance_casadi(self, point):
        """CasADi-compatible signed distance."""
        import casadi as ca

        diff = point - self.center
        dist_to_center = ca.norm_2(diff)
        return dist_to_center - (self.radius + self.safety_margin)

    def __repr__(self) -> str:
        return (
            f"SphereObstacle(center={self.center}, radius={self.radius}, "
            f"safety_margin={self.safety_margin})"
        )


class CylinderObstacle(Obstacle):
    """
    Cylindrical obstacle (vertical axis aligned with z).

    2D constraint in xy-plane: (x-xc)^2 + (y-yc)^2 >= (r + margin)^2
    Combined with vertical bounds: z_min <= z <= z_max
    """

    def __init__(
        self,
        center_xy: np.ndarray,
        radius: float,
        z_min: float = -np.inf,
        z_max: float = np.inf,
        safety_margin: float = 0.5,
    ):
        """
        Initialize cylindrical obstacle.

        Args:
            center_xy: 2D center position [x, y] in xy-plane
            radius: Cylinder radius in meters
            z_min: Lower altitude bound
            z_max: Upper altitude bound
            safety_margin: Additional clearance
        """
        super().__init__(safety_margin)
        self.center_xy = np.asarray(center_xy, dtype=float).flatten()[:2]
        self.radius = float(radius)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        assert self.center_xy.shape == (2,), "Center must be 2D [x, y]"
        assert self.radius > 0, "Radius must be positive"

    def signed_distance(self, point: np.ndarray) -> float:
        """
        Signed distance to cylinder surface.

        Considers both radial distance in xy-plane and vertical bounds.
        """
        point = np.asarray(point).flatten()
        xy = point[:2]
        z = point[2]

        # Radial distance in xy-plane
        radial_dist = np.linalg.norm(xy - self.center_xy)
        radial_signed = radial_dist - (self.radius + self.safety_margin)

        # Vertical distance (if bounded)
        if self.z_min > -np.inf or self.z_max < np.inf:
            # Distance to vertical bounds
            z_signed = max(self.z_min - z, z - self.z_max, 0)
            if z_signed > 0:
                # Outside vertical bounds
                return max(radial_signed, z_signed)

        return radial_signed

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Gradient of signed distance function.

        Points radially outward in xy-plane (ignores z).
        """
        point = np.asarray(point).flatten()
        xy = point[:2]
        diff_xy = xy - self.center_xy
        dist_xy = np.linalg.norm(diff_xy)

        if dist_xy < 1e-10:
            return np.array([1.0, 0.0, 0.0])

        grad_xy = diff_xy / dist_xy
        return np.array([grad_xy[0], grad_xy[1], 0.0])

    def get_center(self) -> np.ndarray:
        z_center = (self.z_min + self.z_max) / 2 if np.isfinite(self.z_min + self.z_max) else 0.0
        return np.array([self.center_xy[0], self.center_xy[1], z_center])

    def get_bounding_radius(self) -> float:
        # Conservative bounding sphere
        if np.isfinite(self.z_min) and np.isfinite(self.z_max):
            half_height = (self.z_max - self.z_min) / 2
            return np.sqrt((self.radius + self.safety_margin) ** 2 + half_height**2)
        return self.radius + self.safety_margin

    def signed_distance_casadi(self, point):
        """CasADi-compatible signed distance (xy-plane only)."""
        import casadi as ca

        xy = point[:2]
        diff_xy = xy - self.center_xy
        dist_xy = ca.norm_2(diff_xy)
        return dist_xy - (self.radius + self.safety_margin)

    def __repr__(self) -> str:
        return (
            f"CylinderObstacle(center_xy={self.center_xy}, radius={self.radius}, "
            f"z=[{self.z_min}, {self.z_max}], safety_margin={self.safety_margin})"
        )


class BoxObstacle(Obstacle):
    """
    Axis-aligned box obstacle.

    Defined by center and half-extents along each axis.
    Uses smooth approximation for differentiability.
    """

    def __init__(
        self,
        center: np.ndarray,
        half_extents: np.ndarray,
        safety_margin: float = 0.5,
    ):
        """
        Initialize box obstacle.

        Args:
            center: 3D center position [x, y, z]
            half_extents: Half-sizes along each axis [wx, wy, wz]
            safety_margin: Additional clearance
        """
        super().__init__(safety_margin)
        self.center = np.asarray(center, dtype=float).flatten()
        self.half_extents = np.asarray(half_extents, dtype=float).flatten()
        assert self.center.shape == (3,), "Center must be 3D"
        assert self.half_extents.shape == (3,), "Half-extents must be 3D"
        assert np.all(self.half_extents > 0), "Half-extents must be positive"

    def signed_distance(self, point: np.ndarray) -> float:
        """
        Signed distance to box surface (including safety margin).

        Uses the standard SDF formula for axis-aligned boxes.
        """
        point = np.asarray(point).flatten()
        # Vector from center to point in local coordinates
        diff = np.abs(point - self.center)
        # Effective half-extents with safety margin
        ext = self.half_extents + self.safety_margin

        # Distance to box surface
        q = diff - ext
        # Outside distance (when any component > 0)
        outside_dist = np.linalg.norm(np.maximum(q, 0))
        # Inside distance (negative, when all components < 0)
        inside_dist = min(np.max(q), 0)

        return outside_dist + inside_dist

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Gradient of signed distance function.

        Computed numerically for robustness.
        """
        point = np.asarray(point).flatten()
        eps = 1e-6
        grad = np.zeros(3)
        for i in range(3):
            point_plus = point.copy()
            point_plus[i] += eps
            point_minus = point.copy()
            point_minus[i] -= eps
            grad[i] = (
                self.signed_distance(point_plus) - self.signed_distance(point_minus)
            ) / (2 * eps)

        # Normalize if non-zero
        norm = np.linalg.norm(grad)
        if norm > 1e-10:
            grad /= norm
        return grad

    def get_center(self) -> np.ndarray:
        return self.center.copy()

    def get_bounding_radius(self) -> float:
        return np.linalg.norm(self.half_extents) + self.safety_margin

    def signed_distance_casadi(self, point):
        """
        CasADi-compatible signed distance.

        Uses soft-minimum approximation for differentiability.
        """
        import casadi as ca

        diff = ca.fabs(point - self.center)
        ext = self.half_extents + self.safety_margin
        q = diff - ext

        # Smooth max using softplus
        outside_dist = ca.norm_2(ca.fmax(q, 0))
        inside_dist = ca.fmin(ca.mmax(q), 0)

        return outside_dist + inside_dist

    def get_corners(self) -> np.ndarray:
        """Get the 8 corner points of the box."""
        corners = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    corner = self.center + np.array([sx, sy, sz]) * self.half_extents
                    corners.append(corner)
        return np.array(corners)

    def __repr__(self) -> str:
        return (
            f"BoxObstacle(center={self.center}, half_extents={self.half_extents}, "
            f"safety_margin={self.safety_margin})"
        )


def create_obstacle_from_config(config: dict) -> Obstacle:
    """
    Factory function to create obstacle from configuration dictionary.

    Args:
        config: Dictionary with 'type' and type-specific parameters

    Returns:
        Obstacle instance
    """
    obs_type = config.get("type", "").lower()
    safety_margin = config.get("safety_margin", 0.5)

    if obs_type == "sphere":
        return SphereObstacle(
            center=np.array(config["center"]),
            radius=config["radius"],
            safety_margin=safety_margin,
        )
    elif obs_type == "cylinder":
        return CylinderObstacle(
            center_xy=np.array(config.get("center", config.get("center_xy", [0, 0])))[:2],
            radius=config["radius"],
            z_min=config.get("z_min", -np.inf),
            z_max=config.get("z_max", np.inf),
            safety_margin=safety_margin,
        )
    elif obs_type == "box":
        return BoxObstacle(
            center=np.array(config["center"]),
            half_extents=np.array(config["half_extents"]),
            safety_margin=safety_margin,
        )
    else:
        raise ValueError(f"Unknown obstacle type: {obs_type}")
