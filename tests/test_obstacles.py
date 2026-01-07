"""Tests for obstacle module."""

import numpy as np
import pytest

from src.obstacles.base import Obstacle
from src.obstacles.static import (
    SphereObstacle,
    CylinderObstacle,
    BoxObstacle,
    create_obstacle_from_config,
)
from src.obstacles.dynamic import (
    DynamicObstacle,
    create_linear_trajectory,
    create_circular_trajectory,
)
from src.obstacles.collision import (
    check_collision,
    minimum_distance,
    check_trajectory_collision,
)


class TestSphereObstacle:
    """Tests for SphereObstacle."""

    def test_creation(self):
        """Test sphere creation."""
        sphere = SphereObstacle(
            center=[5, 0, 5],
            radius=2.0,
            safety_margin=0.5,
        )
        assert np.allclose(sphere.center, [5, 0, 5])
        assert sphere.radius == 2.0
        assert sphere.safety_margin == 0.5

    def test_signed_distance_outside(self):
        """Test signed distance for point outside sphere."""
        sphere = SphereObstacle(center=[0, 0, 0], radius=1.0, safety_margin=0.0)

        # Point at distance 3 from center
        dist = sphere.signed_distance(np.array([3, 0, 0]))
        assert np.isclose(dist, 2.0)  # 3 - 1 = 2

    def test_signed_distance_inside(self):
        """Test signed distance for point inside sphere."""
        sphere = SphereObstacle(center=[0, 0, 0], radius=2.0, safety_margin=0.0)

        # Point at center
        dist = sphere.signed_distance(np.array([0, 0, 0]))
        assert np.isclose(dist, -2.0)

    def test_signed_distance_with_margin(self):
        """Test signed distance includes safety margin."""
        sphere = SphereObstacle(center=[0, 0, 0], radius=1.0, safety_margin=0.5)

        # Point at distance 1.2 from center
        dist = sphere.signed_distance(np.array([1.2, 0, 0]))
        assert dist < 0  # Inside effective radius (1 + 0.5 = 1.5)

    def test_contains(self):
        """Test containment check."""
        sphere = SphereObstacle(center=[0, 0, 0], radius=1.0, safety_margin=0.5)

        assert sphere.contains(np.array([0, 0, 0]))  # Center
        assert sphere.contains(np.array([1.2, 0, 0]))  # Inside margin
        assert not sphere.contains(np.array([2, 0, 0]))  # Outside

    def test_gradient(self):
        """Test gradient direction."""
        sphere = SphereObstacle(center=[0, 0, 0], radius=1.0)

        grad = sphere.gradient(np.array([3, 0, 0]))
        expected = np.array([1, 0, 0])
        assert np.allclose(grad, expected, atol=1e-6)


class TestCylinderObstacle:
    """Tests for CylinderObstacle."""

    def test_creation(self):
        """Test cylinder creation."""
        cylinder = CylinderObstacle(
            center_xy=[5, 0],
            radius=1.0,
            z_min=0,
            z_max=10,
            safety_margin=0.5,
        )
        assert np.allclose(cylinder.center_xy, [5, 0])
        assert cylinder.radius == 1.0

    def test_signed_distance_xy_plane(self):
        """Test signed distance in xy-plane."""
        cylinder = CylinderObstacle(
            center_xy=[0, 0], radius=1.0, safety_margin=0.0
        )

        # Point at (3, 0, 5) - distance 3 in xy, inside z bounds
        dist = cylinder.signed_distance(np.array([3, 0, 5]))
        assert np.isclose(dist, 2.0)  # 3 - 1 = 2

    def test_gradient_points_radially(self):
        """Test gradient points radially in xy-plane."""
        cylinder = CylinderObstacle(center_xy=[0, 0], radius=1.0)

        grad = cylinder.gradient(np.array([3, 4, 5]))  # At (3, 4, z)
        # Should point in (3, 4) direction (normalized)
        expected = np.array([3, 4, 0]) / 5
        assert np.allclose(grad, expected, atol=1e-6)


class TestBoxObstacle:
    """Tests for BoxObstacle."""

    def test_creation(self):
        """Test box creation."""
        box = BoxObstacle(
            center=[5, 0, 5],
            half_extents=[1, 2, 3],
            safety_margin=0.5,
        )
        assert np.allclose(box.center, [5, 0, 5])
        assert np.allclose(box.half_extents, [1, 2, 3])

    def test_signed_distance_outside(self):
        """Test signed distance for point outside box."""
        box = BoxObstacle(
            center=[0, 0, 0], half_extents=[1, 1, 1], safety_margin=0.0
        )

        # Point at (3, 0, 0) - distance 2 from box surface
        dist = box.signed_distance(np.array([3, 0, 0]))
        assert np.isclose(dist, 2.0)

    def test_signed_distance_inside(self):
        """Test signed distance for point inside box."""
        box = BoxObstacle(
            center=[0, 0, 0], half_extents=[2, 2, 2], safety_margin=0.0
        )

        # Point at center
        dist = box.signed_distance(np.array([0, 0, 0]))
        assert dist < 0

    def test_corners(self):
        """Test corner computation."""
        box = BoxObstacle(center=[0, 0, 0], half_extents=[1, 1, 1])
        corners = box.get_corners()
        assert corners.shape == (8, 3)


class TestDynamicObstacle:
    """Tests for DynamicObstacle."""

    def test_linear_trajectory(self):
        """Test linear trajectory creation."""
        traj = create_linear_trajectory(
            initial_position=[0, 0, 5],
            velocity=[1, 0, 0],
        )

        pos_0 = traj(0)
        pos_1 = traj(1)

        assert np.allclose(pos_0, [0, 0, 5])
        assert np.allclose(pos_1, [1, 0, 5])

    def test_circular_trajectory(self):
        """Test circular trajectory creation."""
        traj = create_circular_trajectory(
            center=[0, 0, 5],
            radius=2.0,
            angular_velocity=np.pi,  # Half circle per second
            initial_phase=0,
        )

        pos_0 = traj(0)
        pos_1 = traj(1)  # After half circle

        assert np.allclose(pos_0, [2, 0, 5], atol=1e-6)
        assert np.allclose(pos_1, [-2, 0, 5], atol=1e-6)

    def test_dynamic_obstacle_update(self):
        """Test dynamic obstacle position updates."""
        traj = create_linear_trajectory([0, 0, 5], [1, 0, 0])
        obs = DynamicObstacle(radius=1.0, trajectory_func=traj)

        obs.update_time(0)
        assert np.allclose(obs.current_position, [0, 0, 5])

        obs.update_time(5)
        assert np.allclose(obs.current_position, [5, 0, 5])

    def test_predict_over_horizon(self):
        """Test horizon prediction."""
        traj = create_linear_trajectory([0, 0, 5], [1, 0, 0])
        obs = DynamicObstacle(radius=1.0, trajectory_func=traj)

        predictions = obs.predict_over_horizon(t0=0, dt=1.0, N=5)
        assert len(predictions) == 6  # N+1 positions
        assert np.allclose(predictions[0], [0, 0, 5])
        assert np.allclose(predictions[5], [5, 0, 5])


class TestCollisionChecking:
    """Tests for collision checking functions."""

    def test_check_collision_no_collision(self):
        """Test collision check with no collision."""
        obstacles = [
            SphereObstacle(center=[10, 0, 5], radius=1.0),
        ]

        assert not check_collision(np.array([0, 0, 5]), obstacles)

    def test_check_collision_with_collision(self):
        """Test collision check with collision."""
        obstacles = [
            SphereObstacle(center=[0, 0, 5], radius=2.0),
        ]

        assert check_collision(np.array([0, 0, 5]), obstacles)

    def test_minimum_distance(self):
        """Test minimum distance computation."""
        obstacles = [
            SphereObstacle(center=[5, 0, 0], radius=1.0, safety_margin=0),
            SphereObstacle(center=[10, 0, 0], radius=1.0, safety_margin=0),
        ]

        dist = minimum_distance(np.array([0, 0, 0]), obstacles)
        assert np.isclose(dist, 4.0)  # 5 - 1 = 4

    def test_trajectory_collision_check(self):
        """Test trajectory collision checking."""
        obstacles = [
            SphereObstacle(center=[5, 0, 0], radius=1.0, safety_margin=0),
        ]

        # Trajectory passing through obstacle
        trajectory = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [5, 0, 0],  # Inside obstacle
            [8, 0, 0],
        ])

        collision, idx = check_trajectory_collision(trajectory, obstacles)
        assert collision
        assert idx == 2


class TestObstacleFactory:
    """Tests for obstacle factory functions."""

    def test_create_sphere(self):
        """Test sphere creation from config."""
        config = {
            "type": "sphere",
            "center": [5, 0, 5],
            "radius": 2.0,
            "safety_margin": 0.5,
        }
        obs = create_obstacle_from_config(config)
        assert isinstance(obs, SphereObstacle)

    def test_create_box(self):
        """Test box creation from config."""
        config = {
            "type": "box",
            "center": [5, 0, 5],
            "half_extents": [1, 2, 3],
            "safety_margin": 0.3,
        }
        obs = create_obstacle_from_config(config)
        assert isinstance(obs, BoxObstacle)
