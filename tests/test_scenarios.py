"""Integration tests for simulation scenarios."""

import numpy as np
import pytest
from pathlib import Path

from src.dynamics.uav_state import UAVState
from src.mpc.controller import MPCController
from src.obstacles.static import SphereObstacle, BoxObstacle
from src.obstacles.dynamic import DynamicObstacle, create_linear_trajectory
from src.obstacles.collision import check_collision, minimum_distance
from src.planning.reference import ReferenceGenerator
from src.utils.config_loader import MPCConfig, UAVConfig


@pytest.fixture
def quick_mpc_config():
    """MPC config with short horizon for faster tests."""
    return MPCConfig(
        horizon=15,  # Shorter horizon
        dt=0.1,  # Larger timestep
        obstacle_weight=1000.0,
    )


@pytest.fixture
def uav_config():
    """Standard UAV config."""
    return UAVConfig(
        max_velocity=5.0,
        max_acceleration=3.0,
        max_jerk=20.0,
        max_yaw_rate=1.0,
    )


class TestStaticObstacleScenarios:
    """Integration tests for static obstacle scenarios."""

    def test_single_sphere_avoidance(self, quick_mpc_config, uav_config):
        """Test UAV navigates around single sphere."""
        mpc = MPCController(quick_mpc_config, uav_config)
        ref_gen = ReferenceGenerator(
            horizon=quick_mpc_config.horizon,
            dt=quick_mpc_config.dt,
        )

        # Setup
        state = UAVState(position=[0, 0, 5])
        goal = np.array([20, 0, 5, 0, 0, 0, 0, 0, 0, 0])
        obstacle = SphereObstacle(center=[10, 0, 5], radius=2.0, safety_margin=0.5)

        # Simulate several steps
        trajectory = [state.position.copy()]
        for _ in range(30):
            reference = ref_gen.generate_smooth_reference(state, goal[:3])
            solution = mpc.solve(state, reference, static_obstacles=[obstacle])

            if not solution.success:
                break

            # Apply control
            control = mpc.get_control(solution)
            x = state.to_mpc_state()
            x_next = mpc.dynamics.discrete_dynamics(x, control.to_array())
            state = UAVState.from_mpc_state(x_next)
            trajectory.append(state.position.copy())

        trajectory = np.array(trajectory)

        # Verify no collision occurred
        for pos in trajectory:
            dist = obstacle.signed_distance(pos)
            assert dist > -0.1, f"Collision at {pos}, dist={dist}"

    def test_corridor_navigation(self, quick_mpc_config, uav_config):
        """Test UAV navigates through corridor."""
        mpc = MPCController(quick_mpc_config, uav_config)

        state = UAVState(position=[0, 0, 5])
        goal = np.array([20, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        # Corridor walls
        left_wall = BoxObstacle(
            center=[10, 3, 5],
            half_extents=[8, 0.5, 3],
            safety_margin=0.3,
        )
        right_wall = BoxObstacle(
            center=[10, -3, 5],
            half_extents=[8, 0.5, 3],
            safety_margin=0.3,
        )
        obstacles = [left_wall, right_wall]

        # Simulate
        for _ in range(40):
            solution = mpc.solve(state, goal, static_obstacles=obstacles)
            if not solution.success:
                break

            control = mpc.get_control(solution)
            x_next = mpc.dynamics.discrete_dynamics(
                state.to_mpc_state(), control.to_array()
            )
            state = UAVState.from_mpc_state(x_next)

            # Check no collision
            assert not check_collision(state.position, obstacles)


class TestDynamicObstacleScenarios:
    """Integration tests for dynamic obstacle scenarios."""

    def test_crossing_obstacle(self, quick_mpc_config, uav_config):
        """Test UAV avoids crossing obstacle."""
        mpc = MPCController(quick_mpc_config, uav_config)

        state = UAVState(position=[0, 0, 5])
        goal = np.array([15, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        # Obstacle crossing path perpendicular to UAV
        trajectory = create_linear_trajectory(
            initial_position=[7, -10, 5],
            velocity=[0, 2, 0],  # Moving in +y direction
        )
        obstacle = DynamicObstacle(radius=1.0, trajectory_func=trajectory)

        # Simulate
        t = 0
        for _ in range(50):
            obstacle.update_time(t)
            solution = mpc.solve(
                state, goal, dynamic_obstacles=[obstacle], t_current=t
            )

            if not solution.success:
                break

            control = mpc.get_control(solution)
            x_next = mpc.dynamics.discrete_dynamics(
                state.to_mpc_state(), control.to_array()
            )
            state = UAVState.from_mpc_state(x_next)
            t += mpc.dt

            # Check collision with updated obstacle position
            obstacle.update_time(t)
            dist = obstacle.signed_distance(state.position)
            assert dist > -0.2, f"Collision at t={t}, dist={dist}"

    def test_oncoming_obstacle(self, quick_mpc_config, uav_config):
        """Test UAV avoids head-on obstacle."""
        mpc = MPCController(quick_mpc_config, uav_config)

        state = UAVState(position=[0, 0, 5])
        goal = np.array([20, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        # Obstacle coming toward UAV
        trajectory = create_linear_trajectory(
            initial_position=[25, 0, 5],
            velocity=[-3, 0, 0],  # Moving in -x direction
        )
        obstacle = DynamicObstacle(radius=1.5, trajectory_func=trajectory)

        # Simulate
        t = 0
        min_clearance = float("inf")

        for _ in range(60):
            obstacle.update_time(t)
            solution = mpc.solve(
                state, goal, dynamic_obstacles=[obstacle], t_current=t
            )

            control = mpc.get_control(solution)
            x_next = mpc.dynamics.discrete_dynamics(
                state.to_mpc_state(), control.to_array()
            )
            state = UAVState.from_mpc_state(x_next)
            t += mpc.dt

            obstacle.update_time(t)
            dist = obstacle.signed_distance(state.position)
            min_clearance = min(min_clearance, dist)

            if state.distance_to_point(goal[:3]) < 1.0:
                break

        # Should maintain some clearance
        assert min_clearance > -0.5


class TestReferenceGenerator:
    """Tests for reference trajectory generation."""

    def test_constant_reference(self):
        """Test constant reference generation."""
        gen = ReferenceGenerator(horizon=10, dt=0.1)
        goal = np.array([10, 5, 8, 0, 0, 0, 0, 0, 0, 0])

        ref = gen.generate_constant_reference(goal)

        assert ref.shape == (11, 10)
        assert np.allclose(ref[0], goal)
        assert np.allclose(ref[-1], goal)

    def test_linear_reference(self):
        """Test linear interpolation reference."""
        gen = ReferenceGenerator(horizon=10, dt=0.1, max_velocity=5.0)
        state = UAVState(position=[0, 0, 5])
        goal_pos = np.array([10, 0, 5])

        ref = gen.generate_linear_reference(state, goal_pos)

        assert ref.shape == (11, 10)
        # First position should be current
        assert np.allclose(ref[0, :3], state.position, atol=0.1)

    def test_smooth_reference(self):
        """Test smooth polynomial reference."""
        gen = ReferenceGenerator(horizon=20, dt=0.1, max_velocity=3.0)
        state = UAVState(position=[0, 0, 5], velocity=[1, 0, 0])
        goal_pos = np.array([5, 0, 5])

        ref = gen.generate_smooth_reference(state, goal_pos)

        assert ref.shape == (21, 10)
        # Trajectory should be smooth (no sudden jumps)
        velocities = ref[:, 3:6]
        for i in range(len(velocities) - 1):
            delta_v = np.linalg.norm(velocities[i + 1] - velocities[i])
            assert delta_v < 2.0  # Reasonable velocity change


class TestPerformanceMetrics:
    """Tests for verifying performance requirements."""

    @pytest.mark.slow
    def test_mpc_solve_time(self, quick_mpc_config, uav_config):
        """Test MPC solve time is within bounds."""
        mpc = MPCController(quick_mpc_config, uav_config)

        state = UAVState(position=[0, 0, 5])
        goal = np.array([10, 0, 5, 0, 0, 0, 0, 0, 0, 0])
        obstacle = SphereObstacle(center=[5, 0, 5], radius=1.0)

        solve_times = []
        for _ in range(10):
            solution = mpc.solve(state, goal, static_obstacles=[obstacle])
            solve_times.append(solution.solve_time)

            # Update state
            control = mpc.get_control(solution)
            x_next = mpc.dynamics.discrete_dynamics(
                state.to_mpc_state(), control.to_array()
            )
            state = UAVState.from_mpc_state(x_next)

        avg_solve_time = np.mean(solve_times)
        max_solve_time = np.max(solve_times)

        # Target: <50ms for 20Hz operation
        assert avg_solve_time < 0.1  # 100ms average (relaxed for CI)
        assert max_solve_time < 0.2  # 200ms max (relaxed)

    def test_tracking_error(self, quick_mpc_config, uav_config):
        """Test tracking error stays bounded."""
        mpc = MPCController(quick_mpc_config, uav_config)
        ref_gen = ReferenceGenerator(
            horizon=quick_mpc_config.horizon, dt=quick_mpc_config.dt
        )

        state = UAVState(position=[0, 0, 5])
        goal = np.array([5, 0, 5])

        errors = []
        for _ in range(30):
            reference = ref_gen.generate_smooth_reference(state, goal)
            solution = mpc.solve(state, reference)
            control = mpc.get_control(solution)

            x_next = mpc.dynamics.discrete_dynamics(
                state.to_mpc_state(), control.to_array()
            )
            state = UAVState.from_mpc_state(x_next)

            error = state.distance_to_point(goal)
            errors.append(error)

            if error < 0.5:
                break

        # Should converge to goal
        assert errors[-1] < 1.0
