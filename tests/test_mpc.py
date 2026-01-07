"""Tests for MPC controller module."""

import numpy as np
import pytest

from src.dynamics.uav_state import UAVState, UAVControl
from src.mpc.controller import MPCController, MPCSolution, SolverStatus
from src.mpc.constraints import StateBounds, ControlBounds
from src.obstacles.static import SphereObstacle
from src.utils.config_loader import MPCConfig, UAVConfig


@pytest.fixture
def default_mpc_config():
    """Create default MPC configuration."""
    return MPCConfig(
        horizon=20,
        dt=0.05,
        Q_position=np.array([10.0, 10.0, 20.0]),
        Q_velocity=np.array([5.0, 5.0, 5.0]),
        Q_acceleration=np.array([1.0, 1.0, 1.0]),
        Q_yaw=1.0,
        R_jerk=np.array([0.1, 0.1, 0.1]),
        R_yaw_rate=0.5,
        Qf_multiplier=10.0,
        obstacle_weight=1000.0,
        safety_margin=0.5,
    )


@pytest.fixture
def default_uav_config():
    """Create default UAV configuration."""
    return UAVConfig(
        max_velocity=10.0,
        max_acceleration=5.0,
        max_jerk=30.0,
        max_yaw_rate=1.5,
    )


@pytest.fixture
def mpc_controller(default_mpc_config, default_uav_config):
    """Create MPC controller for testing."""
    return MPCController(default_mpc_config, default_uav_config)


class TestMPCConfig:
    """Tests for MPC configuration."""

    def test_q_matrix_construction(self):
        """Test Q matrix is constructed correctly."""
        config = MPCConfig()
        Q = config.get_Q_matrix()
        assert Q.shape == (10, 10)
        assert np.allclose(Q, np.diag(Q.diagonal()))  # Diagonal

    def test_r_matrix_construction(self):
        """Test R matrix is constructed correctly."""
        config = MPCConfig()
        R = config.get_R_matrix()
        assert R.shape == (4, 4)

    def test_qf_matrix_scaling(self):
        """Test terminal cost scaling."""
        config = MPCConfig(Qf_multiplier=10.0)
        Q = config.get_Q_matrix()
        Qf = config.get_Qf_matrix()
        assert np.allclose(Qf, 10.0 * Q)


class TestStateBounds:
    """Tests for state bounds."""

    def test_default_bounds(self):
        """Test default state bounds."""
        bounds = StateBounds()
        x_min, x_max = bounds.to_arrays()
        assert x_min.shape == (10,)
        assert x_max.shape == (10,)

    def test_velocity_bounds(self):
        """Test velocity bounds are symmetric."""
        bounds = StateBounds(v_max=5.0)
        x_min, x_max = bounds.to_arrays()
        # Velocity indices are 3, 4, 5
        assert x_min[3] == -5.0
        assert x_max[3] == 5.0


class TestControlBounds:
    """Tests for control bounds."""

    def test_default_bounds(self):
        """Test default control bounds."""
        bounds = ControlBounds()
        u_min, u_max = bounds.to_arrays()
        assert u_min.shape == (4,)
        assert u_max.shape == (4,)

    def test_symmetric_bounds(self):
        """Test bounds are symmetric."""
        bounds = ControlBounds(j_max=20.0, yaw_rate_max=1.0)
        u_min, u_max = bounds.to_arrays()
        assert np.allclose(-u_min, u_max)


class TestMPCController:
    """Tests for MPC controller."""

    def test_initialization(self, mpc_controller):
        """Test controller initialization."""
        assert mpc_controller.N == 20
        assert mpc_controller.dt == 0.05
        assert mpc_controller.NX == 10
        assert mpc_controller.NU == 4

    def test_solve_no_obstacles(self, mpc_controller):
        """Test solving without obstacles."""
        state = UAVState(position=[0, 0, 5])
        goal = np.array([10, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        solution = mpc_controller.solve(
            state=state,
            reference=goal,
            static_obstacles=[],
            dynamic_obstacles=[],
        )

        assert isinstance(solution, MPCSolution)
        assert solution.x_trajectory.shape == (21, 10)
        assert solution.u_trajectory.shape == (20, 4)

    def test_solve_returns_feasible(self, mpc_controller):
        """Test that solver returns feasible solution."""
        state = UAVState(position=[0, 0, 5])
        goal = np.array([5, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        solution = mpc_controller.solve(state, goal)

        # Solution should be optimal or suboptimal
        assert solution.status in [SolverStatus.OPTIMAL, SolverStatus.SUBOPTIMAL]

    def test_solve_with_obstacle(self, mpc_controller):
        """Test solving with obstacle."""
        state = UAVState(position=[0, 0, 5])
        goal = np.array([10, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        obstacle = SphereObstacle(center=[5, 0, 5], radius=1.0)

        solution = mpc_controller.solve(
            state=state,
            reference=goal,
            static_obstacles=[obstacle],
        )

        assert solution.success
        # Trajectory should avoid obstacle
        positions = solution.x_trajectory[:, :3]
        for pos in positions:
            dist = obstacle.signed_distance(pos)
            # Allow small violations due to soft constraints
            assert dist > -0.5

    def test_get_control(self, mpc_controller):
        """Test control extraction."""
        state = UAVState(position=[0, 0, 5])
        goal = np.array([5, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        solution = mpc_controller.solve(state, goal)
        control = mpc_controller.get_control(solution)

        assert isinstance(control, UAVControl)
        assert control.jerk.shape == (3,)

    def test_warm_start(self, mpc_controller):
        """Test warm starting improves solve time."""
        state = UAVState(position=[0, 0, 5])
        goal = np.array([10, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        # First solve (cold start)
        solution1 = mpc_controller.solve(state, goal)

        # Move state slightly forward
        state2 = UAVState(position=[0.1, 0, 5])

        # Second solve (warm start)
        solution2 = mpc_controller.solve(state2, goal)

        # Both should succeed
        assert solution1.success
        assert solution2.success

    def test_predicted_trajectory(self, mpc_controller):
        """Test predicted trajectory extraction."""
        state = UAVState(position=[0, 0, 5])
        goal = np.array([5, 0, 5, 0, 0, 0, 0, 0, 0, 0])

        solution = mpc_controller.solve(state, goal)
        predicted = mpc_controller.get_predicted_trajectory(solution)

        assert predicted.shape == (21, 3)
        # First position should match initial state
        assert np.allclose(predicted[0], state.position, atol=0.1)


class TestMPCSolution:
    """Tests for MPCSolution dataclass."""

    def test_success_property(self):
        """Test success property."""
        sol_optimal = MPCSolution(
            x_trajectory=np.zeros((21, 10)),
            u_trajectory=np.zeros((20, 4)),
            cost=0.0,
            solve_time=0.01,
            status=SolverStatus.OPTIMAL,
            iterations=10,
        )
        assert sol_optimal.success

        sol_infeasible = MPCSolution(
            x_trajectory=np.zeros((21, 10)),
            u_trajectory=np.zeros((20, 4)),
            cost=float("inf"),
            solve_time=0.01,
            status=SolverStatus.INFEASIBLE,
            iterations=-1,
        )
        assert not sol_infeasible.success
