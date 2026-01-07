"""Tests for UAV dynamics module."""

import numpy as np
import pytest

from src.dynamics.uav_state import UAVState, UAVControl, SimulationLog
from src.dynamics.point_mass_model import PointMassModel
from src.dynamics.integrators import UAVSimulator, euler_step, rk4_step


class TestUAVState:
    """Tests for UAVState class."""

    def test_creation_default(self):
        """Test default state creation."""
        state = UAVState()
        assert np.allclose(state.position, [0, 0, 0])
        assert np.allclose(state.velocity, [0, 0, 0])
        assert np.allclose(state.acceleration, [0, 0, 0])
        assert state.yaw == 0.0

    def test_creation_with_values(self):
        """Test state creation with specific values."""
        state = UAVState(
            position=[1, 2, 3],
            velocity=[4, 5, 6],
            acceleration=[0.1, 0.2, 0.3],
            yaw=np.pi / 4,
        )
        assert np.allclose(state.position, [1, 2, 3])
        assert np.allclose(state.velocity, [4, 5, 6])
        assert state.x == 1.0
        assert state.y == 2.0
        assert state.z == 3.0

    def test_to_mpc_state(self):
        """Test conversion to MPC state vector."""
        state = UAVState(
            position=[1, 2, 3],
            velocity=[4, 5, 6],
            acceleration=[7, 8, 9],
            yaw=0.5,
        )
        mpc_state = state.to_mpc_state()
        assert mpc_state.shape == (10,)
        assert np.allclose(mpc_state, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0.5])

    def test_from_mpc_state(self):
        """Test creation from MPC state vector."""
        mpc_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0.5])
        state = UAVState.from_mpc_state(mpc_state)
        assert np.allclose(state.position, [1, 2, 3])
        assert np.allclose(state.velocity, [4, 5, 6])
        assert np.allclose(state.acceleration, [7, 8, 9])
        assert np.isclose(state.yaw, 0.5)

    def test_yaw_wrapping(self):
        """Test that yaw is wrapped to [-pi, pi]."""
        state = UAVState(position=[0, 0, 0], yaw=3 * np.pi)
        assert -np.pi <= state.yaw <= np.pi

    def test_speed_computation(self):
        """Test speed property."""
        state = UAVState(velocity=[3, 4, 0])
        assert np.isclose(state.speed, 5.0)

    def test_distance_to(self):
        """Test distance computation."""
        state1 = UAVState(position=[0, 0, 0])
        state2 = UAVState(position=[3, 4, 0])
        assert np.isclose(state1.distance_to(state2), 5.0)


class TestUAVControl:
    """Tests for UAVControl class."""

    def test_creation_default(self):
        """Test default control creation."""
        control = UAVControl()
        assert np.allclose(control.jerk, [0, 0, 0])
        assert control.yaw_rate == 0.0

    def test_from_array(self):
        """Test creation from array."""
        u = np.array([1, 2, 3, 0.5])
        control = UAVControl.from_array(u)
        assert np.allclose(control.jerk, [1, 2, 3])
        assert control.yaw_rate == 0.5

    def test_to_array(self):
        """Test conversion to array."""
        control = UAVControl(jerk=[1, 2, 3], yaw_rate=0.5)
        u = control.to_array()
        assert np.allclose(u, [1, 2, 3, 0.5])


class TestPointMassModel:
    """Tests for PointMassModel class."""

    def test_initialization(self):
        """Test model initialization."""
        model = PointMassModel(dt=0.05)
        assert model.dt == 0.05
        assert model.NX == 10
        assert model.NU == 4

    def test_discrete_matrices_shape(self):
        """Test discrete matrix shapes."""
        model = PointMassModel(dt=0.05)
        Ad, Bd = model.get_discrete_matrices()
        assert Ad.shape == (10, 10)
        assert Bd.shape == (10, 4)

    def test_zero_control_constant_velocity(self):
        """Test that zero control maintains constant velocity."""
        model = PointMassModel(dt=0.1)

        # Initial state with velocity, zero acceleration
        x0 = np.array([0, 0, 5, 1, 0, 0, 0, 0, 0, 0])  # Moving in x
        u = np.zeros(4)

        x1 = model.discrete_dynamics(x0, u)

        # Position should increase by v*dt
        assert np.isclose(x1[0], 0.1, atol=1e-6)  # x += vx*dt
        # Velocity should remain constant
        assert np.isclose(x1[3], 1.0, atol=1e-6)

    def test_jerk_affects_acceleration(self):
        """Test that jerk input changes acceleration."""
        model = PointMassModel(dt=0.1)

        x0 = np.zeros(10)
        x0[2] = 5  # Start at z=5
        u = np.array([10, 0, 0, 0])  # Jerk in x direction

        x1 = model.discrete_dynamics(x0, u)

        # Acceleration should increase
        assert x1[6] > 0  # ax > 0

    def test_step_method(self):
        """Test step method with UAVState/UAVControl."""
        model = PointMassModel(dt=0.05)

        state = UAVState(position=[0, 0, 5], velocity=[1, 0, 0])
        control = UAVControl(jerk=[0, 0, 0], yaw_rate=0)

        new_state = model.step(state, control)

        # Position should advance
        assert new_state.x > 0

    def test_trajectory_simulation(self):
        """Test trajectory simulation."""
        model = PointMassModel(dt=0.05)

        x0 = np.array([0, 0, 5, 1, 0, 0, 0, 0, 0, 0])
        controls = np.zeros((10, 4))

        trajectory = model.simulate_trajectory(x0, controls)

        assert trajectory.shape == (11, 10)
        assert np.allclose(trajectory[0], x0)
        # Position should increase monotonically
        assert all(trajectory[i + 1, 0] > trajectory[i, 0] for i in range(10))


class TestIntegrators:
    """Tests for integrator functions."""

    def test_euler_step(self):
        """Test Euler integration."""
        # Simple linear dynamics: dx/dt = u
        def dynamics(x, u):
            return u

        x = np.array([0.0])
        u = np.array([1.0])
        dt = 0.1

        x_new = euler_step(dynamics, x, u, dt)
        assert np.isclose(x_new[0], 0.1)

    def test_rk4_step(self):
        """Test RK4 integration."""
        # Simple exponential growth: dx/dt = x
        def dynamics(x, u):
            return x

        x = np.array([1.0])
        u = np.array([0.0])
        dt = 0.1

        x_new = rk4_step(dynamics, x, u, dt)
        # Should approximate e^0.1 ≈ 1.1052
        assert np.isclose(x_new[0], np.exp(0.1), rtol=1e-4)


class TestUAVSimulator:
    """Tests for UAVSimulator class."""

    def test_discrete_integration(self):
        """Test discrete integration method."""
        model = PointMassModel(dt=0.05)
        simulator = UAVSimulator(model, integration_method="discrete")

        state = UAVState(position=[0, 0, 5])
        control = UAVControl.zero()

        new_state = simulator.step(state, control)
        assert isinstance(new_state, UAVState)

    def test_simulation_sequence(self):
        """Test simulating a sequence of controls."""
        model = PointMassModel(dt=0.05)
        simulator = UAVSimulator(model)

        initial_state = UAVState(position=[0, 0, 5])
        controls = [UAVControl.zero() for _ in range(10)]

        states = simulator.simulate(initial_state, controls)
        assert len(states) == 11


class TestSimulationLog:
    """Tests for SimulationLog class."""

    def test_append_and_retrieve(self):
        """Test logging and retrieval."""
        log = SimulationLog()

        for i in range(5):
            state = UAVState(position=[i, 0, 5])
            control = UAVControl.zero()
            log.append(t=i * 0.1, state=state, control=control)

        assert len(log) == 5
        trajectory = log.get_position_trajectory()
        assert trajectory.shape == (5, 3)
        assert np.allclose(trajectory[:, 0], [0, 1, 2, 3, 4])
