"""Numerical integrators for UAV dynamics simulation."""

import numpy as np
from typing import Callable

from .uav_state import UAVState, UAVControl
from .point_mass_model import PointMassModel


def euler_step(
    dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Euler (forward) integration step.

    x[k+1] = x[k] + dt * f(x[k], u[k])

    Args:
        dynamics_func: Function f(x, u) -> x_dot (continuous dynamics)
        x: Current state
        u: Control input
        dt: Timestep

    Returns:
        Next state
    """
    x_dot = dynamics_func(x, u)
    return x + dt * x_dot


def rk4_step(
    dynamics_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    4th-order Runge-Kutta integration step.

    Args:
        dynamics_func: Function f(x, u) -> x_dot (continuous dynamics)
        x: Current state
        u: Control input (assumed constant over timestep)
        dt: Timestep

    Returns:
        Next state
    """
    k1 = dynamics_func(x, u)
    k2 = dynamics_func(x + 0.5 * dt * k1, u)
    k3 = dynamics_func(x + 0.5 * dt * k2, u)
    k4 = dynamics_func(x + dt * k3, u)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class UAVSimulator:
    """
    High-level simulator for UAV dynamics.

    Provides convenient interface for stepping the UAV state forward
    using different integration methods.
    """

    def __init__(
        self,
        model: PointMassModel,
        integration_method: str = "discrete",
    ):
        """
        Initialize the simulator.

        Args:
            model: Point mass dynamics model
            integration_method: One of 'discrete', 'euler', 'rk4'
        """
        self.model = model
        self.integration_method = integration_method

        if integration_method not in ["discrete", "euler", "rk4"]:
            raise ValueError(
                f"Unknown integration method: {integration_method}. "
                "Choose from 'discrete', 'euler', 'rk4'"
            )

    def step(self, state: UAVState, control: UAVControl) -> UAVState:
        """
        Propagate state forward by one timestep.

        Args:
            state: Current UAV state
            control: Control input

        Returns:
            New UAV state
        """
        x = state.to_mpc_state()
        u = control.to_array()

        if self.integration_method == "discrete":
            # Use pre-computed discrete matrices (exact for linear system)
            x_next = self.model.discrete_dynamics(x, u)
        elif self.integration_method == "euler":
            x_next = euler_step(
                self.model.continuous_dynamics, x, u, self.model.dt
            )
            x_next[9] = self._wrap_angle(x_next[9])
        elif self.integration_method == "rk4":
            x_next = rk4_step(
                self.model.continuous_dynamics, x, u, self.model.dt
            )
            x_next[9] = self._wrap_angle(x_next[9])

        return UAVState.from_mpc_state(x_next)

    def simulate(
        self, initial_state: UAVState, controls: list[UAVControl]
    ) -> list[UAVState]:
        """
        Simulate trajectory from initial state with given control sequence.

        Args:
            initial_state: Starting state
            controls: List of control inputs

        Returns:
            List of states (length = len(controls) + 1)
        """
        states = [initial_state.copy()]
        state = initial_state

        for control in controls:
            state = self.step(state, control)
            states.append(state)

        return states

    def simulate_with_feedback(
        self,
        initial_state: UAVState,
        controller_func: Callable[[UAVState, float], UAVControl],
        duration: float,
        dt: float = None,
    ) -> tuple[list[UAVState], list[UAVControl], list[float]]:
        """
        Simulate with a feedback controller.

        Args:
            initial_state: Starting state
            controller_func: Function (state, time) -> control
            duration: Simulation duration in seconds
            dt: Override timestep (uses model.dt if None)

        Returns:
            Tuple of (states, controls, times)
        """
        if dt is None:
            dt = self.model.dt

        times = np.arange(0, duration + dt / 2, dt)
        states = [initial_state.copy()]
        controls = []
        state = initial_state

        for t in times[:-1]:
            control = controller_func(state, t)
            controls.append(control)
            state = self.step(state, control)
            states.append(state)

        return states, controls, times.tolist()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi


def create_simulator(
    dt: float = 0.05, integration_method: str = "discrete"
) -> UAVSimulator:
    """
    Factory function to create a UAV simulator.

    Args:
        dt: Timestep in seconds
        integration_method: Integration method ('discrete', 'euler', 'rk4')

    Returns:
        Configured UAVSimulator instance
    """
    model = PointMassModel(dt=dt)
    return UAVSimulator(model=model, integration_method=integration_method)
