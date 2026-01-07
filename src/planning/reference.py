"""Reference trajectory generation for MPC."""

import numpy as np
from typing import Optional, Tuple

from ..dynamics.uav_state import UAVState
from .waypoints import Waypoint, WaypointManager


class ReferenceGenerator:
    """
    Generates reference trajectories for MPC.

    Produces smooth reference states over the prediction horizon
    based on current state and goal/waypoints.
    """

    def __init__(
        self,
        horizon: int,
        dt: float,
        max_velocity: float = 5.0,
        smoothing_factor: float = 0.8,
    ):
        """
        Initialize reference generator.

        Args:
            horizon: MPC prediction horizon (N)
            dt: Time step
            max_velocity: Maximum reference velocity
            smoothing_factor: Blend factor for velocity profiles (0-1)
        """
        self.horizon = horizon
        self.dt = dt
        self.max_velocity = max_velocity
        self.smoothing_factor = smoothing_factor

    def generate_constant_reference(
        self,
        goal_state: np.ndarray,
    ) -> np.ndarray:
        """
        Generate constant reference (goal state at all horizon steps).

        Args:
            goal_state: 10D target state

        Returns:
            (N+1, 10) reference trajectory
        """
        return np.tile(goal_state.reshape(1, -1), (self.horizon + 1, 1))

    def generate_linear_reference(
        self,
        current_state: UAVState,
        goal_position: np.ndarray,
        goal_yaw: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate linear interpolation to goal position.

        Creates a trajectory that smoothly moves from current state
        to goal position with velocity tapering near the goal.

        Args:
            current_state: Current UAV state
            goal_position: Target 3D position
            goal_yaw: Target yaw angle (optional)

        Returns:
            (N+1, 10) reference trajectory
        """
        goal_position = np.asarray(goal_position).flatten()
        if goal_yaw is None:
            goal_yaw = current_state.yaw

        # Compute direction and distance to goal
        direction = goal_position - current_state.position
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            # Already at goal
            goal_state = self._create_hover_state(goal_position, goal_yaw)
            return self.generate_constant_reference(goal_state)

        # Normalized direction
        direction_unit = direction / distance

        # Time to reach goal at max velocity
        time_to_goal = distance / self.max_velocity

        # Generate trajectory
        reference = np.zeros((self.horizon + 1, 10))

        for k in range(self.horizon + 1):
            t = k * self.dt
            progress = min(1.0, t / max(time_to_goal, 1e-6))

            # Position: linear interpolation
            pos = current_state.position + progress * direction

            # Velocity: taper near goal
            if progress < 0.8:
                vel = self.max_velocity * direction_unit
            else:
                # Smooth deceleration
                vel_magnitude = self.max_velocity * (1.0 - progress) / 0.2
                vel = vel_magnitude * direction_unit

            # Acceleration and yaw
            acc = np.zeros(3)
            yaw = self._interpolate_angle(current_state.yaw, goal_yaw, progress)

            reference[k] = np.concatenate([pos, vel, acc, [yaw]])

        return reference

    def generate_velocity_reference(
        self,
        current_state: UAVState,
        target_velocity: np.ndarray,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate reference for velocity tracking.

        Args:
            current_state: Current state
            target_velocity: Desired velocity vector
            duration: How long to maintain velocity (None = whole horizon)

        Returns:
            (N+1, 10) reference trajectory
        """
        target_velocity = np.asarray(target_velocity).flatten()

        reference = np.zeros((self.horizon + 1, 10))
        pos = current_state.position.copy()

        for k in range(self.horizon + 1):
            t = k * self.dt

            # Check if past duration
            if duration is not None and t > duration:
                vel = np.zeros(3)
            else:
                vel = target_velocity
                pos = pos + vel * self.dt if k > 0 else pos

            reference[k] = np.concatenate([
                pos,
                vel,
                np.zeros(3),
                [current_state.yaw]
            ])

        return reference

    def generate_smooth_reference(
        self,
        current_state: UAVState,
        goal_position: np.ndarray,
        goal_velocity: Optional[np.ndarray] = None,
        goal_yaw: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate smooth reference using polynomial blending.

        Creates a trajectory with smooth velocity and acceleration
        profiles using quintic polynomial interpolation.

        Args:
            current_state: Current state
            goal_position: Target position
            goal_velocity: Target velocity (default: zero)
            goal_yaw: Target yaw (default: current)

        Returns:
            (N+1, 10) reference trajectory
        """
        goal_position = np.asarray(goal_position).flatten()
        goal_velocity = (
            np.asarray(goal_velocity).flatten()
            if goal_velocity is not None
            else np.zeros(3)
        )
        goal_yaw = goal_yaw if goal_yaw is not None else current_state.yaw

        # Total time for trajectory
        distance = np.linalg.norm(goal_position - current_state.position)
        T = max(distance / self.max_velocity, self.horizon * self.dt)

        reference = np.zeros((self.horizon + 1, 10))

        for k in range(self.horizon + 1):
            t = k * self.dt
            s = min(1.0, t / T)  # Normalized time [0, 1]

            # Quintic polynomial blending (smooth start and end)
            # s_smooth = 6*s^5 - 15*s^4 + 10*s^3
            s_smooth = 6 * s**5 - 15 * s**4 + 10 * s**3

            # Position interpolation
            pos = (1 - s_smooth) * current_state.position + s_smooth * goal_position

            # Velocity blending
            # ds/dt for velocity
            if T > 1e-6:
                ds_dt = (30 * s**4 - 60 * s**3 + 30 * s**2) / T
            else:
                ds_dt = 0
            vel = ds_dt * (goal_position - current_state.position)

            # Clamp velocity magnitude
            vel_mag = np.linalg.norm(vel)
            if vel_mag > self.max_velocity:
                vel = vel * self.max_velocity / vel_mag

            # Blend toward goal velocity at end
            vel = (1 - s_smooth) * vel + s_smooth * goal_velocity

            # Acceleration (approximate)
            acc = np.zeros(3)

            # Yaw interpolation
            yaw = self._interpolate_angle(current_state.yaw, goal_yaw, s_smooth)

            reference[k] = np.concatenate([pos, vel, acc, [yaw]])

        return reference

    def generate_waypoint_reference(
        self,
        current_state: UAVState,
        waypoint_manager: WaypointManager,
    ) -> np.ndarray:
        """
        Generate reference from waypoint sequence.

        Args:
            current_state: Current state
            waypoint_manager: Waypoint manager with target sequence

        Returns:
            (N+1, 10) reference trajectory
        """
        target_state = waypoint_manager.get_target_state()
        if target_state is None:
            # No more waypoints, hover in place
            return self.generate_constant_reference(current_state.to_mpc_state())

        goal_position = target_state[:3]
        goal_yaw = target_state[9]

        return self.generate_smooth_reference(
            current_state,
            goal_position,
            goal_yaw=goal_yaw,
        )

    def _create_hover_state(
        self,
        position: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        """Create hover state at position."""
        return np.concatenate([position, np.zeros(6), [yaw]])

    @staticmethod
    def _interpolate_angle(
        start: float,
        end: float,
        t: float,
    ) -> float:
        """
        Interpolate between two angles (handles wrapping).

        Args:
            start: Start angle
            end: End angle
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated angle
        """
        # Compute shortest angular distance
        diff = end - start
        diff = ((diff + np.pi) % (2 * np.pi)) - np.pi

        return start + t * diff


def generate_reference_to_goal(
    current_state: UAVState,
    goal_position: np.ndarray,
    horizon: int,
    dt: float,
    max_velocity: float = 5.0,
) -> np.ndarray:
    """
    Convenience function to generate reference trajectory to goal.

    Args:
        current_state: Current UAV state
        goal_position: Target position
        horizon: MPC horizon
        dt: Time step
        max_velocity: Maximum velocity

    Returns:
        (N+1, 10) reference trajectory
    """
    generator = ReferenceGenerator(horizon, dt, max_velocity)
    return generator.generate_smooth_reference(current_state, goal_position)
