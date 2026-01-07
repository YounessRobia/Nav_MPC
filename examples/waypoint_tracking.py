#!/usr/bin/env python3
"""
Waypoint Tracking Example

Demonstrates MPC following a sequence of waypoints.

Usage:
    python waypoint_tracking.py           # Static plot only
    python waypoint_tracking.py --animate # With animation
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix matplotlib path BEFORE any visualization imports
import examples._fix_matplotlib  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.uav_state import UAVState, SimulationLog
from src.dynamics.point_mass_model import PointMassModel
from src.mpc.controller import MPCController
from src.planning.waypoints import WaypointManager, Waypoint
from src.planning.reference import ReferenceGenerator
from src.utils.config_loader import MPCConfig, UAVConfig
from src.visualization.plotter_3d import Plotter3D
from src.visualization.animation import TrajectoryAnimator


def main(animate: bool = False):
    """Run waypoint tracking demonstration."""
    print("=== Waypoint Tracking Example ===\n")

    # Configuration
    uav_config = UAVConfig(
        max_velocity=8.0,
        max_acceleration=4.0,
        max_jerk=25.0,
        max_yaw_rate=1.5,
    )

    mpc_config = MPCConfig(
        horizon=25,
        dt=0.05,
        Q_position=np.array([15.0, 15.0, 20.0]),
        Q_velocity=np.array([8.0, 8.0, 8.0]),
        R_jerk=np.array([0.1, 0.1, 0.1]),
        Qf_multiplier=10.0,
    )

    # Initialize components
    mpc = MPCController(mpc_config, uav_config)
    dynamics = PointMassModel(dt=mpc_config.dt)
    ref_gen = ReferenceGenerator(
        horizon=mpc_config.horizon,
        dt=mpc_config.dt,
        max_velocity=uav_config.max_velocity,
    )

    # Initial state (altitude within [5, 25] constraint)
    state = UAVState(position=[0, 0, 10], velocity=[0, 0, 0])

    # Define waypoints (figure-8 pattern at varying altitudes within [5, 25])
    waypoint_manager = WaypointManager()
    waypoints_data = [
        [5, 5, 12],
        [10, 0, 15],
        [5, -5, 12],
        [0, 0, 8],
        [-5, 5, 12],
        [-10, 0, 15],
        [-5, -5, 12],
        [0, 0, 10],  # Return to center
    ]

    for wp_pos in waypoints_data:
        waypoint_manager.add_position(
            position=np.array(wp_pos),
            tolerance=1.0,
        )

    print(f"Start: {state.position}")
    print(f"Waypoints: {len(waypoint_manager)}")
    for i, wp in enumerate(waypoint_manager.waypoints):
        print(f"  {i+1}. {wp.position}")
    print()

    # Simulation
    log = SimulationLog()
    dt = mpc_config.dt
    max_duration = 30.0
    steps = int(max_duration / dt)
    waypoint_times = []

    for step in range(steps):
        t = step * dt

        # Check waypoint progress
        switched = waypoint_manager.update(state.position)
        if switched:
            waypoint_times.append(t)
            wp_idx = waypoint_manager.current_index
            print(f"  t={t:.2f}s: Reached waypoint {wp_idx}")

        # Check if all waypoints completed
        if waypoint_manager.is_complete:
            print(f"\nAll waypoints reached at t={t:.2f}s!")
            break

        # Get current target
        current_target = waypoint_manager.current_waypoint
        if current_target is None:
            break

        # Generate reference toward current waypoint
        reference = ref_gen.generate_smooth_reference(
            state, current_target.position
        )

        # Solve MPC
        solution = mpc.solve(state=state, reference=reference)

        # Apply control
        control = mpc.get_control(solution)
        x_next = dynamics.discrete_dynamics(
            state.to_mpc_state(), control.to_array()
        )
        state = UAVState.from_mpc_state(x_next)

        # Log
        log.append(
            t=t,
            state=state,
            control=control,
            solve_time=solution.solve_time,
        )

        # Progress
        if step % 50 == 0 and current_target:
            dist = state.distance_to_point(current_target.position)
            print(f"  t={t:.2f}s, targeting waypoint {waypoint_manager.current_index+1}, "
                  f"dist={dist:.2f}m")

    # Statistics
    print(f"\n=== Results ===")
    print(f"Final position: {state.position}")
    print(f"Waypoints reached: {waypoint_manager.current_index}/{len(waypoint_manager)}")
    print(f"Total time: {log.times[-1]:.2f}s")
    print(f"Average solve time: {np.mean(log.solve_times)*1000:.2f}ms")

    # Get data
    trajectory = log.get_position_trajectory()
    waypoints_array = waypoint_manager.get_all_positions()
    # Final waypoint as goal
    goal_position = waypoints_array[-1] if len(waypoints_array) > 0 else None

    if animate:
        # === ANIMATED VISUALIZATION ===
        print("\nCreating animation...")

        animator = TrajectoryAnimator(
            interval=50,
            title="Waypoint Tracking - Figure-8 Pattern Animation"
        )
        anim = animator.animate_trajectory(
            states=log.states,
            times=log.times,
            goal_position=goal_position,
            xlim=(-15, 15),
            ylim=(-10, 10),
            zlim=(5, 20),
            show_full_trajectory=True,
        )
        
        # Save animation
        animator.save("waypoint_tracking.gif", fps=20)
        print("Animation saved to waypoint_tracking.gif")
        
        # Show animation interactively
        plt.show(block=True)
    else:
        # === STATIC VISUALIZATION ===
        print("\nPlotting results...")
        plotter = Plotter3D(title="Waypoint Tracking")
        plotter.setup_axes(xlim=(-15, 15), ylim=(-10, 10), zlim=(5, 20))

        # Plot trajectory
        plotter.plot_trajectory(trajectory, label="UAV Trajectory")

        # Plot waypoints
        plotter.plot_waypoints(waypoints_array)

        # Plot start and end
        plotter.ax.scatter([0], [0], [10], c="green", s=150, marker="^", label="Start")
        plotter.ax.scatter(
            [trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
            c="red", s=100, marker="v", label="End"
        )

        # Connect waypoints with dashed line
        plotter.ax.plot(
            waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2],
            "g--", alpha=0.5, linewidth=1, label="Waypoint Path"
        )

        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waypoint Tracking Example")
    parser.add_argument("--animate", action="store_true", help="Show animated visualization")
    args = parser.parse_args()
    main(animate=args.animate)
