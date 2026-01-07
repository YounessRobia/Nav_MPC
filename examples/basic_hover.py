#!/usr/bin/env python3
"""
Basic Hover Example

Demonstrates MPC-based position holding (hovering) at a fixed location.

Usage:
    python basic_hover.py           # Static plot only
    python basic_hover.py --animate # With animation
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
from src.utils.config_loader import MPCConfig, UAVConfig
from src.visualization.plotter_3d import Plotter3D
from src.visualization.animation import TrajectoryAnimator


def main(animate: bool = False):
    """Run basic hover demonstration."""
    print("=== Basic Hover Example ===\n")

    # Configuration
    uav_config = UAVConfig(
        max_velocity=5.0,
        max_acceleration=3.0,
        max_jerk=20.0,
        max_yaw_rate=1.0,
    )

    mpc_config = MPCConfig(
        horizon=20,
        dt=0.05,
        Q_position=np.array([20.0, 20.0, 30.0]),
        Q_velocity=np.array([10.0, 10.0, 10.0]),
        Q_acceleration=np.array([1.0, 1.0, 1.0]),
        R_jerk=np.array([0.1, 0.1, 0.1]),
        Qf_multiplier=15.0,
    )

    # Initialize MPC controller
    mpc = MPCController(mpc_config, uav_config)
    dynamics = PointMassModel(dt=mpc_config.dt)

    # Initial state (slightly perturbed from hover point, altitude within [5, 25])
    state = UAVState(
        position=[0.5, -0.3, 9.8],
        velocity=[0.2, 0.1, -0.1],
        acceleration=[0.0, 0.0, 0.0],
        yaw=0.1,
    )

    # Hover target (altitude within [5, 25] constraint)
    hover_target = np.array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0])

    # Simulation
    log = SimulationLog()
    dt = mpc_config.dt
    duration = 5.0
    steps = int(duration / dt)

    print(f"Initial position: {state.position}")
    print(f"Hover target: {hover_target[:3]}")
    print(f"Simulating for {duration}s...\n")

    for step in range(steps):
        t = step * dt

        # Solve MPC
        solution = mpc.solve(state, hover_target)

        if not solution.success:
            print(f"Warning: MPC failed at step {step}")

        # Extract and apply control
        control = mpc.get_control(solution)
        x_next = dynamics.discrete_dynamics(
            state.to_mpc_state(), control.to_array()
        )
        state = UAVState.from_mpc_state(x_next)

        # Log
        log.append(t=t, state=state, control=control)

        # Progress
        if step % 20 == 0:
            error = np.linalg.norm(state.position - hover_target[:3])
            print(f"  t={t:.2f}s, position error={error:.4f}m")

    # Final error
    final_error = np.linalg.norm(state.position - hover_target[:3])
    print(f"\nFinal position error: {final_error:.4f}m")
    print(f"Final velocity: {state.velocity}")

    # Get trajectory data
    trajectory = log.get_position_trajectory()

    if animate:
        # === ANIMATED VISUALIZATION ===
        print("\nCreating animation...")

        animator = TrajectoryAnimator(
            interval=50,
            title="Basic Hover - Position Stabilization Animation"
        )
        anim = animator.animate_trajectory(
            states=log.states,
            times=log.times,
            goal_position=hover_target[:3],
            xlim=(-2, 2),
            ylim=(-2, 2),
            zlim=(8, 12),
            show_full_trajectory=True,
        )
        
        # Save animation
        animator.save("basic_hover.gif", fps=20)
        print("Animation saved to basic_hover.gif")
        
        # Show animation interactively
        plt.show(block=True)
    else:
        # === STATIC VISUALIZATION ===
        print("\nPlotting results...")
        plotter = Plotter3D(title="Basic Hover - Position Stabilization")
        plotter.setup_axes(xlim=(-2, 2), ylim=(-2, 2), zlim=(8, 12))

        plotter.plot_trajectory(trajectory, label="Actual Trajectory")
        plotter.plot_goal(hover_target[:3])
        plotter.ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
                           c="green", s=100, marker="^", label="Start")

        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Hover Example")
    parser.add_argument("--animate", action="store_true", help="Show animated visualization")
    args = parser.parse_args()
    main(animate=args.animate)
