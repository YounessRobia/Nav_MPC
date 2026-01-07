#!/usr/bin/env python3
"""
Static Obstacle Avoidance Example

Demonstrates MPC navigating around static spherical obstacles.

Usage:
    python static_avoidance.py           # Static plot only
    python static_avoidance.py --animate # With animation
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
from src.obstacles.static import SphereObstacle
from src.obstacles.collision import minimum_distance
from src.planning.reference import ReferenceGenerator
from src.utils.config_loader import MPCConfig, UAVConfig
from src.visualization.plotter_3d import plot_simulation_results
from src.visualization.animation import TrajectoryAnimator


def main(animate: bool = False):
    """Run static obstacle avoidance demonstration."""
    print("=== Static Obstacle Avoidance Example ===\n")

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
        Q_velocity=np.array([5.0, 5.0, 5.0]),
        Q_acceleration=np.array([1.0, 1.0, 1.0]),
        R_jerk=np.array([0.1, 0.1, 0.1]),
        Qf_multiplier=10.0,
        obstacle_weight=2000.0,
        safety_margin=0.5,
    )

    # Initialize components
    mpc = MPCController(mpc_config, uav_config)
    dynamics = PointMassModel(dt=mpc_config.dt)
    ref_gen = ReferenceGenerator(
        horizon=mpc_config.horizon,
        dt=mpc_config.dt,
        max_velocity=uav_config.max_velocity,
    )

    # Initial state and goal (altitude within [5, 25] constraint)
    state = UAVState(position=[0, 0, 10], velocity=[0, 0, 0])
    goal_position = np.array([25, 0, 10])

    # Create obstacles
    obstacles = [
        SphereObstacle(center=[8, 0, 10], radius=2.0, safety_margin=0.5),
        SphereObstacle(center=[15, 2, 10], radius=1.5, safety_margin=0.5),
        SphereObstacle(center=[15, -2, 10], radius=1.5, safety_margin=0.5),
        SphereObstacle(center=[20, 0, 11], radius=1.0, safety_margin=0.5),
    ]

    print(f"Start: {state.position}")
    print(f"Goal: {goal_position}")
    print(f"Obstacles: {len(obstacles)}")
    print()

    # Simulation
    log = SimulationLog()
    dt = mpc_config.dt
    max_duration = 15.0
    steps = int(max_duration / dt)
    min_clearance = float("inf")

    for step in range(steps):
        t = step * dt

        # Generate reference
        reference = ref_gen.generate_smooth_reference(state, goal_position)

        # Solve MPC
        solution = mpc.solve(
            state=state,
            reference=reference,
            static_obstacles=obstacles,
        )

        # Apply control
        control = mpc.get_control(solution)
        x_next = dynamics.discrete_dynamics(
            state.to_mpc_state(), control.to_array()
        )
        state = UAVState.from_mpc_state(x_next)

        # Log
        predicted = mpc.get_predicted_trajectory(solution)
        log.append(
            t=t,
            state=state,
            control=control,
            mpc_trajectory=predicted,
            solve_time=solution.solve_time,
        )

        # Track minimum clearance
        clearance = minimum_distance(state.position, obstacles)
        min_clearance = min(min_clearance, clearance)

        # Progress
        dist_to_goal = np.linalg.norm(state.position - goal_position)
        if step % 30 == 0:
            print(f"  t={t:.2f}s, dist_to_goal={dist_to_goal:.2f}m, "
                  f"clearance={clearance:.2f}m, solve_time={solution.solve_time*1000:.1f}ms")

        # Check termination
        if dist_to_goal < 0.5:
            print(f"\nGoal reached at t={t:.2f}s!")
            break

    # Statistics
    print(f"\n=== Results ===")
    print(f"Final position: {state.position}")
    print(f"Distance to goal: {np.linalg.norm(state.position - goal_position):.3f}m")
    print(f"Minimum obstacle clearance: {min_clearance:.3f}m")
    print(f"Average solve time: {np.mean(log.solve_times)*1000:.2f}ms")

    # Get trajectory data
    trajectory = log.get_position_trajectory()

    if animate:
        # === ANIMATED VISUALIZATION ===
        print("\nCreating animation...")

        animator = TrajectoryAnimator(
            interval=50,
            title="Static Obstacle Avoidance - Animation"
        )
        anim = animator.animate_trajectory(
            states=log.states,
            times=log.times,
            static_obstacles=obstacles,
            goal_position=goal_position,
            predicted_trajectories=log.mpc_trajectories,
            xlim=(-5, 30),
            ylim=(-10, 10),
            zlim=(0, 12),
            show_full_trajectory=True,
        )
        
        # Save animation
        animator.save("static_avoidance.gif", fps=20)
        print("Animation saved to static_avoidance.gif")
        
        # Show animation interactively
        plt.show(block=True)
    else:
        # === STATIC VISUALIZATION ===
        print("\nPlotting results...")
        plot_simulation_results(
            trajectory=trajectory,
            obstacles=obstacles,
            goal_position=goal_position,
            predicted_trajectories=log.mpc_trajectories,
            title="Static Obstacle Avoidance",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static Obstacle Avoidance Example")
    parser.add_argument("--animate", action="store_true", help="Show animated visualization")
    args = parser.parse_args()
    main(animate=args.animate)
