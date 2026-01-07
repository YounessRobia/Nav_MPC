#!/usr/bin/env python3
"""
Dynamic Obstacle Avoidance Example

Demonstrates MPC navigating around moving obstacles with known trajectories.

Usage:
    python dynamic_avoidance.py           # Static plot only
    python dynamic_avoidance.py --animate # With animation
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
from src.obstacles.dynamic import (
    DynamicObstacle,
    create_linear_trajectory,
    create_circular_trajectory,
)
from src.obstacles.collision import minimum_distance
from src.planning.reference import ReferenceGenerator
from src.utils.config_loader import MPCConfig, UAVConfig
from src.visualization.plotter_3d import Plotter3D
from src.visualization.animation import TrajectoryAnimator


def main(animate: bool = False):
    """Run dynamic obstacle avoidance demonstration."""
    print("=== Dynamic Obstacle Avoidance Example ===\n")

    # Configuration
    uav_config = UAVConfig(
        max_velocity=6.0,
        max_acceleration=4.0,
        max_jerk=25.0,
        max_yaw_rate=1.5,
    )

    mpc_config = MPCConfig(
        horizon=30,
        dt=0.05,
        Q_position=np.array([15.0, 15.0, 20.0]),
        Q_velocity=np.array([5.0, 5.0, 5.0]),
        R_jerk=np.array([0.1, 0.1, 0.1]),
        Qf_multiplier=10.0,
        obstacle_weight=2000.0,
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

    # Create dynamic obstacles with known trajectories
    obstacles = [
        # Obstacle crossing path perpendicular to UAV
        DynamicObstacle(
            radius=1.0,
            trajectory_func=create_linear_trajectory(
                initial_position=[12, -12, 10],
                velocity=[0, 2.5, 0],
            ),
            safety_margin=0.5,
        ),
        # Oncoming obstacle
        DynamicObstacle(
            radius=0.8,
            trajectory_func=create_linear_trajectory(
                initial_position=[28, 0.5, 10],
                velocity=[-2.0, 0, 0],
            ),
            safety_margin=0.5,
        ),
        # Orbiting obstacle
        DynamicObstacle(
            radius=0.6,
            trajectory_func=create_circular_trajectory(
                center=[18, 0, 10],
                radius=3.0,
                angular_velocity=0.8,
                initial_phase=0,
            ),
            safety_margin=0.5,
        ),
    ]

    print(f"Start: {state.position}")
    print(f"Goal: {goal_position}")
    print(f"Dynamic obstacles: {len(obstacles)}")
    print("  - Crossing obstacle (linear)")
    print("  - Oncoming obstacle (linear)")
    print("  - Orbiting obstacle (circular)")
    print()

    # Simulation
    log = SimulationLog()
    dt = mpc_config.dt
    max_duration = 15.0
    steps = int(max_duration / dt)
    min_clearance = float("inf")

    for step in range(steps):
        t = step * dt

        # Update dynamic obstacle positions for visualization
        for obs in obstacles:
            obs.update_time(t)

        # Generate reference
        reference = ref_gen.generate_smooth_reference(state, goal_position)

        # Solve MPC (dynamic obstacles evaluated at future times internally)
        solution = mpc.solve(
            state=state,
            reference=reference,
            dynamic_obstacles=obstacles,
            t_current=t,
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

        # Track minimum clearance (at current time)
        clearance = minimum_distance(state.position, obstacles)
        min_clearance = min(min_clearance, clearance)

        # Progress
        dist_to_goal = np.linalg.norm(state.position - goal_position)
        if step % 40 == 0:
            obs_pos = [obs.current_position for obs in obstacles]
            print(f"  t={t:.2f}s, dist_to_goal={dist_to_goal:.2f}m, "
                  f"clearance={clearance:.2f}m")

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
        
        # Reset obstacles for animation
        for obs in obstacles:
            obs.update_time(0)

        animator = TrajectoryAnimator(
            interval=50,
            title="Dynamic Obstacle Avoidance - Animation"
        )
        anim = animator.animate_trajectory(
            states=log.states,
            times=log.times,
            dynamic_obstacles=obstacles,
            goal_position=goal_position,
            predicted_trajectories=log.mpc_trajectories,
            xlim=(-5, 30),
            ylim=(-15, 15),
            zlim=(0, 12),
            show_full_trajectory=True,
        )
        
        # Save animation
        animator.save("dynamic_avoidance.gif", fps=20)
        print("Animation saved to dynamic_avoidance.gif")
        
        # Show animation interactively
        plt.show(block=True)
    else:
        # === STATIC VISUALIZATION ===
        print("\nPlotting results...")

        # Reset obstacles for plotting
        for obs in obstacles:
            obs.update_time(0)

        plotter = Plotter3D(title="Dynamic Obstacle Avoidance")
        plotter.setup_axes(xlim=(-5, 30), ylim=(-15, 15), zlim=(0, 12))

        # Plot trajectory
        plotter.plot_trajectory(trajectory, label="UAV Trajectory")
        plotter.plot_goal(goal_position)

        # Plot obstacle paths
        for i, obs in enumerate(obstacles):
            obs_positions = obs.predict_over_horizon(0, 0.2, 75)
            obs_traj = np.array(obs_positions)
            plotter.ax.plot(
                obs_traj[:, 0], obs_traj[:, 1], obs_traj[:, 2],
                "r--", alpha=0.3, linewidth=1, label=f"Obstacle {i+1} path"
            )

        # Plot obstacles at initial positions
        plotter.plot_obstacles(obstacles, color="orange", alpha=0.5)

        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Obstacle Avoidance Example")
    parser.add_argument("--animate", action="store_true", help="Show animated visualization")
    args = parser.parse_args()
    main(animate=args.animate)
