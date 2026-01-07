#!/usr/bin/env python3
"""
Urban Rescue Mission - Challenging Scenario

A UAV must navigate through an urban environment with:
- Static building obstacles (narrow passages)
- Moving patrol drones (dynamic obstacles)
- Multiple checkpoints (waypoints)
- Time-critical delivery to final destination

This demonstrates MPC handling combined challenges with live animation.

Usage:
    python urban_rescue.py           # Static plot only
    python urban_rescue.py --animate # With live animation
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
from src.obstacles.static import SphereObstacle, BoxObstacle
from src.obstacles.dynamic import (
    DynamicObstacle,
    create_linear_trajectory,
    create_circular_trajectory,
)
from src.obstacles.collision import minimum_distance
from src.planning.waypoints import WaypointManager
from src.planning.reference import ReferenceGenerator
from src.utils.config_loader import MPCConfig, UAVConfig
from src.visualization.plotter_3d import Plotter3D
from src.visualization.animation import TrajectoryAnimator


def main(animate: bool = False):
    """Run urban rescue mission demonstration."""
    print("=" * 60)
    print("       URBAN RESCUE MISSION - MPC Path Planning")
    print("=" * 60)
    print()
    print("Mission: Navigate through urban canyon, avoid patrol drones,")
    print("         reach all checkpoints, and deliver to destination.")
    print()

    # Configuration - tuned for challenging navigation
    uav_config = UAVConfig(
        max_velocity=7.0,
        max_acceleration=4.5,
        max_jerk=30.0,
        max_yaw_rate=2.0,
    )

    mpc_config = MPCConfig(
        horizon=25,
        dt=0.05,
        Q_position=np.array([20.0, 20.0, 28.0]),  # Stronger position tracking
        Q_velocity=np.array([5.0, 5.0, 5.0]),
        R_jerk=np.array([0.05, 0.05, 0.05]),  # Allow more aggressive maneuvers
        Qf_multiplier=15.0,
        obstacle_weight=800.0,  # Reduced to avoid local minima
        safety_margin=0.3,
    )

    # Initialize components
    mpc = MPCController(mpc_config, uav_config)
    dynamics = PointMassModel(dt=mpc_config.dt)
    ref_gen = ReferenceGenerator(
        horizon=mpc_config.horizon,
        dt=mpc_config.dt,
        max_velocity=uav_config.max_velocity,
    )

    # Initial state - starting at base
    state = UAVState(position=[0, 0, 10], velocity=[0, 0, 0])

    # Define mission waypoints - CAREFULLY PLACED to avoid obstacle conflicts
    # Each waypoint is positioned in clear corridors between obstacles
    waypoint_manager = WaypointManager()
    waypoints_data = [
        # Checkpoint 1: Clear entry point
        [7, 2, 10],
        # Checkpoint 2: Navigate ABOVE the central obstacle at z=14
        [14, 0, 17],  # Go high to clear obstacle at [18, 0, 14]
        # Checkpoint 3: Descend after clearing
        [22, 0, 10],
        # Checkpoint 4: Through the gap between buildings (y=0 is clear)
        [30, 0, 10],
        # Checkpoint 5: Final corridor
        [38, 0, 10],
        # Destination: Delivery point
        [45, 0, 10],
    ]

    for wp_pos in waypoints_data:
        waypoint_manager.add_position(
            position=np.array(wp_pos),
            tolerance=2.0,  # Slightly larger tolerance for smoother transitions
        )

    goal_position = np.array(waypoints_data[-1])

    # Create static obstacles (buildings) - positioned to create interesting but solvable paths
    static_obstacles = [
        # Building 1: South side of entry corridor
        SphereObstacle(center=[8, -5, 10], radius=2.0, safety_margin=0.3),

        # Building 2: North side, forces slight south deviation
        SphereObstacle(center=[11, 5, 10], radius=2.0, safety_margin=0.3),

        # Building 3: Central tower - UAV must go OVER this one
        SphereObstacle(center=[18, 0, 12], radius=2.5, safety_margin=0.3),

        # Building 4: Creates canyon walls (gap in middle at y=0)
        SphereObstacle(center=[26, 5, 10], radius=2.0, safety_margin=0.3),
        SphereObstacle(center=[26, -5, 10], radius=2.0, safety_margin=0.3),

        # Building 5: Final approach obstacles
        SphereObstacle(center=[34, 4, 10], radius=1.5, safety_margin=0.3),
        SphereObstacle(center=[34, -4, 11], radius=1.5, safety_margin=0.3),

        # Building 6: Near destination
        SphereObstacle(center=[42, 3, 10], radius=1.2, safety_margin=0.3),
    ]

    # Create dynamic obstacles (patrol drones) - predictable paths
    dynamic_obstacles = [
        # Patrol drone 1: Horizontal sweep in entry zone
        DynamicObstacle(
            radius=0.7,
            trajectory_func=create_linear_trajectory(
                initial_position=[12, -8, 10],
                velocity=[0, 1.5, 0],
            ),
            safety_margin=0.4,
        ),
        # Patrol drone 2: Orbits the canyon gap
        DynamicObstacle(
            radius=0.6,
            trajectory_func=create_circular_trajectory(
                center=[26, 0, 10],
                radius=7.0,
                angular_velocity=0.4,
                initial_phase=0,
            ),
            safety_margin=0.4,
        ),
        # Patrol drone 3: Oncoming from destination
        DynamicObstacle(
            radius=0.7,
            trajectory_func=create_linear_trajectory(
                initial_position=[42, -2, 10],
                velocity=[-1.2, 0.3, 0],
            ),
            safety_margin=0.4,
        ),
    ]

    # Combine all obstacles for collision checking
    all_obstacles = static_obstacles + dynamic_obstacles

    print("Mission Parameters:")
    print(f"  Start position: {state.position}")
    print(f"  Destination: {goal_position}")
    print(f"  Checkpoints: {len(waypoint_manager)}")
    print(f"  Static obstacles (buildings): {len(static_obstacles)}")
    print(f"  Dynamic obstacles (patrol drones): {len(dynamic_obstacles)}")
    print()

    print("Checkpoint Locations:")
    for i, wp in enumerate(waypoint_manager.waypoints):
        label = "DESTINATION" if i == len(waypoint_manager.waypoints) - 1 else f"Checkpoint {i+1}"
        print(f"  {label}: {wp.position}")
    print()

    # Simulation
    log = SimulationLog()
    dt = mpc_config.dt
    max_duration = 25.0
    steps = int(max_duration / dt)
    min_clearance = float("inf")
    waypoint_times = []

    # Stuck detection parameters
    stuck_threshold = 0.3  # If moved less than this in stuck_window, consider stuck
    stuck_window = 2.0  # seconds to check for being stuck
    stuck_window_steps = int(stuck_window / dt)
    position_history = []
    stuck_count = 0
    max_stuck_before_skip = 3  # Skip waypoint after being stuck this many times

    print("Mission Progress:")
    print("-" * 50)

    for step in range(steps):
        t = step * dt

        # Update dynamic obstacle positions
        for obs in dynamic_obstacles:
            obs.update_time(t)

        # Check waypoint progress
        switched = waypoint_manager.update(state.position)
        if switched:
            waypoint_times.append(t)
            wp_idx = waypoint_manager.current_index
            stuck_count = 0  # Reset stuck counter on waypoint reached
            if wp_idx < len(waypoint_manager.waypoints):
                print(f"  ✓ t={t:.2f}s: Reached checkpoint {wp_idx}")
            else:
                print(f"  ★ t={t:.2f}s: DESTINATION REACHED!")

        # Stuck detection - check if UAV hasn't moved much
        position_history.append(state.position.copy())
        if len(position_history) > stuck_window_steps:
            position_history.pop(0)
            displacement = np.linalg.norm(
                np.array(position_history[-1]) - np.array(position_history[0])
            )
            if displacement < stuck_threshold:
                stuck_count += 1
                if stuck_count == 1:
                    print(f"  ⚠ t={t:.2f}s: Potential local minimum detected...")
                if stuck_count >= max_stuck_before_skip * stuck_window_steps:
                    # Skip to next waypoint
                    if waypoint_manager.current_index < len(waypoint_manager.waypoints) - 1:
                        print(f"  → t={t:.2f}s: Skipping stuck waypoint, proceeding to next")
                        waypoint_manager._current_index += 1
                        stuck_count = 0
                        position_history.clear()

        # Check if mission complete
        if waypoint_manager.is_complete:
            print()
            print("=" * 50)
            print("       MISSION ACCOMPLISHED!")
            print("=" * 50)
            break

        # Get current target
        current_target = waypoint_manager.current_waypoint
        if current_target is None:
            break

        # Generate reference toward current waypoint
        reference = ref_gen.generate_smooth_reference(
            state, current_target.position
        )

        # Solve MPC with all obstacles
        solution = mpc.solve(
            state=state,
            reference=reference,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
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

        # Track minimum clearance
        clearance = minimum_distance(state.position, all_obstacles)
        min_clearance = min(min_clearance, clearance)

        # Progress update every 2 seconds
        if step % 40 == 0 and current_target:
            dist = state.distance_to_point(current_target.position)
            wp_label = f"CP{waypoint_manager.current_index + 1}"
            if waypoint_manager.current_index == len(waypoint_manager.waypoints) - 1:
                wp_label = "DEST"
            print(f"  t={t:.2f}s | Target: {wp_label} | Dist: {dist:.1f}m | "
                  f"Clearance: {clearance:.2f}m | Alt: {state.position[2]:.1f}m")

    # Statistics
    print()
    print("=" * 50)
    print("MISSION STATISTICS")
    print("=" * 50)
    print(f"  Final position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]")
    print(f"  Checkpoints reached: {waypoint_manager.current_index}/{len(waypoint_manager)}")
    print(f"  Total flight time: {log.times[-1]:.2f}s")
    print(f"  Minimum obstacle clearance: {min_clearance:.3f}m")
    print(f"  Average MPC solve time: {np.mean(log.solve_times)*1000:.2f}ms")
    print(f"  Max MPC solve time: {np.max(log.solve_times)*1000:.2f}ms")

    # Flight statistics
    velocities = np.array([s.velocity for s in log.states])
    speeds = np.linalg.norm(velocities, axis=1)
    print(f"  Average speed: {np.mean(speeds):.2f} m/s")
    print(f"  Max speed: {np.max(speeds):.2f} m/s")

    # Get trajectory data
    trajectory = log.get_position_trajectory()
    waypoints_array = waypoint_manager.get_all_positions()

    if animate:
        # === ANIMATED VISUALIZATION ===
        print()
        print("Creating animation...")
        print("(Press Ctrl+C to stop)")

        # Reset dynamic obstacles for animation
        for obs in dynamic_obstacles:
            obs.update_time(0)

        animator = TrajectoryAnimator(
            interval=50,
            title="Urban Rescue Mission - MPC Navigation"
        )

        anim = animator.animate_trajectory(
            states=log.states,
            times=log.times,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            goal_position=goal_position,
            predicted_trajectories=log.mpc_trajectories,
            xlim=(-5, 50),
            ylim=(-12, 12),
            zlim=(5, 22),
            show_full_trajectory=True,
        )

        # Save animation
        animator.save("urban_rescue.gif", fps=20)
        print("Animation saved to urban_rescue.gif")

        # Show animation interactively
        plt.show(block=True)
    else:
        # === STATIC VISUALIZATION ===
        print()
        print("Plotting results...")

        # Reset obstacles for plotting
        for obs in dynamic_obstacles:
            obs.update_time(0)

        plotter = Plotter3D(title="Urban Rescue Mission - Flight Path")
        plotter.setup_axes(xlim=(-5, 50), ylim=(-12, 12), zlim=(5, 22))

        # Plot trajectory
        plotter.plot_trajectory(trajectory, label="UAV Flight Path")

        # Plot waypoints
        plotter.plot_waypoints(waypoints_array)

        # Plot static obstacles (buildings)
        plotter.plot_obstacles(static_obstacles, color="gray", alpha=0.6)

        # Plot dynamic obstacle paths
        for i, obs in enumerate(dynamic_obstacles):
            obs_positions = obs.predict_over_horizon(0, 0.2, 125)
            obs_traj = np.array(obs_positions)
            plotter.ax.plot(
                obs_traj[:, 0], obs_traj[:, 1], obs_traj[:, 2],
                "r--", alpha=0.4, linewidth=1, label=f"Drone {i+1} patrol" if i == 0 else None
            )

        # Plot dynamic obstacles at initial positions
        plotter.plot_obstacles(dynamic_obstacles, color="red", alpha=0.5)

        # Plot start and destination
        plotter.ax.scatter([0], [0], [10], c="green", s=200, marker="^", label="Start (Base)")
        plotter.plot_goal(goal_position, label="Destination")

        # Connect waypoints with dashed line
        plotter.ax.plot(
            waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2],
            "g--", alpha=0.4, linewidth=1, label="Mission Route"
        )

        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Urban Rescue Mission - Challenging MPC Scenario"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show live animated visualization"
    )
    args = parser.parse_args()
    main(animate=args.animate)
