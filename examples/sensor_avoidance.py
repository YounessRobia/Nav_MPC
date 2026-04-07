#!/usr/bin/env python3
"""
Sensor-Based Obstacle Avoidance Example

Demonstrates MPC navigation where obstacles are discovered through simulated
sensing rather than known a priori. The UAV has a limited field-of-view sensor
and must detect obstacles within range before avoiding them.

Key Features:
    - Obstacles are unknown initially
    - Detection within configurable FOV cone and range
    - Dynamic obstacle trajectories estimated from observations
    - MPC plans using only discovered obstacles

Usage:
    python sensor_avoidance.py                    # Static plot only
    python sensor_avoidance.py --animate          # With animation
    python sensor_avoidance.py --radius 20        # Custom detection radius
    python sensor_avoidance.py --fov 180          # Custom FOV in degrees
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
from src.perception import (
    SensorSimulation,
    SensorConfig,
    TrackerConfig,
)


def main(
    animate: bool = False,
    detection_radius: float = 12.0,
    field_of_view_deg: float = 120.0,
):
    """
    Run sensor-based obstacle avoidance demonstration.

    Args:
        animate: Whether to show animated visualization
        detection_radius: Sensor detection range in meters
        field_of_view_deg: Sensor field of view in degrees
    """
    print("=== Sensor-Based Obstacle Avoidance Example ===\n")
    print("In this mode, obstacles are UNKNOWN initially and discovered")
    print("only when they enter the sensor's detection range and FOV.\n")

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

    # Sensor configuration
    fov_rad = np.deg2rad(field_of_view_deg) / 2  # Half-angle
    sensor_config = SensorConfig(
        detection_radius=detection_radius,
        field_of_view=fov_rad,
        noise_std_position=0.05,
        noise_std_velocity=0.02,
        detection_probability=1.0,
        min_detection_range=0.5,
        detect_behind=False,
    )

    tracker_config = TrackerConfig(
        association_threshold=3.0,
        confidence_decay_rate=0.2,
        forget_dynamic_after=8.0,
    )

    # Initialize components
    mpc = MPCController(mpc_config, uav_config)
    dynamics = PointMassModel(dt=mpc_config.dt)
    ref_gen = ReferenceGenerator(
        horizon=mpc_config.horizon,
        dt=mpc_config.dt,
        max_velocity=uav_config.max_velocity,
    )

    # Initialize sensor simulation
    sensor_sim = SensorSimulation(
        sensor_config=sensor_config,
        tracker_config=tracker_config,
    )

    # Initial state and goal
    state = UAVState(position=[0, 0, 10], velocity=[0, 0, 0])
    goal_position = np.array([30, 0, 10])

    # Create WORLD obstacles (ground truth - unknown to UAV initially)
    world_static = [
        SphereObstacle(center=[10, 2, 10], radius=1.5, safety_margin=0.5),
        SphereObstacle(center=[16, -1, 10], radius=1.2, safety_margin=0.5),
        SphereObstacle(center=[22, 3, 10], radius=1.0, safety_margin=0.5),
    ]

    world_dynamic = [
        # Obstacle crossing perpendicular to flight path
        DynamicObstacle(
            radius=1.0,
            trajectory_func=create_linear_trajectory(
                initial_position=[14, -10, 10],
                velocity=[0, 2.0, 0],
            ),
            safety_margin=0.5,
        ),
        # Oncoming obstacle
        DynamicObstacle(
            radius=0.8,
            trajectory_func=create_linear_trajectory(
                initial_position=[28, 1, 10],
                velocity=[-1.5, 0, 0],
            ),
            safety_margin=0.5,
        ),
        # Orbiting obstacle
        DynamicObstacle(
            radius=0.6,
            trajectory_func=create_circular_trajectory(
                center=[20, -3, 10],
                radius=2.5,
                angular_velocity=0.6,
                initial_phase=0,
            ),
            safety_margin=0.5,
        ),
    ]

    # Set world obstacles (ground truth)
    sensor_sim.set_world(world_static, world_dynamic)

    print(f"Sensor Configuration:")
    print(f"  - Detection radius: {detection_radius:.1f} m")
    print(f"  - Field of view: {field_of_view_deg:.0f} degrees")
    print(f"  - Noise std (position): {sensor_config.noise_std_position:.2f} m")
    print()
    print(f"World (ground truth - hidden from UAV):")
    print(f"  - Static obstacles: {len(world_static)}")
    print(f"  - Dynamic obstacles: {len(world_dynamic)}")
    print()
    print(f"Start: {state.position}")
    print(f"Goal: {goal_position}")
    print()

    # Simulation
    log = SimulationLog()
    dt = mpc_config.dt
    max_duration = 20.0
    steps = int(max_duration / dt)
    min_clearance = float("inf")

    # Track discovery events
    discovery_times = {}

    for step in range(steps):
        t = step * dt

        # Update world dynamic obstacles (ground truth)
        for obs in world_dynamic:
            obs.update_time(t)

        # === PERCEPTION STEP ===
        # Detect obstacles within sensor range and FOV
        discovered_static, discovered_dynamic = sensor_sim.perceive_and_update(
            state, t
        )

        # Log new discoveries
        for event in sensor_sim.discovery_log:
            if event.obstacle_id not in discovery_times:
                discovery_times[event.obstacle_id] = event.time
                print(
                    f"  [t={event.time:.2f}s] DISCOVERED {event.obstacle_type} "
                    f"obstacle #{event.obstacle_id} at {event.position}"
                )

        # Generate reference
        reference = ref_gen.generate_smooth_reference(state, goal_position)

        # === MPC STEP ===
        # Solve using ONLY discovered obstacles
        solution = mpc.solve(
            state=state,
            reference=reference,
            static_obstacles=discovered_static,
            dynamic_obstacles=discovered_dynamic,
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

        # Track clearance using WORLD obstacles (ground truth for safety check)
        all_world = world_static + world_dynamic
        clearance = minimum_distance(state.position, all_world)
        min_clearance = min(min_clearance, clearance)

        # Progress
        dist_to_goal = np.linalg.norm(state.position - goal_position)
        num_static, num_dynamic = sensor_sim.tracker.get_num_tracked()

        if step % 50 == 0:
            print(
                f"  t={t:.2f}s | dist_to_goal={dist_to_goal:.2f}m | "
                f"tracked=[{num_static}S, {num_dynamic}D] | clearance={clearance:.2f}m"
            )

        # Check collision with world obstacles
        if clearance < 0:
            print(f"\n*** COLLISION at t={t:.2f}s! ***")
            break

        # Check termination
        if dist_to_goal < 0.5:
            print(f"\nGoal reached at t={t:.2f}s!")
            break

    # Statistics
    stats = sensor_sim.get_statistics()
    print(f"\n=== Results ===")
    print(f"Final position: {state.position}")
    print(f"Distance to goal: {np.linalg.norm(state.position - goal_position):.3f}m")
    print(f"Minimum obstacle clearance: {min_clearance:.3f}m")
    print(f"Average solve time: {np.mean(log.solve_times)*1000:.2f}ms")
    print()
    print(f"Discovery Statistics:")
    print(f"  - Total obstacles in world: {stats['num_world_static'] + stats['num_world_dynamic']}")
    print(f"  - Obstacles discovered: {stats['num_discoveries']}")
    print(f"  - Static discovered: {stats['total_static_discovered']}")
    print(f"  - Dynamic discovered: {stats['total_dynamic_discovered']}")

    # Visualization
    trajectory = log.get_position_trajectory()

    if animate:
        print("\nCreating animation...")

        # Reset for animation
        for obs in world_dynamic:
            obs.update_time(0)

        animator = TrajectoryAnimator(
            interval=50,
            title="Sensor-Based Obstacle Avoidance"
        )
        anim = animator.animate_trajectory(
            states=log.states,
            times=log.times,
            static_obstacles=world_static,
            dynamic_obstacles=world_dynamic,
            goal_position=goal_position,
            predicted_trajectories=log.mpc_trajectories,
            xlim=(-5, 35),
            ylim=(-15, 15),
            zlim=(0, 12),
            show_full_trajectory=True,
        )

        animator.save("sensor_avoidance.gif", fps=20)
        print("Animation saved to sensor_avoidance.gif")

        plt.show(block=True)
    else:
        print("\nPlotting results...")

        # Reset for plotting
        for obs in world_dynamic:
            obs.update_time(0)

        fig = plt.figure(figsize=(14, 6))

        # 3D trajectory plot
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_title("UAV Trajectory with Sensor-Based Avoidance")

        # Plot trajectory
        ax1.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            "b-", linewidth=2, label="UAV Trajectory"
        )
        ax1.scatter(*trajectory[0], c="green", s=100, marker="o", label="Start")
        ax1.scatter(*goal_position, c="red", s=100, marker="*", label="Goal")

        # Plot static obstacles (world)
        for i, obs in enumerate(world_static):
            center = obs.get_center()
            radius = obs.radius
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax1.plot_surface(x, y, z, color="gray", alpha=0.3)

        # Plot dynamic obstacle paths
        for i, obs in enumerate(world_dynamic):
            obs_positions = obs.predict_over_horizon(0, 0.2, 100)
            obs_traj = np.array(obs_positions)
            ax1.plot(
                obs_traj[:, 0], obs_traj[:, 1], obs_traj[:, 2],
                "r--", alpha=0.4, linewidth=1
            )
            # Initial position
            center = obs.get_position_at_time(0)
            radius = obs.radius
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax1.plot_surface(x, y, z, color="orange", alpha=0.4)

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_xlim(-5, 35)
        ax1.set_ylim(-15, 15)
        ax1.set_zlim(0, 12)
        ax1.legend()

        # 2D top-down view with sensor FOV
        ax2 = fig.add_subplot(122)
        ax2.set_title("Top-Down View with Sensor FOV")

        # Plot trajectory
        ax2.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2, label="Trajectory")
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=100, marker="o")
        ax2.scatter(goal_position[0], goal_position[1], c="red", s=100, marker="*")

        # Plot static obstacles
        for obs in world_static:
            center = obs.get_center()
            circle = plt.Circle(
                (center[0], center[1]), obs.radius,
                color="gray", alpha=0.5
            )
            ax2.add_patch(circle)

        # Plot dynamic obstacle paths
        for obs in world_dynamic:
            obs_positions = obs.predict_over_horizon(0, 0.2, 100)
            obs_traj = np.array(obs_positions)
            ax2.plot(obs_traj[:, 0], obs_traj[:, 1], "r--", alpha=0.4)
            center = obs.get_position_at_time(0)
            circle = plt.Circle(
                (center[0], center[1]), obs.radius,
                color="orange", alpha=0.5
            )
            ax2.add_patch(circle)

        # Draw sensor FOV at a few positions along trajectory
        fov_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4]
        for idx in fov_indices:
            pos = trajectory[idx]
            # Approximate yaw from trajectory direction
            if idx < len(trajectory) - 1:
                direction = trajectory[idx + 1] - trajectory[idx]
                yaw = np.arctan2(direction[1], direction[0])
            else:
                yaw = 0

            # Draw FOV cone
            angles = np.linspace(yaw - fov_rad, yaw + fov_rad, 30)
            x_fov = [pos[0]] + [pos[0] + detection_radius * np.cos(a) for a in angles] + [pos[0]]
            y_fov = [pos[1]] + [pos[1] + detection_radius * np.sin(a) for a in angles] + [pos[1]]
            ax2.fill(x_fov, y_fov, color="cyan", alpha=0.1)
            ax2.plot(x_fov, y_fov, "c-", alpha=0.3, linewidth=0.5)

        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_xlim(-5, 35)
        ax2.set_ylim(-15, 15)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("sensor_avoidance_result.png", dpi=150)
        print("Plot saved to sensor_avoidance_result.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensor-Based Obstacle Avoidance Example"
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Show animated visualization"
    )
    parser.add_argument(
        "--radius", type=float, default=12.0,
        help="Detection radius in meters (default: 12.0)"
    )
    parser.add_argument(
        "--fov", type=float, default=120.0,
        help="Field of view in degrees (default: 120.0)"
    )
    args = parser.parse_args()

    main(
        animate=args.animate,
        detection_radius=args.radius,
        field_of_view_deg=args.fov,
    )
