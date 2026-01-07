#!/usr/bin/env python3
"""
Swarm Avoidance - Random Dynamic Obstacles Scenario

A UAV must navigate through a swarm of obstacles with randomized trajectories.
Each run generates different obstacle paths, testing the MPC's adaptability.

Usage:
    python swarm_avoidance.py           # Static plot only
    python swarm_avoidance.py --animate # With live animation
    python swarm_avoidance.py --seed 42 # Reproducible random trajectories
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
from src.obstacles.dynamic import DynamicObstacle
from src.obstacles.collision import minimum_distance
from src.planning.reference import ReferenceGenerator
from src.utils.config_loader import MPCConfig, UAVConfig
from src.visualization.plotter_3d import Plotter3D
from src.visualization.animation import TrajectoryAnimator


def create_random_linear_trajectory(rng, bounds, speed_range):
    """Create a random linear trajectory within bounds."""
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']

    # Random start position
    start = np.array([
        rng.uniform(x_min, x_max),
        rng.uniform(y_min, y_max),
        rng.uniform(z_min, z_max),
    ])

    # Random velocity direction and speed
    direction = rng.uniform(-1, 1, size=3)
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    speed = rng.uniform(speed_range[0], speed_range[1])
    velocity = direction * speed

    def trajectory(t):
        return start + velocity * t

    return trajectory, start, velocity


def create_random_circular_trajectory(rng, bounds, speed_range):
    """Create a random circular trajectory within bounds."""
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']

    # Random center - closer to flight path
    center = np.array([
        rng.uniform(x_min + 2, x_max - 2),
        rng.uniform(y_min + 2, y_max - 2),
        rng.uniform(z_min + 1, z_max - 1),
    ])

    # Smaller, faster orbits for tighter space
    radius = rng.uniform(1.5, 3.5)
    angular_vel = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
    phase = rng.uniform(0, 2 * np.pi)

    # Random orbit plane (mostly horizontal with some tilt)
    tilt = rng.uniform(-0.3, 0.3)

    def trajectory(t):
        angle = angular_vel * t + phase
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + radius * tilt * np.sin(angle)
        return np.array([x, y, z])

    return trajectory, center, radius


def create_random_sinusoidal_trajectory(rng, bounds, speed_range):
    """Create a random sinusoidal (wave) trajectory."""
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']

    # Random start - within tight bounds
    start = np.array([
        rng.uniform(x_min, x_max),
        rng.uniform(y_min, y_max),
        rng.uniform(z_min, z_max),
    ])

    # Random base velocity - faster for challenge
    base_speed = rng.uniform(speed_range[0], speed_range[1])
    direction = rng.choice(['x', 'y'])

    # Sinusoidal parameters - tighter oscillations
    amplitude = rng.uniform(1.5, 4.0)
    frequency = rng.uniform(0.5, 1.2)

    def trajectory(t):
        if direction == 'x':
            x = start[0] + base_speed * t
            y = start[1] + amplitude * np.sin(frequency * t)
            z = start[2]
        else:
            x = start[0] + amplitude * np.sin(frequency * t)
            y = start[1] + base_speed * t
            z = start[2]
        return np.array([x, y, z])

    return trajectory, start, base_speed


def main(animate: bool = False, seed: int = None):
    """Run swarm avoidance demonstration."""
    print("=" * 60)
    print("       SWARM AVOIDANCE - Random Dynamic Obstacles")
    print("=" * 60)
    print()

    # Set random seed for reproducibility
    if seed is not None:
        print(f"Using random seed: {seed}")
        rng = np.random.default_rng(seed)
    else:
        seed = np.random.randint(0, 10000)
        print(f"Generated random seed: {seed} (use --seed {seed} to reproduce)")
        rng = np.random.default_rng(seed)
    print()

    # Configuration
    uav_config = UAVConfig(
        max_velocity=6.0,
        max_acceleration=4.0,
        max_jerk=25.0,
        max_yaw_rate=1.5,
    )

    mpc_config = MPCConfig(
        horizon=25,
        dt=0.05,
        Q_position=np.array([18.0, 18.0, 22.0]),
        Q_velocity=np.array([5.0, 5.0, 5.0]),
        R_jerk=np.array([0.08, 0.08, 0.08]),
        Qf_multiplier=12.0,
        obstacle_weight=1000.0,
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

    # Initial state and goal - tighter corridor
    state = UAVState(position=[0, 0, 10], velocity=[0, 0, 0])
    goal_position = np.array([30, 0, 10])

    # Define bounds for obstacle generation - TIGHT space for challenge
    bounds = {
        'x': (5, 25),      # Narrower x range, obstacles closer to path
        'y': (-6, 6),      # Much narrower y corridor
        'z': (8, 13),      # Tighter altitude band
    }
    speed_range = (1.2, 3.0)  # Faster obstacles

    # Generate random dynamic obstacles - more obstacles in tighter space
    num_obstacles = 10
    dynamic_obstacles = []
    obstacle_info = []

    print(f"Generating {num_obstacles} random obstacles...")
    print("-" * 50)

    for i in range(num_obstacles):
        # Randomly choose trajectory type
        traj_type = rng.choice(['linear', 'circular', 'sinusoidal'], p=[0.4, 0.35, 0.25])
        radius = rng.uniform(0.6, 1.2)  # Larger obstacles

        if traj_type == 'linear':
            traj_func, start, vel = create_random_linear_trajectory(rng, bounds, speed_range)
            info = f"Linear: start={start.round(1)}, vel={vel.round(1)}"
        elif traj_type == 'circular':
            traj_func, center, orbit_r = create_random_circular_trajectory(rng, bounds, speed_range)
            info = f"Circular: center={center.round(1)}, radius={orbit_r:.1f}"
        else:
            traj_func, start, speed = create_random_sinusoidal_trajectory(rng, bounds, speed_range)
            info = f"Sinusoidal: start={start.round(1)}, speed={speed:.1f}"

        obs = DynamicObstacle(
            radius=radius,
            trajectory_func=traj_func,
            safety_margin=0.4,
        )
        dynamic_obstacles.append(obs)
        obstacle_info.append((i + 1, traj_type, info))
        print(f"  Obstacle {i+1} (r={radius:.2f}m): {traj_type}")

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
    collision_count = 0

    print("Navigation Progress:")
    print("-" * 50)

    for step in range(steps):
        t = step * dt

        # Update dynamic obstacle positions
        for obs in dynamic_obstacles:
            obs.update_time(t)

        # Generate reference
        reference = ref_gen.generate_smooth_reference(state, goal_position)

        # Solve MPC
        solution = mpc.solve(
            state=state,
            reference=reference,
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
        clearance = minimum_distance(state.position, dynamic_obstacles)
        min_clearance = min(min_clearance, clearance)

        if clearance < 0:
            collision_count += 1

        # Progress
        dist_to_goal = np.linalg.norm(state.position - goal_position)
        if step % 50 == 0:
            print(f"  t={t:.2f}s | Dist to goal: {dist_to_goal:.1f}m | "
                  f"Clearance: {clearance:.2f}m | Pos: [{state.position[0]:.1f}, "
                  f"{state.position[1]:.1f}, {state.position[2]:.1f}]")

        # Check termination
        if dist_to_goal < 1.0:
            print()
            print("=" * 50)
            print("       GOAL REACHED!")
            print("=" * 50)
            break

    # Statistics
    print()
    print("=" * 50)
    print("MISSION STATISTICS")
    print("=" * 50)
    final_dist = np.linalg.norm(state.position - goal_position)
    print(f"  Final position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]")
    print(f"  Distance to goal: {final_dist:.3f}m")
    print(f"  Goal reached: {'Yes' if final_dist < 1.0 else 'No'}")
    print(f"  Total flight time: {log.times[-1]:.2f}s")
    print(f"  Minimum clearance: {min_clearance:.3f}m")
    if collision_count > 0:
        print(f"  Collision frames: {collision_count}")
    print(f"  Average MPC solve time: {np.mean(log.solve_times)*1000:.2f}ms")

    # Flight statistics
    velocities = np.array([s.velocity for s in log.states])
    speeds = np.linalg.norm(velocities, axis=1)
    print(f"  Average speed: {np.mean(speeds):.2f} m/s")
    print(f"  Max speed: {np.max(speeds):.2f} m/s")

    # Get trajectory data
    trajectory = log.get_position_trajectory()

    if animate:
        # === ANIMATED VISUALIZATION ===
        print()
        print("Creating animation...")

        # Reset dynamic obstacles for animation
        for obs in dynamic_obstacles:
            obs.update_time(0)

        animator = TrajectoryAnimator(
            interval=50,
            title=f"Swarm Avoidance (seed={seed})"
        )

        anim = animator.animate_trajectory(
            states=log.states,
            times=log.times,
            dynamic_obstacles=dynamic_obstacles,
            goal_position=goal_position,
            predicted_trajectories=log.mpc_trajectories,
            xlim=(-3, 35),
            ylim=(-10, 10),
            zlim=(5, 16),
            show_full_trajectory=True,
        )

        # Save animation
        animator.save(f"swarm_avoidance_seed{seed}.gif", fps=20)
        print(f"Animation saved to swarm_avoidance_seed{seed}.gif")

        # Show animation interactively
        plt.show(block=True)
    else:
        # === STATIC VISUALIZATION ===
        print()
        print("Plotting results...")

        # Reset obstacles for plotting
        for obs in dynamic_obstacles:
            obs.update_time(0)

        plotter = Plotter3D(title=f"Swarm Avoidance (seed={seed})")
        plotter.setup_axes(xlim=(-3, 35), ylim=(-10, 10), zlim=(5, 16))

        # Plot trajectory
        plotter.plot_trajectory(trajectory, label="UAV Flight Path")
        plotter.plot_goal(goal_position)

        # Plot obstacle trajectories
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(dynamic_obstacles)))
        for i, obs in enumerate(dynamic_obstacles):
            obs_positions = obs.predict_over_horizon(0, 0.1, 150)
            obs_traj = np.array(obs_positions)
            plotter.ax.plot(
                obs_traj[:, 0], obs_traj[:, 1], obs_traj[:, 2],
                '--', color=colors[i], alpha=0.5, linewidth=1,
                label=f"Obs {i+1}" if i < 3 else None
            )

        # Plot obstacles at initial positions
        plotter.plot_obstacles(dynamic_obstacles, color="red", alpha=0.5)

        # Plot start
        plotter.ax.scatter([0], [0], [10], c="green", s=150, marker="^", label="Start")

        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Swarm Avoidance - Random Dynamic Obstacles"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show live animated visualization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible obstacle trajectories"
    )
    args = parser.parse_args()
    main(animate=args.animate, seed=args.seed)
