#!/usr/bin/env python3
"""
Random Sensor-Based Obstacle Avoidance Scenario

Generates random static and dynamic obstacles in the environment.
The UAV must navigate to the goal while discovering obstacles through
its limited FOV sensor.

Features:
    - Configurable number of random static and dynamic obstacles
    - Random obstacle positions, sizes, and velocities
    - Seed for reproducibility
    - Enhanced visualization with FOV cone and detection events

Usage:
    python random_sensor_avoidance.py                         # Default settings
    python random_sensor_avoidance.py --animate               # With animation
    python random_sensor_avoidance.py --n-static 5 --n-dynamic 8
    python random_sensor_avoidance.py --seed 42               # Reproducible
    python random_sensor_avoidance.py --difficulty hard       # More obstacles
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix matplotlib path
import examples._fix_matplotlib  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.uav_state import UAVState, SimulationLog
from src.dynamics.point_mass_model import PointMassModel
from src.mpc.controller import MPCController
from src.obstacles.static import SphereObstacle
from src.obstacles.dynamic import DynamicObstacle
from src.obstacles.collision import minimum_distance
from src.planning.reference import ReferenceGenerator
from src.utils.config_loader import MPCConfig, UAVConfig
from src.perception import SensorSimulation, SensorConfig, TrackerConfig


@dataclass
class ScenarioConfig:
    """Configuration for random scenario generation."""

    # Environment bounds
    x_min: float = 0.0
    x_max: float = 40.0
    y_min: float = -15.0
    y_max: float = 15.0
    z_flight: float = 10.0  # Flight altitude

    # Obstacle counts
    n_static: int = 5
    n_dynamic: int = 5

    # Static obstacle parameters
    static_radius_min: float = 0.8
    static_radius_max: float = 2.0
    static_safety_margin: float = 0.5

    # Dynamic obstacle parameters
    dynamic_radius_min: float = 0.6
    dynamic_radius_max: float = 1.5
    dynamic_speed_min: float = 0.5
    dynamic_speed_max: float = 3.0
    dynamic_safety_margin: float = 0.5

    # Clearance from start and goal
    start_clearance: float = 5.0
    goal_clearance: float = 5.0

    # Random seed
    seed: Optional[int] = None


def generate_random_scenario(
    config: ScenarioConfig,
    start_position: np.ndarray,
    goal_position: np.ndarray,
) -> Tuple[List[SphereObstacle], List[DynamicObstacle]]:
    """
    Generate random static and dynamic obstacles.

    Args:
        config: Scenario configuration
        start_position: UAV start position
        goal_position: Goal position

    Returns:
        Tuple of (static_obstacles, dynamic_obstacles)
    """
    rng = np.random.default_rng(config.seed)

    static_obstacles = []
    dynamic_obstacles = []

    def is_valid_position(pos: np.ndarray, radius: float, existing: List) -> bool:
        """Check if position is valid (not colliding with start/goal/existing)."""
        # Check clearance from start
        if np.linalg.norm(pos - start_position) < config.start_clearance + radius:
            return False

        # Check clearance from goal
        if np.linalg.norm(pos - goal_position) < config.goal_clearance + radius:
            return False

        # Check clearance from existing obstacles
        for obs in existing:
            if hasattr(obs, 'center'):
                obs_center = obs.center
                obs_radius = obs.radius
            else:
                obs_center = obs.get_position_at_time(0)
                obs_radius = obs.radius

            if np.linalg.norm(pos - obs_center) < (radius + obs_radius + 1.0):
                return False

        return True

    # Generate static obstacles
    attempts = 0
    max_attempts = 1000

    while len(static_obstacles) < config.n_static and attempts < max_attempts:
        attempts += 1

        # Random position along the path corridor
        x = rng.uniform(config.x_min + 5, config.x_max - 5)
        y = rng.uniform(config.y_min + 3, config.y_max - 3)
        z = config.z_flight + rng.uniform(-1, 1)

        pos = np.array([x, y, z])
        radius = rng.uniform(config.static_radius_min, config.static_radius_max)

        if is_valid_position(pos, radius, static_obstacles + dynamic_obstacles):
            obs = SphereObstacle(
                center=pos,
                radius=radius,
                safety_margin=config.static_safety_margin,
            )
            static_obstacles.append(obs)

    # Generate dynamic obstacles
    attempts = 0

    while len(dynamic_obstacles) < config.n_dynamic and attempts < max_attempts:
        attempts += 1

        # Random initial position (can start outside visible range)
        x = rng.uniform(config.x_min, config.x_max)
        y = rng.uniform(config.y_min - 10, config.y_max + 10)
        z = config.z_flight + rng.uniform(-2, 2)

        pos = np.array([x, y, z])
        radius = rng.uniform(config.dynamic_radius_min, config.dynamic_radius_max)

        if is_valid_position(pos, radius, static_obstacles + dynamic_obstacles):
            # Random velocity direction and magnitude
            speed = rng.uniform(config.dynamic_speed_min, config.dynamic_speed_max)

            # Bias velocity to cross the flight path
            vel_patterns = [
                # Crossing perpendicular
                lambda: np.array([rng.uniform(-0.5, 0.5), rng.choice([-1, 1]), 0]) * speed,
                # Moving along path
                lambda: np.array([rng.choice([-1, 1]), rng.uniform(-0.3, 0.3), 0]) * speed,
                # Random direction
                lambda: rng.normal(0, 1, 3) / np.linalg.norm(rng.normal(0, 1, 3) + 1e-6) * speed,
            ]

            velocity = rng.choice(vel_patterns)()
            velocity[2] *= 0.1  # Reduce vertical velocity

            # Create trajectory function (closure captures pos and velocity)
            initial_pos = pos.copy()
            vel = velocity.copy()

            def make_trajectory(p, v):
                def trajectory(t):
                    return p + v * t
                return trajectory

            obs = DynamicObstacle(
                radius=radius,
                trajectory_func=make_trajectory(initial_pos, vel),
                safety_margin=config.dynamic_safety_margin,
            )
            dynamic_obstacles.append(obs)

    print(f"Generated {len(static_obstacles)} static and {len(dynamic_obstacles)} dynamic obstacles")

    return static_obstacles, dynamic_obstacles


def run_simulation(
    animate: bool = False,
    n_static: int = 5,
    n_dynamic: int = 5,
    detection_radius: float = 12.0,
    field_of_view_deg: float = 120.0,
    seed: Optional[int] = None,
    difficulty: str = "medium",
):
    """
    Run the random sensor-based avoidance simulation.

    Args:
        animate: Whether to show animation
        n_static: Number of static obstacles
        n_dynamic: Number of dynamic obstacles
        detection_radius: Sensor detection range
        field_of_view_deg: Sensor FOV in degrees
        seed: Random seed for reproducibility
        difficulty: 'easy', 'medium', or 'hard'
    """
    print("=" * 60)
    print("  Random Sensor-Based Obstacle Avoidance Scenario")
    print("=" * 60)
    print()

    # Adjust parameters based on difficulty
    difficulty_params = {
        "easy": {"n_static": 3, "n_dynamic": 2, "speed_max": 1.5},
        "medium": {"n_static": 5, "n_dynamic": 5, "speed_max": 2.5},
        "hard": {"n_static": 8, "n_dynamic": 8, "speed_max": 3.5},
        "extreme": {"n_static": 12, "n_dynamic": 12, "speed_max": 4.0},
    }

    if difficulty in difficulty_params:
        params = difficulty_params[difficulty]
        if n_static == 5 and n_dynamic == 5:  # Use defaults from difficulty
            n_static = params["n_static"]
            n_dynamic = params["n_dynamic"]
        speed_max = params["speed_max"]
    else:
        speed_max = 2.5

    # UAV configuration
    uav_config = UAVConfig(
        max_velocity=6.0,
        max_acceleration=4.0,
        max_jerk=25.0,
        max_yaw_rate=1.5,
    )

    # MPC configuration
    mpc_config = MPCConfig(
        horizon=30,
        dt=0.05,
        Q_position=np.array([15.0, 15.0, 20.0]),
        Q_velocity=np.array([5.0, 5.0, 5.0]),
        R_jerk=np.array([0.1, 0.1, 0.1]),
        Qf_multiplier=10.0,
        obstacle_weight=2500.0,
    )

    # Sensor configuration
    fov_rad = np.deg2rad(field_of_view_deg) / 2
    sensor_config = SensorConfig(
        detection_radius=detection_radius,
        field_of_view=fov_rad,
        noise_std_position=0.05,
        noise_std_velocity=0.03,
        detection_probability=1.0,
        min_detection_range=0.5,
    )

    tracker_config = TrackerConfig(
        association_threshold=3.0,
        confidence_decay_rate=0.15,
        forget_dynamic_after=12.0,
    )

    # Scenario configuration
    scenario_config = ScenarioConfig(
        n_static=n_static,
        n_dynamic=n_dynamic,
        dynamic_speed_max=speed_max,
        seed=seed,
    )

    # Initialize components
    mpc = MPCController(mpc_config, uav_config)
    dynamics = PointMassModel(dt=mpc_config.dt)
    ref_gen = ReferenceGenerator(
        horizon=mpc_config.horizon,
        dt=mpc_config.dt,
        max_velocity=uav_config.max_velocity,
    )

    sensor_sim = SensorSimulation(sensor_config, tracker_config)

    # Start and goal positions
    start_position = np.array([0.0, 0.0, 10.0])
    goal_position = np.array([35.0, 0.0, 10.0])

    # Generate random obstacles
    print(f"Scenario: {difficulty.upper()} difficulty")
    print(f"Random seed: {seed if seed else 'None (random)'}")
    world_static, world_dynamic = generate_random_scenario(
        scenario_config, start_position, goal_position
    )

    sensor_sim.set_world(world_static, world_dynamic)

    print()
    print(f"Sensor: radius={detection_radius}m, FOV={field_of_view_deg}°")
    print(f"Start: {start_position}")
    print(f"Goal: {goal_position}")
    print()

    # Initial state
    state = UAVState(position=start_position.copy(), velocity=[0, 0, 0])

    # Simulation
    log = SimulationLog()
    dt = mpc_config.dt
    max_duration = 25.0
    steps = int(max_duration / dt)
    min_clearance = float("inf")

    print("Starting simulation...")
    print("-" * 50)

    for step in range(steps):
        t = step * dt

        # Update world obstacles
        for obs in world_dynamic:
            obs.update_time(t)

        # Perception step
        discovered_static, discovered_dynamic = sensor_sim.perceive_and_update(state, t)

        # Log discoveries
        for event in sensor_sim.discovery_log:
            if event.time == t:
                print(f"  [t={t:5.2f}s] DETECTED {event.obstacle_type:7s} obstacle at "
                      f"({event.position[0]:5.1f}, {event.position[1]:5.1f})")

        # Generate reference
        reference = ref_gen.generate_smooth_reference(state, goal_position)

        # MPC solve
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

        # Safety check with world obstacles
        all_world = world_static + world_dynamic
        clearance = minimum_distance(state.position, all_world)
        min_clearance = min(min_clearance, clearance)

        # Progress
        dist_to_goal = np.linalg.norm(state.position - goal_position)
        num_s, num_d = sensor_sim.tracker.get_num_tracked()

        if step % 60 == 0:
            print(f"  t={t:5.2f}s | dist={dist_to_goal:5.2f}m | "
                  f"tracked=[{num_s}S,{num_d}D] | clearance={clearance:5.2f}m")

        # Collision check
        if clearance < 0:
            print(f"\n*** COLLISION at t={t:.2f}s! ***")
            break

        # Goal check
        if dist_to_goal < 0.5:
            print(f"\n*** GOAL REACHED at t={t:.2f}s! ***")
            break

    # Results
    print("-" * 50)
    stats = sensor_sim.get_statistics()
    final_dist = np.linalg.norm(state.position - goal_position)

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Final distance to goal: {final_dist:.3f} m")
    print(f"  Minimum clearance: {min_clearance:.3f} m")
    print(f"  Average MPC solve time: {np.mean(log.solve_times)*1000:.2f} ms")
    print()
    print(f"  Discovery Statistics:")
    print(f"    - World obstacles: {stats['num_world_static']} static, "
          f"{stats['num_world_dynamic']} dynamic")
    print(f"    - Discovered: {stats['total_static_discovered']} static, "
          f"{stats['total_dynamic_discovered']} dynamic")
    print(f"    - Discovery rate: "
          f"{stats['num_discoveries']}/{stats['num_world_static']+stats['num_world_dynamic']} "
          f"({100*stats['num_discoveries']/(stats['num_world_static']+stats['num_world_dynamic']):.0f}%)")
    print()

    # Visualization
    trajectory = log.get_position_trajectory()

    if animate:
        print("Creating animation with sensor visualization...")

        from src.visualization.sensor_animation import SensorTrajectoryAnimator

        # Reset obstacles
        for obs in world_dynamic:
            obs.update_time(0)

        animator = SensorTrajectoryAnimator(
            figsize=(16, 7),
            interval=50,
            title=f"Sensor-Based Avoidance ({difficulty.title()} - Seed: {seed})"
        )

        anim = animator.animate_trajectory(
            states=log.states,
            times=log.times,
            world_static=world_static,
            world_dynamic=world_dynamic,
            discovery_events=sensor_sim.discovery_log,
            sensor_config=sensor_config,
            goal_position=goal_position,
            predicted_trajectories=log.mpc_trajectories,
            xlim=(-5, 45),
            ylim=(-20, 20),
            zlim=(0, 15),
        )

        animator.save("random_sensor_avoidance.gif", fps=20)
        print("Animation saved to random_sensor_avoidance.gif")

        plt.show(block=True)

    else:
        print("Creating static visualization...")

        # Reset obstacles
        for obs in world_dynamic:
            obs.update_time(0)

        fig = plt.figure(figsize=(16, 7), facecolor='#1a1a2e')

        # 3D plot
        ax3d = fig.add_subplot(121, projection='3d', facecolor='#16213e')
        ax3d.set_title(f"3D Trajectory ({difficulty.title()})", color='white', fontweight='bold')

        # Plot trajectory
        ax3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                 'b-', linewidth=2, label='UAV Path')
        ax3d.scatter(*start_position, c='green', s=150, marker='o', label='Start')
        ax3d.scatter(*goal_position, c='red', s=150, marker='*', label='Goal')

        # Plot static obstacles
        for obs in world_static:
            center = obs.get_center()
            radius = obs.radius
            u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax3d.plot_surface(x, y, z, color='gray', alpha=0.4)

        # Plot dynamic obstacle paths
        for obs in world_dynamic:
            positions = obs.predict_over_horizon(0, 0.3, 80)
            pos_arr = np.array(positions)
            ax3d.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2],
                     'r--', alpha=0.4, linewidth=1)
            # Initial position
            init_pos = obs.get_position_at_time(0)
            ax3d.scatter(*init_pos, c='orange', s=50, alpha=0.7)

        ax3d.set_xlabel('X [m]', color='white')
        ax3d.set_ylabel('Y [m]', color='white')
        ax3d.set_zlabel('Z [m]', color='white')
        ax3d.tick_params(colors='white')
        ax3d.set_xlim(-5, 45)
        ax3d.set_ylim(-20, 20)
        ax3d.set_zlim(0, 15)
        ax3d.legend(facecolor='#1a1a2e', labelcolor='white')

        # 2D top-down plot
        ax2d = fig.add_subplot(122, facecolor='#16213e')
        ax2d.set_title("Top-Down View with Sensor FOV", color='white', fontweight='bold')

        # Plot trajectory
        ax2d.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
        ax2d.scatter(start_position[0], start_position[1], c='green', s=150, marker='o')
        ax2d.scatter(goal_position[0], goal_position[1], c='red', s=150, marker='*')

        # Plot static obstacles
        for obs in world_static:
            center = obs.get_center()
            circle = plt.Circle((center[0], center[1]), obs.radius,
                               color='gray', alpha=0.5)
            ax2d.add_patch(circle)

        # Plot dynamic obstacles
        for obs in world_dynamic:
            positions = obs.predict_over_horizon(0, 0.3, 80)
            pos_arr = np.array(positions)
            ax2d.plot(pos_arr[:, 0], pos_arr[:, 1], 'r--', alpha=0.4, linewidth=1)
            init_pos = obs.get_position_at_time(0)
            circle = plt.Circle((init_pos[0], init_pos[1]), obs.radius,
                               color='orange', alpha=0.5)
            ax2d.add_patch(circle)

        # Draw FOV at key positions
        from matplotlib.patches import Wedge
        fov_indices = np.linspace(0, len(trajectory)-1, 6, dtype=int)

        for idx in fov_indices:
            pos = trajectory[idx]
            # Compute heading from next position
            if idx < len(trajectory) - 1:
                direction = trajectory[min(idx+5, len(trajectory)-1)] - pos
                yaw = np.arctan2(direction[1], direction[0])
            else:
                yaw = 0

            wedge = Wedge(
                (pos[0], pos[1]), detection_radius,
                np.rad2deg(yaw - fov_rad), np.rad2deg(yaw + fov_rad),
                color='cyan', alpha=0.1, edgecolor='cyan', linewidth=1
            )
            ax2d.add_patch(wedge)

        ax2d.set_xlabel('X [m]', color='white')
        ax2d.set_ylabel('Y [m]', color='white')
        ax2d.tick_params(colors='white')
        ax2d.set_xlim(-5, 45)
        ax2d.set_ylim(-20, 20)
        ax2d.set_aspect('equal')
        ax2d.grid(True, alpha=0.3, color='gray')

        plt.tight_layout()
        plt.savefig('random_sensor_avoidance_result.png', dpi=150, facecolor='#1a1a2e')
        print("Plot saved to random_sensor_avoidance_result.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Sensor-Based Obstacle Avoidance Scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python random_sensor_avoidance.py --difficulty easy
  python random_sensor_avoidance.py --n-static 10 --n-dynamic 10 --seed 42
  python random_sensor_avoidance.py --animate --difficulty hard
  python random_sensor_avoidance.py --radius 8 --fov 90  # Harder sensing
        """
    )

    parser.add_argument("--animate", action="store_true",
                       help="Show animated visualization with FOV")
    parser.add_argument("--n-static", type=int, default=5,
                       help="Number of static obstacles (default: 5)")
    parser.add_argument("--n-dynamic", type=int, default=5,
                       help="Number of dynamic obstacles (default: 5)")
    parser.add_argument("--radius", type=float, default=12.0,
                       help="Detection radius in meters (default: 12.0)")
    parser.add_argument("--fov", type=float, default=120.0,
                       help="Field of view in degrees (default: 120.0)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--difficulty", type=str, default="medium",
                       choices=["easy", "medium", "hard", "extreme"],
                       help="Difficulty preset (default: medium)")

    args = parser.parse_args()

    run_simulation(
        animate=args.animate,
        n_static=args.n_static,
        n_dynamic=args.n_dynamic,
        detection_radius=args.radius,
        field_of_view_deg=args.fov,
        seed=args.seed,
        difficulty=args.difficulty,
    )
