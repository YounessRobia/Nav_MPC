#!/usr/bin/env python3
"""
MPC UAV Path Planning - Main Simulation Entry Point

This script runs the MPC-based path planning simulation with
obstacle avoidance for quadrotor UAVs.
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.dynamics.uav_state import UAVState, UAVControl, SimulationLog
from src.dynamics.point_mass_model import PointMassModel
from src.dynamics.integrators import UAVSimulator
from src.mpc.controller import MPCController, MPCSolution
from src.obstacles.base import Obstacle
from src.obstacles.static import create_obstacle_from_config
from src.obstacles.dynamic import DynamicObstacle, create_dynamic_obstacle_from_config
from src.obstacles.collision import check_state_collision, minimum_distance
from src.planning.reference import ReferenceGenerator
from src.planning.waypoints import WaypointManager, Waypoint
from src.utils.config_loader import (
    load_config,
    UAVConfig,
    MPCConfig,
    ScenarioConfig,
    get_default_config_paths,
)
from src.utils.logging_utils import setup_logger


class Simulation:
    """
    Main simulation class for MPC UAV path planning.

    Orchestrates the interaction between dynamics, MPC controller,
    obstacles, and visualization.
    """

    def __init__(
        self,
        uav_config: UAVConfig,
        mpc_config: MPCConfig,
        scenario_config: ScenarioConfig,
    ):
        """
        Initialize simulation.

        Args:
            uav_config: UAV parameters
            mpc_config: MPC parameters
            scenario_config: Scenario definition
        """
        self.uav_config = uav_config
        self.mpc_config = mpc_config
        self.scenario = scenario_config

        # Setup logger
        self.logger = setup_logger("simulation")

        # Initialize components
        self._setup_dynamics()
        self._setup_mpc()
        self._setup_obstacles()
        self._setup_reference()

        # Simulation state
        self.current_state: Optional[UAVState] = None
        self.current_time: float = 0.0
        self.log = SimulationLog()

    def _setup_dynamics(self):
        """Initialize dynamics model and simulator."""
        self.dynamics = PointMassModel(dt=self.mpc_config.dt)
        self.simulator = UAVSimulator(self.dynamics, integration_method="discrete")

    def _setup_mpc(self):
        """Initialize MPC controller."""
        self.mpc = MPCController(self.mpc_config, self.uav_config)
        self.logger.info(
            f"MPC initialized: horizon={self.mpc_config.horizon}, dt={self.mpc_config.dt}"
        )

    def _setup_obstacles(self):
        """Initialize obstacles from scenario config."""
        self.static_obstacles: List[Obstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # Create static obstacles
        for obs_config in self.scenario.static_obstacles:
            config_dict = {
                "type": obs_config.type,
                "center": obs_config.center.tolist() if obs_config.center is not None else None,
                "radius": obs_config.radius,
                "half_extents": obs_config.half_extents.tolist() if obs_config.half_extents is not None else None,
                "z_min": obs_config.z_min,
                "z_max": obs_config.z_max,
                "safety_margin": obs_config.safety_margin,
            }
            # Remove None values
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            obs = create_obstacle_from_config(config_dict)
            self.static_obstacles.append(obs)

        # Create dynamic obstacles
        for obs_config in self.scenario.dynamic_obstacles:
            config_dict = {
                "type": obs_config.type,
                "radius": obs_config.radius,
                "safety_margin": obs_config.safety_margin,
                "trajectory": obs_config.trajectory,
            }
            obs = create_dynamic_obstacle_from_config(config_dict)
            self.dynamic_obstacles.append(obs)

        self.logger.info(
            f"Obstacles: {len(self.static_obstacles)} static, "
            f"{len(self.dynamic_obstacles)} dynamic"
        )

    def _setup_reference(self):
        """Initialize reference generator and waypoints."""
        self.ref_generator = ReferenceGenerator(
            horizon=self.mpc_config.horizon,
            dt=self.mpc_config.dt,
            max_velocity=self.uav_config.max_velocity,
        )

        # Setup waypoints
        self.waypoint_manager = WaypointManager()
        for wp_pos in self.scenario.waypoints:
            self.waypoint_manager.add_position(wp_pos, tolerance=1.0)

        # Add goal as final waypoint
        self.waypoint_manager.add_position(
            self.scenario.goal_position,
            tolerance=self.scenario.goal_tolerance,
        )

        self.goal_position = self.scenario.goal_position

    def reset(self):
        """Reset simulation to initial state."""
        self.current_state = UAVState(
            position=self.scenario.initial_position.copy(),
            velocity=self.scenario.initial_velocity.copy(),
            acceleration=self.scenario.initial_acceleration.copy(),
            yaw=self.scenario.initial_yaw,
        )
        self.current_time = 0.0
        self.log = SimulationLog()
        self.waypoint_manager.reset()
        self.mpc.reset_warm_start()

        # Reset dynamic obstacles to t=0
        for obs in self.dynamic_obstacles:
            obs.update_time(0.0)

        self.logger.info("Simulation reset to initial state")

    def step(self) -> Tuple[bool, str]:
        """
        Execute one simulation step.

        Returns:
            Tuple of (continue_simulation, status_message)
        """
        # Update dynamic obstacles
        for obs in self.dynamic_obstacles:
            obs.update_time(self.current_time)

        # Check waypoint progress
        self.waypoint_manager.update(self.current_state.position)

        # Generate reference trajectory
        reference = self.ref_generator.generate_smooth_reference(
            self.current_state,
            self.goal_position,
        )

        # Solve MPC
        solution = self.mpc.solve(
            state=self.current_state,
            reference=reference,
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=self.dynamic_obstacles,
            t_current=self.current_time,
        )

        # Extract control
        control = self.mpc.get_control(solution)

        # Log before applying control
        predicted_traj = self.mpc.get_predicted_trajectory(solution)
        self.log.append(
            t=self.current_time,
            state=self.current_state,
            control=control,
            mpc_trajectory=predicted_traj,
            solve_time=solution.solve_time,
            cost=solution.cost,
        )

        # Apply control and integrate dynamics
        self.current_state = self.simulator.step(self.current_state, control)
        self.current_time += self.mpc_config.dt

        # Check termination conditions
        return self._check_termination(solution)

    def _check_termination(
        self, solution: MPCSolution
    ) -> Tuple[bool, str]:
        """
        Check simulation termination conditions.

        Returns:
            Tuple of (continue_simulation, status_message)
        """
        # Check collision
        all_obstacles = self.static_obstacles + self.dynamic_obstacles
        if check_state_collision(self.current_state, all_obstacles):
            return False, "COLLISION"

        # Check goal reached
        dist_to_goal = self.current_state.distance_to_point(self.goal_position)
        if dist_to_goal < self.scenario.goal_tolerance:
            return False, "GOAL_REACHED"

        # Check timeout
        if self.current_time >= self.scenario.duration:
            return False, "TIMEOUT"

        # Check solver failure
        if not solution.success:
            self.logger.warning(f"MPC solver failed: {solution.status}")
            # Continue with zero control
            return True, "SOLVER_WARNING"

        return True, "OK"

    def run(self, visualize: bool = False) -> Tuple[str, SimulationLog]:
        """
        Run complete simulation.

        Args:
            visualize: Whether to show real-time visualization

        Returns:
            Tuple of (final_status, simulation_log)
        """
        self.reset()
        self.logger.info("Starting simulation...")

        status = "RUNNING"
        step_count = 0
        total_solve_time = 0.0

        while True:
            continue_sim, status = self.step()

            # Accumulate stats
            if self.log.solve_times:
                total_solve_time += self.log.solve_times[-1]
            step_count += 1

            # Progress logging
            if step_count % 50 == 0:
                dist = self.current_state.distance_to_point(self.goal_position)
                self.logger.info(
                    f"t={self.current_time:.2f}s, dist_to_goal={dist:.2f}m, "
                    f"avg_solve={1000*total_solve_time/step_count:.1f}ms"
                )

            if not continue_sim:
                break

        # Final stats
        avg_solve_time = total_solve_time / max(step_count, 1)
        min_clearance = float("inf")
        for state in self.log.states:
            clearance = minimum_distance(
                state.position, self.static_obstacles + self.dynamic_obstacles
            )
            min_clearance = min(min_clearance, clearance)

        self.logger.info(f"Simulation complete: {status}")
        self.logger.info(f"  Steps: {step_count}")
        self.logger.info(f"  Duration: {self.current_time:.2f}s")
        self.logger.info(f"  Avg MPC solve time: {1000*avg_solve_time:.2f}ms")
        self.logger.info(f"  Min obstacle clearance: {min_clearance:.3f}m")

        return status, self.log

    def visualize_results(
        self,
        save_path: Optional[str] = None,
        show_animation: bool = False,
    ):
        """
        Visualize simulation results.

        Args:
            save_path: Path to save figure
            show_animation: Whether to show animation
        """
        from src.visualization.plotter_3d import plot_simulation_results
        from src.visualization.animation import create_animation

        # Get data
        trajectory = self.log.get_position_trajectory()
        predicted_trajs = self.log.mpc_trajectories

        # Static plot
        plotter = plot_simulation_results(
            trajectory=trajectory,
            obstacles=self.static_obstacles + self.dynamic_obstacles,
            goal_position=self.goal_position,
            waypoints=self.waypoint_manager.get_all_positions(),
            predicted_trajectories=predicted_trajs,
            title=f"Simulation: {self.scenario.name}",
            save_path=save_path,
        )

        if show_animation:
            import matplotlib
            matplotlib.use('TkAgg')  # Switch to interactive backend
            import matplotlib.pyplot as plt
            
            # Reset dynamic obstacles for animation
            for obs in self.dynamic_obstacles:
                obs.update_time(0.0)

            anim = create_animation(
                states=self.log.states,
                times=self.log.times,
                static_obstacles=self.static_obstacles,
                dynamic_obstacles=self.dynamic_obstacles,
                goal_position=self.goal_position,
                predicted_trajectories=predicted_trajs,
            )
            
            # Save animation GIF
            anim.save("trajectory_animation.gif", writer="pillow", fps=20)
            self.logger.info("Animation saved to trajectory_animation.gif")
            
            # Show animation interactively
            plt.show(block=True)
        else:
            plotter.show()


def run_simulation(
    scenario_path: str,
    uav_config_path: Optional[str] = None,
    mpc_config_path: Optional[str] = None,
    visualize: bool = True,
    save_plot: Optional[str] = None,
    animate: bool = False,
) -> Tuple[str, SimulationLog]:
    """
    Run simulation from configuration files.

    Args:
        scenario_path: Path to scenario YAML
        uav_config_path: Path to UAV config (optional)
        mpc_config_path: Path to MPC config (optional)
        visualize: Show visualization
        save_plot: Path to save plot
        animate: Create animation

    Returns:
        Tuple of (status, log)
    """
    # Load configs
    if uav_config_path is None or mpc_config_path is None:
        default_uav, default_mpc = get_default_config_paths()
        uav_config_path = uav_config_path or str(default_uav)
        mpc_config_path = mpc_config_path or str(default_mpc)

    uav_config, mpc_config, scenario_config = load_config(
        uav_config_path, mpc_config_path, scenario_path
    )

    # Create and run simulation
    sim = Simulation(uav_config, mpc_config, scenario_config)
    status, log = sim.run()

    # Visualize
    if visualize:
        sim.visualize_results(save_path=save_plot, show_animation=animate)

    return status, log


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MPC UAV Path Planning Simulation"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="config/scenarios/single_sphere.yaml",
        help="Path to scenario config file",
    )
    parser.add_argument(
        "--uav-config",
        type=str,
        default=None,
        help="Path to UAV config file",
    )
    parser.add_argument(
        "--mpc-config",
        type=str,
        default=None,
        help="Path to MPC config file",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save result plot",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animation GIF",
    )

    args = parser.parse_args()

    status, log = run_simulation(
        scenario_path=args.scenario,
        uav_config_path=args.uav_config,
        mpc_config_path=args.mpc_config,
        visualize=not args.no_viz,
        save_plot=args.save_plot,
        animate=args.animate,
    )

    print(f"\nSimulation finished with status: {status}")
    return 0 if status == "GOAL_REACHED" else 1


if __name__ == "__main__":
    exit(main())
