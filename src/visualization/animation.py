"""Animation utilities for trajectory visualization."""

import sys

# Fix matplotlib version conflict - remove system path before imports
_conflicting_path = '/usr/lib/python3/dist-packages'
if _conflicting_path in sys.path:
    sys.path.remove(_conflicting_path)
    _removed = True
else:
    _removed = False

# Clear any cached mpl_toolkits from wrong location
for _mod in [k for k in sys.modules if k.startswith('mpl_toolkits')]:
    del sys.modules[_mod]

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for animation display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - register 3d projection
from typing import List, Optional, Callable

if _removed:
    sys.path.append(_conflicting_path)

from ..obstacles.base import Obstacle
from ..obstacles.dynamic import DynamicObstacle
from ..dynamics.uav_state import UAVState
from .plotter_3d import Plotter3D


class TrajectoryAnimator:
    """
    Animator for UAV trajectory playback.

    Creates smooth animations of simulation results with
    support for dynamic obstacles.
    """

    def __init__(
        self,
        figsize: tuple = (12, 9),
        interval: int = 50,  # ms between frames
        title: str = "UAV Path Planning Animation",
    ):
        """
        Initialize animator.

        Args:
            figsize: Figure size
            interval: Animation interval in milliseconds
            title: Animation title
        """
        self.figsize = figsize
        self.interval = interval
        self.title = title
        self.plotter = None
        self.animation = None

    def animate_trajectory(
        self,
        states: List[UAVState],
        times: List[float],
        static_obstacles: List[Obstacle] = None,
        dynamic_obstacles: List[DynamicObstacle] = None,
        goal_position: Optional[np.ndarray] = None,
        predicted_trajectories: Optional[List[np.ndarray]] = None,
        xlim: tuple = (-5, 25),
        ylim: tuple = (-15, 15),
        zlim: tuple = (0, 15),
        show_trail: bool = True,
        trail_length: int = 50,
        show_full_trajectory: bool = True,
    ) -> FuncAnimation:
        """
        Create animation of trajectory.

        Args:
            states: List of UAVState objects
            times: List of timestamps
            static_obstacles: Static obstacles
            dynamic_obstacles: Dynamic obstacles
            goal_position: Goal position marker
            predicted_trajectories: MPC predictions at each timestep
            xlim, ylim, zlim: Axis limits
            show_trail: Show trajectory trail
            trail_length: Number of past positions to show
            show_full_trajectory: Show complete trajectory as faded background

        Returns:
            FuncAnimation object
        """
        static_obstacles = static_obstacles or []
        dynamic_obstacles = dynamic_obstacles or []

        # Create figure and axes with dark background for better visuals
        fig = plt.figure(figsize=self.figsize, facecolor='#1a1a2e')
        ax = fig.add_subplot(111, projection="3d", facecolor='#16213e')
        ax.set_title(self.title, color='white', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("X [m]", color='white')
        ax.set_ylabel("Y [m]", color='white')
        ax.set_zlabel("Z [m]", color='white')
        
        # Style axis
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.grid(True, alpha=0.3)

        # Store all positions for trail
        positions = np.array([s.position for s in states])

        # Plot full trajectory as faded background line
        if show_full_trajectory:
            ax.plot(
                positions[:, 0], positions[:, 1], positions[:, 2],
                color='#4a4a6a', linewidth=1.5, alpha=0.4, linestyle='--',
                label="Full Path"
            )

        # Plot static obstacles
        for obs in static_obstacles:
            self._plot_obstacle(ax, obs, color="#e94560", alpha=0.4)

        # Plot goal with glow effect
        if goal_position is not None:
            ax.scatter(
                [goal_position[0]], [goal_position[1]], [goal_position[2]],
                c="#00ff88", s=300, marker="*", edgecolors='white', linewidths=1,
                label="Goal", zorder=10
            )
            # Add goal ring
            theta = np.linspace(0, 2*np.pi, 50)
            ring_r = 1.0
            ring_x = goal_position[0] + ring_r * np.cos(theta)
            ring_y = goal_position[1] + ring_r * np.sin(theta)
            ring_z = np.full_like(theta, goal_position[2])
            ax.plot(ring_x, ring_y, ring_z, color='#00ff88', alpha=0.5, linewidth=2)

        # Plot start position
        start_pos = positions[0]
        ax.scatter(
            [start_pos[0]], [start_pos[1]], [start_pos[2]],
            c="#ffa500", s=150, marker="o", edgecolors='white', linewidths=1,
            label="Start", zorder=10
        )

        # Initialize artists - UAV as larger marker
        (uav_marker,) = ax.plot([], [], [], 'o', color='#00d4ff', markersize=12, 
                                 markeredgecolor='white', markeredgewidth=2, label="UAV")
        (trail_line,) = ax.plot([], [], [], '-', color='#00d4ff', linewidth=3, alpha=0.9)
        (predicted_line,) = ax.plot([], [], [], '--', color='#ffff00', linewidth=2, 
                                     alpha=0.7, label="MPC Prediction")

        # Dynamic obstacle markers
        dyn_markers = []
        for i, obs in enumerate(dynamic_obstacles):
            (marker,) = ax.plot([], [], [], 'o', color='#ff6b6b', markersize=10,
                                markeredgecolor='white', markeredgewidth=1)
            dyn_markers.append(marker)

        # Status text box
        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color='white',
                              fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
        
        # Progress text
        progress_text = ax.text2D(0.02, 0.88, "", transform=ax.transAxes, color='#00ff88',
                                  fontsize=10)

        def init():
            uav_marker.set_data([], [])
            uav_marker.set_3d_properties([])
            trail_line.set_data([], [])
            trail_line.set_3d_properties([])
            predicted_line.set_data([], [])
            predicted_line.set_3d_properties([])
            time_text.set_text("")
            progress_text.set_text("")
            for marker in dyn_markers:
                marker.set_data([], [])
                marker.set_3d_properties([])
            return [uav_marker, trail_line, predicted_line, time_text, progress_text] + dyn_markers

        def update(frame):
            # Update UAV position
            state = states[frame]
            uav_marker.set_data([state.x], [state.y])
            uav_marker.set_3d_properties([state.z])

            # Update trail - show traveled path
            if show_trail:
                trail_pos = positions[: frame + 1]
                trail_line.set_data(trail_pos[:, 0], trail_pos[:, 1])
                trail_line.set_3d_properties(trail_pos[:, 2])

            # Update predicted trajectory
            if predicted_trajectories and frame < len(predicted_trajectories):
                pred = predicted_trajectories[frame]
                if pred is not None and len(pred) > 0:
                    predicted_line.set_data(pred[:, 0], pred[:, 1])
                    predicted_line.set_3d_properties(pred[:, 2])

            # Update dynamic obstacles
            t = times[frame]
            for i, obs in enumerate(dynamic_obstacles):
                obs.update_time(t)
                pos = obs.current_position
                dyn_markers[i].set_data([pos[0]], [pos[1]])
                dyn_markers[i].set_3d_properties([pos[2]])

            # Update status text
            vel = np.linalg.norm(state.velocity)
            time_text.set_text(f"Time: {t:.2f}s | Velocity: {vel:.1f} m/s")
            
            # Update progress
            progress_pct = (frame + 1) / len(states) * 100
            if goal_position is not None:
                dist_to_goal = np.linalg.norm(state.position - goal_position)
                progress_text.set_text(f"Progress: {progress_pct:.0f}% | Distance to goal: {dist_to_goal:.1f}m")
            else:
                progress_text.set_text(f"Progress: {progress_pct:.0f}%")

            # Rotate view slowly for 3D effect
            ax.view_init(elev=25, azim=45 + frame * 0.3)

            return [uav_marker, trail_line, predicted_line, time_text, progress_text] + dyn_markers

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            blit=False,
            repeat=True,
        )

        ax.legend(loc="upper right", facecolor='#1a1a2e', edgecolor='gray', 
                  labelcolor='white', fontsize=9)
        
        plt.tight_layout()

        return self.animation

    def _plot_obstacle(self, ax, obstacle, color="red", alpha=0.3):
        """Plot obstacle on given axes."""
        from ..obstacles.static import SphereObstacle, CylinderObstacle, BoxObstacle

        if isinstance(obstacle, SphereObstacle):
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            r = obstacle.radius + obstacle.safety_margin
            x = obstacle.center[0] + r * np.outer(np.cos(u), np.sin(v))
            y = obstacle.center[1] + r * np.outer(np.sin(u), np.sin(v))
            z = obstacle.center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

        elif isinstance(obstacle, BoxObstacle):
            # Simplified box plot
            center = obstacle.center
            ext = obstacle.half_extents + obstacle.safety_margin
            corners = []
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        corners.append(center + np.array([dx, dy, dz]) * ext)
            corners = np.array(corners)
            # Just plot edges
            ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c=color, alpha=alpha)

    def show(self, block: bool = True):
        """Display the animation."""
        plt.show(block=block)

    def save(
        self,
        filename: str,
        fps: int = 20,
        dpi: int = 150,
        writer: str = "pillow",
    ):
        """
        Save animation to file.

        Args:
            filename: Output filename (e.g., 'animation.gif' or 'animation.mp4')
            fps: Frames per second
            dpi: Resolution
            writer: Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)
        """
        if self.animation is not None:
            self.animation.save(filename, writer=writer, fps=fps, dpi=dpi)


def create_animation(
    states: List[UAVState],
    times: List[float],
    static_obstacles: List[Obstacle] = None,
    dynamic_obstacles: List[DynamicObstacle] = None,
    goal_position: Optional[np.ndarray] = None,
    predicted_trajectories: Optional[List[np.ndarray]] = None,
    interval: int = 50,
) -> FuncAnimation:
    """
    Convenience function to create trajectory animation.

    Args:
        states: List of UAV states
        times: List of timestamps
        static_obstacles: Static obstacles
        dynamic_obstacles: Dynamic obstacles
        goal_position: Goal marker
        predicted_trajectories: MPC predictions
        interval: Animation interval in ms

    Returns:
        FuncAnimation object
    """
    animator = TrajectoryAnimator(interval=interval)

    # Compute axis limits
    positions = np.array([s.position for s in states])
    margin = 5
    xlim = (positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    ylim = (positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    zlim = (0, max(positions[:, 2].max() + 5, 15))

    return animator.animate_trajectory(
        states=states,
        times=times,
        static_obstacles=static_obstacles,
        dynamic_obstacles=dynamic_obstacles,
        goal_position=goal_position,
        predicted_trajectories=predicted_trajectories,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )
