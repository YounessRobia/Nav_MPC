"""3D visualization using Matplotlib."""

import sys

# Fix for matplotlib version conflict between system and pip installations
# Remove system path and clear cached mpl_toolkits modules before any matplotlib import
_conflicting_path = '/usr/lib/python3/dist-packages'
if _conflicting_path in sys.path:
    sys.path.remove(_conflicting_path)
    _removed_system_path = True
else:
    _removed_system_path = False

# Clear any cached mpl_toolkits imports from wrong location
_modules_to_remove = [k for k in sys.modules if k.startswith('mpl_toolkits')]
for mod in _modules_to_remove:
    del sys.modules[mod]

import numpy as np
import matplotlib
# Try to use a non-interactive backend first, fall back if needed
try:
    matplotlib.use('Agg')  # Non-interactive backend for saving
except:
    pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed to register 3d projection
from typing import List, Optional, Tuple

# Restore system path after imports
if _removed_system_path:
    sys.path.append(_conflicting_path)

from ..obstacles.base import Obstacle
from ..obstacles.static import SphereObstacle, CylinderObstacle, BoxObstacle
from ..obstacles.dynamic import DynamicObstacle
from ..dynamics.uav_state import UAVState


class Plotter3D:
    """
    3D visualization for UAV path planning.

    Provides methods for plotting trajectories, obstacles,
    and UAV state in a 3D Matplotlib figure.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 9),
        title: str = "UAV Path Planning",
    ):
        """
        Initialize 3D plotter.

        Args:
            figsize: Figure size (width, height)
            title: Figure title
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(title)

        # Plot handles for updating
        self._trajectory_line = None
        self._predicted_line = None
        self._uav_marker = None
        self._velocity_arrow = None
        self._reference_line = None
        self._obstacle_artists = []

        # Default style
        self.trajectory_color = "blue"
        self.predicted_color = "cyan"
        self.reference_color = "green"
        self.uav_color = "red"

    def setup_axes(
        self,
        xlim: Tuple[float, float] = (-5, 25),
        ylim: Tuple[float, float] = (-15, 15),
        zlim: Tuple[float, float] = (0, 15),
    ):
        """
        Configure axis limits and labels.

        Args:
            xlim: X-axis limits
            ylim: Y-axis limits
            zlim: Z-axis limits
        """
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        try:
            self.ax.set_box_aspect([1, 1, 0.5])
        except AttributeError:
            pass  # Older matplotlib versions

    def plot_trajectory(
        self,
        trajectory: np.ndarray,
        color: Optional[str] = None,
        label: str = "Trajectory",
        linewidth: float = 2.0,
    ):
        """
        Plot a trajectory path.

        Args:
            trajectory: (N, 3) or (N, 10) array of positions/states
            color: Line color
            label: Legend label
            linewidth: Line width
        """
        trajectory = np.asarray(trajectory)
        if trajectory.shape[1] >= 3:
            positions = trajectory[:, :3]
        else:
            positions = trajectory

        color = color or self.trajectory_color
        self._trajectory_line, = self.ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=linewidth,
            label=label,
        )

    def plot_predicted_trajectory(
        self,
        trajectory: np.ndarray,
        color: Optional[str] = None,
        alpha: float = 0.5,
    ):
        """
        Plot MPC predicted trajectory.

        Args:
            trajectory: (N, 3) or (N, 10) array
            color: Line color
            alpha: Transparency
        """
        trajectory = np.asarray(trajectory)
        positions = trajectory[:, :3] if trajectory.shape[1] >= 3 else trajectory
        color = color or self.predicted_color

        if self._predicted_line is not None:
            self._predicted_line.remove()

        self._predicted_line, = self.ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=alpha,
            label="Predicted",
        )

    def plot_reference(
        self,
        reference: np.ndarray,
        color: Optional[str] = None,
        alpha: float = 0.3,
    ):
        """
        Plot reference trajectory.

        Args:
            reference: (N, 3) or (N, 10) array
            color: Line color
            alpha: Transparency
        """
        reference = np.asarray(reference)
        positions = reference[:, :3] if reference.shape[1] >= 3 else reference
        color = color or self.reference_color

        if self._reference_line is not None:
            self._reference_line.remove()

        self._reference_line, = self.ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=alpha,
            label="Reference",
        )

    def plot_uav(
        self,
        state: UAVState,
        size: float = 0.5,
        show_velocity: bool = True,
    ):
        """
        Plot UAV position and optional velocity vector.

        Args:
            state: Current UAV state
            size: Marker size
            show_velocity: Whether to show velocity arrow
        """
        # Remove old markers
        if self._uav_marker is not None:
            self._uav_marker.remove()

        # Plot UAV position
        self._uav_marker = self.ax.scatter(
            [state.x],
            [state.y],
            [state.z],
            c=self.uav_color,
            s=100 * size,
            marker="o",
            label="UAV",
        )

        # Plot velocity arrow using quiver
        if show_velocity and np.linalg.norm(state.velocity) > 0.1:
            scale = 0.5
            self.ax.quiver(
                state.x, state.y, state.z,
                state.velocity[0] * scale,
                state.velocity[1] * scale,
                state.velocity[2] * scale,
                color="orange",
                arrow_length_ratio=0.3,
            )

    def plot_sphere_obstacle(
        self,
        obstacle: SphereObstacle,
        color: str = "red",
        alpha: float = 0.3,
        resolution: int = 20,
    ):
        """
        Plot spherical obstacle.

        Args:
            obstacle: SphereObstacle instance
            color: Surface color
            alpha: Transparency
            resolution: Sphere resolution
        """
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        r = obstacle.radius + obstacle.safety_margin
        x = obstacle.center[0] + r * np.outer(np.cos(u), np.sin(v))
        y = obstacle.center[1] + r * np.outer(np.sin(u), np.sin(v))
        z = obstacle.center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))

        surface = self.ax.plot_surface(
            x, y, z, color=color, alpha=alpha, linewidth=0
        )
        self._obstacle_artists.append(surface)

    def plot_cylinder_obstacle(
        self,
        obstacle: CylinderObstacle,
        color: str = "red",
        alpha: float = 0.3,
        resolution: int = 20,
    ):
        """
        Plot cylindrical obstacle.

        Args:
            obstacle: CylinderObstacle instance
            color: Surface color
            alpha: Transparency
            resolution: Cylinder resolution
        """
        theta = np.linspace(0, 2 * np.pi, resolution)
        r = obstacle.radius + obstacle.safety_margin

        z_min = obstacle.z_min if np.isfinite(obstacle.z_min) else 0
        z_max = obstacle.z_max if np.isfinite(obstacle.z_max) else 20
        z = np.linspace(z_min, z_max, 10)

        theta_grid, z_grid = np.meshgrid(theta, z)
        x = obstacle.center_xy[0] + r * np.cos(theta_grid)
        y = obstacle.center_xy[1] + r * np.sin(theta_grid)

        surface = self.ax.plot_surface(
            x, y, z_grid, color=color, alpha=alpha, linewidth=0
        )
        self._obstacle_artists.append(surface)

    def plot_box_obstacle(
        self,
        obstacle: BoxObstacle,
        color: str = "red",
        alpha: float = 0.3,
    ):
        """
        Plot box obstacle using simple wireframe/scatter.

        Args:
            obstacle: BoxObstacle instance
            color: Surface color
            alpha: Transparency
        """
        center = obstacle.center
        ext = obstacle.half_extents + obstacle.safety_margin

        # Define vertices
        vertices = []
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    vertices.append(center + np.array([dx, dy, dz]) * ext)
        vertices = np.array(vertices)

        # Plot edges instead of faces for compatibility
        edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),  # x-parallel
            (0, 2), (1, 3), (4, 6), (5, 7),  # y-parallel
            (0, 4), (1, 5), (2, 6), (3, 7),  # z-parallel
        ]
        for i, j in edges:
            self.ax.plot(
                [vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]],
                color=color, alpha=alpha, linewidth=1
            )

    def plot_obstacles(
        self,
        obstacles: List[Obstacle],
        color: str = "red",
        alpha: float = 0.3,
    ):
        """
        Plot all obstacles.

        Args:
            obstacles: List of obstacles
            color: Surface color
            alpha: Transparency
        """
        for obs in obstacles:
            if isinstance(obs, SphereObstacle):
                self.plot_sphere_obstacle(obs, color, alpha)
            elif isinstance(obs, CylinderObstacle):
                self.plot_cylinder_obstacle(obs, color, alpha)
            elif isinstance(obs, BoxObstacle):
                self.plot_box_obstacle(obs, color, alpha)
            elif isinstance(obs, DynamicObstacle):
                # Plot as sphere at current position
                sphere = SphereObstacle(
                    center=obs.current_position,
                    radius=obs.radius,
                    safety_margin=obs.safety_margin,
                )
                self.plot_sphere_obstacle(sphere, "orange", alpha)

    def plot_goal(
        self,
        position: np.ndarray,
        color: str = "green",
        size: float = 1.0,
    ):
        """
        Plot goal position marker.

        Args:
            position: 3D goal position
            color: Marker color
            size: Marker size
        """
        self.ax.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            c=color,
            s=150 * size,
            marker="*",
            label="Goal",
        )

    def plot_waypoints(
        self,
        waypoints: np.ndarray,
        color: str = "purple",
        size: float = 0.5,
    ):
        """
        Plot waypoint markers.

        Args:
            waypoints: (N, 3) array of waypoint positions
            color: Marker color
            size: Marker size
        """
        waypoints = np.asarray(waypoints)
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, -1)

        self.ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            waypoints[:, 2],
            c=color,
            s=80 * size,
            marker="D",
            label="Waypoints",
        )

    def clear_obstacles(self):
        """Remove all obstacle artists."""
        for artist in self._obstacle_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._obstacle_artists = []

    def add_legend(self, loc: str = "upper right"):
        """Add legend to plot."""
        self.ax.legend(loc=loc)

    def show(self, block: bool = True):
        """Display the figure."""
        plt.show(block=block)

    def save(self, filename: str, dpi: int = 150):
        """Save figure to file."""
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def close(self):
        """Close the figure."""
        plt.close(self.fig)


def plot_simulation_results(
    trajectory: np.ndarray,
    obstacles: List[Obstacle],
    goal_position: np.ndarray,
    waypoints: Optional[np.ndarray] = None,
    predicted_trajectories: Optional[List[np.ndarray]] = None,
    title: str = "Simulation Results",
    save_path: Optional[str] = None,
):
    """
    Convenience function to plot complete simulation results.

    Args:
        trajectory: Actual trajectory (N, 3) or (N, 10)
        obstacles: List of obstacles
        goal_position: Goal position
        waypoints: Optional waypoints
        predicted_trajectories: Optional list of MPC predictions
        title: Plot title
        save_path: Optional path to save figure
    """
    # Switch to interactive backend for display
    plt.switch_backend('TkAgg')

    plotter = Plotter3D(title=title)

    # Compute axis limits from trajectory
    trajectory = np.asarray(trajectory)
    positions = trajectory[:, :3] if trajectory.shape[1] >= 3 else trajectory

    x_margin = 5
    xlim = (positions[:, 0].min() - x_margin, positions[:, 0].max() + x_margin)
    ylim = (positions[:, 1].min() - x_margin, positions[:, 1].max() + x_margin)
    zlim = (0, max(positions[:, 2].max() + 5, 15))

    plotter.setup_axes(xlim, ylim, zlim)
    plotter.plot_obstacles(obstacles)
    plotter.plot_trajectory(positions)
    plotter.plot_goal(goal_position)

    if waypoints is not None and len(waypoints) > 0:
        plotter.plot_waypoints(waypoints)

    # Plot sample of predicted trajectories
    if predicted_trajectories:
        step = max(1, len(predicted_trajectories) // 5)
        for i in range(0, len(predicted_trajectories), step):
            pred = predicted_trajectories[i]
            plotter.ax.plot(
                pred[:, 0], pred[:, 1], pred[:, 2],
                "c--", alpha=0.2, linewidth=0.5
            )

    plotter.add_legend()

    if save_path:
        plotter.save(save_path)

    return plotter
