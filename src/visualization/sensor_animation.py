"""Animation utilities for sensor-based trajectory visualization."""

import sys

# Fix matplotlib version conflict
_conflicting_path = '/usr/lib/python3/dist-packages'
if _conflicting_path in sys.path:
    sys.path.remove(_conflicting_path)
    _removed = True
else:
    _removed = False

for _mod in [k for k in sys.modules if k.startswith('mpl_toolkits')]:
    del sys.modules[_mod]

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Optional, Dict, Tuple, Any

if _removed:
    sys.path.append(_conflicting_path)

from ..obstacles.base import Obstacle
from ..obstacles.dynamic import DynamicObstacle
from ..dynamics.uav_state import UAVState
from ..perception.sensor_simulation import DiscoveryEvent


class SensorTrajectoryAnimator:
    """
    Animator for UAV trajectory with sensor visualization.

    Shows:
    - FOV cone following UAV heading
    - Detection range circle
    - Flash effect on obstacle discovery
    - Discovered vs undiscovered obstacles (different colors)
    """

    def __init__(
        self,
        figsize: tuple = (16, 7),
        interval: int = 50,
        title: str = "Sensor-Based Obstacle Avoidance",
    ):
        self.figsize = figsize
        self.interval = interval
        self.title = title
        self.animation = None

    def animate_trajectory(
        self,
        states: List[UAVState],
        times: List[float],
        world_static: List[Obstacle],
        world_dynamic: List[DynamicObstacle],
        discovery_events: List[DiscoveryEvent],
        sensor_config: Any,
        goal_position: Optional[np.ndarray] = None,
        predicted_trajectories: Optional[List[np.ndarray]] = None,
        xlim: tuple = (-5, 35),
        ylim: tuple = (-15, 15),
        zlim: tuple = (0, 15),
    ) -> FuncAnimation:
        """
        Create animation with sensor visualization.

        Args:
            states: List of UAVState objects
            times: List of timestamps
            world_static: All static obstacles (ground truth)
            world_dynamic: All dynamic obstacles (ground truth)
            discovery_events: List of DiscoveryEvent from SensorSimulation
            sensor_config: SensorConfig with detection_radius and field_of_view
            goal_position: Goal position marker
            predicted_trajectories: MPC predictions at each timestep
            xlim, ylim, zlim: Axis limits
        """
        # Build discovery time map
        discovery_times = {e.obstacle_id: e.time for e in discovery_events}

        # Create figure with two subplots: 3D view and 2D top-down
        fig = plt.figure(figsize=self.figsize, facecolor='#1a1a2e')

        # 3D trajectory view
        ax3d = fig.add_subplot(121, projection='3d', facecolor='#16213e')
        ax3d.set_title("3D Trajectory View", color='white', fontsize=12, fontweight='bold')
        ax3d.set_xlim(xlim)
        ax3d.set_ylim(ylim)
        ax3d.set_zlim(zlim)
        ax3d.set_xlabel("X [m]", color='white')
        ax3d.set_ylabel("Y [m]", color='white')
        ax3d.set_zlabel("Z [m]", color='white')
        ax3d.tick_params(colors='white')
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.grid(True, alpha=0.3)

        # 2D top-down view with sensor FOV
        ax2d = fig.add_subplot(122, facecolor='#16213e')
        ax2d.set_title("Top-Down View with Sensor FOV", color='white', fontsize=12, fontweight='bold')
        ax2d.set_xlim(xlim)
        ax2d.set_ylim(ylim)
        ax2d.set_xlabel("X [m]", color='white')
        ax2d.set_ylabel("Y [m]", color='white')
        ax2d.tick_params(colors='white')
        ax2d.set_aspect('equal')
        ax2d.grid(True, alpha=0.3, color='gray')

        # Store positions
        positions = np.array([s.position for s in states])

        # Plot full trajectory as faded background (3D)
        ax3d.plot(
            positions[:, 0], positions[:, 1], positions[:, 2],
            color='#4a4a6a', linewidth=1.5, alpha=0.3, linestyle='--'
        )

        # Plot full trajectory (2D)
        ax2d.plot(
            positions[:, 0], positions[:, 1],
            color='#4a4a6a', linewidth=1.5, alpha=0.3, linestyle='--'
        )

        # Plot goal
        if goal_position is not None:
            ax3d.scatter(*goal_position, c='#00ff88', s=200, marker='*',
                        edgecolors='white', linewidths=1, zorder=10)
            ax2d.scatter(goal_position[0], goal_position[1], c='#00ff88',
                        s=200, marker='*', edgecolors='white', linewidths=1, zorder=10)

        # Plot start
        ax3d.scatter(*positions[0], c='#ffa500', s=100, marker='o',
                    edgecolors='white', linewidths=1, zorder=10)
        ax2d.scatter(positions[0, 0], positions[0, 1], c='#ffa500', s=100,
                    marker='o', edgecolors='white', linewidths=1, zorder=10)

        # Initialize animated elements
        # 3D elements
        (uav_marker_3d,) = ax3d.plot([], [], [], 'o', color='#00d4ff', markersize=12,
                                     markeredgecolor='white', markeredgewidth=2)
        (trail_3d,) = ax3d.plot([], [], [], '-', color='#00d4ff', linewidth=2, alpha=0.8)
        (predicted_3d,) = ax3d.plot([], [], [], '--', color='#ffff00', linewidth=2, alpha=0.7)

        # 2D elements
        (uav_marker_2d,) = ax2d.plot([], [], 'o', color='#00d4ff', markersize=12,
                                     markeredgecolor='white', markeredgewidth=2)
        (trail_2d,) = ax2d.plot([], [], '-', color='#00d4ff', linewidth=2, alpha=0.8)
        (predicted_2d,) = ax2d.plot([], [], '--', color='#ffff00', linewidth=2, alpha=0.7)

        # FOV wedge (2D) - will be updated each frame
        fov_wedge = None
        detection_circle = None

        # Static obstacle markers (will change color when discovered)
        static_markers_3d = []
        static_markers_2d = []
        static_obstacle_ids = []

        for i, obs in enumerate(world_static):
            center = obs.get_center()
            radius = obs.radius
            obs_id = id(obs)
            static_obstacle_ids.append(obs_id)

            # 3D sphere (simplified as scatter)
            marker_3d = ax3d.scatter([center[0]], [center[1]], [center[2]],
                                     c='#666666', s=100, alpha=0.5, marker='o')
            static_markers_3d.append(marker_3d)

            # 2D circle
            circle = Circle((center[0], center[1]), radius,
                           color='#666666', alpha=0.5, fill=True)
            ax2d.add_patch(circle)
            static_markers_2d.append(circle)

        # Dynamic obstacle markers
        dyn_markers_3d = []
        dyn_markers_2d = []
        dyn_trails_2d = []
        dyn_obstacle_ids = []

        for i, obs in enumerate(world_dynamic):
            obs_id = id(obs)
            dyn_obstacle_ids.append(obs_id)

            # 3D marker
            (marker_3d,) = ax3d.plot([], [], [], 'o', color='#666666',
                                     markersize=10, alpha=0.5)
            dyn_markers_3d.append(marker_3d)

            # 2D marker
            (marker_2d,) = ax2d.plot([], [], 'o', color='#666666',
                                     markersize=10, alpha=0.5)
            dyn_markers_2d.append(marker_2d)

            # 2D trail (predicted path)
            (trail,) = ax2d.plot([], [], '--', color='#ff6b6b',
                                linewidth=1, alpha=0.3)
            dyn_trails_2d.append(trail)

        # Detection flash circles (appear briefly when obstacle discovered)
        flash_circles = []

        # Text displays
        time_text = fig.text(0.5, 0.95, "", ha='center', color='white',
                            fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

        discovery_text = ax2d.text(0.02, 0.95, "", transform=ax2d.transAxes,
                                   color='#00ff88', fontsize=10, fontweight='bold',
                                   verticalalignment='top')

        stats_text = ax2d.text(0.02, 0.05, "", transform=ax2d.transAxes,
                              color='white', fontsize=9,
                              verticalalignment='bottom')

        # Store discovered obstacle IDs
        discovered_ids = set()

        # Sensor parameters
        detection_radius = sensor_config.detection_radius
        fov_angle = sensor_config.field_of_view  # half-angle

        def init():
            uav_marker_3d.set_data([], [])
            uav_marker_3d.set_3d_properties([])
            trail_3d.set_data([], [])
            trail_3d.set_3d_properties([])
            predicted_3d.set_data([], [])
            predicted_3d.set_3d_properties([])

            uav_marker_2d.set_data([], [])
            trail_2d.set_data([], [])
            predicted_2d.set_data([], [])

            for marker in dyn_markers_3d:
                marker.set_data([], [])
                marker.set_3d_properties([])
            for marker in dyn_markers_2d:
                marker.set_data([], [])
            for trail in dyn_trails_2d:
                trail.set_data([], [])

            time_text.set_text("")
            discovery_text.set_text("")
            stats_text.set_text("")

            return []

        def update(frame):
            nonlocal fov_wedge, detection_circle, discovered_ids

            state = states[frame]
            t = times[frame]
            pos = state.position

            # Compute heading from velocity or use yaw
            vel = state.velocity
            if np.linalg.norm(vel[:2]) > 0.1:
                yaw = np.arctan2(vel[1], vel[0])
            else:
                yaw = state.yaw

            # Update UAV markers
            uav_marker_3d.set_data([pos[0]], [pos[1]])
            uav_marker_3d.set_3d_properties([pos[2]])

            uav_marker_2d.set_data([pos[0]], [pos[1]])

            # Update trail
            trail_pos = positions[:frame+1]
            trail_3d.set_data(trail_pos[:, 0], trail_pos[:, 1])
            trail_3d.set_3d_properties(trail_pos[:, 2])

            trail_2d.set_data(trail_pos[:, 0], trail_pos[:, 1])

            # Update predicted trajectory
            if predicted_trajectories and frame < len(predicted_trajectories):
                pred = predicted_trajectories[frame]
                if pred is not None and len(pred) > 0:
                    predicted_3d.set_data(pred[:, 0], pred[:, 1])
                    predicted_3d.set_3d_properties(pred[:, 2])
                    predicted_2d.set_data(pred[:, 0], pred[:, 1])

            # Update FOV wedge (remove old, add new)
            if fov_wedge is not None:
                fov_wedge.remove()
            if detection_circle is not None:
                detection_circle.remove()

            # Draw FOV wedge
            fov_start = np.rad2deg(yaw - fov_angle)
            fov_end = np.rad2deg(yaw + fov_angle)
            fov_wedge = Wedge(
                (pos[0], pos[1]), detection_radius,
                fov_start, fov_end,
                color='#00d4ff', alpha=0.15, linewidth=2,
                edgecolor='#00d4ff'
            )
            ax2d.add_patch(fov_wedge)

            # Draw detection range circle (faded)
            detection_circle = Circle(
                (pos[0], pos[1]), detection_radius,
                fill=False, color='#00d4ff', alpha=0.3,
                linewidth=1, linestyle=':'
            )
            ax2d.add_patch(detection_circle)

            # Check for new discoveries and update obstacle colors
            newly_discovered = []
            for event in discovery_events:
                if event.time <= t and event.obstacle_id not in discovered_ids:
                    discovered_ids.add(event.obstacle_id)
                    newly_discovered.append(event)

            # Update static obstacle colors
            for i, obs in enumerate(world_static):
                obs_id = id(obs)
                # Find matching discovery event
                is_discovered = any(
                    e.obstacle_id == i or
                    (e.obstacle_type == 'static' and
                     np.linalg.norm(e.position - obs.get_center()) < 2.0)
                    for e in discovery_events if e.time <= t
                )

                color = '#e94560' if is_discovered else '#666666'
                alpha = 0.7 if is_discovered else 0.3

                static_markers_2d[i].set_facecolor(color)
                static_markers_2d[i].set_alpha(alpha)

            # Update dynamic obstacles
            for i, obs in enumerate(world_dynamic):
                obs.update_time(t)
                dyn_pos = obs.current_position

                # Check if discovered
                is_discovered = any(
                    e.obstacle_type == 'dynamic' and
                    np.linalg.norm(e.position - obs.get_position_at_time(e.time)) < 3.0
                    for e in discovery_events if e.time <= t
                )

                color = '#ff6b6b' if is_discovered else '#666666'
                alpha = 0.8 if is_discovered else 0.4

                dyn_markers_3d[i].set_data([dyn_pos[0]], [dyn_pos[1]])
                dyn_markers_3d[i].set_3d_properties([dyn_pos[2]])
                dyn_markers_3d[i].set_color(color)
                dyn_markers_3d[i].set_alpha(alpha)

                dyn_markers_2d[i].set_data([dyn_pos[0]], [dyn_pos[1]])
                dyn_markers_2d[i].set_color(color)
                dyn_markers_2d[i].set_alpha(alpha)

                # Show predicted path for discovered dynamic obstacles
                if is_discovered:
                    future_pos = obs.predict_over_horizon(t, 0.2, 25)
                    future_arr = np.array(future_pos)
                    dyn_trails_2d[i].set_data(future_arr[:, 0], future_arr[:, 1])
                    dyn_trails_2d[i].set_alpha(0.5)
                else:
                    dyn_trails_2d[i].set_data([], [])

            # Flash effect for newly discovered obstacles
            for flash in flash_circles:
                flash.remove()
            flash_circles.clear()

            for event in newly_discovered:
                flash = Circle(
                    (event.position[0], event.position[1]), 3.0,
                    fill=False, color='#00ff88', alpha=0.8,
                    linewidth=3
                )
                ax2d.add_patch(flash)
                flash_circles.append(flash)

            # Update text
            vel_mag = np.linalg.norm(state.velocity)
            time_text.set_text(f"Time: {t:.2f}s | Velocity: {vel_mag:.1f} m/s")

            # Discovery text
            if newly_discovered:
                disc_str = ", ".join([f"{e.obstacle_type}" for e in newly_discovered])
                discovery_text.set_text(f"DETECTED: {disc_str}")
            else:
                # Fade out after 0.5s
                discovery_text.set_text("")

            # Stats
            num_disc_static = sum(1 for e in discovery_events if e.time <= t and e.obstacle_type == 'static')
            num_disc_dynamic = sum(1 for e in discovery_events if e.time <= t and e.obstacle_type == 'dynamic')
            stats_text.set_text(
                f"Discovered: {num_disc_static}/{len(world_static)} static, "
                f"{num_disc_dynamic}/{len(world_dynamic)} dynamic"
            )

            # Slowly rotate 3D view
            ax3d.view_init(elev=25, azim=45 + frame * 0.2)

            return []

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=self.interval,
            blit=False,
            repeat=True,
        )

        plt.tight_layout()
        return self.animation

    def save(self, filename: str, fps: int = 20, dpi: int = 150, writer: str = "pillow"):
        """Save animation to file."""
        if self.animation is not None:
            self.animation.save(filename, writer=writer, fps=fps, dpi=dpi)

    def show(self, block: bool = True):
        """Display the animation."""
        plt.show(block=block)
