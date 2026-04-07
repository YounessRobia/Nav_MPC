"""High-level sensor simulation wrapper for MPC integration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..dynamics.uav_state import UAVState
from ..obstacles.base import Obstacle
from ..obstacles.dynamic import DynamicObstacle
from .sensor_model import SensorModel, SensorConfig, Detection
from .obstacle_tracker import ObstacleTracker, TrackerConfig


@dataclass
class DiscoveryEvent:
    """Record of an obstacle discovery."""

    time: float
    """Time of discovery."""

    obstacle_id: int
    """ID of discovered obstacle."""

    obstacle_type: str
    """'static' or 'dynamic'."""

    position: np.ndarray
    """Position at discovery."""


class SensorSimulation:
    """
    High-level wrapper for sensor-based obstacle perception.

    Integrates the sensor model and obstacle tracker to provide
    a simple interface for the MPC simulation loop.

    Usage:
        sensor_sim = SensorSimulation()
        sensor_sim.set_world(static_obstacles, dynamic_obstacles)

        # In simulation loop:
        discovered_static, discovered_dynamic = sensor_sim.perceive_and_update(
            robot_state, current_time
        )
        solution = mpc.solve(..., static_obstacles=discovered_static,
                           dynamic_obstacles=discovered_dynamic, ...)
    """

    def __init__(
        self,
        sensor_config: SensorConfig = None,
        tracker_config: TrackerConfig = None,
    ):
        """
        Initialize sensor simulation.

        Args:
            sensor_config: Configuration for sensor model
            tracker_config: Configuration for obstacle tracker
        """
        self.sensor = SensorModel(sensor_config or SensorConfig())
        self.tracker = ObstacleTracker(tracker_config or TrackerConfig())

        # World (ground truth - hidden from UAV)
        self.world_static: List[Obstacle] = []
        self.world_dynamic: List[DynamicObstacle] = []

        # Discovery log for analysis
        self.discovery_log: List[DiscoveryEvent] = []
        self._known_obstacle_ids: set = set()

    def set_world(
        self,
        static_obstacles: List[Obstacle],
        dynamic_obstacles: List[DynamicObstacle],
    ):
        """
        Set ground truth world state.

        Args:
            static_obstacles: All static obstacles in the world
            dynamic_obstacles: All dynamic obstacles in the world
        """
        self.world_static = list(static_obstacles)
        self.world_dynamic = list(dynamic_obstacles)

    def perceive_and_update(
        self,
        robot_state: UAVState,
        current_time: float,
    ) -> Tuple[List[Obstacle], List[DynamicObstacle]]:
        """
        Simulate perception and return obstacles for MPC.

        This is the main entry point called each simulation step.

        Args:
            robot_state: Current UAV state
            current_time: Current simulation time

        Returns:
            Tuple of (discovered_static, discovered_dynamic) for MPC
        """
        # Get detections from sensor
        detections = self.sensor.detect(
            robot_position=robot_state.position,
            robot_yaw=robot_state.yaw,
            static_obstacles=self.world_static,
            dynamic_obstacles=self.world_dynamic,
            t=current_time,
        )

        # Log new discoveries
        self._log_discoveries(detections, current_time)

        # Update tracker
        self.tracker.update(detections, current_time)

        # Return obstacles for MPC
        return (
            self.tracker.get_static_obstacles_for_mpc(),
            self.tracker.get_dynamic_obstacles_for_mpc(),
        )

    def _log_discoveries(self, detections: List[Detection], current_time: float):
        """Log newly discovered obstacles."""
        for detection in detections:
            if detection.obstacle_id not in self._known_obstacle_ids:
                self._known_obstacle_ids.add(detection.obstacle_id)
                event = DiscoveryEvent(
                    time=current_time,
                    obstacle_id=detection.obstacle_id,
                    obstacle_type="static" if detection.is_static else "dynamic",
                    position=detection.position.copy(),
                )
                self.discovery_log.append(event)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get discovery and tracking statistics.

        Returns:
            Dictionary with perception metrics
        """
        tracker_stats = self.tracker.get_statistics()
        return {
            **tracker_stats,
            "num_world_static": len(self.world_static),
            "num_world_dynamic": len(self.world_dynamic),
            "num_discoveries": len(self.discovery_log),
            "detection_radius": self.sensor.config.detection_radius,
            "field_of_view": self.sensor.config.field_of_view,
        }

    def get_discovery_log(self) -> List[DiscoveryEvent]:
        """Get list of discovery events."""
        return self.discovery_log.copy()

    def get_tracked_positions_for_visualization(
        self, current_time: float
    ) -> List[Tuple[np.ndarray, float, bool, float]]:
        """
        Get tracked obstacle positions for visualization.

        Args:
            current_time: Current time for dynamic obstacle prediction

        Returns:
            List of (position, radius, is_static, confidence) tuples
        """
        return self.tracker.get_all_tracked_positions(current_time)

    def get_world_obstacle_positions(
        self, current_time: float
    ) -> List[Tuple[np.ndarray, float, bool]]:
        """
        Get ground truth world obstacle positions (for visualization).

        Args:
            current_time: Current time for dynamic obstacle positions

        Returns:
            List of (position, radius, is_static) tuples
        """
        positions = []

        for obs in self.world_static:
            center = obs.get_center()
            radius = obs.get_bounding_radius()
            positions.append((center, radius, True))

        for obs in self.world_dynamic:
            center = obs.get_position_at_time(current_time)
            radius = obs.get_bounding_radius()
            positions.append((center, radius, False))

        return positions

    def is_obstacle_discovered(self, obstacle) -> bool:
        """
        Check if an obstacle has been discovered.

        Args:
            obstacle: Obstacle to check

        Returns:
            True if obstacle has been detected at least once
        """
        obs_id = id(obstacle)
        return obs_id in self.sensor._obstacle_id_map

    def get_sensor_config(self) -> SensorConfig:
        """Get sensor configuration."""
        return self.sensor.config

    def get_tracker_config(self) -> TrackerConfig:
        """Get tracker configuration."""
        return self.tracker.config

    def reset(self):
        """Reset perception state (keeps world obstacles)."""
        self.sensor.reset()
        self.tracker.reset()
        self.discovery_log.clear()
        self._known_obstacle_ids.clear()

    def __repr__(self) -> str:
        num_static, num_dynamic = self.tracker.get_num_tracked()
        return (
            f"SensorSimulation(world=[{len(self.world_static)}S, "
            f"{len(self.world_dynamic)}D], tracked=[{num_static}S, {num_dynamic}D])"
        )


def create_sensor_simulation_from_config(
    perception_config: Dict[str, Any]
) -> SensorSimulation:
    """
    Factory function to create SensorSimulation from config dict.

    Args:
        perception_config: Dictionary with perception parameters

    Returns:
        Configured SensorSimulation
    """
    sensor_cfg = perception_config.get("sensor", {})
    tracker_cfg = perception_config.get("tracker", {})

    sensor_config = SensorConfig(
        detection_radius=sensor_cfg.get("detection_radius", 15.0),
        field_of_view=sensor_cfg.get("field_of_view", np.pi / 2),
        noise_std_position=sensor_cfg.get("noise_std_position", 0.1),
        noise_std_velocity=sensor_cfg.get("noise_std_velocity", 0.05),
        detection_probability=sensor_cfg.get("detection_probability", 1.0),
        min_detection_range=sensor_cfg.get("min_detection_range", 0.5),
        detect_behind=sensor_cfg.get("detect_behind", False),
    )

    tracker_config = TrackerConfig(
        association_threshold=tracker_cfg.get("association_threshold", 3.0),
        min_confidence_threshold=tracker_cfg.get("min_confidence_threshold", 0.1),
        confidence_decay_rate=tracker_cfg.get("confidence_decay_rate", 0.2),
        forget_static_after=tracker_cfg.get("forget_static_after", float("inf")),
        forget_dynamic_after=tracker_cfg.get("forget_dynamic_after", 10.0),
    )

    return SensorSimulation(sensor_config, tracker_config)
