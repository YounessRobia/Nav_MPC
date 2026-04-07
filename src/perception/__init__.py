"""
Perception module for sensor-based obstacle discovery.

This module provides simulated sensor perception for MPC-based UAV planning,
allowing obstacles to be discovered within a detection radius rather than
being known a priori.

Main Components:
    - SensorModel: Simulates sensor detection with configurable FOV and range
    - TrackedObstacle: Discovered static obstacle with confidence tracking
    - TrackedDynamicObstacle: Discovered dynamic obstacle with velocity estimation
    - ObstacleTracker: Manages discovered obstacles and track lifecycle
    - SensorSimulation: High-level wrapper for easy MPC integration

Usage Example:
    from src.perception import SensorSimulation, SensorConfig, TrackerConfig

    # Create sensor simulation
    sensor_sim = SensorSimulation(
        sensor_config=SensorConfig(detection_radius=12.0, field_of_view=np.pi/2),
        tracker_config=TrackerConfig(forget_dynamic_after=10.0),
    )

    # Set world obstacles (ground truth)
    sensor_sim.set_world(static_obstacles, dynamic_obstacles)

    # In simulation loop:
    discovered_static, discovered_dynamic = sensor_sim.perceive_and_update(
        robot_state, current_time
    )
    solution = mpc.solve(
        state=robot_state,
        reference=reference,
        static_obstacles=discovered_static,
        dynamic_obstacles=discovered_dynamic,
        t_current=current_time,
    )
"""

from .sensor_model import (
    SensorModel,
    SensorConfig,
    Detection,
)

from .tracked_obstacle import (
    TrackedObstacle,
    TrackedDynamicObstacle,
    create_tracked_static_from_detection,
    create_tracked_dynamic_from_detection,
)

from .obstacle_tracker import (
    ObstacleTracker,
    TrackerConfig,
)

from .sensor_simulation import (
    SensorSimulation,
    DiscoveryEvent,
    create_sensor_simulation_from_config,
)

__all__ = [
    # Sensor model
    "SensorModel",
    "SensorConfig",
    "Detection",
    # Tracked obstacles
    "TrackedObstacle",
    "TrackedDynamicObstacle",
    "create_tracked_static_from_detection",
    "create_tracked_dynamic_from_detection",
    # Tracker
    "ObstacleTracker",
    "TrackerConfig",
    # High-level interface
    "SensorSimulation",
    "DiscoveryEvent",
    "create_sensor_simulation_from_config",
]
