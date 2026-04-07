"""Obstacle tracker for managing discovered obstacles."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..obstacles.base import Obstacle
from ..obstacles.dynamic import DynamicObstacle
from .sensor_model import Detection
from .tracked_obstacle import (
    TrackedObstacle,
    TrackedDynamicObstacle,
    create_tracked_static_from_detection,
    create_tracked_dynamic_from_detection,
)


@dataclass
class TrackerConfig:
    """Configuration for obstacle tracker."""

    association_threshold: float = 3.0
    """Maximum distance to associate detection with existing track (meters)."""

    min_confidence_threshold: float = 0.1
    """Remove tracks below this confidence level."""

    confidence_decay_rate: float = 0.2
    """Confidence decay per second when not observed."""

    forget_static_after: float = float("inf")
    """Seconds to keep static obstacles after last observation (inf = never forget)."""

    forget_dynamic_after: float = 10.0
    """Seconds to keep dynamic obstacles after last observation."""


class ObstacleTracker:
    """
    Manages discovered obstacles and their tracking state.

    Maintains separate dictionaries for static and dynamic tracked obstacles.
    Uses nearest-neighbor association for matching detections to existing tracks.
    """

    def __init__(self, config: TrackerConfig = None):
        """
        Initialize obstacle tracker.

        Args:
            config: Tracker configuration parameters
        """
        self.config = config or TrackerConfig()

        self.tracked_static: Dict[int, TrackedObstacle] = {}
        self.tracked_dynamic: Dict[int, TrackedDynamicObstacle] = {}

        # Statistics
        self._total_static_discovered = 0
        self._total_dynamic_discovered = 0

    def update(self, detections: List[Detection], current_time: float):
        """
        Process new detections and update tracking state.

        Steps:
        1. Separate static and dynamic detections
        2. Associate detections with existing tracks
        3. Update matched tracks
        4. Create new tracks for unmatched detections
        5. Decay confidence for unobserved tracks
        6. Remove stale tracks

        Args:
            detections: List of new detections
            current_time: Current simulation time
        """
        # Separate detections by type
        static_detections = [d for d in detections if d.is_static]
        dynamic_detections = [d for d in detections if not d.is_static]

        # Process static detections
        self._process_static_detections(static_detections, current_time)

        # Process dynamic detections
        self._process_dynamic_detections(dynamic_detections, current_time)

        # Decay and cleanup
        self._decay_and_cleanup(current_time)

    def _process_static_detections(
        self, detections: List[Detection], current_time: float
    ):
        """Process static obstacle detections."""
        matched_track_ids = set()

        for detection in detections:
            # Try to associate with existing track
            track_id = self._find_matching_static_track(detection)

            if track_id is not None:
                # Update existing track
                self.tracked_static[track_id].update(detection)
                matched_track_ids.add(track_id)
            else:
                # Create new track
                new_track = create_tracked_static_from_detection(detection)
                self.tracked_static[detection.obstacle_id] = new_track
                self._total_static_discovered += 1

    def _process_dynamic_detections(
        self, detections: List[Detection], current_time: float
    ):
        """Process dynamic obstacle detections."""
        matched_track_ids = set()

        for detection in detections:
            # Try to associate with existing track
            track_id = self._find_matching_dynamic_track(detection, current_time)

            if track_id is not None:
                # Update existing track
                self.tracked_dynamic[track_id].update(detection)
                matched_track_ids.add(track_id)
            else:
                # Create new track
                new_track = create_tracked_dynamic_from_detection(detection)
                self.tracked_dynamic[detection.obstacle_id] = new_track
                self._total_dynamic_discovered += 1

    def _find_matching_static_track(
        self, detection: Detection
    ) -> Optional[int]:
        """
        Find existing static track matching this detection.

        Uses obstacle_id for direct matching, falls back to nearest neighbor.

        Args:
            detection: New detection

        Returns:
            Track ID if match found, None otherwise
        """
        # Direct ID match (same underlying obstacle)
        if detection.obstacle_id in self.tracked_static:
            return detection.obstacle_id

        # Nearest neighbor by position
        min_dist = self.config.association_threshold
        best_match = None

        for track_id, track in self.tracked_static.items():
            dist = np.linalg.norm(
                track.obstacle.get_center() - detection.position
            )
            if dist < min_dist:
                min_dist = dist
                best_match = track_id

        return best_match

    def _find_matching_dynamic_track(
        self, detection: Detection, current_time: float
    ) -> Optional[int]:
        """
        Find existing dynamic track matching this detection.

        Uses predicted position for association.

        Args:
            detection: New detection
            current_time: Current time for prediction

        Returns:
            Track ID if match found, None otherwise
        """
        # Direct ID match
        if detection.obstacle_id in self.tracked_dynamic:
            return detection.obstacle_id

        # Nearest neighbor using predicted position
        min_dist = self.config.association_threshold
        best_match = None

        for track_id, track in self.tracked_dynamic.items():
            predicted_pos = track.predict(current_time)
            dist = np.linalg.norm(predicted_pos - detection.position)
            if dist < min_dist:
                min_dist = dist
                best_match = track_id

        return best_match

    def _decay_and_cleanup(self, current_time: float):
        """Decay confidence and remove stale tracks."""
        # Decay static tracks
        tracks_to_remove = []
        for track_id, track in self.tracked_static.items():
            track.decay_confidence(current_time, self.config.confidence_decay_rate)

            time_since_seen = current_time - track.last_seen
            if (
                track.confidence < self.config.min_confidence_threshold
                or time_since_seen > self.config.forget_static_after
            ):
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracked_static[track_id]

        # Decay dynamic tracks
        tracks_to_remove = []
        for track_id, track in self.tracked_dynamic.items():
            track.decay_confidence(current_time, self.config.confidence_decay_rate)

            time_since_seen = current_time - track.last_seen
            if (
                track.confidence < self.config.min_confidence_threshold
                or time_since_seen > self.config.forget_dynamic_after
            ):
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracked_dynamic[track_id]

    def get_static_obstacles_for_mpc(self) -> List[Obstacle]:
        """
        Get static obstacles for MPC constraints.

        Returns:
            List of discovered static obstacles
        """
        return [
            track.get_obstacle_for_mpc()
            for track in self.tracked_static.values()
        ]

    def get_dynamic_obstacles_for_mpc(self) -> List[DynamicObstacle]:
        """
        Get dynamic obstacles for MPC with estimated trajectories.

        Returns:
            List of DynamicObstacle with constant-velocity trajectories
        """
        return [
            track.to_dynamic_obstacle()
            for track in self.tracked_dynamic.values()
        ]

    def get_all_tracked_positions(
        self, t: float
    ) -> List[Tuple[np.ndarray, float, bool, float]]:
        """
        Get all tracked positions and metadata for visualization.

        Args:
            t: Current time for dynamic obstacle prediction

        Returns:
            List of (position, radius, is_static, confidence) tuples
        """
        positions = []

        for track in self.tracked_static.values():
            center = track.obstacle.get_center()
            radius = track.obstacle.get_bounding_radius()
            positions.append((center, radius, True, track.confidence))

        for track in self.tracked_dynamic.values():
            center = track.predict(t)
            radius = track.radius + track.safety_margin
            positions.append((center, radius, False, track.confidence))

        return positions

    def get_num_tracked(self) -> Tuple[int, int]:
        """
        Get number of currently tracked obstacles.

        Returns:
            Tuple of (num_static, num_dynamic)
        """
        return len(self.tracked_static), len(self.tracked_dynamic)

    def get_statistics(self) -> Dict:
        """
        Get tracking statistics.

        Returns:
            Dictionary with tracking metrics
        """
        return {
            "num_static_tracked": len(self.tracked_static),
            "num_dynamic_tracked": len(self.tracked_dynamic),
            "total_static_discovered": self._total_static_discovered,
            "total_dynamic_discovered": self._total_dynamic_discovered,
        }

    def reset(self):
        """Reset tracker to initial state."""
        self.tracked_static.clear()
        self.tracked_dynamic.clear()
        self._total_static_discovered = 0
        self._total_dynamic_discovered = 0

    def __repr__(self) -> str:
        return (
            f"ObstacleTracker(static={len(self.tracked_static)}, "
            f"dynamic={len(self.tracked_dynamic)})"
        )
