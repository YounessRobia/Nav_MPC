"""Reference trajectory and waypoint management."""

from .waypoints import Waypoint, WaypointManager
from .reference import ReferenceGenerator

__all__ = ["Waypoint", "WaypointManager", "ReferenceGenerator"]
