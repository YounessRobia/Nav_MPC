"""Configuration loading and dataclasses for UAV and MPC parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class PositionBounds:
    """Position bounds for geofencing."""

    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = 5.0   # Minimum altitude [m]
    z_max: float = 25.0  # Maximum altitude [m]


@dataclass
class UAVConfig:
    """UAV physical parameters and constraints."""

    mass: float = 1.5
    max_velocity: float = 10.0
    max_acceleration: float = 5.0
    max_jerk: float = 30.0
    max_yaw_rate: float = 1.5
    position_bounds: PositionBounds = field(default_factory=PositionBounds)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UAVConfig":
        """Create UAVConfig from dictionary."""
        bounds_data = data.pop("position_bounds", {})
        bounds = PositionBounds(**bounds_data) if bounds_data else PositionBounds()
        return cls(position_bounds=bounds, **data)


@dataclass
class SolverConfig:
    """IPOPT solver configuration."""

    max_iter: int = 100
    tolerance: float = 1e-6
    print_level: int = 0
    warm_start: bool = True


@dataclass
class MPCConfig:
    """MPC controller parameters."""

    horizon: int = 30
    dt: float = 0.05

    # State cost weights
    Q_position: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 10.0, 20.0])
    )
    Q_velocity: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0]))
    Q_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0])
    )
    Q_yaw: float = 1.0

    # Control cost weights
    R_jerk: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))
    R_yaw_rate: float = 0.5

    # Control smoothness weights
    S_jerk: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    S_yaw_rate: float = 0.1

    # Terminal cost multiplier
    Qf_multiplier: float = 10.0

    # Obstacle avoidance
    obstacle_weight: float = 1000.0
    safety_margin: float = 0.5

    # Solver settings
    solver: SolverConfig = field(default_factory=SolverConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MPCConfig":
        """Create MPCConfig from dictionary."""
        # Convert lists to numpy arrays
        array_fields = [
            "Q_position",
            "Q_velocity",
            "Q_acceleration",
            "R_jerk",
            "S_jerk",
        ]
        for key in array_fields:
            if key in data and isinstance(data[key], list):
                data[key] = np.array(data[key])

        # Handle solver config
        solver_data = data.pop("solver", {})
        solver = SolverConfig(**solver_data) if solver_data else SolverConfig()

        return cls(solver=solver, **data)

    def get_Q_matrix(self) -> np.ndarray:
        """Construct full Q matrix (10x10 diagonal)."""
        return np.diag(
            np.concatenate(
                [self.Q_position, self.Q_velocity, self.Q_acceleration, [self.Q_yaw]]
            )
        )

    def get_R_matrix(self) -> np.ndarray:
        """Construct full R matrix (4x4 diagonal)."""
        return np.diag(np.concatenate([self.R_jerk, [self.R_yaw_rate]]))

    def get_Qf_matrix(self) -> np.ndarray:
        """Construct terminal cost matrix."""
        return self.Qf_multiplier * self.get_Q_matrix()


@dataclass
class ObstacleConfig:
    """Configuration for a single obstacle."""

    type: str  # 'sphere', 'cylinder', 'box'
    center: Optional[np.ndarray] = None
    radius: Optional[float] = None
    half_extents: Optional[np.ndarray] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    safety_margin: float = 0.5
    # For dynamic obstacles
    trajectory: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObstacleConfig":
        """Create ObstacleConfig from dictionary."""
        data = data.copy()
        if "center" in data:
            data["center"] = np.array(data["center"])
        if "half_extents" in data:
            data["half_extents"] = np.array(data["half_extents"])
        if "initial_position" in data.get("trajectory", {}):
            data["trajectory"]["initial_position"] = np.array(
                data["trajectory"]["initial_position"]
            )
        if "velocity" in data.get("trajectory", {}):
            data["trajectory"]["velocity"] = np.array(data["trajectory"]["velocity"])
        return cls(**data)


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario."""

    name: str = "Unnamed Scenario"
    description: str = ""

    # Simulation parameters
    duration: float = 20.0
    dt: float = 0.05

    # Initial state
    initial_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 5.0])
    )
    initial_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    initial_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    initial_yaw: float = 0.0

    # Goal
    goal_position: np.ndarray = field(
        default_factory=lambda: np.array([20.0, 0.0, 5.0])
    )
    goal_tolerance: float = 0.5

    # Waypoints
    waypoints: List[np.ndarray] = field(default_factory=list)

    # Obstacles
    static_obstacles: List[ObstacleConfig] = field(default_factory=list)
    dynamic_obstacles: List[ObstacleConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioConfig":
        """Create ScenarioConfig from dictionary."""
        config = cls()

        config.name = data.get("name", config.name)
        config.description = data.get("description", config.description)

        # Simulation params
        sim = data.get("simulation", {})
        config.duration = sim.get("duration", config.duration)
        config.dt = sim.get("dt", config.dt)

        # Initial state
        init = data.get("initial_state", {})
        if "position" in init:
            config.initial_position = np.array(init["position"])
        if "velocity" in init:
            config.initial_velocity = np.array(init["velocity"])
        if "acceleration" in init:
            config.initial_acceleration = np.array(init["acceleration"])
        config.initial_yaw = init.get("yaw", config.initial_yaw)

        # Goal
        goal = data.get("goal", {})
        if "position" in goal:
            config.goal_position = np.array(goal["position"])
        config.goal_tolerance = goal.get("tolerance", config.goal_tolerance)

        # Waypoints
        waypoints_data = data.get("waypoints", [])
        config.waypoints = [
            np.array(wp.get("position", wp) if isinstance(wp, dict) else wp)
            for wp in waypoints_data
        ]

        # Static obstacles
        static_obs = data.get("static_obstacles", [])
        config.static_obstacles = [ObstacleConfig.from_dict(o) for o in static_obs]

        # Dynamic obstacles
        dynamic_obs = data.get("dynamic_obstacles", [])
        config.dynamic_obstacles = [ObstacleConfig.from_dict(o) for o in dynamic_obs]

        return config


def load_yaml(filepath: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def load_config(
    uav_config_path: str | Path | None = None,
    mpc_config_path: str | Path | None = None,
    scenario_path: str | Path | None = None,
) -> Tuple[UAVConfig, MPCConfig, Optional[ScenarioConfig]]:
    """
    Load configuration files.

    Args:
        uav_config_path: Path to UAV config YAML
        mpc_config_path: Path to MPC config YAML
        scenario_path: Path to scenario config YAML (optional)

    Returns:
        Tuple of (UAVConfig, MPCConfig, ScenarioConfig or None)
    """
    # Load UAV config
    if uav_config_path:
        uav_data = load_yaml(uav_config_path)
        uav_config = UAVConfig.from_dict(uav_data)
    else:
        uav_config = UAVConfig()

    # Load MPC config
    if mpc_config_path:
        mpc_data = load_yaml(mpc_config_path)
        mpc_config = MPCConfig.from_dict(mpc_data)
    else:
        mpc_config = MPCConfig()

    # Load scenario config
    scenario_config = None
    if scenario_path:
        scenario_data = load_yaml(scenario_path)
        scenario_config = ScenarioConfig.from_dict(scenario_data)

    return uav_config, mpc_config, scenario_config


def get_default_config_paths() -> Tuple[Path, Path]:
    """Get default paths for config files relative to package root."""
    package_root = Path(__file__).parent.parent.parent
    config_dir = package_root / "config"
    return config_dir / "uav_config.yaml", config_dir / "mpc_config.yaml"
