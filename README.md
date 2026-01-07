# MPC-UAV-Planner

A Model Predictive Control (MPC) framework for autonomous UAV path planning with real-time obstacle avoidance.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CasADi](https://img.shields.io/badge/CasADi-3.6+-green.svg)](https://web.casadi.org/)

## Overview

This project implements an optimization-based Model Predictive Controller for quadrotor UAV navigation in complex environments. The framework handles both static and dynamic obstacles while ensuring smooth, dynamically feasible trajectories through jerk-controlled triple integrator dynamics.

### Key Features

- **Real-time MPC**: Receding horizon control at 20Hz (50ms timestep)
- **Multi-obstacle avoidance**: Spheres, cylinders, boxes, and moving obstacles
- **Smooth trajectories**: Jerk control ensures actuator safety
- **Constraint handling**: Velocity, acceleration, altitude, and safety margins
- **Dynamic obstacles**: Predicts moving obstacle positions over the planning horizon
- **3D Visualization**: Real-time animated trajectory visualization

## Results

### Swarm Avoidance

Navigation through a swarm of randomly moving dynamic obstacles demonstrating the MPC's ability to handle unpredictable multi-agent environments.

<p align="center">
  <img src="results/swarm_avoidance_seed3041.gif" alt="Swarm Avoidance" width="700"/>
</p>

### Urban Rescue Mission

Complex multi-objective scenario combining static building obstacles with moving patrol drones. The UAV must visit multiple waypoints while avoiding all obstacles.

<p align="center">
  <img src="results/urban_rescue.gif" alt="Urban Rescue Mission" width="700"/>
</p>

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YounessRobia/Nav_MPC.git
cd Nav_MPC

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Usage

### Running Scenarios

```bash
# Basic hover stabilization
python examples/basic_hover.py

# Static obstacle avoidance
python examples/static_avoidance.py

# Dynamic obstacle avoidance
python examples/dynamic_avoidance.py

# Waypoint tracking
python examples/waypoint_tracking.py

# Swarm avoidance
python examples/swarm_avoidance.py

# Urban rescue mission
python examples/urban_rescue.py
```

## Dependencies

- **CasADi**: Symbolic math and automatic differentiation
- **IPOPT**: Interior-point nonlinear optimizer (via CasADi)
- **NumPy/SciPy**: Numerical computing
- **Matplotlib**: Visualization and animation
- **PyYAML**: Configuration management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Youness El Mrhasli** - [YounessRobia](https://github.com/YounessRobia)

