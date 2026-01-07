"""Visualization and animation tools."""

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

from .plotter_3d import Plotter3D
from .animation import TrajectoryAnimator

if _removed:
    sys.path.append(_conflicting_path)

__all__ = ["Plotter3D", "TrajectoryAnimator"]
