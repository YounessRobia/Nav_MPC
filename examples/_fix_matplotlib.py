"""
Matplotlib path fix for visualization imports.

Import this module BEFORE any matplotlib or visualization imports to fix
the version conflict between system and pip-installed matplotlib.
"""

import sys

# Fix matplotlib version conflict - remove system path before imports
_conflicting_path = '/usr/lib/python3/dist-packages'
if _conflicting_path in sys.path:
    sys.path.remove(_conflicting_path)
    _removed = True
else:
    _removed = False

# Clear any cached mpl_toolkits from wrong location
for _mod in [k for k in list(sys.modules.keys()) if k.startswith('mpl_toolkits')]:
    del sys.modules[_mod]

# Now safe to import matplotlib
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend

# Restore system path after fix is applied
if _removed:
    sys.path.append(_conflicting_path)
