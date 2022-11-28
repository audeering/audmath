from audmath.core.api import (
    db,
    inverse_db,
    inverse_normal_distribution,
    rms,
)

# Discourage from audmath import *
__all__ = []

# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
