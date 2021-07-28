from audmath.core.api import (
    inverse_normal_distribution,
)

# Discourage from audeer import *
__all__ = []

# Dynamically get the version of the installed module
try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:  # pragma: no cover
    pkg_resources = None  # pragma: no cover
finally:
    del pkg_resources
