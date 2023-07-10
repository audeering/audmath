from audmath.core.api import db
from audmath.core.api import duration_in_seconds
from audmath.core.api import inverse_db
from audmath.core.api import inverse_normal_distribution
from audmath.core.api import rms
from audmath.core.api import samples
from audmath.core.api import window


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
