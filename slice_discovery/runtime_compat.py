from __future__ import annotations

import sys


def ensure_numpy_pickle_compat() -> None:
    import numpy.core
    import numpy.core.multiarray

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
