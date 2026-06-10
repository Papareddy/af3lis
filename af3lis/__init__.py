"""af3lis — AlphaFold 3 mirror of boltzlis for protein-interface screens.

Produces a TSV schema byte-identical to boltzlis in ``flat`` aggregation mode
(``iLIS/LIS/cLIS/iLIA/LIA/cLIA/actifpTM/ipSAE/PEAK/ipTM/pTM/pLDDT_i/pLDDT_j``
+ ``n_models``), and a per-seed-aware schema in the default ``per_seed`` mode.
Drop-in for ECT x CCR4-NOT screens that previously ran via boltzlis.

House metric: ``PEAK`` = ``1 - min-interchain-PAE / 30``.
Confident hit: ``PEAK >= 0.7 AND iLIS >= 0.22``.

Public API (re-exported for boltzlis muscle-memory compatibility):

Sequence + JSON build
    - ``resolve_chain``   (fetch.py)
    - ``build_grid``      (json_build.py)
    - ``build_complex``   (json_build.py)

AF3 output IO
    - ``FoldResult``      (af3_io.py)
    - ``load_sample``     (af3_io.py)

Aggregation
    - ``collect_all``     (collect.py)
"""

from __future__ import annotations

__version__ = "0.1.0"

# Sequence resolution + AF3 input JSON emission.
from af3lis.fetch import resolve_chain
from af3lis.json_build import build_complex, build_grid

# AF3 output discovery + per-sample loading.
from af3lis.af3_io import FoldResult, load_sample

# Aggregation entrypoint (TSV writer).
from af3lis.collect import collect_all

__all__ = [
    "__version__",
    "resolve_chain",
    "build_grid",
    "build_complex",
    "FoldResult",
    "load_sample",
    "collect_all",
]
