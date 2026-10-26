"""Configuration: paths of the external data, the working directory and the results, read from configs/default.yaml
(plus an optional override file given by the environment variable OTTR_CONFIG).

Every module takes its paths from `cfg()` instead of hard-coding them, so the whole pipeline can be moved by editing one file.

Layout of the working directory (`paths.work`, git-ignored):
    inputs/<ds>.h5ad            spots x genes (raw counts of the model genes) + obs (slice, layer, region, column, depth ...)
    inputs/<ds>_columns.csv     one row per cortical column: area, centroid, SA index
    runs/input/<ds>/            per-section raw h5ad files (symlinks) - the training input of INSPIRE
    embeddings/<ds>/<name>__seed0.npy   latent of every model (rows in the order of inputs/<ds>.h5ad)
    anchors/                    FGW anchor specs (pickles)
    cache/                      metric caches (one json per embedding)
"""
import os
from functools import lru_cache
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root


def _merge(a, b):
    for k, v in b.items():
        a[k] = _merge(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


class Config(dict):
    """dict with attribute access and helpers for the derived paths."""

    def __getattr__(self, k):
        v = self[k]
        return Config(v) if isinstance(v, dict) else v

    def path(self, key):
        """Absolute path of paths.<key> (relative entries are relative to the repository root)."""
        p = self['paths'][key]
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    # ---- working directory
    @property
    def work(self): return self.path('work')
    def inputs(self, ds): return os.path.join(self.work, 'inputs', f'{ds}.h5ad')
    def columns(self, ds): return os.path.join(self.work, 'inputs', f'{ds}_columns.csv')
    def run_input(self, ds): return os.path.join(self.work, 'runs', 'input', ds)
    def emb_dir(self, ds): return os.path.join(self.work, 'embeddings', ds)
    def emb(self, ds, name): return os.path.join(self.emb_dir(ds), f'{name}__seed0.npy')
    @property
    def anchors(self): return os.path.join(self.work, 'anchors')
    @property
    def cache(self): return os.path.join(self.work, 'cache')

    # ---- results
    @property
    def results(self): return self.path('results')
    def models(self, name=''): return os.path.join(self.results, 'models', name)
    def pictures(self, name=''): return os.path.join(self.results, 'pictures', name)
    def tables(self, name=''): return os.path.join(self.results, 'tables', name)
    def model3d(self, name=''): return os.path.join(self.results, '3d_model', name)

    # ---- previous project (reference for the reproduction tests)
    def ref(self, *parts): return os.path.join(self['paths']['reference'], *parts)


@lru_cache(maxsize=None)
def cfg():
    """The merged configuration (cached)."""
    c = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'default.yaml')))
    extra = os.environ.get('OTTR_CONFIG')
    if extra:
        c = _merge(c, yaml.safe_load(open(extra if os.path.isabs(extra) else os.path.join(ROOT, extra))))
    return Config(c)


def ensure(*dirs):
    """Create directories (and return the first one) - small helper used by the writers."""
    for d in dirs: os.makedirs(d, exist_ok=True)
    return dirs[0] if dirs else None
