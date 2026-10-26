"""Shared fixtures: a synthetic working directory (tests/synthetic.py) with the configuration pointing to it."""
import os, sys
import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(scope='session')
def syn(tmp_path_factory):
    """Synthetic workspace; sets OTTR_CONFIG for the whole session and returns the truth."""
    import synthetic
    from ottr import config
    root = str(tmp_path_factory.mktemp('syn'))
    cfgp, truth = synthetic.make(root)
    old = os.environ.get('OTTR_CONFIG'); os.environ['OTTR_CONFIG'] = cfgp; config.cfg.cache_clear()
    yield dict(root=root, ds=synthetic.DS, **truth)
    if old is None: os.environ.pop('OTTR_CONFIG', None)
    else: os.environ['OTTR_CONFIG'] = old
    config.cfg.cache_clear()


def have_reference():
    from ottr.config import cfg
    return os.path.isdir(cfg().ref('results'))
