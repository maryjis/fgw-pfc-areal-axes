"""Metrics on cases with a known answer (synthetic data)."""
import numpy as np
import pandas as pd


def test_eta_and_rank():
    from ottr.metrics.series import eta, rank
    g = np.repeat(['a', 'b'], 50)
    assert eta(np.r_[np.zeros(50), np.ones(50)][:, None], g) == 1.0            # all variance between groups
    rng = np.random.default_rng(0); assert eta(rng.normal(size=(10000, 1)), rng.integers(0, 2, 10000)) < 0.01
    df = pd.DataFrame(dict(model=['a', 'b'], mixing_all=[1, 0], region_nmi_in_section=[1, 0], area_transfer=[1, 0], layer_heldout=[1, 0], SA=[1, 0], section_nmi=[0, 1]))
    T = rank(df); assert T.index[0] == 'a' and T.loc['a', 'score'] == 1 and T.loc['b', 'score'] == 0


def test_mixing_layer_area(syn):
    from ottr.metrics.evaluate import Context
    from ottr.config import cfg
    import anndata as ad
    ctx = Context(syn['ds']); o = ad.read_h5ad(cfg().inputs(syn['ds']), backed='r').obs; rng = np.random.default_rng(0)
    sec = pd.get_dummies(o.slice).values.astype(float); lay = pd.get_dummies(o.layer).values.astype(float)
    sep = ctx.evaluate(np.c_[sec * 10, lay] + rng.normal(0, 0.01, (len(o), sec.shape[1] + lay.shape[1])))
    mix = ctx.evaluate(lay + rng.normal(0, 0.3, lay.shape))
    m_sep = sep['align: cross-slice share'] / sep['align: cross-slice share, chance']
    m_mix = mix['align: cross-slice share'] / mix['align: cross-slice share, chance']
    assert m_sep < 0.05 and m_mix > 0.9                                         # separated sections vs perfect mixing
    assert mix['layer: held-out slice (spot kNN)'] > 0.95                       # layer one-hot -> layers transfer
    tru = ctx.evaluate(np.load(cfg().emb(syn['ds'], 'truth')))
    assert tru['region: column area transfer, slice z-scored'] > 0.9            # planted regions -> areas transfer
    assert tru['SA: ridge on held-out slice (Spearman)'] > 0.9 and tru['SA: best |Spearman| PC1-5 vs SA index'] > 0.7
