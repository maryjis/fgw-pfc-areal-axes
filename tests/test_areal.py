"""The areal-axis model recovers planted region components and finds no reproducible axis in pure noise (synthetic data)."""
import numpy as np
import pandas as pd


def _pb(R, model):
    return R[(R.model == model) & (R.analysis == 'pseudobulk')].set_index('comp')


def test_recovers_planted_axes(syn):
    from ottr.areal import multi_axes
    R = multi_axes.run([('truth', 'truth'), ('random', 'random')], ds=syn['ds'], tag='_t')
    Z = np.load(f"{multi_axes.axes_dir()}/{syn['ds']}_t_truth.npz", allow_pickle=True)
    mu = pd.DataFrame(Z['pb_mu'], index=Z['pb_regions']).loc[syn['mu'].index]
    for k, col in enumerate(['mu1', 'mu2', 'mu3']):
        r = abs(np.corrcoef(mu[k], syn['mu'][col])[0, 1])
        assert r >= 0.95, (col, r)          # axis k+1 = planted component k+1 (amplitudes 1.0 > 0.7 > 0.5)
    pb = _pb(R, 'truth')
    assert pb.loc[[1, 2, 3], 'splithalf'].min() >= 0.9
    assert pb.loc[[1, 2, 3], 'var_share'].sum() >= 0.6
    assert pb.loc[1, 'reading'] == 'SA axis'   # mu_1 drives the column SA index


def test_no_axis_in_noise(syn):
    from ottr.areal import multi_axes
    R = pd.read_csv(f"{syn['root']}/results/tables/axes_{syn['ds']}_t_comp.csv")
    pb = _pb(R, 'random')
    assert pb.splithalf.max() < 0.8, pb.splithalf.round(2).tolist()
    assert pb.var_share.max() < 0.35        # no dominant direction among the region means of noise
