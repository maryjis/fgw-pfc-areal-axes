import sys, os, numpy as np, pandas as pd
sys.path.insert(0, 'tests'); sys.path.insert(0, 'src')
import synthetic
from ottr import config
out = []
for seed in range(8):
    root = os.path.join(sys.argv[1], f'syn{seed}'); cfgp, truth = synthetic.make(root, seed=seed)
    os.environ['OTTR_CONFIG'] = cfgp; config.cfg.cache_clear()
    from ottr.areal import multi_axes
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        R = multi_axes.run([('random', 'random')], ds=synthetic.DS, tag='_n')
    pb = R[R.analysis == 'pseudobulk']
    out.append(pb.splithalf.round(2).tolist()); print(seed, pb.splithalf.round(2).tolist(), pb.var_share.round(2).tolist(), list(pb.reading.str[:20]), flush=True)
