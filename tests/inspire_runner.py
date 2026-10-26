"""Train one implementation of Model_LGCN for a few steps on pickled inputs and save its weights and losses (CPU).
python inspire_runner.py <impl: orig|bench|new> <inputs.pkl> <options.json> <out.npz> [package dir]
orig  = the original INSPIRE package (github.com/jiazhao97/INSPIRE), bench = the previous project's benchmark copy,
new   = ottr.inspire. The packages are both called INSPIRE, so each implementation runs in its own process."""
import sys, json, pickle, os
import numpy as np

impl, inp, optf, out = sys.argv[1:5]
os.environ['CUDA_VISIBLE_DEVICES'] = ''                       # CPU: deterministic
import torch
torch.use_deterministic_algorithms(True)
st_list = pickle.load(open(inp, 'rb')); opt = json.load(open(optf))
steps = opt.pop('n_training_steps'); bs = opt.pop('batch_size'); spec = opt.pop('fgw_spec_file', None)
if spec: opt['fgw_spec'] = pickle.load(open(spec, 'rb'))
if impl == 'new':
    from ottr.inspire.model import Model_LGCN
else:
    sys.path.insert(0, sys.argv[5]); from INSPIRE.model import Model_LGCN
m = Model_LGCN(adata_st_list=st_list, n_spatial_factors=5, n_training_steps=steps, batch_size=bs, seed=0, **opt)
m.step_interval = 10 ** 9
log = out.replace('.npz', '_loss.csv')
if impl == 'orig': m.train(st_list)
else: m.train(st_list, loss_log_path=log, loss_log_interval=1)
sd = {k: v.detach().cpu().numpy() for k, v in m.net.state_dict().items()}
sd.update({f'disc{d}.{k}': v.detach().cpu().numpy() for d, net in m.discriminator.items() for k, v in net.state_dict().items()})
np.savez(out, **sd)
