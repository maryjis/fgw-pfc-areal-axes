"""Build the reference workspace work_ref/ (configs/reference.yaml): inputs = the inputs of the current work/ (identical to the
previous project's, tests/test_reference.py::test_inputs), embeddings = symlinks to the previous project's stored embeddings
(same names). Running the analysis modules with OTTR_CONFIG=configs/reference.yaml then recomputes every deterministic result
of the previous project with the new code (scripts/tools/run_reference.sh)."""
import os, glob, sys, yaml
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from ottr.config import ROOT, cfg

REF = cfg().ref(); W = os.path.join(ROOT, 'work_ref')
WORK = os.path.join(ROOT, yaml.safe_load(open(os.path.join(ROOT, 'configs', 'default.yaml')))['paths']['work'])   # the main work dir
DS = ['pfcserL', 'devP0', 'devP32', 'dev3M', 'dev7M']
EMB = {'pfcserL': ['curve_r1.5_centred', 'encslice_gan1_s20000', 'fgw_joint_a07c_c770_k2_s20000', 'fgw2d_joint_a07c_c770_k2_s20000',
                   'fgw2d_joint_a07c_c770_k2_direct_s20000', 'fgw2d_joint_a07c_c770_k2_directnb_s20000'],
       **{d: ['adultenc_fgw2d20k', 'adultencNF_fgw2d20k', 'adultencNB_fgw2d20k'] for d in DS[1:]}}
os.makedirs(f'{W}/inputs', exist_ok=True)
for ds in DS:
    for f in (f'{ds}.h5ad', f'{ds}_columns.csv'):
        dst = f'{W}/inputs/{f}'
        if os.path.islink(dst) and not os.path.exists(dst): os.remove(dst)   # broken / looping link
        if not os.path.lexists(dst): os.symlink(os.path.join(WORK, 'inputs', f), dst)
    os.makedirs(f'{W}/embeddings/{ds}', exist_ok=True)
    for e in EMB[ds]:
        for ext in ('npy', 'json'):
            src = f'{REF}/embeddings/{ds}/{e}__seed0.{ext}'; dst = f'{W}/embeddings/{ds}/{e}__seed0.{ext}'
            if os.path.exists(src) and not os.path.lexists(dst): os.symlink(src, dst)
print('reference workspace', W)
