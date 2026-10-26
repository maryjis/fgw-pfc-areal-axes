"""Step 00: check that the external inputs and the two Python environments exist (prints what is missing)."""
import os, sys, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from ottr.config import cfg
c = cfg(); ok = True
for k in ('raw_adult', 'raw_dev', 'dev_annotation', 'cdepth_sa', 'sa_model', 'svg_genes', 'atlas_segments', 'atlas_mask', 'meshes'):
    p = c.path(k); e = os.path.exists(p); ok &= e; print(f"{'ok ' if e else 'MISSING'}  {k:16s} {p}")
for s in c.series['sections']:
    p = f"{c.path('raw_adult')}/{s}_b100_anndata.h5ad"
    if not os.path.exists(p): ok = False; print('MISSING section', p)
for name, py, mods in (('train', c['python']['train'], 'torch,ot,scanpy,anndata,sklearn,yaml'), ('analysis', c['python']['analysis'], 'ot,anndata,sklearn,umap,harmonypy,matplotlib,yaml')):
    r = subprocess.run([py, '-c', f'import {mods}; import torch; print(torch.cuda.is_available())' if name == 'train' else f'import {mods}'], capture_output=True, text=True)
    ok &= r.returncode == 0; print(f"{'ok ' if r.returncode == 0 else 'BROKEN'}  python.{name} {py} {r.stdout.strip()} {r.stderr.strip()[-200:]}")
print('all inputs present' if ok else 'fix the paths in configs/default.yaml'); sys.exit(0 if ok else 1)
