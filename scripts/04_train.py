"""Step 04: train the models of the benchmark on the serial series (GPU, about 8-10 h per model; one model per GPU).

  fgw1d_factors   FGW 1D column anchors, 30 spatial factors            (spec anchors/fgw_spec_<ds>_a07c_k2.pkl)
  fgw2d_factors   FGW 2D (column x depth) anchors, 30 spatial factors  (spec anchors/fgw2d_spec_<ds>_a07c_k2.pkl)
  fgw2d_poisson   FGW 2D, no factors, Poisson counts
  fgw2d_nb        FGW 2D, no factors, negative binomial counts
  inspire_gan     INSPIRE baseline with adversarial alignment (no FGW)
  classic         original INSPIRE (optional: --with_classic)
Writes the job list work/runs/jobs_train.tsv and runs the GPU queue (results/models/<model>/, logs/train_<model>.log).
usage: python scripts/04_train.py [--models a,b] [--steps 20000] [--with_classic] [--dry_run]"""
import os, sys, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from ottr.config import cfg, ROOT, ensure

MODELS = {'fgw1d_factors': 'fgw_spec', 'fgw2d_factors': 'fgw2d_spec', 'fgw2d_poisson': 'fgw2d_spec', 'fgw2d_nb': 'fgw2d_spec',
          'inspire_gan': None, 'classic': None}

if __name__ == '__main__':
    P = argparse.ArgumentParser(); P.add_argument('--models', default='fgw1d_factors,fgw2d_factors,fgw2d_poisson,fgw2d_nb,inspire_gan')
    P.add_argument('--steps', type=int, default=20000); P.add_argument('--with_classic', action='store_true'); P.add_argument('--dry_run', action='store_true')
    a = P.parse_args(); c = cfg(); ds = c.series['name']; py = c['python']['train']
    names = a.models.split(',') + (['classic'] if a.with_classic else [])
    jobs = []
    for m in names:
        spec = f"--fgw_spec {c.anchors}/{MODELS[m]}_{ds}_a07c_k2.pkl" if MODELS[m] else ''
        if MODELS[m] and not os.path.exists(f"{c.anchors}/{MODELS[m]}_{ds}_a07c_k2.pkl"): sys.exit(f'missing anchor spec for {m}: run step 03')
        cmd = (f"{py} -m ottr.inspire.train --preset {m} --slice_dir {c.run_input(ds)} --svg_genes_file {c.path('svg_genes')} "
               f"--n_training_steps {a.steps} --seed 0 --out {c.models(m)} {spec}")
        jobs.append(f'train_{m}\t{cmd}\t{c["gpu"]["min_free_mb"]}')
    jf = os.path.join(ensure(os.path.join(c.work, 'runs')), 'jobs_train.tsv'); open(jf, 'w').write('\n'.join(jobs) + '\n')
    print(open(jf).read())
    if a.dry_run: sys.exit(0)
    subprocess.run([py, '-m', 'ottr.utils.gpu_queue', jf, '--gpus', ','.join(map(str, c['gpu']['ids'])), '--per_gpu', str(c['gpu']['per_gpu']),
                    '--min_free_mb', str(c['gpu']['min_free_mb']), '--state', os.path.join(c.work, 'runs', 'state'), '--logs', os.path.join(ROOT, 'logs'),
                    '--stop_when_empty', '--cwd', ROOT], check=True)
