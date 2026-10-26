"""Tiny GPU job queue.  python -m ottr.utils.gpu_queue jobs.tsv [--gpus 0,1] [--per_gpu 1] [--min_free_mb 18000] [--stop_when_empty]

jobs.tsv: <name>\t<shell command>[\t<required free MB>]  - the command sees CUDA_VISIBLE_DEVICES and $GPU.
A job is skipped if <state>/<name>.done exists; failures leave <state>/<name>.failed; logs go to <logs>/<name>.log.
A job starts on the least loaded GPU that has fewer than per_gpu jobs of this queue and enough free memory (the GPUs may be
shared with other users). The file is re-read every cycle, so jobs can be appended while the queue runs."""
import os, time, argparse, subprocess


def free_mb(g):
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', str(g)], capture_output=True, text=True).stdout.strip()
    return int(out) if out else 0


def read_jobs(path, default_mb):
    jobs = []
    for line in open(path):
        if line.strip() and not line.startswith('#'):
            f = line.rstrip('\n').split('\t'); jobs.append((f[0], f[1], int(f[2]) if len(f) > 2 and f[2] else default_mb))
    return jobs


def main():
    P = argparse.ArgumentParser(); P.add_argument('jobs'); P.add_argument('--gpus', default='0,1'); P.add_argument('--per_gpu', type=int, default=1)
    P.add_argument('--min_free_mb', type=int, default=18000); P.add_argument('--state', default='work/runs/state'); P.add_argument('--logs', default='logs')
    P.add_argument('--stop_when_empty', action='store_true'); P.add_argument('--cwd', default='.')
    a = P.parse_args(); gpus = [int(g) for g in a.gpus.split(',')]; os.makedirs(a.state, exist_ok=True); os.makedirs(a.logs, exist_ok=True)
    running = {}
    while True:
        for n, (p, g) in list(running.items()):
            r = p.poll()
            if r is not None:
                open(f'{a.state}/{n}.{"done" if r == 0 else "failed"}', 'w').write(f'exit {r}\n'); del running[n]
                print(time.strftime('%H:%M'), 'finished', n, 'exit', r, flush=True)
        pending = [j for j in read_jobs(a.jobs, a.min_free_mb) if j[0] not in running
                   and not os.path.exists(f'{a.state}/{j[0]}.done') and not os.path.exists(f'{a.state}/{j[0]}.failed')]
        if not pending and not running and a.stop_when_empty:
            print('queue empty'); break
        for n, cmd, need in pending:
            load = {g: sum(1 for _, gg in running.values() if gg == g) for g in gpus}
            cand = [g for g in gpus if load[g] < a.per_gpu and free_mb(g) >= need]
            if not cand: continue
            g = min(cand, key=lambda x: load[x])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), GPU=str(g), OPENBLAS_NUM_THREADS='16', OMP_NUM_THREADS='16')
            running[n] = (subprocess.Popen(cmd, shell=True, stdout=open(f'{a.logs}/{n}.log', 'w'), stderr=subprocess.STDOUT, env=env, cwd=a.cwd), g)
            print(time.strftime('%H:%M'), 'started', n, 'on GPU', g, flush=True)
            time.sleep(60)   # let the job allocate its memory before the next free-memory check
        time.sleep(30)


if __name__ == '__main__':
    main()
