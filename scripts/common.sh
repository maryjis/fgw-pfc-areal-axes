# shared settings of the step scripts: interpreters from configs/default.yaml, threads, python path
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=$(awk '$1=="analysis:"{print $2}' configs/default.yaml)     # analysis environment (deterministic numerics, 8 BLAS threads)
PYT=$(awk '$1=="train:"{print $2}' configs/default.yaml)       # training environment (GPU; also Leiden: scanpy 1.11)
export PYTHONPATH="$ROOT/src" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
mkdir -p logs
DS=$(awk '/^series:/{f=1} f && $1=="name:"{print $2; exit}' configs/default.yaml)
