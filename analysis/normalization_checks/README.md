# Normalization and split-half checks of the areal-axis model

Exploratory scripts, copied as they were run (paths still point to the scratch directory of the session; `config.yaml` is the
`OTTR_CONFIG` override that sent all outputs there). Not part of the pipeline.

| file | what |
|---|---|
| `noise_check.py` | areal model on the pure-noise latent of `tests/synthetic.py`, 8 seeds: how often noise gives an "areal axis" |
| `depth_check.py`, `depth_real.py` | correlation of the axes with the region-level sequencing depth (SimR and the real series) |
| `pooled_test.py` → `axes.csv`, `truth.csv`, `run.log` | mean of logs vs log of sums (sum counts, divide by summed library size, then log1p) on SimR / SimK: pseudobulks and smoothed PCA r 1.5 |
| `pooled_axis1.py` | the reproducible axis of the pooled SimR fit is one region (A8C); split-half without it |
| `resim.py` → `resim_axes.csv`, `resim.log` | SimR redrawn with seeds 1-3 and with equal depth for every spot |
| `viz_meanlog_pooled.py` → `meanlog_vs_pooled.png` | mean expression over genes: mean of logs vs log of sums, by region and on a SimR section |

Findings: the false SA axis of SimR (and "areal axis A46D ↔ A8b") is produced by the region-level depth through log1p(CP10k)
averaged after the log; it reappears identically for every seed and disappears with log of sums. Independently, the split-half
criterion (|r| of region scores) passes axes made by single small noisy regions (A8C, A14C, A6Vb, OPAl, A24c), also with equal depth.
