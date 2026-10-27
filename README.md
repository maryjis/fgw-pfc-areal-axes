# optimal_transport__tr_axeses

Optimal-transport (fused Gromov–Wasserstein) alignment of serial spatial-transcriptomics sections inside INSPIRE, and the
**areal axes** of the marmoset prefrontal cortex (PFC) that the aligned latent reveals — in adults and across development
(P0, P32, 3 months, 7 months).

The repository contains the methods as separate modules, the step scripts that run them in order, tests that check that the
refactored code reproduces the results of the original project, and the results (figures, tables, 3D pages, presentation).

---

## 1. The idea in short

**Problem.** A serial series of coronal sections (27 sections, T402–T428, Stereo-seq bin100, ~13 000 bins per section) has
strong section effects (depth of sequencing, cutting angle, tissue quality). A latent space that mixes the sections well is easy
to get (adversarial alignment, Harmony) but usually at the price of the biology: neighbouring cortical areas and layers get mixed
too. We want a latent where **the same piece of cortex in neighbouring sections lands in the same place**, without using the
region labels.

**Anchors from the geometry of the cortex.** The cortex of every section is cut into **columns** (≈200 µm wide strips across
the layers, built from the manual layers by a Laplace depth field and its streamlines) and, for the 2D variant, every column into
8 **depth bins**. Between two neighbouring sections we compute a **fused Gromov–Wasserstein coupling** `T` between these nodes:

```
min_T  (1 − α) Σ_ij M_ij T_ij  +  α Σ_ijkl (C^A_ik − C^B_jl)² T_ij T_kl      T 1 = p,  Tᵀ 1 = q  (uniform)
```

* `M` — distance of the node profiles in a training-free space (smoothed PCA, see below), so similar expression is matched;
* `C` — distance **along the cortex** (1D) or in the **column × depth plane** in mm (2D), so the order of the columns (and
  of the depth) is preserved — a coupling cannot jump from orbital to dorsal cortex just because two columns look alike;
* `α = 0.7` weights structure against features; sections that form a ring use a circular distance.

**FGW inside training.** INSPIRE (graph auto-encoder with a Poisson count model through spatial factors) is trained with an
extra loss: the latent mean of every node of section A is pulled towards the nodes of B that `T` assigns to it (cross-entropy
between `T`'s row and the softmax of cosine similarities, both directions, weight λ = 770). In **joint** mode `T` is recomputed
from the current latent every 500 steps (after a 2 000-step warm-up), so the coupling and the latent refine each other. The FGW
models use no adversarial loss.

**Smoothed PCA** — PCA of log-expression averaged over a disk of radius 1.5 grid steps (≈9 bins, 75 µm), centred per section —
is the training-free reference and the space in which the FGW anchors are computed. The radius is the smallest one that already
gives the full layer signal (§4.2).

**Areal-axis model.** In the aligned latent we look for directions that order the **regions** consistently across layers and
sections:

```
x_trl = f_l + θ c_t + Σ_m μ_m(r) g_ml + ε
```

`x_trl` is the mean latent of section t, region r, layer l; `f_l` (layer profile) and `θ c_t` (sequencing depth of the section)
are removed; an SVD of the region × (layer, dimension) matrix of the residual gives the region scores `μ_m(r)` of areal axis m
(shared by all layers and sections) and its loading `g_ml` in every layer. A split-half test (odd vs even sections) tells which
axes are reproducible. The axes are compared with the frontal PCs of an independent segment atlas (Huang et al. 2026) and carried
to the developing brain by applying the **adult encoder** to the young sections.

---

## 2. Repository layout

```
configs/default.yaml        all paths (raw data, annotation, atlas, meshes, working dir), sections, interpreters, GPU settings
configs/reference.yaml      reference workspace for the reproduction tests
envs/train.yml, analysis.yml  the two Python environments (versions used for the results)
src/ottr/                   the package (python -m ottr.<module> ...)
  config.py                 reads the configuration; every path comes from here
  registry.py               the models of the benchmark, their labels and groupings
  utils/                    helpers: columns.py (cortical columns, depth), smoothing.py (disk mean on the grid, FFT),
                            labels.py (region -> area, marker programmes), gpu_queue.py (GPU job queue)
  data/                     inputs: series.py (adult series), dev.py (P0-7M), sa_index.py (frozen adult SA classifier)
  smoothed_pca/             curve.py (smoothed PCA for any radius), select.py (choice of the radius)
  fgw/                      common.py (dataset, columns, profiles, section pairs), spec1d.py (column FGW), spec2d.py (column x depth FGW)
  inspire/                  networks.py, model.py (INSPIRE LGCN + FGW + extensions + classic mode), inputs.py (features), train.py (CLI)
  metrics/                  evaluate.py (global metrics), nmi_within.py (Leiden NMI split), series.py (series metrics + rank),
                            recon_r2.py (reconstruction R2)
  areal/                    multi_axes.py (spot / column PCA, areal-axis model, L2/L3), spot_axes.py (spot-level model),
                            development.py (adult axes on P0-7M)
  atlas/                    frontal_pca.py (frontal PCA of the segment atlas), flatmap.py (our spots and axes on the atlas
                            segments), match.py (Spearman of our axes with the atlas PCs)
  viz/                      style.py, umap_series.py, axes_maps.py (axes on tissue, regions on axes), dev.py, summary.py,
                            brain3d.py + templates/ (interactive 3D pages)
scripts/                    00_check ... 12_presentation (the pipeline in order), run_all.sh, common.sh,
                            R/ (export of the atlas from its RDS files), tools/ (reference workspace)
tests/                      pytest: synthetic tests, INSPIRE equivalence tests, reference (reproduction) tests
results/                    models/ (checkpoints, latents; not in git), pictures/, tables/, 3d_model/, presentation/
```

What each module does and why:

| module | what | why |
|---|---|---|
| `data.series`, `data.dev` | QC exactly as INSPIRE (no MT genes, ≥200 genes per bin, ≥3 bins per gene); the 2 000 SVG genes; cortical columns from the manual layers; SA index of every column from the frozen adult classifier; young brains: annotation joined on coordinates, adult gene order | one input table per dataset that every later step reads |
| `utils.columns` | Laplace depth (0 at the pia, 1 at the white matter), mid-depth centre line, streamlines → columns of ≈405 units | nodes of FGW 1D/2D and the units of the column analyses |
| `smoothed_pca` | disk mean of log1p CP10k (radius r), PCA-32, centred / Harmony variants; radius rule | anchor space for FGW and the no-training baseline |
| `fgw.spec1d / spec2d` | FGW couplings between the nodes of neighbouring sections | the anchors of the alignment loss |
| `inspire` | INSPIRE + FGW loss; presets for every model; encode-only for new data; reconstructions | the trained latents |
| `metrics` | mixing, region / area / layer transfer to held-out sections, SA, Leiden NMI within sections and groups, η², rank | which model aligns sections without losing areas and layers |
| `areal.multi_axes` | main axes by four methods; the areal-axis model with split-half, readings of every axis | what the latent encodes beyond the layers |
| `areal.spot_axes` | the same idea on single bins (no pseudobulks) | robustness of the axes to the pseudobulk step |
| `areal.development` | adult axes on the young columns, conservation, variance share, sample trends with age, top genes | which adult axes already exist at birth |
| `atlas` | frontal PCA of the atlas; our spots placed in the atlas segments of T408–T428; Spearman per atlas PC | independent validation of the axes |
| `viz` | figures and 3D pages | results/pictures, results/3d_model |

---

## 3. Installation

```bash
conda env create -f envs/train.yml        # GPU; torch 2.5.1 (CUDA 12.1)
conda env create -f envs/analysis.yml
# set the interpreters (python.train, python.analysis) and the data paths in configs/default.yaml, then
python scripts/00_check.py                # lists missing inputs
```

External inputs (read only): adult sections with manual layers and atlas regions (`paths.raw_adult`), the young PFC sections
(`paths.raw_dev`) with their annotation tables (`dev_annotation`, `cdepth_sa`), the frozen adult SA classifier (`sa_model`), the
atlas export (`atlas_segments`, `atlas_mask`; produced from the atlas RDS files by `scripts/R/*.R`) and the 3D meshes (`meshes`).
The working directory `work/` and `results/models/` are generated.

---

## 4. Running

```bash
bash scripts/run_all.sh                   # all steps; --from=N to resume, --dry-run to list the steps
```

| step | script | what | time |
|---|---|---|---|
| 00 | `00_check.py` | inputs and environments present | 1 min |
| 01 | `01_prepare_data.sh` | `work/inputs/<ds>.h5ad`, `<ds>_columns.csv` for the series and P0–7M | 15 min |
| 02 | `02_smoothed_pca.sh [--select]` | smoothed PCA r 1.5 (+ the radius curve and its choice) | 3 min (+60) |
| 03 | `03_fgw_anchors.sh` | FGW 1D and 2D specs, 45 section pairs (±1, ±2) | 5 min |
| 04 | `04_train.py` | 5 models × 20 000 steps (GPU queue, one model per GPU, ~20 GB each) | 8–20 h per model |
| 05 | `05_collect_encode.py` | latents → `work/embeddings`; P0–7M encoded by the adult FGW 2D models; reconstructions | 15 min (GPU) |
| 06 | `06_metrics.sh` | metrics, Leiden NMI, series metrics, rank, reconstruction R² | 90 min |
| 07 | `07_axes.sh` | axes by four methods, areal-axis model, spot-level model | 60 min |
| 08 | `08_atlas.sh` | atlas frontal PCA, flat-map projection, correspondence | 20 min |
| 09 | `09_development.sh` | adult axes in P0–7M | 30 min |
| 10 | `10_figures.py` | figures and 3D pages | 30 min |
| 11 | `11_compare_with_previous.py` | retrained models vs the stored results of the previous project | 1 min |
| 12 | `12_presentation.py [--embed]` | `results/presentation/index.html` | 1 min |

Models (`04_train.py`, presets of `ottr.inspire.train`):

| key | model |
|---|---|
| `fgw1d_factors` | FGW 1D (columns), joint, α 0.7, λ 770, 30 spatial factors, patch-variance 10 |
| `fgw2d_factors` | FGW 2D (column × depth), otherwise as above |
| `fgw2d_poisson` | FGW 2D, counts decoded directly from the latent (no factors), Poisson |
| `fgw2d_nb` | FGW 2D, no factors, negative-binomial counts with a per-gene dispersion |
| `inspire_gan` | INSPIRE baseline: all-pairs discriminators (GAN 1), section one-hot in the encoder |
| `classic` | original INSPIRE (optional, `--with_classic`) |

### 4.1 Normalisation (as in the original INSPIRE)
The encoder input is `[log1p(CP10k), graph-smoothed log1p(CP10k)]` of the model genes (CP10k over all genes of the bin); the
feature decoder reconstructs these values. The raw counts of the model genes and their sum (library size) enter only the count
likelihood: `log λ = log(β · softmax(factors)) + γ_section` (or the direct count decoder), rate = library × λ.

### 4.2 Choice of the smoothing radius
`ottr.smoothed_pca.select`: the smallest radius on the plateau of the layer transfer to a held-out section (within 2% of the
best value on the curve 0–7 grid steps). Larger disks blur the layers and remove the section mixing; the anchors need the
layers. The rule gives r = 1.5 on the adult5 and P0 benchmarks and on the series. On the series the layer-transfer maximum is
at r = 2.5 (0.799 vs 0.789 at 1.5) — a flat plateau — while the cross-section neighbour share drops from 0.41 to 0.11, and
r = 1.5 also has the best combined score (`results/tables/smoothing_radius_pfcserL.csv`).

---

## 5. Tests

```bash
pytest -q                                   # all; GPU tests are skipped without a GPU
bash scripts/tools/run_reference.sh         # reference workspace for the reproduction tests (~90 min), then
pytest -q tests/test_reference.py
```

* `test_inspire.py` — **classic mode = the original INSPIRE package**: identical weights after 20 training steps (CPU);
  every mode of the project (FGW joint with factors, no factors Poisson / NB, INSPIRE + GAN) = the previous benchmark copy:
  identical weights and loss logs.
* `test_areal.py` — synthetic sections with three planted region components: the areal-axis model recovers them (|r| ≥ 0.95,
  split-half ≥ 0.9); pure noise gives no reproducible axis.
* `test_metrics.py` — mixing, layer and area transfer, SA, η² and the rank on cases with a known answer.
* `test_columns_fgw.py` — Laplace columns on a curved ribbon; the FGW 2D coupling of identical geometries is diagonal.
* `test_reference.py` — **level 1 of the reproduction**: the new code reproduces the stored inputs, smoothed PCA, FGW specs,
  metrics, axes, atlas and development tables of the previous project exactly (allclose 1e-7), run on the stored embeddings.
* `scripts/11_compare_with_previous.py` — **level 2**: the retrained models against the previous ones (metrics |Δ| ≤ 0.02,
  reproducible axes |r| ≥ 0.95).

Exactness needs the same numerical stack: the analysis steps run in the analysis environment with 8 BLAS threads, the Leiden
steps (evaluate, nmi_within) in the training environment (scanpy 1.11), as the original results were produced.

---

## 6. What is in git

Code, configuration, tests, `results/pictures`, `results/tables`, `results/presentation`. Not in git (rebuilt by the
pipeline): `work/` (inputs, embeddings, anchors, caches), `results/models/` (checkpoints and latents, ~1 GB per model),
`results/3d_model/*.html` (10–12 MB each; `python scripts/10_figures.py 3d`).

---

## 7. Results (this run, series T402–T428, 20 000 steps)

| place | model | mixing | region NMI in section | area transfer | layer transfer | SA | section NMI | score |
|---|---|---|---|---|---|---|---|---|
| 1 | FGW 2D · factors | 0.786 | 0.174 | 0.785 | 0.803 | 0.936 | 0.037 | 0.855 |
| 2 | FGW 1D · factors | 0.666 | 0.189 | 0.708 | 0.797 | 0.936 | 0.107 | 0.786 |
| 3 | FGW 2D · no factors · Poisson | 0.908 | 0.135 | 0.817 | 0.751 | 0.904 | 0.026 | 0.603 |
| 4 | FGW 2D · no factors · NB | 0.928 | 0.140 | 0.832 | 0.735 | 0.916 | 0.026 | 0.602 |
| 5 | INSPIRE (GAN) | 0.549 | 0.167 | 0.557 | 0.790 | 0.918 | 0.210 | 0.514 |
| 6 | smoothed PCA | 0.425 | 0.184 | 0.593 | 0.789 | 0.677 | 0.266 | 0.367 |

Areal axes of FGW 2D with factors: 1 SA (64%, split-half 0.99), 2 arch of axis 1 (17%), 3 OPAl ↔ A24b (7%), 4 rostro-caudal,
5 A14R ↔ A13L; all five reproducible (split-half ≥ 0.81). Figures: `results/pictures`, deck: `results/presentation/index.html`,
3D pages: `results/3d_model` (rebuild: `python scripts/10_figures.py 3d`).

## 8. Reproduction status

* Level 1 (tests/test_reference.py, 27 tests): the new code reproduces the stored inputs, smoothed PCA, FGW specs, metrics,
  axes, atlas and development tables exactly. One deliberate change: the atlas flat-map assignment of the original script broke
  ties by Python's random set order (≈9% of spots, and the choice among four equally good grid offsets, were not
  reproducible); ties now go to the smallest segment id, and the downstream projection is verified on the stored assignment.
  k-means NMI within L3/L5 (not a ranking metric) varies in the 4th digit with OpenMP threads.
* Training: identical to the previous training script (GPU test) and, in classic mode, to the original INSPIRE package.
  The retrained NB and INSPIRE (GAN) models — trained from scratch in both projects — give the same metrics to 15 digits.
* Level 2 (`scripts/11_compare_with_previous.py`, `results/tables/reproduction_summary.csv`): all reproducible areal axes,
  column PCs 1–3 and atlas correspondences pass (|r| ≥ 0.95); metrics within ±0.02 for 5 of 6 models. FGW 1D (old model:
  7 500 steps + continuation, new: one 20 000-step run) is just outside on two metrics: area transfer 0.683 → 0.708 (+0.025),
  section NMI 0.086 → 0.107 (+0.022).
* Tests: 40 passed (`bash scripts/tools/run_tests.sh`, report `results/tables/test_report.csv`).
