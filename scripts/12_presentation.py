"""Step 12: HTML presentation of the results -> results/presentation/index.html (open in a browser; arrow keys / swipe).
Built from results/pictures and results/tables, so it always shows the current results; figures are referenced as
../pictures/<name>.png (the folder results/ is self-contained). --embed writes a single-file copy with the figures inlined
(results/presentation/deck_standalone.html, for sending as one file)."""
import os, sys, base64, html, re
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from ottr.config import cfg, ensure
from ottr.registry import MODELS, FGW2D

c = cfg(); OUT = ensure(os.path.join(c.results, 'presentation')); PIC = c.pictures(); T = c.tables
EMBED = '--embed' in sys.argv
esc = html.escape


def img(name, alt):
    p = os.path.join(PIC, f'{name}.png')
    if not os.path.exists(p): return f'<div class="missing">figure {esc(name)} not built yet</div>'
    src = 'data:image/png;base64,' + base64.b64encode(open(p, 'rb').read()).decode() if EMBED else f'../pictures/{name}.png'
    return f'<img src="{src}" alt="{esc(alt)}" loading="lazy">'


def table(df, fmt=3, cls=''):
    h = ''.join(f'<th>{esc(str(x))}</th>' for x in df.columns)
    rows = ''.join('<tr>' + ''.join(f'<td>{(f"{v:.{fmt}f}" if isinstance(v, float) else esc(str(v)))}</td>' for v in r) + '</tr>' for r in df.itertuples(index=False))
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table>'


def read(name):
    p = T(name); return pd.read_csv(p) if os.path.exists(p) else None


slides = []
S = lambda title, body, kicker='': slides.append((kicker, title, body))

S('FGW column-anchor alignment of serial sections and the areal axes of the marmoset PFC',
  '<p class="lead">Serial adult series T402–T428 (27 sections, 343 707 bins) · Stereo-seq bin100 · reproduced with the repository '
  '<code>optimal_transport__tr_axeses</code></p><ul><li>FGW 1D / 2D anchors inside INSPIRE training</li><li>Smoothed PCA as the training-free reference</li>'
  '<li>Areal-axis model of the latent, compared with the atlas frontal PCs</li><li>Adult axes carried to P0 – 7M by the adult encoder</li></ul>', 'Results')

S('Pipeline', '<ol class="steps">' + ''.join(f'<li><b>{a}</b> {b}</li>' for a, b in [
    ('01 data', 'QC as INSPIRE, SVG genes, Laplace-streamline columns, SA index of every column'),
    ('02 smoothed PCA', 'disk mean r = 1.5 grid steps, PCA-32, centred per section; radius by the layer-transfer rule'),
    ('03 FGW anchors', 'couplings between neighbouring sections (±1, ±2): 1D columns or 2D column × depth nodes'),
    ('04 training', 'INSPIRE + FGW loss, 20 000 steps; no-factor variants with Poisson / NB counts; INSPIRE + GAN baseline'),
    ('05 encode', 'the adult encoders applied to P0, P32, 3M, 7M (no training on the young data)'),
    ('06 metrics', 'mixing, region / area / layer transfer, SA, Leiden NMI split; rank; reconstruction R²'),
    ('07 axes', 'spot PCA, column PCA, areal-axis model (pseudobulk), spot-level model'),
    ('08 atlas', 'frontal PCA of the segment atlas; our axes on the atlas flat map; Spearman per fPC'),
    ('09 development', 'adult axes on the young columns: conservation, variance share, trends with age'),
    ('10–12', 'figures, 3D pages, this deck; 11 = comparison with the previous results')]) + '</ol>', 'Overview')

S('FGW anchors: aligning the cortex of two sections without labels',
  '<div class="two"><div><p>Nodes of a section: <b>1D</b> cortical columns (~200 µm wide, complete L2–L6) or <b>2D</b> column × depth-bin blocks (8 bins).</p>'
  '<p>Coupling <i>T</i> between the nodes of sections A and B: balanced fused Gromov–Wasserstein</p>'
  '<p class="eq">min<sub>T</sub> (1 − α) Σ M<sub>ij</sub>T<sub>ij</sub> + α Σ (C<sup>A</sup><sub>ik</sub> − C<sup>B</sup><sub>jl</sub>)² T<sub>ij</sub>T<sub>kl</sub></p>'
  '<p>M: distance of node profiles (smoothed PCA); C: distance along the cortex (1D) or in the column × depth plane (2D, mm); α = 0.7.</p></div>'
  '<div><p>Training loss (weight λ = 770): the latent mean of node <i>i</i> in A is pulled to the nodes of B that T assigns to it</p>'
  '<p class="eq">L = − Σ<sub>j</sub> T<sub>ij</sub>/T<sub>i·</sub> log softmax<sub>j</sub>(cos(z<sub>A,i</sub>, z<sub>B,j</sub>)/τ)</p>'
  '<p><b>Joint mode:</b> T is recomputed from the current latent every 500 steps (after 2 000), so coupling and latent are refined together. '
  'No adversarial loss in the FGW models.</p></div></div>', 'Methods')

S('INSPIRE and what was changed',
  '<ul><li>Base: INSPIRE LGCN (Zhao et al.): encoder of [log1p CP10k, graph-smoothed copy] → 32-dim latent; decoder reconstructs the features; '
  'counts ~ Poisson(library × β·softmax(factors) × e<sup>γ<sub>section</sub></sup>)</li>'
  '<li>Added: second decoder for the smoothed features; patch-variance anti-collapse of the factors; FGW anchors (1D / 2D, joint); '
  'direct count decoder without factors; negative-binomial counts; all-pairs discriminators; encode-only / warm start</li>'
  '<li><b>Classic mode</b> (chain of S−1 discriminators, original β normalisation) gives weights identical to the original package after training '
  '(tests/test_inspire.py); every project mode gives weights identical to the previous benchmark copy</li></ul>', 'Methods')

R = read(f"smoothing_radius_{c.series['name']}.csv")
S('Smoothed PCA: choice of the radius', img('smoothing_radius', 'metrics along the smoothing radius') +
  ('<p class="note">Rule: the smallest radius on the plateau of the layer transfer (within 2% of its best value). ' +
   (f'Chosen r = {R[R.chosen].r.iloc[0]:g} grid steps.' if R is not None else '') + '</p>'), 'Methods')

S('Areal-axis model',
  '<p class="eq">x<sub>trl</sub> = f<sub>l</sub> + θ c<sub>t</sub> + Σ<sub>m</sub> μ<sub>m</sub>(r) g<sub>ml</sub> + ε</p>'
  '<ul><li>x<sub>trl</sub>: mean latent of section t, region r, layer l (≥ 5 bins), dimensions scaled to unit SD</li>'
  '<li>f<sub>l</sub> layer profile and θ c<sub>t</sub> sequencing depth of the section are removed; the residual region × (layer, dim) matrix is decomposed by SVD</li>'
  '<li>μ<sub>m</sub>(r): score of region r on areal axis m, shared by all layers and sections; g<sub>ml</sub>: its loading in layer l</li>'
  '<li>split-half: |r| of μ between fits on the odd and the even sections</li></ul>'
  '<p class="note">Synthetic test (tests/test_areal.py): three planted region components are recovered with |r| ≥ 0.95; pure noise gives no axis with split-half ≥ 0.8.</p>', 'Methods')

S('Training losses', img('training_losses', 'training losses'), 'Training')
RR = read('recon_r2.csv')
S('Reconstruction R²', img('recon_r2', 'reconstruction R2') + (table(RR[['model', 'counts_spot_pooled_R2', 'counts_pseudobulk_pooled_R2', 'features_genes_pooled_R2', 'features_axf_pooled_R2']]) if RR is not None else ''), 'Training')

RK = read('series_rank.csv')
S('Benchmark: metrics and rank', img('metrics_rank', 'metrics and rank') + (f'<p class="note">Best model: <b>{esc(RK.iloc[0, 0])}</b> (score {RK.score.iloc[0]:.3f}).</p>' if RK is not None else ''), 'Benchmark')
S('UMAP of the latents', img('umap_series', 'UMAP by section, region, layer'), 'Benchmark')

for k, tag, ml, title in FGW2D:
    S(f'Areal axes · {title}', img(f'regions_{k}_pseudobulk', 'regions on the areal axes'), 'Axes')
    S(f'Areal axes on the tissue · {title}', img(f'tissue_{k}_pseudobulk', 'areal axes on five sections'), 'Axes')
S('Column PCA on the tissue · FGW 2D with factors', img('tissue_fgw2d_factors_column', 'column PCs on five sections'), 'Axes')

A = read('atlas/axes_vs_frontal_models.csv')
best = ''
if A is not None:
    b = A[A.analysis == 'pseudobulk areal'][['model', 'comp', 'reading', 'best_fPC', 'rho_best', 'rho_fPC2', 'rho_fPC4', 'rho_fPC9', 'rho_fPC10']]
    best = table(b, 2, 'small')
S('Our axes vs the frontal PCs of the atlas', img('axes_vs_atlas', 'Spearman of our axes with atlas fPCs') + best, 'Atlas')

for k, tag, ml, title in FGW2D[:1] + FGW2D[1:]:
    S(f'Development · {title}', '<div class="two">' + img(f'dev_conservation_{k}', 'conservation across ages') + img(f'dev_age_{k}', 'sample medians by age') + '</div>', 'Development')
S('Adult areal axes on the young sections · FGW 2D with factors', img('dev_spatial_fgw2d_factors_pseudobulk', 'areal axes on adult and young sections'), 'Development')

TR = read('test_report.csv'); RS = read('reproduction_summary.csv')
body = ''
if TR is not None: body += '<h3>Tests (pytest)</h3>' + table(TR.groupby('file').outcome.value_counts().unstack(fill_value=0).reset_index(), 0, 'small')
if RS is not None: body += '<h3>Retrained models vs previous results</h3>' + table(RS, 0, 'small')
S('Verification', body or '<p>run the tests and step 11</p>', 'Reproduction')

S('Interactive 3D pages', '<ul><li><a href="../3d_model/adult_atlas_pcs_vs_axes_3d.html">Adult: atlas frontal PCs vs our axes</a> (three FGW 2D models, four analyses)</li>'
  '<li><a href="../3d_model/development_axes_3d.html">Development: adult areal axes at P0, P32, 3M, 7M, adult</a> with genes and the closest atlas PC</li></ul>'
  '<p class="note">Standalone HTML files (data and plotly.js embedded); rebuild with <code>python scripts/10_figures.py 3d</code>.</p>', 'Resources')

sec = '\n'.join(f'<section class="slide" id="s{i + 1}" aria-label="{esc(t)}"><div class="kicker">{esc(k)}</div><h2>{esc(t)}</h2><div class="body">{b}</div>'
                f'<div class="num">{i + 1} / {len(slides)}</div></section>' for i, (k, t, b) in enumerate(slides))
page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FGW Axes Results</title>
<style>
:root {{ --bg:#f4f5f7; --panel:#fff; --fg:#17202e; --muted:#5b6577; --line:#dfe3ea; --accent:#2b5fb4; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#11151c; --panel:#1a202a; --fg:#e8ecf2; --muted:#9aa4b5; --line:#2c3442; --accent:#7aa7f0; }} }}
:root[data-theme="dark"] {{ --bg:#11151c; --panel:#1a202a; --fg:#e8ecf2; --muted:#9aa4b5; --line:#2c3442; --accent:#7aa7f0; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.slide {{ display:none; min-height:100vh; padding:28px 40px 48px; max-width:1500px; margin:0 auto; position:relative; }}
.slide.on {{ display:block; }} h2 {{ font-size:28px; margin:4px 0 16px; line-height:1.2; }}
.kicker {{ color:var(--accent); font-weight:600; text-transform:uppercase; letter-spacing:.06em; font-size:13px; }}
.body img {{ max-width:100%; max-height:74vh; display:block; margin:0 auto; background:#fff; border-radius:6px; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; align-items:start; }} .two img {{ max-height:62vh; }}
.eq {{ font-family:Georgia, serif; font-size:20px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 14px; overflow-x:auto; }}
.lead {{ font-size:20px; }} .note {{ color:var(--muted); }} .missing {{ padding:40px; border:1px dashed var(--line); color:var(--muted); text-align:center; }}
.steps li {{ margin:6px 0; }} code {{ background:var(--panel); padding:1px 5px; border-radius:4px; border:1px solid var(--line); }}
table {{ border-collapse:collapse; margin:14px auto; font-size:14px; background:var(--panel); }} th, td {{ border:1px solid var(--line); padding:4px 8px; text-align:right; }}
th:first-child, td:first-child {{ text-align:left; }} table.small {{ font-size:12.5px; }}
.num {{ position:absolute; right:40px; bottom:16px; color:var(--muted); font-size:13px; }}
nav {{ position:fixed; left:0; right:0; bottom:0; display:flex; justify-content:center; gap:8px; padding:8px; pointer-events:none; }}
nav button {{ pointer-events:auto; font:inherit; padding:6px 14px; border-radius:6px; border:1px solid var(--line); background:var(--panel); color:var(--fg); cursor:pointer; }}
a {{ color:var(--accent); }}
@media (max-width:800px) {{ .slide {{ padding:16px 16px 56px; }} .two {{ grid-template-columns:1fr; }} h2 {{ font-size:22px; }} .num {{ right:16px; }} }}
@media print {{ .slide {{ display:block; page-break-after:always; min-height:auto; }} nav {{ display:none; }} }}
</style></head><body>
{sec}
<nav><button id="prev" aria-label="previous slide">←</button><button id="next" aria-label="next slide">→</button></nav>
<script>
const S = [...document.querySelectorAll('.slide')]; let i = Math.max(0, S.findIndex(s => '#' + s.id === location.hash));
const show = n => {{ i = Math.min(S.length - 1, Math.max(0, n)); S.forEach((s, k) => s.classList.toggle('on', k === i)); history.replaceState(null, '', '#' + S[i].id); }};
document.addEventListener('keydown', e => {{ if (['ArrowRight', 'PageDown', ' '].includes(e.key)) show(i + 1); if (['ArrowLeft', 'PageUp'].includes(e.key)) show(i - 1); }});
document.getElementById('next').onclick = () => show(i + 1); document.getElementById('prev').onclick = () => show(i - 1);
let x0 = null; addEventListener('touchstart', e => x0 = e.touches[0].clientX); addEventListener('touchend', e => {{ if (x0 === null) return; const d = e.changedTouches[0].clientX - x0; if (Math.abs(d) > 50) show(i + (d < 0 ? 1 : -1)); x0 = null; }});
show(i);
</script></body></html>'''
out = os.path.join(OUT, 'deck_standalone.html' if EMBED else 'index.html'); open(out, 'w').write(page); print('saved', out, len(slides), 'slides', round(len(page) / 1e6, 1), 'MB')
