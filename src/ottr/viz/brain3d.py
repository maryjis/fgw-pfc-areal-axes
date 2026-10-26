"""Interactive 3D pages of the marmoset brain (step 10) -> results/3d_model/.

atlas page        (templates/brain3d_atlas.html): left brain = region means of an atlas frontal PC (fPC1-10, over the frontal
                  segments); right brain = region means of one of our axes per model and analysis (areal axes on spots, column PCA,
                  spot PCA, spot-level model) over the adult series; the Spearman rho of the axis with every fPC.
development page  (templates/brain3d_development.html): per model, areal axes 1-5 on the brain at every age (adults per fine
                  region, P0-7M per coarse area, column means of the areal score in the adult coordinates); sample medians and
                  variance share per age; top-30 genes of each end with their mean z per age; inset = the closest atlas fPC
                  (top 3).
Meshes: Brain/MINDS BMA 2.0, right hemisphere (paths.meshes; our A8Av = BMA A8aV). Pages are standalone HTML files with the
data and Plotly (plotly.js 2.35.0, cached in work/vendor) embedded."""
import os, json, urllib.request
import numpy as np
import pandas as pd
import anndata as ad
from ..config import cfg, ensure
from ..utils.labels import AREA as FINE2AREA, COARSE_DEV
from ..areal.multi_axes import axes_dir
from ..atlas.frontal_pca import FRONTAL, FNAME

LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']; K = 5; BMA = {'A8Av': 'A8aV'}; AGES = ['P0', 'P32', '3M', '7M', 'adult']
TEMPLATES = os.path.join(os.path.dirname(__file__), 'templates')
PLOTLY = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.0/plotly.min.js'


class Meshes:
    def __init__(self):
        M3 = cfg().path('meshes'); meta = json.load(open(f'{M3}/meshes.json')); q = meta['quant']
        self.mn, self.scl = np.array(q['min']), np.array(q['scale']); self.buf = {f: open(f'{M3}/{fn}', 'rb').read() for f, fn in meta['files'].items()}
        self.idx = {x['id']: x for x in meta['meshes']}; self.labels = json.load(open(f'{M3}/regions.json'))['labels']
        self.key = {v['key']: int(k) for k, v in self.labels.items() if v['hemi'] == 'R'}

    def mesh(self, mid):
        x = self.idx[mid]; b = self.buf[x['file']]; v = np.frombuffer(b, np.uint16, x['vn'] * 3, x['vo']).reshape(-1, 3) * self.scl + self.mn
        f = np.frombuffer(b, np.uint16 if x['it'] == 16 else np.uint32, x['in'], x['io']).reshape(-1, 3)
        return dict(x=np.round(v[:, 0], 2).tolist(), y=np.round(v[:, 1], 2).tolist(), z=np.round(v[:, 2], 2).tolist(), i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist())

    def region(self, g):
        return dict(self.mesh(f'L:{self.key[g]}'), centroid=self.labels[str(self.key[g])]['centroid'])


def atlas_data(models):
    """models: [dict(key, label, axes_tag, axes_label, emb, spot_tag, match_label)] -> data dict of the atlas page."""
    c = cfg(); ds = c.series['name']; ms = Meshes(); seg = c.path('atlas_segments')
    S = pd.read_csv(f'{seg}/segments_meta.csv', index_col=0); F = pd.read_csv(c.tables('atlas/frontal_pca_scores.csv'), index_col=0)
    fr = S.index[S.VL_region.isin(FRONTAL)]; reg_of = S.loc[fr, 'region'].map(lambda r: BMA.get(r, r))
    FR = F.loc[fr].groupby(reg_of).mean(); FR = FR[FR.index.isin(list(ms.key))]; zone = S.loc[fr].groupby(reg_of).VL_region.agg(lambda x: x.mode().iloc[0])
    atlas = {k: {r: round(float(FR.loc[r, f'fPC{k}']), 3) for r in FR.index} for k in range(1, 11)}
    o = ad.read_h5ad(c.inputs(ds), backed='r').obs; lay = o.layer.astype(str).values; reg = o.region.astype(str).values
    sec = o.slice.astype(str).values; col = o.col.values.astype(int); dep = o.depth.values.astype(float)
    okl = np.isin(lay, LAYERS); okr = ~np.isin(reg, ['nan', 'None', 'remove', 'unassigned']); okm = okl & np.isfinite(dep)
    CORR = pd.read_csv(c.tables('atlas/axes_vs_frontal_models.csv'))
    AN = [('areal', 'pseudobulk areal', 'pseudobulk'), ('column', 'column PCA', 'column'), ('spot', 'spot PCA', 'spot'), ('spotmodel', 'spot-level model', None)]
    out_models = {}
    for m in models:
        Z = np.load(os.path.join(axes_dir(), f"{ds}{m['axes_tag']}_{m['axes_label'].replace(' ', '_')}.npz"), allow_pickle=True); kf = Z['keepf']
        T = pd.read_csv(c.tables(f"axes_{ds}{m['axes_tag']}_comp.csv")); T = T[T.model == m['axes_label']]
        E = np.load(c.emb(ds, m['emb'])).astype(np.float64)[:, kf]; pb = np.full((len(E), K), np.nan)
        for j, l in enumerate(LAYERS):
            q = lay == l; pb[q] = (E[q] - E[q].mean(0)) @ Z['pb_G'][:K, j].T
        d = dict(zip(Z['col_keys'], Z['col_scores'][:, :K].astype(float))); colsp = np.array([d.get(f'{s}|{cc}', [np.nan] * K) for s, cc in zip(sec, col)], float)
        sm = np.full((len(E), K), np.nan); sm[okm] = np.load(os.path.join(axes_dir(), f"spot_axes_scores{m['spot_tag']}_layer+depth_{m['axes_label'].replace(' ', '_')}.npy"))[:, :K]
        SPOT = {'areal': pb, 'column': colsp, 'spot': Z['spot_scores'][:, :K].astype(float), 'spotmodel': sm}
        SA_ = pd.read_csv(c.tables(f"spot_axes_{ds}{m['spot_tag']}.csv")); SA_ = SA_[(SA_.variant == 'layer+depth') & (SA_.model == m['axes_label'])]
        axes = []
        for kind, an_label, ta in AN:
            q = okl & okr; RM = pd.DataFrame(SPOT[kind][q]).groupby(reg[q]).mean(); RM.index = [BMA.get(r, r) for r in RM.index]
            for k in range(K):
                cr = CORR[(CORR.model == m['match_label']) & (CORR.analysis == an_label) & (CORR.comp == k + 1)].iloc[0]
                if ta: t = T[(T.analysis == ta) & (T.comp == k + 1)].iloc[0]; reading, share = str(t.reading), float(t.var_share)
                else: reading, share = str(cr.reading), float(SA_[SA_.component == k + 1].iloc[0].var_share)
                axes.append(dict(id=f'{kind}|{k + 1}', analysis=an_label, k=k + 1, reading=reading, share=round(share, 3), rho=[round(float(cr[f'rho_fPC{j}']), 3) for j in range(1, 11)],
                                 values={r: round(float(RM.loc[r, k]), 4) for r in RM.index if r in ms.key and np.isfinite(RM.loc[r, k])}))
        out_models[m['key']] = dict(label=m['label'], axes=axes); print('atlas page', m['key'], flush=True)
    A = pd.read_csv(c.tables('atlas/frontal_pca_annotation.csv')).set_index('pc'); v1 = F.loc[fr, 'fPC1'].var()
    fshare = {k: round(float(A.var_share_frontal[1] * F.loc[fr, f'fPC{k}'].var() / v1), 4) for k in range(1, 11)}
    regs = sorted(set(FR.index) | {r for mm in out_models.values() for a in mm['axes'] for r in a['values']})
    return dict(fpcs=[dict(k=k, name=FNAME[k], share=fshare[k]) for k in range(1, 11)], atlas=atlas, zone={r: str(zone.get(r, '')) for r in regs},
                models=out_models, model_order=[m['key'] for m in models], shell=ms.mesh('L:0'), regions={g: ms.region(g) for g in regs if g in ms.key})


def development_data(models, atlas_json=None):
    """models: [(AxesModel, display label, match label)] -> data dict of the development page."""
    from ..areal.development import LATENT
    c = cfg(); ds = c.series['name']; ms = Meshes()
    o = ad.read_h5ad(c.inputs(ds), backed='r').obs; lay = o.layer.astype(str).values; sec = o.slice.astype(str).values; col = o.col.values.astype(int)
    AC = pd.read_csv(c.columns(ds)); AC['slice'] = AC.slice.astype(str); AC = AC.set_index(['slice', 'col'])
    CORR = pd.read_csv(c.tables('atlas/axes_vs_frontal_models.csv')); out, regs = {}, None
    for M, label, mlabel in models:
        Z = M.npz; kf = Z['keepf']; PG = Z['pb_G'][:K]; T = M.comp; T = T[T.analysis == 'pseudobulk'].set_index('comp')
        E = np.load(c.emb(ds, M.adult_emb)).astype(np.float64)[:, kf]; sc = np.full((len(E), K), np.nan)
        for j, l in enumerate(LAYERS):
            q = lay == l; sc[q] = (E[q] - E[q].mean(0)) @ PG[:, j].T
        m = (col >= 0) & np.isin(lay, LAYERS[1:]); df = pd.DataFrame(sc[m], columns=range(K)); df['s'] = sec[m]; df['c'] = col[m]; cm = df.groupby(['s', 'c']).mean()
        cm['region'] = [AC.area_level3.get(k, 'nan') for k in cm.index]; adult = cm.groupby('region').mean(); adult = adult[adult.index.isin(list(FINE2AREA))]
        regs = list(adult.index); area_of = {r: COARSE_DEV.get(FINE2AREA[r], 'nan') for r in regs}
        vals = {'adult': {k: {BMA.get(r, r): round(float(adult.loc[r, k]), 3) for r in regs} for k in range(K)}}
        for a in AGES[:-1]:
            C = pd.read_csv(M.out(f'dev{a}_columns.csv')); C = C[C.latent == LATENT]; am = C.groupby('area')[[f'pseudobulk_{k + 1}' for k in range(K)]].mean()
            vals[a] = {k: {BMA.get(r, r): (round(float(am.loc[area_of[r], f'pseudobulk_{k + 1}']), 3) if area_of[r] in am.index else None) for r in regs} for k in range(K)}
        SM = pd.read_csv(M.out('age_areal_samples_median.csv')); CS = pd.read_csv(M.out('conservation.csv')); CS = CS[CS.analysis == 'pseudobulk']
        G = pd.read_csv(M.out('age_areal_genes_median.csv'))
        axes = [dict(k=k + 1, reading=str(T.loc[k + 1].reading), share=float(T.loc[k + 1].var_share), splithalf=float(T.loc[k + 1].splithalf)) for k in range(K)]
        for ax_ in axes:
            cr = CORR[(CORR.model == mlabel) & (CORR.analysis == 'pseudobulk areal') & (CORR.comp == ax_['k'])].iloc[0]
            rho = [float(cr[f'rho_fPC{j}']) for j in range(1, 11)]; jb = int(np.argmax(np.abs(rho)))
            ax_['fpc'] = jb + 1; ax_['fpc_rho'] = round(rho[jb], 3); ax_['fpc_top'] = [[int(j) + 1, round(rho[j], 3)] for j in np.argsort(-np.abs(rho))[:3]]
        out[M.key] = dict(label=label, axes=axes, values=vals,
                          trend={k: {a: [round(float(x), 3) for x in SM[SM.age == a][f'axis{k + 1}']] for a in AGES} for k in range(K)},
                          share={k: {a: round(float(CS[(CS.comp == k + 1) & (CS.age == a)].share.iloc[0]), 4) for a in AGES} for k in range(K)},
                          genes={k: {end: [dict(g=r.gene, r=round(r.loading_r, 2), rho=round(r.rho_age_dev, 2), z=[round(float(getattr(r, f'z_{a}')), 2) for a in AGES])
                                           for r in G[(G.axis == k + 1) & (G.end == end)].sort_values('rank').itertuples()] for end in ('positive', 'negative')} for k in range(K)})
        print('development page', M.key, flush=True)
    rb = [BMA.get(r, r) for r in regs]
    d = dict(ages=AGES, models=out, model_order=[M.key for M, _, _ in models], area={BMA.get(r, r): COARSE_DEV.get(FINE2AREA[r], 'nan') for r in regs},
             shell=ms.mesh('L:0'), regions={g: ms.region(g) for g in rb if g in ms.key})
    if atlas_json: d.update(atlas=atlas_json['atlas'], fpcs=atlas_json['fpcs'], aregions=atlas_json['regions'])
    return d


def plotly_js():
    f = os.path.join(ensure(os.path.join(cfg().work, 'vendor')), 'plotly-2.35.0.min.js')
    if not os.path.exists(f): urllib.request.urlretrieve(PLOTLY, f)
    return open(f).read()


def build_page(template, data, out_name):
    """Template + data -> standalone HTML in results/3d_model/ (data and Plotly embedded, opens from disk)."""
    t = open(os.path.join(TEMPLATES, template)).read()
    page = t.replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
    tag = f'<script src="{PLOTLY}"></script>'; assert tag in page
    body = page.replace(tag, '<script>' + plotly_js().replace('</script', '<\\/script') + '</script>')
    te = body.index('</title>') + len('</title>'); head, rest = body[:te], body[te:]; se = rest.index('</style>') + len('</style>')
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + rest[:se] + '\n<style>html{color-scheme:light}body{margin:0}[hidden]{display:none!important}</style>\n</head>\n<body>\n' + rest[se:] + '\n</body>\n</html>\n')
    p = os.path.join(ensure(cfg().model3d()), out_name); open(p, 'w').write(html); print('saved', p, round(len(html) / 1e6, 1), 'MB', flush=True)
    return p
