"""Final cross-dataset verification: bilateral-signal agreement with hemibrain
gold standard + hemispheric (L vs R) difference reproducibility."""
import sys, os, pickle
import numpy as np
import pandas as pd
from scipy import stats

DATA = 'data'

def norm_side(s):
    return {'L': 'left', 'R': 'right', 'left': 'left', 'right': 'right'}.get(str(s).strip(), None)

def glom(t):
    t = str(t)
    return t[4:] if t.startswith('ORN_') else None

def cratio(c, i):
    c = np.asarray(c, float); i = np.asarray(i, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = c / i
    r[(i == 0) & (c == 0)] = np.nan
    return r

# ---- loaders (same logic as notebook) ----
def load_mcns():
    with open(f'{DATA}/Male_CNS/neurons.pkl', 'rb') as f:
        meta = pickle.load(f)[0].copy()
    side = meta['rootSide'].map(norm_side).fillna(meta['somaSide'].map(norm_side))
    meta['side'] = side
    cls = meta['class'].astype(str); typ = meta['flywireType'].astype(str)
    meta['role'] = np.select([typ.map(lambda t: str(t).startswith('ORN_')), cls.eq('ALPN'), cls.eq('ALLN')],
                             ['ORN', 'ALPN', 'ALLN'], default=None)
    meta['glomerulus'] = np.where(meta['role'] == 'ORN', typ.map(glom), None)
    def alfield(roi, key, field):
        d = roi.get(key, {}) if isinstance(roi, dict) else {}
        return d.get(field, 0) or 0
    orns = meta[(meta['role'] == 'ORN') & meta['side'].notna() & meta['glomerulus'].notna()].copy()
    orns['alL'] = orns['roiInfo'].apply(lambda r: alfield(r, 'AL(L)', 'downstream'))
    orns['alR'] = orns['roiInfo'].apply(lambda r: alfield(r, 'AL(R)', 'downstream'))
    out = pd.concat([
        pd.DataFrame({'glomerulus': orns['glomerulus'].values, 'orn_side': orns['side'].values,
                      'syn_AL': 'left', 'orn_id': orns['bodyId'].values, 'weight': orns['alL'].values, 'kind': 'out'}),
        pd.DataFrame({'glomerulus': orns['glomerulus'].values, 'orn_side': orns['side'].values,
                      'syn_AL': 'right', 'orn_id': orns['bodyId'].values, 'weight': orns['alR'].values, 'kind': 'out'}),
    ], ignore_index=True)
    postn = meta[meta['role'].isin(['ALPN', 'ALLN']) & meta['side'].notna()]
    roi = pd.read_feather(f'{DATA}/Male_CNS/orn_to_alpnln_AL_roi_adj.feather')
    id2role = meta.set_index('bodyId')['role']; id2side = meta.set_index('bodyId')['side']
    id2glom = orns.set_index('bodyId')['glomerulus']
    roi = roi.copy(); roi['kind'] = roi['bodyId_post'].map(id2role).map({'ALPN': 'PN', 'ALLN': 'LN'})
    roi = roi.dropna(subset=['kind'])
    e = pd.DataFrame({'glomerulus': roi['bodyId_pre'].map(id2glom).values, 'orn_side': roi['bodyId_pre'].map(id2side).values,
                      'syn_AL': roi['roi'].map({'AL(L)': 'left', 'AL(R)': 'right'}).values,
                      'orn_id': roi['bodyId_pre'].values, 'weight': roi['weight'].values, 'kind': roi['kind'].values})
    r = pd.concat([out, e], ignore_index=True).dropna(subset=['glomerulus', 'orn_side', 'syn_AL'])
    r['animal'] = 'MCNS'; return r

_NP_SIDE = {'AL_L': 'left', 'AL_R': 'right'}
def _cave_syn(nb, conn_csv, chunksize=None):
    nb = nb.copy(); nb['id'] = nb['id'].astype('int64')
    orns = nb[(nb['role'] == 'ORN') & nb['side'].notna() & nb['glomerulus'].notna()]
    orn_ids = set(orns['id'])
    pn = set(nb[nb['role'] == 'ALPN']['id']); ln = set(nb[nb['role'] == 'ALLN']['id'])
    id2side = nb.set_index('id')['side']; id2glom = orns.set_index('id')['glomerulus']
    parts = []
    reader = pd.read_csv(conn_csv, usecols=['pre_root_id', 'post_root_id', 'neuropil', 'syn_count'], chunksize=chunksize)
    for ch in (reader if chunksize else [reader]):
        c = ch[ch['pre_root_id'].isin(orn_ids) & ch['neuropil'].isin(['AL_L', 'AL_R'])]
        if len(c) == 0:
            continue
        c = c.assign(glomerulus=c['pre_root_id'].map(id2glom), orn_side=c['pre_root_id'].map(id2side),
                     syn_AL=c['neuropil'].map(_NP_SIDE))
        o = c.groupby(['glomerulus', 'orn_side', 'syn_AL', 'pre_root_id'], as_index=False)['syn_count'].sum()
        o['kind'] = 'out'; parts.append(o.rename(columns={'pre_root_id': 'orn_id', 'syn_count': 'weight'}))
        for kind, ids in [('PN', pn), ('LN', ln)]:
            cc = c[c['post_root_id'].isin(ids)]
            if len(cc):
                g = cc.groupby(['glomerulus', 'orn_side', 'syn_AL', 'pre_root_id'], as_index=False)['syn_count'].sum()
                g['kind'] = kind; parts.append(g.rename(columns={'pre_root_id': 'orn_id', 'syn_count': 'weight'}))
    return pd.concat(parts, ignore_index=True)

def load_fafb():
    nb = pd.read_csv(f'{DATA}/FlyWire/classification.csv.gz').rename(columns={'root_id': 'id', 'hemibrain_type': 'type'})
    nb['side'] = nb['side'].map(norm_side); typ = nb['type'].astype(str)
    nb['role'] = np.select([typ.map(lambda t: str(t).startswith('ORN_')), nb['class'].eq('ALPN'), nb['class'].eq('ALLN')],
                           ['ORN', 'ALPN', 'ALLN'], default=None)
    nb['glomerulus'] = np.where(nb['role'] == 'ORN', typ.map(glom), None)
    r = _cave_syn(nb, f'{DATA}/FlyWire/connections_princeton_no_threshold.csv.gz', chunksize=3_000_000)
    r['animal'] = 'FAFB'; return r

def load_banc():
    nb = pd.read_csv(f'{DATA}/BANC/neurons.csv.gz').rename(columns={'Root ID': 'id', 'Primary Cell Type': 'type', 'Soma side': 'side', 'Class': 'class'})
    nb['side'] = nb['side'].map(norm_side); cls = nb['class'].astype(str)
    nb['role'] = np.select([cls.eq('olfactory_receptor_neuron'), cls.eq('antennal_lobe_projection_neuron'),
                            cls.eq('antennal_lobe_local_neuron')], ['ORN', 'ALPN', 'ALLN'], default=None)
    nb['glomerulus'] = np.where(nb['role'] == 'ORN', nb['type'].astype(str).map(glom), None)
    r = _cave_syn(nb, f'{DATA}/BANC/connections_princeton.csv.gz')
    r['animal'] = 'BANC'; return r

def load_hemibrain():
    return pd.read_feather(f'{DATA}/Hemibrain/orn_lat_syn.feather')

# ---- build everything ----
syn = pd.concat([load_mcns(), load_fafb(), load_banc(), load_hemibrain()], ignore_index=True)
syn['series'] = syn['animal'] + '-' + syn['syn_AL'].map({'left': 'L', 'right': 'R'})
syn['ipsi'] = syn['orn_side'] == syn['syn_AL']
syn['metric'] = syn['kind'].map({'out': 'R_contra', 'PN': 'P_PN', 'LN': 'P_LN'})

agg = (syn.groupby(['animal', 'series', 'glomerulus', 'metric', 'ipsi'])['weight'].sum().unstack('ipsi', fill_value=0))
for col in [True, False]:
    if col not in agg: agg[col] = 0
agg = agg.rename(columns={True: 'ipsi_syn', False: 'contra_syn'}).reset_index()
common = sorted(set.intersection(*agg.groupby('animal')['glomerulus'].agg(set).tolist()))

o = syn[(syn['kind'] == 'out') & (syn['animal'] != 'hemibrain') & syn['glomerulus'].isin(common)].copy()
o['role'] = np.where(o['syn_AL'] == o['orn_side'], 'ipsi', 'contra')
po = (o.pivot_table(index=['animal', 'orn_side', 'orn_id', 'glomerulus'], columns='role', values='weight', fill_value=0).reset_index())
po['r'] = cratio(po['contra'], po['ipsi'])

# hemibrain gold standard
out = syn[(syn['animal'] == 'hemibrain') & (syn['kind'] == 'out')]
gm = out.groupby(['glomerulus', 'orn_side'])['weight'].sum().unstack(fill_value=0).rename(columns={'left': 'contra', 'right': 'ipsi'})
gm['mirror'] = gm['contra'] / gm['ipsi'].replace(0, np.nan)

# per-ORN bilateral fraction per dataset
bf = po.groupby(['animal', 'glomerulus'], as_index=False).apply(lambda d: pd.Series({'bil_frac': (d['contra'] > d['ipsi']).mean()}))
wide = bf.pivot(index='glomerulus', columns='animal', values='bil_frac').join(gm['mirror'])

print('=== Per-ORN bilateral fraction vs hemibrain mirror ratio (spearman) ===')
for a in ['MCNS', 'FAFB', 'BANC']:
    sub = wide.dropna(subset=[a, 'mirror'])
    rho, p = stats.spearmanr(sub[a], sub['mirror'])
    print(f'{a}: n={len(sub)}  rho={rho:.3f} (p={p:.2e})')

# ---- hemispheric (L vs R ORN-soma) difference per glomerulus, per dataset ----
print('\n=== L-vs-R per-ORN median difference (med_L - med_R), common glomeruli ===')
med = po.groupby(['animal', 'glomerulus', 'orn_side'])['r'].median().unstack('orn_side')
med['diff'] = med['left'] - med['right']
med = med.dropna(subset=['diff'])
lr = med['diff'].unstack('animal').reindex(common)
consist = lr.dropna()
print('sign-consistent (FAFB & BANC same sign) among glomeruli in both:')
agree = ((consist['FAFB'] > 0) & (consist['BANC'] > 0)) | ((consist['FAFB'] < 0) & (consist['BANC'] < 0))
print('FAFB/BANC agree on direction:', agree.sum(), '/', len(consist), '| p(binomial vs 0.5):',
      stats.binomtest(agree.sum(), len(consist), 0.5).pvalue)
print(consist.round(2).to_string())
