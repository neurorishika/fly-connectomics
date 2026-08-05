"""Systematic dataset-quality audit for the ORN-lateralization analysis.

Checks, per dataset and per metric (R_contra, P_PN, P_LN):
  1. ORN / PN / LN side-label quality (count balance, label source agreement)
  2. per-AL-series completeness (L vs R synapse totals)
  3. per-glomerulus L-vs-R ratio asymmetry (uniform offset -> artifact)
  4. cross-series concordance (Spearman) -- which series are outliers

Run: .venv/bin/python scripts/audit_dataset_quality.py   (from repo root)
"""
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

# ---------------- loaders (identical logic to the notebook) ----------------
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
    r['animal'] = 'FAFB'; return r, nb

def load_banc():
    nb = pd.read_csv(f'{DATA}/BANC/neurons.csv.gz').rename(columns={'Root ID': 'id', 'Primary Cell Type': 'type', 'Soma side': 'side', 'Class': 'class'})
    nb['side'] = nb['side'].map(norm_side); cls = nb['class'].astype(str)
    nb['role'] = np.select([cls.eq('olfactory_receptor_neuron'), cls.eq('antennal_lobe_projection_neuron'),
                            cls.eq('antennal_lobe_local_neuron')], ['ORN', 'ALPN', 'ALLN'], default=None)
    nb['glomerulus'] = np.where(nb['role'] == 'ORN', nb['type'].astype(str).map(glom), None)
    r = _cave_syn(nb, f'{DATA}/BANC/connections_princeton.csv.gz')
    r['animal'] = 'BANC'; return r, nb

def load_hemibrain():
    return pd.read_feather(f'{DATA}/Hemibrain/orn_lat_syn.feather')

# ---------------- build synapse table + summary ----------------
syn = pd.concat([load_mcns(), load_fafb()[0], load_banc()[0], load_hemibrain()], ignore_index=True)
syn['series'] = syn['animal'] + '-' + syn['syn_AL'].map({'left': 'L', 'right': 'R'})
syn['ipsi'] = syn['orn_side'] == syn['syn_AL']
syn['metric'] = syn['kind'].map({'out': 'R_contra', 'PN': 'P_PN', 'LN': 'P_LN'})

agg = (syn.groupby(['animal', 'series', 'glomerulus', 'metric', 'ipsi'])['weight'].sum().unstack('ipsi', fill_value=0))
for col in [True, False]:
    if col not in agg: agg[col] = 0
agg = agg.rename(columns={True: 'ipsi_syn', False: 'contra_syn'}).reset_index()
common = sorted(set.intersection(*agg.groupby('animal')['glomerulus'].agg(set).tolist()))
print(f'common glomeruli across 4 animals: {len(common)}')

print('\n================ 1. SIDE-LABEL QUALITY ================')
# MCNS: rootSide vs somaSide agreement for PN/LN
with open(f'{DATA}/Male_CNS/neurons.pkl', 'rb') as f:
    meta = pickle.load(f)[0].copy()
cls = meta['class'].astype(str)
meta['role'] = np.where(cls.eq('ALPN'), 'ALPN', np.where(cls.eq('ALLN'), 'ALLN', None))
pn = meta[meta['role'].isin(['ALPN', 'ALLN'])].copy()
agree = (pn['rootSide'].map(norm_side) == pn['somaSide'].map(norm_side)).mean()
print(f'MCNS PN/LN: rootSide==somaSide agreement: {agree*100:.1f}%  (n={len(pn)})')
print('MCNS PN/LN rootSide counts:', pn['rootSide'].value_counts(dropna=False).to_dict())
print('MCNS PN/LN somaSide counts:', pn['somaSide'].value_counts(dropna=False).to_dict())

# per-dataset PN/LN count balance (soma-side column)
def pn_ln_balance(nb, sidecol):
    b = nb[nb['role'].isin(['ALPN', 'ALLN'])].copy()
    c = pd.crosstab(b['role'], b[sidecol])
    for col in ['left', 'right']:
        if col not in c:
            c[col] = 0
    c['L/(L+R)'] = c['left'] / (c['left'] + c['right'])
    return c

for a, nb, sidecol in [('FAFB', load_fafb()[1], 'side'), ('BANC', load_banc()[1], 'side')]:
    print(f'\n{a} PN/LN balance:\n{pn_ln_balance(nb, sidecol).to_string()}')

print('\n================ 2. PER-SERIES TOTAL SYNAPSES (completeness) ================')
for m in ['R_contra', 'P_PN', 'P_LN']:
    t = agg[agg['metric'] == m].groupby('series')[['ipsi_syn', 'contra_syn']].sum()
    t['tot'] = t['ipsi_syn'] + t['contra_syn']
    t['ratio'] = t['contra_syn'] / t['ipsi_syn'].replace(0, np.nan)
    print(f'\n{m}:\n{t.round(1).to_string()}')
    # L vs R balance within animals that have both
    for a in ['MCNS', 'FAFB', 'BANC']:
        ls = f'{a}-L'; rs = f'{a}-R'
        if ls in t.index and rs in t.index:
            lr = t.loc[ls, 'tot'] / t.loc[rs, 'tot']
            print(f'  {a} L/R total ratio: {lr:.3f}')

print('\n================ 3. L-vs-R RATIO ASYMMETRY PER GLOMERULUS ================')
for m in ['R_contra', 'P_PN', 'P_LN']:
    sub = agg[agg['metric'] == m]
    sub = sub.copy()
    sub['ratio'] = cratio(sub['contra_syn'], sub['ipsi_syn'])
    piv = sub.pivot_table(index=['animal', 'glomerulus'], columns='hemisphere', values='ratio') if 'hemisphere' in sub.columns else None
    # hemisphere from series
    sub['hemi'] = sub['series'].str[-1]
    piv = sub.pivot_table(index=['animal', 'glomerulus'], columns='hemi', values='ratio')
    piv['asym'] = piv['L'] - piv['R']
    piv = piv.dropna(subset=['asym'])
    print(f'\n{m}:')
    for a in ['MCNS', 'FAFB', 'BANC']:
        d = piv.xs(a, level='animal')['asym'].dropna()
        print(f'  {a}: mean|asym|={d.abs().mean():.3f}  #L>R={int((d>0).sum())}/{len(d)}  #L<R={int((d<0).sum())}/{len(d)}')

print('\n================ 4. CROSS-SERIES CONCORDANCE (Spearman, common glomeruli) ================')
for m in ['R_contra', 'P_PN', 'P_LN']:
    sub = agg[agg['metric'] == m].copy()
    sub['ratio'] = cratio(sub['contra_syn'], sub['ipsi_syn'])
    wide = sub.pivot(index='glomerulus', columns='series', values='ratio')
    series = ['MCNS-L', 'MCNS-R', 'FAFB-L', 'FAFB-R', 'BANC-L', 'BANC-R', 'hemibrain-R']
    print(f'\n{m}:')
    hdr = '        ' + ''.join(f'{s:>10s}' for s in series)
    print(hdr)
    for s1 in series:
        row = f'{s1:>8s}'
        for s2 in series:
            if s1 == s2:
                row += f'{"1.00":>10s}'
                continue
            d = wide[[s1, s2]].replace(np.inf, np.nan).dropna()
            if len(d) >= 5:
                rho = float(d[s1].corr(d[s2], method='spearman'))
                row += f'{rho:>10.2f}'
            else:
                row += f'{"--":>10s}'
        print(row)
    # same-animal L-vs-R correlation (self-consistency)
    for a in ['MCNS', 'FAFB', 'BANC']:
        ls, rs = f'{a}-L', f'{a}-R'
        d = wide[[ls, rs]].replace(np.inf, np.nan).dropna()
        rho, p = stats.spearmanr(d[ls], d[rs])
        print(f'  {a}: L-vs-R rho={rho:.3f} (n={len(d)}, p={p:.2e})')
