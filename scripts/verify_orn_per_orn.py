"""Independent re-derivation of the per-ORN contra/ipsi table (notebook cell 16 logic,
re-implemented from the raw data tables) plus artifact probes.

Run from repo root:  .venv/bin/python scripts/verify_orn_per_orn.py
"""
import os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('analysis'))
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
EPS = 1e-9

_SIDE = {'L': 'left', 'R': 'right', 'left': 'left', 'right': 'right'}
def norm_side(s):
    return _SIDE.get(str(s).strip(), None)

def glom(t):
    t = str(t)
    return t[4:] if t.startswith('ORN_') else None

def is_orn(t):
    return str(t).startswith('ORN_')

def cratio(contra, ipsi):
    contra = np.asarray(contra, float); ipsi = np.asarray(ipsi, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = contra / ipsi
    r[(ipsi == 0) & (contra == 0)] = np.nan
    return r

# ---------------- loaders (same as notebook) ----------------
def load_mcns():
    with open(f'{DATA}/Male_CNS/neurons.pkl', 'rb') as f:
        meta = pickle.load(f)[0].copy()
    side = meta['rootSide'].map(norm_side).fillna(meta['somaSide'].map(norm_side))
    meta['side'] = side
    cls = meta['class'].astype(str); typ = meta['flywireType'].astype(str)
    meta['role'] = np.select([typ.map(is_orn), cls.eq('ALPN'), cls.eq('ALLN')],
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
    cache = f'{DATA}/Male_CNS/orn_to_alpnln_AL_roi_adj.feather'
    roi = pd.read_feather(cache)
    id2role = meta.set_index('bodyId')['role']
    id2side = meta.set_index('bodyId')['side']
    id2glom = orns.set_index('bodyId')['glomerulus']
    roi = roi.copy()
    roi['kind'] = roi['bodyId_post'].map(id2role).map({'ALPN': 'PN', 'ALLN': 'LN'})
    roi = roi.dropna(subset=['kind'])
    e = pd.DataFrame({
        'glomerulus': roi['bodyId_pre'].map(id2glom).values,
        'orn_side': roi['bodyId_pre'].map(id2side).values,
        'syn_AL': roi['roi'].map({'AL(L)': 'left', 'AL(R)': 'right'}).values,
        'orn_id': roi['bodyId_pre'].values, 'weight': roi['weight'].values, 'kind': roi['kind'].values,
    })
    r = pd.concat([out, e], ignore_index=True).dropna(subset=['glomerulus', 'orn_side', 'syn_AL'])
    r['animal'] = 'MCNS'
    return r, orns

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
        c = c.assign(glomerulus=c['pre_root_id'].map(id2glom),
                     orn_side=c['pre_root_id'].map(id2side),
                     syn_AL=c['neuropil'].map(_NP_SIDE))
        o = c.groupby(['glomerulus', 'orn_side', 'syn_AL', 'pre_root_id'], as_index=False)['syn_count'].sum()
        o['kind'] = 'out'
        parts.append(o.rename(columns={'pre_root_id': 'orn_id', 'syn_count': 'weight'}))
        for kind, ids in [('PN', pn), ('LN', ln)]:
            cc = c[c['post_root_id'].isin(ids)]
            if len(cc):
                g = cc.groupby(['glomerulus', 'orn_side', 'syn_AL', 'pre_root_id'], as_index=False)['syn_count'].sum()
                g['kind'] = kind
                parts.append(g.rename(columns={'pre_root_id': 'orn_id', 'syn_count': 'weight'}))
    return pd.concat(parts, ignore_index=True)

def load_fafb():
    nb = pd.read_csv(f'{DATA}/FlyWire/classification.csv.gz').rename(columns={'root_id': 'id', 'hemibrain_type': 'type'})
    nb['side'] = nb['side'].map(norm_side); typ = nb['type'].astype(str)
    nb['role'] = np.select([typ.map(is_orn), nb['class'].eq('ALPN'), nb['class'].eq('ALLN')],
                           ['ORN', 'ALPN', 'ALLN'], default=None)
    nb['glomerulus'] = np.where(nb['role'] == 'ORN', typ.map(glom), None)
    r = _cave_syn(nb, f'{DATA}/FlyWire/connections_princeton_no_threshold.csv.gz', chunksize=3_000_000)
    r['animal'] = 'FAFB'; return r, nb

def load_banc():
    nb = pd.read_csv(f'{DATA}/BANC/neurons.csv.gz').rename(columns={'Root ID': 'id', 'Primary Cell Type': 'type', 'Soma side': 'side', 'Class': 'class'})
    nb['side'] = nb['side'].map(norm_side); cls = nb['class'].astype(str)
    nb['role'] = np.select([cls.eq('olfactory_receptor_neuron'),
                            cls.eq('antennal_lobe_projection_neuron'),
                            cls.eq('antennal_lobe_local_neuron')], ['ORN', 'ALPN', 'ALLN'], default=None)
    nb['glomerulus'] = np.where(nb['role'] == 'ORN', nb['type'].astype(str).map(glom), None)
    r = _cave_syn(nb, f'{DATA}/BANC/connections_princeton.csv.gz')
    r['animal'] = 'BANC'; return r, nb

def load_hemibrain():
    cache = f'{DATA}/Hemibrain/orn_lat_syn.feather'
    return pd.read_feather(cache)

if __name__ == '__main__':
    print('loading...')
    mcns, mcns_orns = load_mcns()
    fafb, fafb_nb = load_fafb()
    banc, banc_nb = load_banc()
    hb = load_hemibrain()

    syn = pd.concat([mcns, fafb, banc, hb], ignore_index=True)
    syn['series'] = syn['animal'] + '-' + syn['syn_AL'].map({'left': 'L', 'right': 'R'})
    syn['ipsi'] = syn['orn_side'] == syn['syn_AL']
    syn['metric'] = syn['kind'].map({'out': 'R_contra', 'PN': 'P_PN', 'LN': 'P_LN'})

    # common glomeruli across all 4 animals
    agg = (syn.groupby(['animal', 'series', 'glomerulus', 'metric', 'ipsi'])['weight'].sum()
              .unstack('ipsi', fill_value=0))
    for col in [True, False]:
        if col not in agg: agg[col] = 0
    agg = agg.rename(columns={True: 'ipsi_syn', False: 'contra_syn'}).reset_index()
    common = sorted(set.intersection(*agg.groupby('animal')['glomerulus'].agg(set).tolist()))
    print(f'common glomeruli: {len(common)}')

    # ---- per-ORN table (cell 16 logic) ----
    o = syn[(syn['kind'] == 'out') & (syn['animal'] != 'hemibrain') & syn['glomerulus'].isin(common)].copy()
    o['role'] = np.where(o['syn_AL'] == o['orn_side'], 'ipsi', 'contra')
    po = (o.pivot_table(index=['animal', 'orn_side', 'orn_id', 'glomerulus'],
                        columns='role', values='weight', fill_value=0).reset_index())
    for c in ['ipsi', 'contra']:
        if c not in po: po[c] = 0
    po['r'] = cratio(po['contra'], po['ipsi'])
    po = po.dropna(subset=['r'])
    print(f'per-ORN rows: {len(po)}  (MCNS {sum(po.animal=="MCNS")}, FAFB {sum(po.animal=="FAFB")}, BANC {sum(po.animal=="BANC")})')
    print(f'non-finite r (fully contra, ipsi=0): {(~np.isfinite(po.r)).sum()}')

    # SIDE-METADATA SANITY: bulk output side vs soma side
    print('\n=== SIDE SANITY: ORNs whose bulk output is on the OPPOSITE side of their soma ===')
    for a in ['MCNS', 'FAFB', 'BANC']:
        d = po[po.animal == a]
        bulk_contra = d.contra > d.ipsi
        print(f'{a}: {bulk_contra.sum()} / {len(d)} ORNs have contra > ipsi '
              f'({bulk_contra.mean()*100:.1f}%)')

    # top deviant ORNs per animal
    print('\n=== TOP 15 deviant ORNs per animal (highest finite contra/ipsi) ===')
    for a in ['MCNS', 'FAFB', 'BANC']:
        d = po[po.animal == a].copy()
        d['r'] = d['r'].replace(np.inf, np.nan)
        top = d.sort_values('r', ascending=False).head(15)
        print(f'\n-- {a} --')
        print(top[['glomerulus', 'orn_side', 'orn_id', 'ipsi', 'contra', 'r']].to_string(index=False))

    # which glomeruli have ORNs with contra > ipsi (bilateral ORN evidence)
    print('\n=== Glomeruli containing >=1 ORN with contra > ipsi, by animal ===')
    for a in ['MCNS', 'FAFB', 'BANC']:
        d = po[po.animal == a]
        g = d[d.contra > d.ipsi].groupby('glomerulus')['orn_id'].nunique()
        print(f'{a}: {sorted(g.index.tolist())}')

    # L vs R distribution differences per glomerulus
    print('\n=== L-vs-R soma ORN distribution difference (per glomerulus, MCNS) ===')
    from scipy import stats
    d = po[po.animal == 'MCNS']
    for gl in sorted(common):
        sub = d[d.glomerulus == gl]
        L = sub[sub.orn_side == 'left']['r'].replace(np.inf, np.nan).dropna()
        R = sub[sub.orn_side == 'right']['r'].replace(np.inf, np.nan).dropna()
        if len(L) < 2 or len(R) < 2: continue
        med_diff = np.nanmedian(L) - np.nanmedian(R)
        if abs(med_diff) > 0.2:
            ks = stats.ks_2samp(L, R)
            print(f'{gl}: n_L={len(L)} n_R={len(R)} med_L={np.nanmedian(L):.3f} med_R={np.nanmedian(R):.3f} '
                  f'diff={med_diff:+.3f}  KS p={ks.pvalue:.2e}')
