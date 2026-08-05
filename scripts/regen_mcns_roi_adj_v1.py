#!/usr/bin/env python
"""Regenerate ``data/Male_CNS/orn_to_alpnln_AL_roi_adj.feather`` from
neuPrint ``male-cns:v1.0``.

Runs the ORN-lateralization notebook's own ``load_mcns()`` code (cells 1+3)
from the repo root, so the ``from neuprint import ...`` inside the notebook's
cache-fallback resolves to the installed ``neuprint-python`` package rather
than the repo's ``analysis/neuprint/`` directory. Requires the v1.0
``data/Male_CNS/neurons.pkl`` (see scripts/regen_mcns_neurons_v1.py).

Run from the repo root:
    poetry run python scripts/regen_mcns_roi_adj_v1.py
"""
import json
import os

nb = json.load(open('analysis/ORN-lateralization-multi.ipynb'))

# sanity: the notebook must already target male-cns:v1.0
assert "dataset='male-cns:v1.0'" in ''.join(nb['cells'][3]['source'])
assert not os.path.exists('data/Male_CNS/orn_to_alpnln_AL_roi_adj.feather'), \
    'cache already exists — move it aside first if you want to refetch'

ns = {}
for idx in (1, 3):  # cell 1 = helpers, cell 3 = load_mcns()
    src = ''.join(nb['cells'][idx]['source'])
    exec(compile(src, f'<notebook cell {idx}>', 'exec'), ns)
ns['DATA'] = 'data'  # notebook runs from analysis/, we run from the repo root

print('calling load_mcns() (fetches ORN->ALPN/ALLN adjacencies from male-cns:v1.0) ...')
r = ns['load_mcns']()
print('load_mcns() OK —', r.shape, 'rows')

import pandas as pd
roi = pd.read_feather('data/Male_CNS/orn_to_alpnln_AL_roi_adj.feather')
print('saved data/Male_CNS/orn_to_alpnln_AL_roi_adj.feather', roi.shape)
print('columns:', list(roi.columns))
print(roi.head(3).to_string())
