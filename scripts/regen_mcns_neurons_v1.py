#!/usr/bin/env python
"""Regenerate ``data/Male_CNS/neurons.pkl`` from neuPrint ``male-cns:v1.0``.

Mirrors analysis/neuprint/male-cns.ipynb (cells 1-2), which created the
original v0.9 cache, but targets the v1.0 dataset. The v0.9 cache is kept
alongside as ``data/Male_CNS/neurons_v0.9.pkl``.

Run from the repo root:
    poetry run python scripts/regen_mcns_neurons_v1.py
"""
import os
import pickle

from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria as NC, fetch_neurons, set_default_client

load_dotenv()
TOKEN = os.getenv('NEUPRINT_TOKEN')
if not TOKEN:
    raise SystemExit('NEUPRINT_TOKEN missing from .env')

cl = Client('https://neuprint.janelia.org', dataset='male-cns:v1.0', token=TOKEN)
set_default_client(cl)

print('fetching all neurons from male-cns:v1.0 ...')
neurons = fetch_neurons(NC())
print('fetched', neurons[0].shape)

with open('data/Male_CNS/neurons.pkl', 'wb') as f:
    pickle.dump(neurons, f)
print('saved data/Male_CNS/neurons.pkl')
