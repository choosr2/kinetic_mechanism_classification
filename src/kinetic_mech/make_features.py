"""Step 1 (preprocessing): compute the 136 features for the training, validation and test sets.

    training    the 4,950,000 training experiments in 2,416 batches of 2,048, each batch sampled with the authors'
                random points and noise (dataset.training_draws); the last 2,032 are left out -> 4,947,968 rows
    validation  the 50,000 validation experiments in batches of 25, sampled the same way
    test        the five released test sets (already sampled and noised), 100,000 experiments each

Outputs in data/processed/ (float32; NaN where a test cannot be computed):
    train_features.npy, train_labels.npy, train_draws.csv (points and noise of every batch)
    validation_features.npy, validation_labels.npy, validation_conditions.csv
    test_<p>pts_<n>pct_features.npy, test_labels.npy
The training step saves its progress and can be stopped and restarted.

Usage:  uv run python -m kinetic_mech.make_features [--sets train,validation,test]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from . import config as C
from .dataset import apply_draw, load_test, load_train, sampled_validation, training_draws
from .features import COLUMNS, compute_features

CHUNK = 64          # training batches read from disk at a time


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def make_train():
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    prog_file = C.PROCESSED / 'train_progress.json'
    prog = json.load(open(prog_file)) if prog_file.exists() else {'batches_done': 0}
    if prog.get('complete'):
        log('training features: already done')
        return
    x1, x2, y = load_train()
    draws, blocks = training_draws(len(y))
    nb = C.N_TRAIN_BATCHES
    n = nb * C.BATCH
    F = open_memmap(C.PROCESSED / 'train_features.npy', mode='r+' if prog['batches_done'] else 'w+',
                    dtype=np.float32, shape=(n, len(COLUMNS)))
    if not prog['batches_done']:
        np.save(C.PROCESSED / 'train_labels.npy', y[:n])
        pd.DataFrame({'batch': range(nb), 'n_pts': [len(r) - 1 for r, _ in draws[:nb]],
                      'noise': [float(s) for _, s in draws[:nb]]}).to_csv(C.PROCESSED / 'train_draws.csv', index=False)
    log('training features: %d batches of %d, starting at batch %d' % (nb, C.BATCH, prog['batches_done']))
    t0 = time.time()
    for start in range(prog['batches_done'], nb, CHUNK):
        stop = min(start + CHUNK, nb)
        block = np.array(x2[start * C.BATCH:stop * C.BATCH])        # one sequential read
        for i in range(start, stop):
            rows, sd = draws[i]
            b = apply_draw(block[(i - start) * C.BATCH:(i - start + 1) * C.BATCH], rows, sd, blocks)
            F[i * C.BATCH:(i + 1) * C.BATCH] = compute_features(x1[i * C.BATCH:(i + 1) * C.BATCH], b).to_numpy(np.float32)
        F.flush()
        prog['batches_done'] = stop
        json.dump(prog, open(prog_file, 'w'))
        log('  %d / %d batches (%.1f min)' % (stop, nb, (time.time() - t0) / 60))
    prog['complete'] = True
    json.dump(prog, open(prog_file, 'w'))


def validation_features():
    """Features of the sampled validation set. Batches with the same number of points are computed together
    (every feature is per experiment, so this gives the same values as one batch at a time)."""
    x1, batches, y, n_pts, noise = sampled_validation()
    by_n = {}
    for i, b in batches:
        by_n.setdefault(b.shape[1], []).append((i, b))
    parts = []
    for items in by_n.values():
        idx = np.concatenate([np.arange(i * C.VAL_BATCH, (i + 1) * C.VAL_BATCH) for i, _ in items])
        F = compute_features(x1[idx], np.concatenate([b for _, b in items]))
        F.index = idx
        parts.append(F)
    F = pd.concat(parts).sort_index()
    return F.to_numpy(np.float32), y, pd.DataFrame({'n_pts': n_pts, 'noise': noise})


def make_validation():
    F, y, cond = validation_features()
    np.save(C.PROCESSED / 'validation_features.npy', F)
    np.save(C.PROCESSED / 'validation_labels.npy', y)
    cond.to_csv(C.PROCESSED / 'validation_conditions.csv', index=False)
    log('validation features: %s' % (F.shape,))


def make_test():
    for pts, noise in C.TEST_CONDITIONS:
        x1, x2, y = load_test(pts, noise)
        F = compute_features(x1, x2).to_numpy(np.float32)
        np.save(C.PROCESSED / ('test_%dpts_%dpct_features.npy' % (pts, noise)), F)
        np.save(C.PROCESSED / 'test_labels.npy', y)
        log('test %d points, %d%% noise: %s' % (pts, noise, F.shape))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sets', default='train,validation,test')
    args = ap.parse_args()
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    steps = {'train': make_train, 'validation': make_validation, 'test': make_test}
    for s in args.sets.split(','):
        steps[s]()


if __name__ == '__main__':
    main()
