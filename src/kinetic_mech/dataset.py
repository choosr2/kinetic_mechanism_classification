"""Load the Burés & Larrosa (Nature 2023) dataset and sample it as the authors did.

Each experiment ("sample") has
    x1  (4,)       catalyst loading [cat]0 of runs 1-4 (mole fraction; 0.01-0.10)
    x2  (21, 12)   21 time points (row 0 is t = 0) x [t, S, P] for run 1 | run 2 | run 3 | run 4
    y   (1,)       the mechanism, 0-19 (M1-M20)
Runs 1-3 start with S = 1, P = 0 at rising catalyst loading; run 4 is the same-excess run (S0 + P0 = 1).
Times and concentrations are normalised within each experiment.

Training and validation are cut into batches. Every batch keeps a random number of time points (the same rows for
the whole batch) and gets a random noise level, as in the authors' BatchGenerator. The test sets were released
already sampled and noised.
"""
import pickle
import random

import numpy as np

from . import config as C

SUFFIX = '_M1_M20_train_val_test_set.pkl'


def raw_path(name):
    return C.RAW / (name + SUFFIX)


def load_pickle(name):
    """x1_train, x2_val, y_test, ... as stored by the authors."""
    with open(raw_path(name), 'rb') as f:
        return pickle.load(f)


def memmap_x2_train(n_samples):
    """x2_train is about 10 GB, so it is memory-mapped instead of loaded. numpy pickles hold the raw buffer as one
    BINBYTES8 opcode (0x8e + 8-byte length); the bytes after it are the array."""
    nbytes = n_samples * 21 * 12 * 8
    with open(raw_path('x2_train'), 'rb') as f:
        head = f.read(1024)
    key = b'\x8e' + nbytes.to_bytes(8, 'little')
    pos = head.find(key)
    if pos < 0:
        raise ValueError('could not find the array inside x2_train')
    return np.memmap(raw_path('x2_train'), dtype=np.float64, mode='r', offset=pos + len(key),
                     shape=(n_samples, 21, 12))


def load_train():
    """x1 (n, 4), memory-mapped x2 (n, 21, 12) and labels (n,) of the 4,950,000 training experiments."""
    y = np.asarray(load_pickle('y_train')).ravel().astype(int)
    return np.asarray(load_pickle('x1_train')), memmap_x2_train(len(y)), y


def load_test(pts, noise):
    """One released test set: x1, x2 and labels of its 100,000 experiments."""
    x2 = load_pickle('x2_test')[pts][noise]
    return np.asarray(load_pickle('x1_test')), np.asarray(x2), np.asarray(load_pickle('y_test')).ravel().astype(int)


# --------------------------------------------------------------------------------------------- the authors' sampler
def batch_draws(n_samples, batch, seed, noise_seed=None, n_rows=21):
    """One pass of the authors' BatchGenerator.

    Returns, for each full batch, (rows kept, noise level %), and one Gaussian noise block per noise level. Draws are
    made in the authors' order from np.random.default_rng(seed): number of points, which of the 20 later rows (t = 0
    is always kept), then the noise level. A last partial batch is dropped, as in the authors' code. The noise blocks
    (batch x 20 points x 8 concentration columns) are reused by every batch with that level."""
    gauss = random.Random(seed if noise_seed is None else noise_seed).gauss
    width = len(C.CONC_COLS)
    blocks = {sd: np.reshape([gauss(0, sd / 100) for _ in range(batch * max(C.TPS) * width)],
                             (batch, max(C.TPS), width))
              for sd in C.NOISE if sd != 0}
    rs = np.random.default_rng(seed)
    draws = []
    for _ in range(n_samples // batch):
        n = rs.choice(C.TPS)
        rows = np.sort(np.append(rs.choice(n_rows - 1, n, replace=False) + 1, [0]))
        sd = rs.choice(C.NOISE)
        draws.append((rows, sd))
    return draws, blocks


def apply_draw(x2_batch, rows, sd, blocks):
    """Keep the batch's rows and add its noise block to S and P (not to t = 0). No clipping, as in the paper."""
    b = np.array(x2_batch[:, rows])
    if sd != 0:
        b[:, 1:, C.CONC_COLS] += blocks[sd][:len(b), :len(rows) - 1]
    return b


def training_draws(n_samples=4_950_000):
    """Points and noise of the 2,416 training batches of 2,048."""
    return batch_draws(n_samples, C.BATCH, C.TRAIN_RNG_SEED, noise_seed=C.TRAIN_NOISE_SEED)


def sampled_validation():
    """The 50,000 validation experiments cut into batches of 25 and sampled as in the authors' train.py.
    Returns x1, a list of (batch index, sampled x2 batch) and labels, plus each experiment's points and noise."""
    x1, x2 = np.asarray(load_pickle('x1_val')), np.asarray(load_pickle('x2_val'))
    y = np.asarray(load_pickle('y_val')).ravel().astype(int)
    draws, blocks = batch_draws(len(x2), C.VAL_BATCH, C.VAL_SEED)
    batches = [(i, apply_draw(x2[i * C.VAL_BATCH:(i + 1) * C.VAL_BATCH], rows, sd, blocks))
               for i, (rows, sd) in enumerate(draws)]
    n_pts = np.repeat([len(r) - 1 for r, _ in draws], C.VAL_BATCH)
    noise = np.repeat([float(sd) for _, sd in draws], C.VAL_BATCH)
    return x1, batches, y[:len(n_pts)], n_pts, noise
