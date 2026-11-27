# Data

The data are not in this repository (about 10 GB). They are the simulated kinetic experiments released with

> Burés J, Larrosa I (2023). Organic reaction mechanism classification using machine learning. *Nature* 613, 689–695.
> doi:10.1038/s41586-022-05639-4

Download the code and data archive of that paper (doi:10.48420/16965271) and put these nine files in `data/raw/`:

```
x1_train_M1_M20_train_val_test_set.pkl   x2_train_M1_M20_train_val_test_set.pkl   y_train_M1_M20_train_val_test_set.pkl
x1_val_M1_M20_train_val_test_set.pkl     x2_val_M1_M20_train_val_test_set.pkl     y_val_M1_M20_train_val_test_set.pkl
x1_test_M1_M20_train_val_test_set.pkl    x2_test_M1_M20_train_val_test_set.pkl    y_test_M1_M20_train_val_test_set.pkl
```

If the files are somewhere else, point the code at them instead of copying:

```powershell
$env:KMC_RAW_DIR = "D:\path\to\the\pickles"
```

| Folder | Contents |
|---|---|
| `raw/` | the pickles above, never modified |
| `processed/` | feature matrices written by `python -m kinetic_mech.make_features` (about 2.8 GB) |

Each experiment has four runs of substrate S and product P over time: runs 1–3 at rising catalyst loading
(1–10 mol%) and run 4, the same-excess run. 4,950,000 training, 50,000 validation and 100,000 test experiments;
every mechanism (M1–M20) is 5% of each split.
