# Data

NSL-KDD is **not** committed to git. It's ~21 MB raw, and it's a standard public
dataset you can re-download any time.

## Layout

- `raw/` - the original `.txt` files as released by UNB
  (KDDTrain+.txt, KDDTest+.txt). Downloaded automatically on first run.
- `processed/` - cleaned CSVs produced by `main.py` (labels parsed, difficulty
  column dropped).

## Get it

Run the pipeline once:

```bash
python main.py
```

`src/data/load.py` downloads both files into `raw/` from the
[HoaNP/NSL-KDD-DataSet](https://github.com/HoaNP/NSL-KDD-DataSet) mirror when
they're missing.

## Columns

41 features + `label` + `difficulty`. Feature groups:

- basic TCP connection features (duration, bytes, protocol, service, flag)
- content features (failed logins, root shell, etc.)
- traffic features aggregated per host / per service (count, error rates)

Attack labels map to 4 families: DoS, Probe, R2L, U2R.