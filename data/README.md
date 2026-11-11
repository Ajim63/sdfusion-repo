# Data

The CRSP and Compustat data used in the paper are licensed through WRDS and
cannot be redistributed, so this folder ships empty. Place your own files here.
Everything in this folder except this README is ignored by git.

## Expected files

All three files are in long format with one header row.

**`characteristics.csv`** - one row per firm-month

| column | meaning |
|---|---|
| `permno` | firm identifier |
| `date` | month the characteristics are observed (`1980-01`, `1980-01-31`, `198001` and `19800131` are all accepted) |
| one column per characteristic | raw values; missing entries left empty |
| optional `industry`, size column | used only to order firms in the cross-sectional scan |

**`returns.csv`** - one row per firm-month

| column | meaning |
|---|---|
| `permno` | firm identifier |
| `date` | month in which the return is realised |
| `ret` | monthly excess return (return minus the one-month T-bill rate) |

**`macro.csv`** - one row per month

| column | meaning |
|---|---|
| `date` | month |
| one column per macroeconomic variable | raw values |

Column names are configurable under `data:` in the YAML config.

## Timing

Characteristics and macro variables dated month *t* are paired with the return
of month *t+1*. They must therefore be known by the end of month *t* (apply your
own reporting lags to accounting variables before writing the file). If your
returns file already stores the next-month return on the row dated *t*, set
`returns_are_forward: true`.

## Trying the code without WRDS access

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python train.py --config configs/smoke.yaml
python evaluate.py --run outputs/smoke
```

The synthetic data are random numbers in the right layout. They are for checking
that the pipeline runs and carry no economic meaning.
