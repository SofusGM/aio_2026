# Project 1: Cars

## Requirements
Python 3.11 with `pandas numpy scipy matplotlib seaborn statsmodels linearmodels`. Pretty standard stuff!

```
pip install pandas numpy scipy matplotlib seaborn statsmodels linearmodels
```

## Generate all results
Only `src/` and `data/` are needed:

```
python src/run_me.py
```

All tables and figures are written to `res/`, which is created if it doesn't exist. Test output, such as the F-tests in `characteristics_test.py`, is printed in the terminal. Each script in `src/` can also be run on its own, but in the order listed in `run_me.py`.

Note: `market_size.py` adds predicted household counts to `data/cars.csv`, and `demand_models.py` writes `data/demand_model_parameters.csv`.
