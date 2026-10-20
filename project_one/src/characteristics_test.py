"""Joint robust F-test of the car characteristics in a log-price regression with market-year FE."""

# %% Load packages
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
import parts

DATA = parts.DATA
MARKET_SIZE = 'hh'

# %% Read in data and build the variables used in demand_models.py

cars = pd.read_csv(DATA / 'cars.csv')
cars['ma_ye'] = cars['ma'].astype(str) + '_' + cars['ye'].astype(str) # market-year identifier for fixed effects
cars['hpwt'] = cars['hp'] / cars['we'] * 100   # performance: kW per 100 kg (as in BLP)
cars['size'] = cars['le'] * cars['wi'] / 1e4   # size: length x width in m^2 (as in BLP)
cars['market_size'], cars['s'], cars['s0'] = parts.market_size_function(data=cars, type=MARKET_SIZE)

# %% Joint F-test of the characteristics, robust standard errors

for y in ['np.log(eurpr)',]:
    m = smf.ols(f'{y} ~ home + hpwt + size + li + C(ma_ye)', data=cars).fit(cov_type='HC1')
    print(y, m.f_test('home = 0, hpwt = 0, size = 0, li = 0'))

# %%
