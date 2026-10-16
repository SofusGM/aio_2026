""" 
Cars: Demand Models
This script estimates two demand models: a simple logit and a 1-level nested logit model of the
consumer demand for cars (same as demand_models.py, but with one nest level instead of two).
"""

# %% Load packages
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from pathlib import Path
import seaborn as sns 
import parts
sns.set_theme()

DATA = Path('../data') # notebooks run from their own folder (src/)
RES = Path('../res/demand models')
RES.mkdir(parents=True, exist_ok=True)

# BASELINE DECISIONS

# 0. Market size
MARKET_SIZE = 'hh'

# 1. Price variable
PRICE_VAR = 'eurpr'     # choose between 'pr', price-to-income `princ`, or price in common currency `eurpr`
LOG_PRICE = True        # choose between True or False

# 2. Controls and fixed effects
CONTROLS = 'home + hpwt + size + li'   # domestic bias, performance (kW/100kg), size (m^2), fuel use: one measure each, as hp, we, le, wi, cy, sp are highly collinear
FE = 'ma_ye + brd'    # market-year fixed effects (column built from ma and ye after reading data)

# 3. Instrument(s)
INSTRUMENTS = 'log_count_segment + distance_segment_hpwt + distance_segment_size + distance_segment_li'   # choose from 'log_exchange_rate', 'log_producer_prices', 'log_producer_prices_sdr', 'sum_weight_subgroup', 'log_count_subgroup', 'log_count_segment', 'distance_segment_<x>' (see parts.instruments_function)
INSTRUMENTS_SIMPLE = 'log_exchange_rate + sum_weight_subgroup' # Select instruments for simple logit here
INSTRUMENTS_ON = True  # choose between True or False

# 4. Substitution patterns (nest)
NEST = 'cla'
SUBGROUP = 'home'   # not part of the model: only used to build instruments such as log_count_subgroup and sum_weight_subgroup

# %% Read in data

cars = pd.read_csv(DATA / 'cars.csv') # this reads the *balanced* dataset (i.e. J = 40 products per market always)
lbl_vars = pd.read_csv(DATA / 'labels_variables.csv', index_col=0) # labels on variables (same as in project1.pdf)
lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels() # the values that variables take (not relevant for all)

cars['ma_ye'] = cars['ma'].astype(str) + '_' + cars['ye'].astype(str) # market-year identifier for fixed effects
cars['hpwt'] = cars['hp'] / cars['we'] * 100   # performance: kW per 100 kg (as in BLP)
cars['size'] = cars['le'] * cars['wi'] / 1e4   # size: length x width in m^2 (as in BLP)

print(cars.shape)

# %%  Estimate market size, shares and outside share

cars['market_size'], cars['s'], cars['s0'] = parts.market_size_function(data=cars, type=MARKET_SIZE)

print(cars.columns)

# %%  Price variable used in mean utility

PRICE = f'log_{PRICE_VAR}' if LOG_PRICE else PRICE_VAR   # price regressor
if LOG_PRICE:
    cars[PRICE] = np.log(cars[PRICE_VAR])

# %% Construct right hand side in regression

FE_TERMS = ' + '.join(f'C({f.strip()})' for f in FE.split('+'))  # one set of dummies per fixed effect

if INSTRUMENTS_ON:
    rhs = f'1 + {CONTROLS} + {FE_TERMS} + [{PRICE} ~ {INSTRUMENTS_SIMPLE}]'
    rhs_nested = f'1 + {CONTROLS} + {FE_TERMS} + [{PRICE} + ln_s_j_g ~ {INSTRUMENTS}]'
else:
    rhs = f'1 + {PRICE} + {CONTROLS} + {FE_TERMS}'
    rhs_nested = f'1 + {PRICE} + {CONTROLS} + {FE_TERMS} + [ln_s_j_g ~ {INSTRUMENTS}]'
    
# %% Actually compute the instruments

INSTR = [t.strip() for t in INSTRUMENTS.split('+')]
INSTR_SIMPLE = [t.strip() for t in INSTRUMENTS_SIMPLE.split('+')]
for t in dict.fromkeys(INSTR + INSTR_SIMPLE):   # every instrument used by either model, once
    cars[t] = parts.instruments_function(cars, t, NEST, SUBGROUP)  # stored in cars so the rhs formula can use them

# %% Compute nested logit with one level

# estimation sample: rows with all controls and the instruments of both models observed (same sample for both)
sample = cars[INSTR + INSTR_SIMPLE + [c.strip() for c in CONTROLS.split('+')]].notna().all(axis=1).values

# ln(s_j/s_0) = x_j beta - alpha p_j + rho ln(s_{j|g}) + xi_j is linear, so 2SLS gives
# beta, alpha and rho together with SEs that account for estimating rho (shares from the full data)
cars['s_g'] = cars.groupby(['ma', 'ye', NEST])['s'].transform('sum')
cars['ln_s_j_g'] = np.log(cars['s'] / cars['s_g'])   # product share within its nest -> rho
cars['delta'] = np.log(cars['s']) - np.log(cars['s0'])

fit_1level = IV2SLS.from_formula(f'delta ~ {rhs_nested}', data=cars.loc[sample]).fit()
rho_hat = fit_1level.params['ln_s_j_g']
print(fit_1level.params[[k for k in fit_1level.params.index if not k.startswith('C(')]])

# %% Compute simple logit with the same sample

fit_simple = IV2SLS.from_formula(f'delta ~ {rhs}', data=cars.loc[sample]).fit()
print(fit_simple.params[[k for k in fit_simple.params.index if not k.startswith('C(')]])

# %% First-stage F-tests: are the excluded instruments jointly relevant for each endogenous regressor? (rule of thumb F > 10)
# Robust first-stage Wald statistics are chi2(#instruments), so they are divided by #instruments to get F.
# With two endogenous regressors in the nested logit, each F ignores how the instruments are shared across them.

first_stage_fits = ({'Simple logit': (fit_simple, INSTR_SIMPLE)} if INSTRUMENTS_ON else {}) | {'Nested logit': (fit_1level, INSTR)}
for name, (fit, instr) in first_stage_fits.items():
    diag = fit.first_stage.diagnostics
    diag['F'] = diag['f.stat'] / len(instr)
    print(f'\n{name} first stage ({", ".join(instr)}):')
    print(diag[['partial.rsquared', 'F', 'f.pval']].round(3))

# %% Instrument grid: every combination from GRID_POOL, same controls, FE and nest as above
# Fast 2SLS with the exogenous regressors (constant, controls, FE dummies) partialled out (Frisch-Waugh), robust SEs.
# Nested logit: 2-4 instruments for (price, ln_s_j_g); simple logit: 1-3 instruments for price.
# ok = alpha < 0 and significant, 0 < rho < 1 (nested), first-stage F > 10, overidentification test not rejected.

from itertools import combinations

CHARS = [c.strip() for c in CONTROLS.split('+') if c.strip() != 'home']  # characteristics for the differentiation IVs
GRID_POOL = (['log_exchange_rate', 'log_producer_prices', 'log_producer_prices_sdr', 'sum_weight_subgroup', 'log_count_subgroup',
              'log_count_segment'] + [f'distance_segment_{x}' for x in CHARS])
CTRL = [c.strip() for c in CONTROLS.split('+')]
for t in GRID_POOL:
    cars[t] = parts.instruments_function(cars, t, NEST, SUBGROUP)

def tsls(y, X, Z):
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    bread = np.linalg.inv(Xh.T @ Xh)
    b = bread @ Xh.T @ y
    e = y - X @ b
    return b, np.sqrt(np.diag(bread @ ((Xh * e[:, None]).T @ (Xh * e[:, None])) @ bread)), e

partialled = {}   # one partialling per estimation sample (instruments with missing values change the sample)
def grid_data(combo):
    key = tuple(t for t in combo if cars[t].isna().any())
    if key not in partialled:
        m = cars[CTRL + list(key)].notna().all(axis=1).values
        d = cars[m]
        W = np.column_stack([np.ones(len(d)), d[CTRL].values,
                             pd.get_dummies(d[[f.strip() for f in FE.split('+')]].astype(str), drop_first=True, dtype=float).values])
        cols = ['delta', PRICE, 'ln_s_j_g'] + GRID_POOL
        V = d[cols].fillna(0).values   # NaN only in instruments outside this combo
        partialled[key] = (m, pd.DataFrame(V - W @ np.linalg.lstsq(W, V, rcond=None)[0], columns=cols), np.linalg.matrix_rank(W))
    return partialled[key]

grid = []
for model, endog, sizes in [('nested', [PRICE, 'ln_s_j_g'], (2, 3, 4)), ('simple', [PRICE], (1, 2, 3))]:
    for combo in (cb for r in sizes for cb in combinations(GRID_POOL, r)):
        m, D, kw = grid_data(combo)
        y, X, Z = D['delta'].values, D[endog].values, D[list(combo)].values
        b, se, e = tsls(y, X, Z)
        rho = b[1] if model == 'nested' else 0.0
        r = 1 / (1 - rho)
        own = b[0] * (r - (r - 1) * cars.loc[m, 's'] / cars.loc[m, 's_g'] - cars.loc[m, 's'])   # own-price elasticity
        F = parts.first_stage_F(X, Z, kw)
        grid.append(dict(model=model, instruments=' + '.join(combo), nobs=int(m.sum()), dropped=len(cars) - int(m.sum()),
                         alpha=b[0], alpha_se=se[0],
                         rho=b[1] if model == 'nested' else np.nan, rho_se=se[1] if model == 'nested' else np.nan,
                         F_price=F[0], F_rho=F[1] if model == 'nested' else np.nan, F_min=min(F),
                         overid_p=parts.overid_pval(y, X, Z, e), elast_median=own.median()))

grid = pd.DataFrame(grid)
grid['ok'] = ((grid.alpha / grid.alpha_se < -1.96) & (grid.F_min > 10) & ~(grid.overid_p < 0.05)
              & ((grid.model == 'simple') | ((grid.rho > 0) & (grid.rho < 1))))
for model in ['nested', 'simple']:
    g = grid[grid.model == model].sort_values(['ok', 'F_min'], ascending=False)
    g.to_csv(RES / f'instrument_grid_{model}_2.csv', index=False)
    print(f'\n{model}: {g.ok.sum()} of {len(g)} combinations ok')
    print(g[g.ok].head(15).round(3).to_string(index=False))

# %% Summary across runs of the ok regressions only: alphas, rhos and F-tests; the simple and nested logit always share the same controls.
# Rerunning with other CONTROLS replaces only the rows for that control set, so the file builds up over runs.

summary_file = RES / 'instrument_grid_summary_2.csv'
summary = grid[grid.ok].assign(controls=CONTROLS)[['controls', 'model', 'instruments', 'alpha', 'alpha_se', 'rho', 'rho_se', 'F_price', 'F_rho']]
if summary_file.exists():
    old = pd.read_csv(summary_file)
    summary = pd.concat([old[old.controls != CONTROLS], summary], ignore_index=True)
summary.to_csv(summary_file, index=False)

# %% Matched pairs of ok regressions: every simple-logit instrument is also used in the nested logit (the nested may add more)

ok_simple, ok_nested = grid[grid.ok & (grid.model == 'simple')], grid[grid.ok & (grid.model == 'nested')]
pairs_ok = [dict(controls=CONTROLS, instruments_simple=s.instruments, instruments_nested=n.instruments,
                 alpha_simple=s.alpha, alpha_simple_se=s.alpha_se, F_price_simple=s.F_price,
                 alpha_nested=n.alpha, alpha_nested_se=n.alpha_se, rho=n.rho, rho_se=n.rho_se,
                 F_price_nested=n.F_price, F_rho=n.F_rho)
            for s in ok_simple.itertuples() for n in ok_nested.itertuples()
            if set(s.instruments.split(' + ')) <= set(n.instruments.split(' + '))]

matched_file = RES / 'instrument_grid_matched_2.csv'
matched = pd.DataFrame(pairs_ok, columns=['controls', 'instruments_simple', 'instruments_nested', 'alpha_simple', 'alpha_simple_se',
                                          'F_price_simple', 'alpha_nested', 'alpha_nested_se', 'rho', 'rho_se', 'F_price_nested', 'F_rho'])
if matched_file.exists():
    old = pd.read_csv(matched_file)
    matched = pd.concat([old[old.controls != CONTROLS], matched], ignore_index=True)
matched.to_csv(matched_file, index=False)
print(f'\n{len(pairs_ok)} matched pairs of ok regressions (simple instruments a subset of nested instruments)')

# %%
