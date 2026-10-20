import pandas as pd
import numpy as np
from pathlib import Path
import parts
import matplotlib.pyplot as plt

DATA = parts.DATA
RES = parts.RES / 'diversionratios'
RES.mkdir(parents=True, exist_ok=True)


MARKET_SIZE = 'hh'
PRICE_VAR, LOG_PRICE = 'eurpr', True
SEGMENT = 'cla'
MARKET_YE = 99


cars = pd.read_csv(DATA / 'cars.csv')


# Calculate shares on the full dataset, before selecting a market
cars['market_size'], cars['s'], cars['s0'] = parts.market_size_function(data=cars, type=MARKET_SIZE)
cars['segment_share'] = (cars.groupby(['ma', 'ye', SEGMENT])['s'].transform('sum'))

lbl_vars = pd.read_csv(DATA / 'labels_variables.csv', index_col=0)['label']
lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels()



# load in parameters from estimation code
params = pd.read_csv(DATA / 'demand_model_parameters.csv', index_col=0)

alpha_simple = params.loc['alpha', 'Simple logit']
alpha_extended = params.loc['alpha', 'Extended logit']
rho_extended = params.loc['log_within_segment_share', 'Extended logit']


# elasticities, as in demand_models.py
def elasticity(pairs, alpha, rho):
    r = 1 / (1 - rho)
    own = pairs['co_j'] == pairs['co_k']
    same_segment = pairs[f'{SEGMENT}_j'] == pairs[f'{SEGMENT}_k']
    dp = 1 if LOG_PRICE else pairs[PRICE_VAR]

    return -alpha * dp * (r * own - (r - 1) * same_segment * pairs['s'] / pairs['segment_share']  - pairs['s'])

results_list = []

# %% Diversion ratios between VW and Fiat in each market, year MARKET_YE

# Setup
firm_codes = {name: code for code, name in lbl_vals['firm'].items()}
VW, FIAT = firm_codes['VW'], firm_codes['Fiat']
MODELS = {'simple logit': (alpha_simple, 0), 'extended logit': (alpha_extended, rho_extended)}

cols = ['ma', 'ye', 'co', 'frm', 'type', SEGMENT]
year = cars[cars['ye'] == MARKET_YE]

# every pair (j, k) in the same market; s and segment_share belong to k, s_j to the responding car j
pairs = year[cols + ['s']].rename(columns={'s': 's_j'}).merge(
    year[cols + ['s', 'segment_share', PRICE_VAR]], on=['ma', 'ye'], suffixes=('_j', '_k'))


# define diversion: the price of ONE car k (of from_firm) rises by 1%;
# D_k = share of k's lost sales that go to to_firm's cars = -(sum_{j in to_firm} dS_j) / dS_k
def product_diversion(p, from_firm, to_firm):
    change = p['E'] * p['s_j']                                            # change in s_j from a 1% rise in p_k
    own = change[p['co_j'] == p['co_k']].groupby(p['co_k']).sum()         # change in s_k itself (negative)
    to_firm_cars = change[p['frm_j'] == to_firm].groupby(p['co_k']).sum()
    cars_A = p[(p['frm_k'] == from_firm) & (p['co_j'] == p['co_k'])].set_index('co_k')
    D = -to_firm_cars.reindex(cars_A.index, fill_value=0) / own[cars_A.index]
    return pd.DataFrame({'type': cars_A['type_k'], 's': cars_A['s_j'], 'D': 100 * D})

product_rows = []
for model, (alpha, rho) in MODELS.items():
    pairs['E'] = elasticity(pairs, alpha, rho)
    for ma, p in pairs.groupby('ma'):
        res = {}
        for name, (A, B) in {'VW to Fiat': (VW, FIAT), 'Fiat to VW': (FIAT, VW)}.items():
            d = product_diversion(p, A, B)
            res[name] = np.average(d['D'], weights=d['s'])                # sales-weighted average over A's cars
            product_rows.append(d.assign(market=lbl_vals['market'][ma], model=model, direction=name))
        results_list.append({'market': lbl_vals['market'][ma], 'model': model, **res})

# diversion per car
products = pd.concat(product_rows)
products.to_csv(RES / 'diversion_by_car.csv', index=False)

# per-market summary table
diversion_table = pd.DataFrame(results_list).pivot(index='market', columns='model', values=['VW to Fiat', 'Fiat to VW'])
diversion_table = diversion_table.swaplevel(axis=1)[list(MODELS)]   # columns: model -> direction
print(diversion_table.round(2))


# %% LaTeX table
lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
         rf'\caption{{Diversion ratios between VW and Fiat, 19{MARKET_YE} (\%)}}', r'\label{tab:diversion}',
         r'\begin{tabular}{lcccc}', r'\toprule',
         r' & \multicolumn{2}{c}{Simple logit} & \multicolumn{2}{c}{Extended logit} \\',
         r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
         r'Market & VW $\to$ Fiat & Fiat $\to$ VW & VW $\to$ Fiat & Fiat $\to$ VW \\', r'\midrule']
for market, row in diversion_table.iterrows():
    lines.append(market + ' & ' + ' & '.join(f'{v:.2f}' for v in row.values) + r' \\')
lines += [r'\bottomrule',
          r'\multicolumn{5}{p{0.85\linewidth}}{\footnotesize \textit{Note:} Share of a car\textquoteright s lost sales that go to the other firm when its own price rises by 1\%, '
          r'averaged over the cars of the first firm with sales weights.} \\',
          r'\end{tabular}', r'\end{table}']
(RES / 'diversion_ratios.tex').write_text('\n'.join(lines) + '\n')