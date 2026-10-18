import pandas as pd
import numpy as np
from pathlib import Path
import parts
import matplotlib.pyplot as plt

DATA = Path('project_one/data') 
RES = Path('project_one/res/merger')
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


#define elasticityies from param values
def elasticity(pairs, alpha, rho):
    r = 1 / (1 - rho)
    own = pairs['co_j'] == pairs['co_k']
    same_segment = pairs[f'{SEGMENT}_j'] == pairs[f'{SEGMENT}_k']
    dp = 1 if LOG_PRICE else pairs[PRICE_VAR]

    return -alpha * dp * (r * own - (r - 1) * same_segment * pairs['s'] / pairs['segment_share']  - pairs['s'])

results_list = []


#divesion ratioes between VW and Fiat in each market year_YE:

#Setup
firm_codes = {name: code for code, name in lbl_vals['firm'].items()}
VW, FIAT = firm_codes['VW'], firm_codes['Fiat']
MODELS = {'simple logit': (alpha_simple, 0), 'extended logit': (alpha_extended, rho_extended)}

cols = ['ma', 'ye', 'co', 'frm', SEGMENT]
year = cars[cars['ye'] == MARKET_YE]

# every pair (j, k) in the same market; s and segment_share belong to k, s_j to the responding car j
pairs = year[cols + ['s']].rename(columns={'s': 's_j'}).merge(
    year[cols + ['s', 'segment_share', PRICE_VAR]], on=['ma', 'ye'], suffixes=('_j', '_k'))

# define diversion
def diversion(p, from_firm, to_firm):
    price_up = p['frm_k'] == from_firm                  # the cars whose price rises
    change = p['E'] * p['s_j']                          # change in s_j from a 1% rise in p_k
    lost = -change[price_up & (p['frm_j'] == from_firm)].sum()
    gained = change[price_up & (p['frm_j'] == to_firm)].sum()
    return gained / lost

for model, (alpha, rho) in MODELS.items():
    pairs['E'] = elasticity(pairs, alpha, rho)
    for ma, p in pairs.groupby('ma'):
        results_list.append({'market': lbl_vals['market'][ma], 'model': model,
                             'VW to Fiat': 100 * diversion(p, VW, FIAT),
                             'Fiat to VW': 100 * diversion(p, FIAT, VW)})


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
          r'\multicolumn{5}{p{0.85\linewidth}}{\footnotesize \textit{Note:} Share of the sales the first firm loses '
          r'that go to the second firm when it raises the price of all its cars by 1\%.} \\',
          r'\end{tabular}', r'\end{table}']
(RES / 'diversion_ratios.tex').write_text('\n'.join(lines) + '\n')



