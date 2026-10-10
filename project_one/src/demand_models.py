""" 
Cars: Demand Models
This script estimates two demand models: a simple logit and and extendend logit model of the
consumer demand for cars.
"""

# %% Load packages
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
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
MARKET_SIZE = 'pop' # choose between whole population divided by 3 'pop' or others.

# 1. Price variable
PRICE_VAR = 'eurpr'     # choose between 'pr', price-to-income `princ`, or price in common currency `eurpr`
LOG_PRICE = True        # choose between True or False

# 2. Controls and fixed effects
CONTROLS = 'cy + hp + we + le + li + sp'
FE = 'co'

# 3. Instrument(s)
INSTRUMENTS = 'log_xexr + we_rival + n_sub'   # choose from 'log_xexr', 'hausman', 'we_rival', 'n_sub', 'n_nest' (see parts.instruments_function)
INSTRUMENTS_ON = True  # choose between True or False

# 4. Substitution patterns (nests)
NEST = 'home'
SUBGROUP = 'cla'

# %% Read in data

cars = pd.read_csv(DATA / 'cars.csv') # this reads the *balanced* dataset (i.e. J = 40 products per market always)
lbl_vars = pd.read_csv(DATA / 'labels_variables.csv', index_col=0) # labels on variables (same as in project1.pdf)
lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels() # the values that variables take (not relevant for all)

print(cars.shape)

# %%  Estimate market size, shares and outside share

cars['market_size'], cars['s'], cars['s0'] = parts.market_size_function(data=cars, type=MARKET_SIZE)

print(cars.columns)

# %%  Price variable used in mean utility

if LOG_PRICE:
    cars['log_'+PRICE_VAR] = np.log(cars[PRICE_VAR]) 

# %% Construct right hand side in regression

if INSTRUMENTS_ON:
    rhs = f'1 + {CONTROLS} + C({FE}) + [log_{PRICE_VAR} ~ {INSTRUMENTS}]'
else:
    rhs = f'1 + log_{PRICE_VAR} + {CONTROLS} + C({FE})'
    
# %% Actually compute the instruments

INSTR = [t.strip() for t in INSTRUMENTS.split('+')]
for t in INSTR:
    cars[t] = parts.instruments_function(cars, t, NEST, SUBGROUP)  # stored in cars so the rhs formula can use them

# %% Compute nested logit with two levels

# estimation sample: rows with all controls and instruments observed
sample = cars[INSTR + [c.strip() for c in CONTROLS.split('+')]].notna().all(axis=1).values

# X = regressors, Z = exogenous regressors + instruments (delta is a placeholder to parse the formula)
mod = IV2SLS.from_formula(f'delta ~ {rhs}', data=cars[sample].assign(delta=0.0))
X = np.column_stack([mod.exog.ndarray, mod.endog.ndarray])
Z = np.column_stack([mod.exog.ndarray, cars.loc[sample, INSTR].values])
W = np.linalg.inv(Z.T @ Z)

res = minimize(parts.Q_nested, x0=[0.5, 0.3], args=(cars, X, Z, W, NEST, SUBGROUP, sample),
               method='L-BFGS-B', bounds=[(0.0, 0.95)] * 2)
print(res)
rho_hat = res.x


# %% Linear parameters at rho_hat
cars['delta'] = parts.invert_demand_nest_2levels(cars, rho_hat, NEST, SUBGROUP)
fit_2levels = IV2SLS.from_formula(f'delta ~ {rhs}', data=cars.loc[sample]).fit()

theta1 = fit_2levels.params[[k for k in fit_2levels.params.index if not k.startswith('C(')]]
est_2 = pd.concat([theta1, pd.Series(rho_hat, index=['rho_1', 'rho_2'])]).to_frame('Estimate')
print(est_2)

# %% Compute simple logit with the same sample

cars['delta'] = np.log(cars['s']) - np.log(cars['s0'])
fit_simple = IV2SLS.from_formula(f'delta ~ {rhs}', data=cars.loc[sample]).fit()
fit_2levels.params

# %% Save the two models side by side in a LaTeX table

esc = lambda v: v.replace('_', r'\_')  # variable names as they are, escaped for LaTeX

rows = [f'log_{PRICE_VAR}' if LOG_PRICE else PRICE_VAR] + [c.strip() for c in CONTROLS.split('+')]
fits = [fit_simple, fit_2levels]

tex = [r'\begin{table}[htbp]', r'\centering', r'\caption{Demand estimates}', r'\label{tab:demand_models}',
       r'\begin{tabular}{lcc}', r'\toprule', r' & Simple logit & Nested logit \\', r'\midrule']
for k in rows:
    tex.append(f'{esc(k)} & ' + ' & '.join(f'{f.params[k]:.3f}' for f in fits) + r' \\')
    tex.append(' & ' + ' & '.join(f'({f.std_errors[k]:.3f})' for f in fits) + r' \\')
tex += [rf'$\rho_1$ ({SUBGROUP}) & & {rho_hat[0]:.3f} \\',
        rf'$\rho_2$ ({NEST}) & & {rho_hat[1]:.3f} \\',
        r'\midrule',
        rf'{FE} FE & $\checkmark$ & $\checkmark$ \\',
        rf'Price instrumented & {"Yes" if INSTRUMENTS_ON else "No"} & {"Yes" if INSTRUMENTS_ON else "No"} \\',
        rf'Observations & {fit_simple.nobs} & {fit_2levels.nobs} \\',
        r'\bottomrule',
        r'\multicolumn{3}{p{0.75\linewidth}}{\footnotesize \textit{Note:} Robust standard errors in parentheses. '
        rf'Dependent variable is delta, market size = {MARKET_SIZE}. '
        rf'Nested logit nests on {NEST} and {SUBGROUP}; $\rho$ estimated by GMM with '
        + ', '.join(esc(z) for z in INSTR) + r' as instruments.} \\',
        r'\end{tabular}', r'\end{table}']

(RES / 'demand_models.tex').write_text('\n'.join(tex) + '\n')

# %% Calculate own- and cross-price elasticity matrix

# elast_logit = pd.DataFrame(np.nan, index=range(1,J+1), columns=range(1,J+1))
# alpha = -res_logit.params[f'log_{PRICE_VAR}']
# for j in range(1, J+1): # response category
#     for k in range(1, J+1): # category with price change
#         s_k = cars.xs(k, level='j')['s'] # share of category k: (N,) array
#         if (j == k):
#             elast_ijk = -alpha * (1 - s_k) # own-price
#         else:
#             elast_ijk = alpha * s_k # cross-price

#         elast_logit.loc[j,k] = elast_ijk.mean() # average over observations

# # prettify
# elast_logit.index = elast_logit.index.map(rename_num2name)
# elast_logit.columns = elast_logit.columns.map(rename_num2name)
# sns.heatmap(elast_logit, cmap='coolwarm', annot=True, fmt=".1%",  annot_kws={"size": 6});
# plt.xlabel('Good k (has price change)'); plt.ylabel('Good j (response)');





# elast_simple = pd.DataFrame(results_ols1.params['p']*np.eye(J), index=labels, columns=labels)

# price_coefs = results_ols2.params[[f'C(j)[{j}]:p' for j in range(1, J+1)    ]]
# elast_2levels = pd.DataFrame(np.diag(price_coefs), index=labels, columns=labels)

# elast_ols3 = pd.DataFrame(tab_ols3.iloc[:, 1:].values, index=labels, columns=labels)




# Path('img').mkdir(exist_ok=True) # figures are written next to this notebook
# fig,axs = plt.subplots(3, 2, figsize=(8,10))

# for ax, elast, title in zip(axs.flatten(),
#                           [elast_ols1, elast_ols2],
#                           ['OLS 1', 'OLS 2', 'OLS 3', 'Logit', 'AIDS']):
#     # reset index and columns from names to integers
#     elast = elast.reset_index(drop=True).rename(columns={c: i for i,c in enumerate(elast.columns)})
#     sns.heatmap(elast, cmap='coolwarm', annot=True, fmt=".1%", annot_kws={"size": 5}, ax=ax);
#     ax.set_title(title);
# # remove axs[2,1
# axs[2,1].axis('off')
# plt.tight_layout();
# plt.savefig('demand models/elasticities_demand_models.pdf', bbox_inches='tight')