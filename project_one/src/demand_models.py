"""Simple logit and extended logit (nested by segment), both estimated by 2SLS."""

# %% Load packages
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import parts
sns.set_theme()

DATA = parts.DATA
RES = parts.RES / 'demand models'
RES.mkdir(parents=True, exist_ok=True)

# BASELINE DECISIONS

# 0. Market size
MARKET_SIZE = 'hh'

# 1. Price variable
PRICE_VAR, LOG_PRICE = 'eurpr', True   # price in common currency, in logs

# 2. Controls and fixed effects
CONTROLS = 'home + hpwt + size + li'   # one measure each; hp, we, le, wi, cy, sp are highly collinear
FE = 'ma_ye + brd'                     # market-year and brand fixed effects

# 3. Substitution patterns
SEGMENT = 'cla'      # the nest
SUBGROUP = 'home'    # only used to build subgroup instruments, not part of the model

# 4. Instruments (see parts.instruments_function)
INSTRUMENTS_SIMPLE   = 'log_exchange_rate'
INSTRUMENTS_EXTENDED = 'log_exchange_rate + log_count_segment + distance_segment_hpwt + distance_segment_size'
INSTRUMENTS_SIMPLE_ROBUST = 'log_exchange_rate + log_count_segment + distance_segment_hpwt + distance_segment_size'
# %% Read in data

cars = pd.read_csv(DATA / 'cars.csv') # balanced: 40 products per market-year
lbl_vars = pd.read_csv(DATA / 'labels_variables.csv', index_col=0)
lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels()

cars['ma_ye'] = cars['ma'].astype(str) + '_' + cars['ye'].astype(str) # market-year identifier for fixed effects
cars['hpwt'] = cars['hp'] / cars['we'] * 100   # performance: kW per 100 kg (as in BLP)
cars['size'] = cars['le'] * cars['wi'] / 1e4   # size: length x width in m^2 (as in BLP)

print(cars.shape)

# %%  Estimate market size, shares and outside share

cars['market_size'], cars['s'], cars['s0'] = parts.market_size_function(data=cars, type=MARKET_SIZE)

# %%  Price variable used in mean utility

PRICE = f'log_{PRICE_VAR}' if LOG_PRICE else PRICE_VAR   # price regressor
if LOG_PRICE:
    cars[PRICE] = np.log(cars[PRICE_VAR])

# %% Dependent variable and within-segment share

cars['segment_share'] = cars.groupby(['ma', 'ye', SEGMENT])['s'].transform('sum')
cars['log_within_segment_share'] = np.log(cars['s'] / cars['segment_share'])   # its coefficient is rho
cars['log_share_ratio'] = np.log(cars['s']) - np.log(cars['s0'])

# %% Instruments and estimation sample

split = lambda terms: [t.strip() for t in terms.split('+')]
CTRL, FE_COLS = split(CONTROLS), split(FE)
INSTR = {'simple': split(INSTRUMENTS_SIMPLE), 'extended': split(INSTRUMENTS_EXTENDED),
         'simple_robust': split(INSTRUMENTS_SIMPLE_ROBUST)}
ALL_INSTR = list(dict.fromkeys(sum(INSTR.values(), [])))
for t in ALL_INSTR:
    cars[t] = parts.instruments_function(cars, t, SEGMENT, SUBGROUP)  # stored in cars so the formulas can use them

# same sample for both models: rows with all controls and instruments observed
sample = cars[CTRL + ALL_INSTR].notna().all(axis=1).values
data = cars.loc[sample]

# %% Estimate the models

FE_TERMS = ' + '.join(f'C({f})' for f in FE_COLS)  # one set of dummies per fixed effect
EXOG = f'1 + {CONTROLS} + {FE_TERMS}'

fit_simple = IV2SLS.from_formula(f'log_share_ratio ~ {EXOG} + [{PRICE} ~ {INSTRUMENTS_SIMPLE}]', data=data).fit()
fit_extended = IV2SLS.from_formula(f'log_share_ratio ~ {EXOG} + [{PRICE} + log_within_segment_share ~ {INSTRUMENTS_EXTENDED}]', data=data).fit()
fit_simple_robust = IV2SLS.from_formula(f'log_share_ratio ~ {EXOG} + [{PRICE} ~ {INSTRUMENTS_SIMPLE_ROBUST}]', data=data).fit() # all instruments

alpha_simple = - fit_simple.params[PRICE]       # alpha > 0 means demand falls with price
alpha_extended = - fit_extended.params[PRICE]
rho_extended = fit_extended.params['log_within_segment_share']

for name, fit in [('Simple logit', fit_simple), ('Extended logit', fit_extended)]:
    print(f'\n{name}:')
    print(fit.params[[k for k in fit.params.index if not k.startswith('C(')]])


# save parameters for diversionratios.py and merger.py
def reported_params(fit, alpha):
    params = fit.params[[k for k in fit.params.index if not k.startswith('C(')]].copy()
    params['alpha'] = alpha
    return params

results = pd.concat(
    {
        'Simple logit': reported_params(fit_simple, alpha_simple),
        'Extended logit': reported_params(fit_extended, alpha_extended),
        'Simple logit (robustness)': reported_params(fit_simple_robust, - fit_simple_robust.params[PRICE]),
    },
    axis=1
)

results.to_csv(DATA / 'demand_model_parameters.csv')

# %% Diagnostics: first-stage F (rule of thumb > 10), exogenous regressors partialled out

W = np.column_stack([np.ones(len(data)), data[CTRL].values,
                     pd.get_dummies(data[FE_COLS].astype(str), drop_first=True, dtype=float).values])
partial = lambda cols: data[cols].values - W @ np.linalg.lstsq(W, data[cols].values, rcond=None)[0]
kw = np.linalg.matrix_rank(W)

def iv_diagnostics(endog, instruments):
    return parts.first_stage_F(partial(endog), partial(instruments), kw)

ENDOG_SIMPLE, ENDOG_EXTENDED = [PRICE], [PRICE, 'log_within_segment_share']
diag_simple = iv_diagnostics(ENDOG_SIMPLE, INSTR['simple'])
diag_extended = iv_diagnostics(ENDOG_EXTENDED, INSTR['extended'])
diag_simple_robust = iv_diagnostics(ENDOG_SIMPLE, INSTR['simple_robust'])

print(f'\nSimple logit: first-stage F (price) = {diag_simple[0]:.1f}')
print(f'Extended logit: Sanderson-Windmeijer F price = {diag_extended[0]:.1f}, '
      f'within-segment share = {diag_extended[1]:.1f}')
print(f'Simple logit (robustness): first-stage F (price) = {diag_simple_robust[0]:.1f}')

# %% LaTeX tables

esc = lambda v: v.replace('_', r'\_')
LABELS = {PRICE: 'Log price', 'home': 'Domestic car', 'hpwt': 'Horsepower/weight', 'size': r'Size (m$^2$)',
          'li': 'Fuel consumption', 'log_within_segment_share': r'$\rho$ (segment)'}
FE_LABELS = {'ma_ye': 'Market-year FE', 'brd': 'Brand FE'}

def write_table(file, caption, label, headers, fits, diags, rows, note, instr_keys):
    n = len(fits)
    tex =[r'\begin{table}[htbp]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
           rf'\begin{{tabular}}{{l{"c" * n}}}', r'\toprule', ' & ' + ' & '.join(headers) + r' \\', r'\midrule']
    for k in rows:
        tex.append(LABELS.get(k, esc(k)) + ' & ' + ' & '.join(f'{f.params[k]:.3f}' if k in f.params else '' for f in fits) + r' \\')
        tex.append(' & ' + ' & '.join(f'({f.std_errors[k]:.3f})' if k in f.params else '' for f in fits) + r' \\')
    tex.append(r'\midrule')
    tex += [FE_LABELS.get(f, esc(f) + ' FE') + ' & ' + ' & '.join([r'$\checkmark$'] * n) + r' \\' for f in FE_COLS]
    tex.append('All instruments & ' + ' & '.join(r'$\checkmark$' if set(INSTR[k]) == set(ALL_INSTR) else '' for k in instr_keys) + r' \\')
    tex += ['First-stage F (price / within-segment share) & ' + ' & '.join(' / '.join(f'{x:.1f}' for x in F) for F in diags) + r' \\',
            'Observations & ' + ' & '.join(str(f.nobs) for f in fits) + r' \\',
            r'\bottomrule',
            rf'\multicolumn{{{n + 1}}}{{p{{0.9\linewidth}}}}{{\footnotesize \textit{{Note:}} {note}}} \\',
            r'\end{tabular}', r'\end{table}']
    (RES / file).write_text('\n'.join(tex) + '\n')

list_instr = lambda terms: ', '.join(esc(t) for t in terms)
COMMON_NOTE = (rf'Robust standard errors in parentheses. Dependent variable is $\ln(s_j/s_0)$, market size = {esc(MARKET_SIZE)}. '
               rf'The extended logit is a nested logit with the segment ({SEGMENT}) as nest. '
               'First-stage F: robust F for the simple logits, Sanderson--Windmeijer F for the extended logit. ')

write_table('demand_models.tex', 'Demand estimates', 'tab:demand_models',
            ['Simple logit', 'Extended logit (nested by segment)', 'Simple logit (robustness)'],
            [fit_simple, fit_extended, fit_simple_robust], [diag_simple, diag_extended, diag_simple_robust],
            [PRICE] + CTRL + ['log_within_segment_share'],
            COMMON_NOTE + f'All instruments: {list_instr(ALL_INSTR)}. '
            f'Columns without a checkmark use only the price instrument, {list_instr(INSTR["simple"])}. '
            'The robustness column re-estimates the simple logit with all instruments.',
            ['simple', 'extended', 'simple_robust'])

# %% Calculate own- and cross-price elasticity matrices for the simple and the extended logit

# every pair (j, k) sold in the same market (ma, ye); s and segment_share belong to k
cols = ['ma', 'ye', 'co', SEGMENT]
pairs = cars[cols].merge(cars[cols + ['s', 'segment_share', PRICE_VAR]], on=['ma', 'ye'], suffixes=('_j', '_k'))

def elasticity(pairs, alpha, rho):
    r = 1 / (1 - rho)
    own = pairs['co_j'] == pairs['co_k']
    same_segment = pairs[f'{SEGMENT}_j'] == pairs[f'{SEGMENT}_k']
    dp = 1 if LOG_PRICE else pairs[PRICE_VAR]  # p_k if price enters in levels
    return -alpha * dp * (r * own - (r - 1) * same_segment * pairs['s'] / pairs['segment_share'] - pairs['s'])

pairs['simple'] = elasticity(pairs, alpha_simple, 0)
pairs['extended'] = elasticity(pairs, alpha_extended, rho_extended)
MODELS = {'simple': 'Simple logit', 'extended': 'Extended logit (nested by segment)'}
CBAR = {'label': r'$\mathcal{E}_{jk} = \partial \log s_j \,/\, \partial \log p_k$'}  # notation as in the lecture note

# %% Full matrix: all models, averaged over the markets (ma, ye) where both j and k are sold (never together = black)

fig, axs = plt.subplots(1, 2, figsize=(40, 18))
for ax, (model, title) in zip(axs, MODELS.items()):
    elast = pairs.pivot_table(index='co_j', columns='co_k', values=model, aggfunc='mean')
    # colour scale from cross-price elasticities only
    vmax = np.nanmax(np.abs(elast.values[elast.index.values[:, None] != elast.columns.values[None, :]]))
    ax.set_facecolor('black') # NaN cells show as black
    sns.heatmap(elast, ax=ax, cmap='coolwarm', center=0, vmin=-vmax, vmax=vmax, cbar_kws=CBAR)
    ax.set_title(title); ax.set_xlabel(r'Car model $k$ (price change)'); ax.set_ylabel(r'Car model $j$ (response)')
fig.savefig(RES / 'price_elasticities_all_models.pdf', bbox_inches='tight')

# %% Comparison: top N_PER_SEGMENT models by sales in each of the N_SEGMENTS largest segments, in COUNTRY in YEAR

N_PER_SEGMENT, N_SEGMENTS, YEAR, COUNTRY = 3, 3, 99, 'Italy'
ma_code = {name: code for code, name in lbl_vals['market'].items()}
seg_name = lbl_vals['class']

m = cars[(cars['ma'] == ma_code[COUNTRY]) & (cars['ye'] == YEAR)].set_index('co')
segments = m.groupby('cla')['qu'].sum().nlargest(N_SEGMENTS).index # largest segments by sales
top = (m[m['cla'].isin(segments)].sort_values('qu', ascending=False).groupby('cla').head(N_PER_SEGMENT)
       .sort_values(['cla', 'qu'], ascending=[True, False])) # by segment, then by sales
p = pairs[(pairs['ma'] == ma_code[COUNTRY]) & (pairs['ye'] == YEAR)]

# segment blocks: first/last position of each segment in the sorted list
seg = top['cla'].to_numpy()
bounds = np.flatnonzero(np.diff(seg)) + 1
starts, ends = np.r_[0, bounds], np.r_[bounds, len(seg)]

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
for ax, (model, title) in zip(axs, MODELS.items()):
    elast = (p.pivot_table(index='co_j', columns='co_k', values=model, aggfunc='mean')
             .reindex(index=top.index, columns=top.index))
    elast.index, elast.columns = top['type'], top['type']
    # colour scale from cross-price elasticities only
    vmax = np.nanmax(np.abs(elast.values[~np.eye(len(elast), dtype=bool)]))
    ax.set_facecolor('black')
    sns.heatmap(elast, annot=True, fmt='.2f', annot_kws={'size': 8}, ax=ax, cmap='coolwarm', center=0, vmin=-vmax, vmax=vmax, cbar_kws=CBAR)
    # name the segment blocks along the top
    sec = ax.secondary_xaxis('top')
    sec.set_xticks((starts + ends) / 2, [seg_name[seg[s]] for s in starts])
    sec.tick_params(length=0)
    ax.set_title(title, pad=25); ax.set_xlabel(r'Car model $k$ (price change)'); ax.set_ylabel(r'Car model $j$ (response)')
fig.tight_layout()
fig.savefig(RES / 'price_elasticities_top_models.pdf', bbox_inches='tight')


# %%
