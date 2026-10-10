"""
Cars: Search for instruments
Plots the variation in potential instruments for price and checks their relevance:
  1. destination-country cost-shifters (exchange rate, price indices, VAT)
  2. exporter-country cost-shifters (exchange rate, price indices) and the real exchange rate
  3. Hausman-type: price of the same model in the other markets
  4. BLP-type: characteristics of other products of the same firm / in the same nest, and product counts
Figures are saved in res/instruments/.
"""

# %% Load packages
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import parts
sns.set_theme(style='whitegrid', context='notebook')

DATA = Path('../data') # notebooks run from their own folder (src/)
RES = Path('../res/instruments')
RES.mkdir(parents=True, exist_ok=True)

NEST = 'home'
SUBGROUP = 'cla'
CHARS = ['cy', 'hp', 'we', 'le', 'li', 'sp']   # controls (ac left out: 986 missing)

# %% Read in data

cars = pd.read_csv(DATA / 'cars.csv')
lbl_vars = pd.read_csv(DATA / 'labels_variables.csv', index_col=0)['label']
lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels()

cars['year'] = 1900 + cars['ye']
cars['market'] = cars['ma'].map(lbl_vals['market'])
cars['location'] = cars['loc'].map(lbl_vals['location'])
cars['log_p'] = np.log(cars['eurpr'])

DEST_VARS = ['avdexr', 'avdcpr', 'avdppr', 'tax']   # destination country: vary by (ma, ye)
EXP_VARS = ['avexr', 'avcpr', 'avppr']              # exporter country: vary by (loc, ye)


def lab(v, width=45):
    """'var: description' wrapped for titles/axis labels."""
    return textwrap.fill(f'{v}: {LABELS.get(v, "")}', width)


def year_axis(ax):
    ax.set_xlim(1970, 1999)
    ax.set_xticks(range(1970, 2000, 5))
    ax.set_xlabel('')


def save(fig, name):
    fig.savefig(RES / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.show()

# %% Construct candidate instruments beyond the raw cost-shifters

# real exchange rate: destination currency per unit of exporter currency
cars['log_xexr'] = np.log(cars['xexr'])

# Hausman: mean log price of the same model in the *other* markets in the same year
g = cars.groupby(['co', 'ye'])['log_p']
cars['hausman_p'] = (g.transform('sum') - cars['log_p']) / (g.transform('size') - 1)  # NaN if sold in one market only

# BLP: other products of the same firm / in the same subgroup, and counts.
# (sums over *all* rivals in the market are not used: total - own char is collinear with own char given market-year FE)
f = cars.groupby(['ma', 'ye', 'frm'])
sub = cars.groupby(['ma', 'ye', NEST, SUBGROUP])
blp = pd.DataFrame({'hp_rival_frm': f['hp'].transform('sum') - cars['hp'],
                    'we_rival_frm': f['we'].transform('sum') - cars['we'],
                    'hp_rival_home_cla': sub['hp'].transform('sum') - cars['hp'],
                    'we_rival_home_cla': parts.instruments_function(cars, 'we_rival', NEST, SUBGROUP),
                    'ln_n_frm': np.log(f['qu'].transform('size')),
                    'ln_n_home_cla': parts.instruments_function(cars, 'n_sub', NEST, SUBGROUP)})
cars = pd.concat([cars, blp], axis=1)

LABELS = {**lbl_vars.to_dict(),
          'log_xexr': 'log real exchange rate (avdexr/avexr)',
          'hausman_p': 'avg. log price of same model in other markets, same year',
          'hp_rival_frm': 'sum of hp of other models of the same firm',
          'we_rival_frm': 'sum of weight of other models of the same firm',
          'hp_rival_home_cla': 'sum of hp of other models in same nest/subgroup',
          'we_rival_home_cla': 'sum of weight of other models in same nest/subgroup',
          'ln_n_frm': 'log number of models of the same firm',
          'ln_n_home_cla': 'log number of models in same nest/subgroup'}

LOCATIONS = sorted(cars['location'].dropna().unique())
LOC_COLORS = dict(zip(LOCATIONS, sns.color_palette('tab20', len(LOCATIONS))))

CANDIDATES = {'Destination': DEST_VARS,
              'Exporter': EXP_VARS + ['log_xexr'],
              'Hausman': ['hausman_p'],
              'BLP': list(blp.columns)}

# %% Figure 1: destination-country variables over time, by market

dest = cars.drop_duplicates(['ma', 'ye'])

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for v, ax in zip(DEST_VARS, axes.flat):
    sns.lineplot(data=dest, x='year', y=v, hue='market', ax=ax, legend=False)
    ax.set_title(lab(v, 60), fontsize=11)
    ax.set_ylabel('')
    year_axis(ax)
axes[0, 0].set_yscale('log')
fig.legend(handles=[plt.Line2D([], [], color=c) for c in sns.color_palette(n_colors=dest['market'].nunique())],
           labels=sorted(dest['market'].unique()), loc='lower center', ncol=5, frameon=False)
fig.suptitle('Destination-country cost-shifters (constant within a market-year)', fontsize=14)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, '01_destination_cost_shifters')

# %% Figure 2: exporter-country variables over time, by production location

exp = cars.drop_duplicates(['loc', 'ye'])
exp = exp[~((exp['location'] == 'Yugoslavia') & (exp['avcpr'] > 1e4))]  # one hyperinflation obs (1994) squashes the axis

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
for v, ax in zip(EXP_VARS, axes.flat):
    sns.lineplot(data=exp, x='year', y=v, hue='location', palette=LOC_COLORS, ax=ax, legend=(v == EXP_VARS[-1]))
    ax.set_yscale('log')  # high-inflation countries
    ax.set_title(lab(v), fontsize=11)
    ax.set_ylabel('')
    year_axis(ax)
axes[-1].legend(title='Production location', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
fig.suptitle('Exporter-country cost-shifters (vary across products within a market-year), log scale', fontsize=14)
fig.tight_layout()
save(fig, '02_exporter_cost_shifters')

# %% Figure 3: real exchange rate over time, by market and production location

fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharex=True)
xr = cars.drop_duplicates(['ma', 'loc', 'ye'])
for (m, d), ax in zip(xr.groupby('market'), axes):
    # normalise each (market, location) series to its mean: removes currency units
    d = d.assign(rel=d['log_xexr'] - d.groupby('loc')['log_xexr'].transform('mean'))
    # pivot to a full year grid so gaps in a series are not drawn as straight lines
    wide = d.pivot(index='year', columns='location', values='rel').reindex(range(1970, 2000))
    for loc_name in wide.columns:
        ax.plot(wide.index, wide[loc_name], color=LOC_COLORS[loc_name], label=loc_name)
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_title(f'Market: {m}')
    ax.set_ylabel('log xexr, deviation from mean' if m == 'Belgium' else '')
    year_axis(ax)
fig.legend(handles=[plt.Line2D([], [], color=LOC_COLORS[l]) for l in LOCATIONS], labels=LOCATIONS,
           title='Production location', bbox_to_anchor=(1.0, 0.5), loc='center left', frameon=False)
fig.suptitle(lab('log_xexr', 100) + ', demeaned within market × location', fontsize=14)
fig.tight_layout()
save(fig, '03_real_exchange_rate')

# %% Variation and relevance of all candidates
# variation: share of the variance left after market-year means (0 -> absorbed by market-year FE)
# relevance: partial correlation with log price after market-year FE and characteristics (first-stage style)


def demean(df, cols):
    return df[cols] - df.groupby(['ma', 'ye'])[cols].transform('mean')


rows = []
for cat, vs in CANDIDATES.items():
    for v in vs:
        d = cars.dropna(subset=[v, 'log_p'] + CHARS)
        z = np.log(d[v]) if v in DEST_VARS + EXP_VARS else d[v]
        d = d.assign(z=z)
        dm = demean(d, ['z', 'log_p'] + CHARS)
        X = np.column_stack([dm[CHARS].values, np.ones(len(dm))])
        res = dm[['z', 'log_p']].values - X @ np.linalg.lstsq(X, dm[['z', 'log_p']].values, rcond=None)[0]
        within = dm['z'].var() / d['z'].var()
        corr = np.corrcoef(res.T)[0, 1] if res[:, 0].std() > 1e-10 else np.nan
        rows.append({'var': v, 'category': cat, 'within': within, 'partial_corr': corr, 'N': len(d)})

summary = pd.DataFrame(rows).set_index('var')
summary['label'] = [lab(v, 55) for v in summary.index]
print(summary[['category', 'within', 'partial_corr', 'N']].round(3))

# %% Figure 4: summary of variation and relevance

fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
palette = dict(zip(CANDIDATES, sns.color_palette(n_colors=len(CANDIDATES))))
sns.barplot(data=summary, y='label', x='within', hue='category', palette=palette, dodge=False, ax=axes[0])
sns.barplot(data=summary, y='label', x='partial_corr', hue='category', palette=palette, dodge=False, ax=axes[1], legend=False)

axes[0].set_title('Variation left after market-year FE\n(share of variance)')
axes[0].set_xlim(0, 1)
axes[1].set_title('Relevance: partial corr. with log price\n(after market-year FE + characteristics)')
axes[1].axvline(0, color='grey', lw=0.8)
for i, v in enumerate(summary.index):
    if np.isnan(summary.loc[v, 'partial_corr']):
        axes[1].text(0.01, i, 'absorbed by FE', va='center', fontsize=9, color='grey')
for ax in axes:
    ax.set_xlabel('')
    ax.set_ylabel('')
axes[0].get_legend().remove()
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=palette[c]) for c in CANDIDATES], labels=list(CANDIDATES),
           loc='upper center', ncol=len(CANDIDATES), frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.97))
save(fig, '04_variation_and_relevance')

# %% Figure 5: first-stage style scatter for the main candidates

MAIN = ['log_xexr', 'hausman_p', 'we_rival_home_cla', 'ln_n_home_cla']

fig, axes = plt.subplots(1, len(MAIN), figsize=(18, 4.5))
for v, ax in zip(MAIN, axes):
    d = cars.dropna(subset=[v, 'log_p'] + CHARS)
    dm = demean(d, [v, 'log_p'] + CHARS)
    X = np.column_stack([dm[CHARS].values, np.ones(len(dm))])
    res = dm[[v, 'log_p']].values - X @ np.linalg.lstsq(X, dm[[v, 'log_p']].values, rcond=None)[0]
    sns.regplot(x=res[:, 0], y=res[:, 1], ax=ax, scatter_kws={'s': 4, 'alpha': 0.25},
                line_kws={'color': 'C3'})
    ax.set_xlabel(lab(v, 40), fontsize=10)
    ax.set_title(f'corr = {summary.loc[v, "partial_corr"]:.2f}')
axes[0].set_ylabel('log price (residualised)')
fig.suptitle('Log price vs. instrument, both residualised on market-year FE and characteristics', fontsize=14)
fig.tight_layout()
save(fig, '05_first_stage_scatter')

# %% Coverage of the chosen instruments: how much data do we lose?
# log_xexr has no missing values, but domestically produced cars (xexr = 1) give it no variation at all.
# hausman_p is missing for models sold in only one market that year.
# The controls exclude ac (missing for 986 rows, mostly in the 1970s).

CHOSEN = ['log_xexr', 'hausman_p', 'we_rival_home_cla', 'ln_n_home_cla']
cars['domestic'] = cars['xexr'] == 1

samples = {'All rows': pd.Series(True, index=cars.index),
           'Controls observed (excl. ac)': cars[CHARS].notna().all(axis=1)}
samples['+ 3 instruments (no Hausman)'] = samples['Controls observed (excl. ac)'] & cars[[c for c in CHOSEN if c != 'hausman_p']].notna().all(axis=1)
samples['+ all 4 instruments'] = samples['Controls observed (excl. ac)'] & cars[CHOSEN].notna().all(axis=1)

coverage = pd.DataFrame({k: [s.sum(), s.mean()] for k, s in samples.items()}, index=['N', 'share']).T
print(coverage.round(3))
print('\nShare of rows produced domestically (log_xexr = 0):', round(cars['domestic'].mean(), 3))
print(cars.groupby('market')[['domestic']].mean().assign(hausman_missing=cars['hausman_p'].isna().groupby(cars['market']).mean()).round(2))

# %% Figure 6: data kept by year and where log_xexr has no variation

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

for k, s in samples.items():
    axes[0].plot(range(1970, 2000), s.groupby(cars['year']).mean(), label=k,
                 ls='--' if 'no Hausman' in k else '-')  # dashed: identical to the controls line
axes[0].set_ylim(0, 1.02)
axes[0].set_title('Share of observations kept, by year')
axes[0].legend(frameon=False, loc='lower right')
year_axis(axes[0])

sns.lineplot(data=cars, x='year', y='domestic', hue='market', errorbar=None, ax=axes[1])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('')
axes[1].set_title('Share of models produced in the destination country\n(xexr = 1, so log_xexr gives no variation)')
axes[1].legend(title='', frameon=False, ncol=5, loc='upper center')
year_axis(axes[1])

fig.tight_layout()
save(fig, '06_instrument_coverage')
