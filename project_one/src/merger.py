"""
Cars: Merger simulation (VW group x Fiat group)

For each market (country) in year MARKET_YE and each demand model:
  1. recover marginal costs from the observed prices (multiproduct Bertrand-Nash FOCs),
  2. solve for post-merger prices with the merged ownership matrix,
  3. compute the change in consumer surplus (in money) and the change in HHI,
  4. find the break-even efficiency: the cut kappa in the merging firms' marginal costs,
     c' = c (1 - kappa), at which consumers are exactly as well off as before (dCS = 0)
     (course note, Section 3.3; slides: "report the kappa at which consumers break even").

Ownership is by firm (frm), not brand: frm 26 = VW group (VW, Audi, Seat, Skoda),
frm 4 = Fiat group (Fiat, Lancia, Alfa Romeo, ...). The data's frm codes follow ownership over time.

Consumer surplus. With log price, the log-sum change divided by alpha is in log-price units, not money
(course note, Section 3.5), because the marginal utility of money, alpha/p_j, differs across cars.
We therefore use the first-order (Roy's identity) money measure, per potential consumer (household):
    dCS ~= - sum_j  0.5 * (s_j,pre + s_j,post) * (p_j,post - p_j,pre)
This is the money version of the log-sum change: d logsum = sum_j s_j d delta_j = -alpha sum_j s_j d ln p_j.
It is accurate for small price changes; the log-sum is kept as a cross-check (dW/alpha ~= -sum_j s_j d ln p_j).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.optimize import brentq
import parts

DATA = Path('../data')
RES = Path('../res/merger')
RES.mkdir(parents=True, exist_ok=True)

# Must be the same as in demand_models.py (the parameters were estimated with it)
MARKET_SIZE = 'hh'
PRICE_VAR, LOG_PRICE = 'eurpr', True     # eurpr is in SDR
SEGMENT = 'cla'
MARKET_YE = 99
SDR_TO_EUR = 1.2956                      # project1.pdf: multiply eurpr by 1.2956 to get euros

# Merging firms (frm codes)
VW_FRM, FIAT_FRM = 26, 4
GROUP_LABEL = {VW_FRM: 'VW group', FIAT_FRM: 'Fiat group'}

KAPPA_MAX = 0.5                          # search the break-even cost cut on [0, KAPPA_MAX]

assert LOG_PRICE, 'This script is written for log price in the demand model.'

# %% Load data, shares and parameters

cars = pd.read_csv(DATA / 'cars.csv')
if 'market' not in cars.columns:
    lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels()
    cars['market'] = cars['ma'].map(lbl_vals['market'])

# Shares on the full dataset, before selecting markets
cars['market_size'], cars['s'], cars['s0'] = parts.market_size_function(data=cars, type=MARKET_SIZE)

params = pd.read_csv(DATA / 'demand_model_parameters.csv', index_col=0)
MODELS = {   # name: (alpha, rho)
    'Simple logit': (params.loc['alpha', 'Simple logit'], 0.0),
    'Extended nested logit': (params.loc['alpha', 'Extended logit'],
                              params.loc['log_within_segment_share', 'Extended logit']),
    'Simple logit (robustness)': (params.loc['alpha', 'Simple logit (robustness)'], 0.0),
}
for name, (alpha, rho) in MODELS.items():
    print(f'{name}: alpha = {alpha:.3f}, rho = {rho:.3f}')


# %% Helper functions

def ownership(firms):
    """Ownership matrix: 1 if products j and k are sold by the same firm."""
    firms = np.asarray(firms)
    return (firms[:, None] == firms[None, :]).astype(float)


def nested_logsum(delta, nests, rho):
    """Log-sum welfare index for the simple (rho = 0) or one-level nested logit; outside option delta_0 = 0."""
    log_iv = [(1 - rho) * logsumexp(delta[nests == h] / (1 - rho)) for h in np.unique(nests)]
    return float(logsumexp(np.r_[0.0, log_iv]))


def weighted_average(values, weights):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return np.average(values[ok], weights=weights[ok]) if ok.any() else np.nan


def save_table_png(df, path, title, col_widths=None, fontsize=8.5):
    """Render a DataFrame as a styled table image."""
    cells = [[f'{v:.2f}' if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in df.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(max(10, 1.9 * df.shape[1]), max(3.5, 0.42 * (len(df) + 2))))
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=list(df.columns), colWidths=col_widths,
                     cellLoc='center', bbox=[0, 0, 1, 0.90])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#D5DDE5')
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor('#243B53')
            cell.set_text_props(color='white', weight='bold', va='center')
        else:
            pooled = cells[r - 1][0].startswith('All')
            cell.set_facecolor('#DCE8F2' if pooled else ('#F1F5F9' if r % 2 == 0 else 'white'))
            if pooled:
                cell.set_text_props(weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', color='#243B53', pad=14)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# %% Merger simulation, market by market

product_rows, welfare_rows, breakeven_rows = [], [], []
dCS_functions = {}   # (market, model) -> dCS(kappa) in mill. EUR, for the pooled break-even

for ma in sorted(cars.loc[cars['ye'].eq(MARKET_YE), 'ma'].unique()):
    market = cars[cars['ma'].eq(ma) & cars['ye'].eq(MARKET_YE)].reset_index(drop=True)
    label = market['market'].iloc[0]

    shares = market['s'].to_numpy()
    prices = market[PRICE_VAR].to_numpy()
    M = float(market['market_size'].iloc[0])
    nests = market[SEGMENT].to_numpy()
    nest_id = nests.reshape(1, -1)
    nest_data = {'nest_id': nest_id}
    S_obs = np.r_[market['s0'].iloc[0], shares].reshape(1, -1)

    # Ownership before and after: the two groups become one firm
    firm_pre = market['frm'].to_numpy()
    merging = np.isin(firm_pre, [VW_FRM, FIAT_FRM])
    firm_post = np.where(merging, -1, firm_pre)     # -1 = merged VW-Fiat firm
    H, H_merge = ownership(firm_pre), ownership(firm_post)

    result = market[['ma', 'market', 'ye', 'co', 'type', 'brand', 'frm', SEGMENT, 's']].rename(
        columns={'s': 'share_pre'})
    result['group'] = market['frm'].map(GROUP_LABEL).fillna('Other')
    result['market_size'] = M
    result['price_observed_eur'] = prices * SDR_TO_EUR

    for model, (alpha, rho) in MODELS.items():
        # Mean utilities from observed shares (closed-form nested-logit inversion)
        delta_obs = parts.invert_demand(S_obs, rho=rho, nest_id=nest_id)[0]

        # Marginal costs from the pre-merger FOCs at observed prices
        markup = parts.zeta_logprice(prices, H, delta_obs, prices, nest_data, alpha, rho)
        mc = prices - markup
        if (mc <= 0).any():
            print(f'WARNING {label}, {model}: {(mc <= 0).sum()} products with non-positive marginal cost')

        # Equilibrium prices before (should reproduce observed prices) and after the merger
        p_pre = parts.solve_nash(H, mc, prices, delta_obs, nest_data, alpha, rho)
        p_post = parts.solve_nash(H_merge, mc, prices, delta_obs, nest_data, alpha, rho)
        print(f'{label}, {model}: max |p_pre - p_obs| = {np.max(np.abs(p_pre - prices)):.2e}')

        # Shares at pre- and post-merger prices
        delta_pre = delta_obs - alpha * np.log(p_pre / prices)
        delta_post = delta_obs - alpha * np.log(p_post / prices)
        s_pre = parts.shares_from_delta(delta_pre.reshape(1, -1), nest_data, rho)[0, 1:]
        s_post = parts.shares_from_delta(delta_post.reshape(1, -1), nest_data, rho)[0, 1:]

        # Consumer surplus in money (SDR per household -> EUR), first-order approximation
        dCS_hh_sdr = -np.sum(0.5 * (s_pre + s_post) * (p_post - p_pre))
        spending_hh_sdr = np.sum(s_pre * p_pre)

        # Cross-check with the log-sum: dW/alpha should equal -sum s * d ln p (log-price units)
        dW = nested_logsum(delta_post, nests, rho) - nested_logsum(delta_pre, nests, rho)
        check = -np.sum(0.5 * (s_pre + s_post) * np.log(p_post / p_pre))

        welfare_rows.append({
            'market': label, 'model': model,
            'dCS_per_household_eur': dCS_hh_sdr * SDR_TO_EUR,
            'dCS_total_mill_eur': dCS_hh_sdr * SDR_TO_EUR * M / 1e6,
            'spending_total_mill_eur': spending_hh_sdr * SDR_TO_EUR * M / 1e6,
            'dCS_pct_of_spending': 100 * dCS_hh_sdr / spending_hh_sdr,
            'dlogsum_over_alpha': dW / alpha,
            'first_order_check': check,
            'median_lerner_pct': 100 * np.median(markup / prices),
        })

        # Break-even efficiency: cut the merging firms' marginal costs by kappa, re-solve the
        # post-merger equilibrium and find the kappa at which dCS = 0 (consumers break even).
        def dCS_at(kappa, mc=mc, delta_obs=delta_obs, alpha=alpha, rho=rho,
                   s_pre=s_pre, p_pre=p_pre, merging=merging, H_merge=H_merge, nest_data=nest_data,
                   prices=prices):
            mc_k = mc * (1 - kappa * merging)
            p_k = parts.solve_nash(H_merge, mc_k, prices, delta_obs, nest_data, alpha, rho)
            s_k = parts.shares_from_delta((delta_obs - alpha * np.log(p_k / prices)).reshape(1, -1),
                                          nest_data, rho)[0, 1:]
            return -np.sum(0.5 * (s_pre + s_k) * (p_k - p_pre)), p_k   # SDR per household

        kappa_star = brentq(lambda k: dCS_at(k)[0], 0.0, KAPPA_MAX, xtol=1e-8)
        p_star = dCS_at(kappa_star)[1]
        q_merging = s_pre[merging] * M                                  # pre-merger quantities
        dCS_functions[(label, model)] = (lambda k, f=dCS_at, M=M: f(k)[0] * SDR_TO_EUR * M / 1e6)

        breakeven_rows.append({
            'market': label, 'model': model,
            'kappa_pct': 100 * kappa_star,
            # annual cost savings implied by kappa*: kappa* x marginal cost x quantity, merging products
            'savings_mill_eur': kappa_star * np.sum(mc[merging] * q_merging) * SDR_TO_EUR / 1e6,
            # price change of the merging products at kappa* (share-weighted), for comparison
            'dp_merging_at_kappa_pct': 100 * np.average(p_star[merging] / p_pre[merging] - 1,
                                                       weights=s_pre[merging]),
            'mc_merging_mill_eur': np.sum(mc[merging] * q_merging) * SDR_TO_EUR / 1e6,
        })

        key = {'Simple logit': 'simple', 'Extended nested logit': 'extended',
               'Simple logit (robustness)': 'robust'}[model]
        result[f'{key}_mc_eur'] = mc * SDR_TO_EUR
        result[f'{key}_pre_eur'] = p_pre * SDR_TO_EUR
        result[f'{key}_post_eur'] = p_post * SDR_TO_EUR
        result[f'{key}_change_eur'] = (p_post - p_pre) * SDR_TO_EUR
        result[f'{key}_change_pct'] = 100 * (p_post / p_pre - 1)
        result[f'{key}_share_post'] = s_post

    product_rows.append(result)

results = pd.concat(product_rows, ignore_index=True)
results.to_csv(RES / f'merger_prices_ye_{MARKET_YE}.csv', index=False)


# %% Price effects: summary by market and model (share-weighted)

MODEL_KEYS = {'Simple logit': 'simple', 'Extended nested logit': 'extended',
              'Simple logit (robustness)': 'robust'}

groups = [(m, g) for m, g in results.groupby('market', sort=True)] + [('All markets', results)]
summary_rows = []
for market_name, g in groups:
    # Within a market weight by shares; pooled across markets weight by quantities (share x market size)
    w = g['share_pre'].to_numpy() * (g['market_size'].to_numpy() if market_name == 'All markets' else 1)
    vw, fiat = g['group'].eq('VW group').to_numpy(), g['group'].eq('Fiat group').to_numpy()

    for model, key in MODEL_KEYS.items():
        top = g.loc[g[f'{key}_change_eur'].idxmax()]
        summary_rows.append({
            'market': market_name, 'model': model,
            'avg_pct_change_all': weighted_average(g[f'{key}_change_pct'], w),
            'avg_pct_change_vw_group': weighted_average(g.loc[vw, f'{key}_change_pct'], w[vw]),
            'avg_pct_change_fiat_group': weighted_average(g.loc[fiat, f'{key}_change_pct'], w[fiat]),
            'largest_increase_eur': top[f'{key}_change_eur'],
            'largest_increase_pct': top[f'{key}_change_pct'],
            'product_largest_increase': str(top['type']).title(),
        })

merger_overview = pd.DataFrame(summary_rows)
merger_overview.round(3).to_csv(RES / f'merger_overview_ye_{MARKET_YE}.csv', index=False)

pretty = merger_overview.rename(columns={
    'market': 'Market', 'model': 'Model',
    'avg_pct_change_all': 'All products\n(%)',
    'avg_pct_change_vw_group': 'VW group\n(%)',
    'avg_pct_change_fiat_group': 'Fiat group\n(%)',
    'largest_increase_eur': 'Largest\nincrease (EUR)',
    'largest_increase_pct': 'Largest\nincrease (%)',
    'product_largest_increase': 'Product with\nlargest increase'})

pretty.to_latex(RES / f'merger_overview_ye_{MARKET_YE}.tex', index=False, float_format='%.2f', escape=True,
                caption=f'Merger price effects by market and demand model, {1900 + MARKET_YE}. '
                        'Averages are share-weighted (pooled row: quantity-weighted). '
                        'Prices converted from SDR to EUR.',
                label='tab:merger_overview')
save_table_png(pretty, RES / f'merger_overview_ye_{MARKET_YE}.png',
               f'Merger price effects by market, {1900 + MARKET_YE}',
               col_widths=[0.10, 0.20, 0.10, 0.10, 0.10, 0.11, 0.10, 0.19])


# %% Consumer surplus: by market and model, plus pooled totals

consumer_surplus = pd.DataFrame(welfare_rows)

pooled = []
for model, g in consumer_surplus.groupby('model', sort=False):
    total, spend = g['dCS_total_mill_eur'].sum(), g['spending_total_mill_eur'].sum()
    households = results.drop_duplicates('market')['market_size'].sum()
    pooled.append({'market': 'All markets', 'model': model,
                   'dCS_per_household_eur': total * 1e6 / households,
                   'dCS_total_mill_eur': total, 'spending_total_mill_eur': spend,
                   'dCS_pct_of_spending': 100 * total / spend,
                   'dlogsum_over_alpha': np.nan, 'first_order_check': np.nan,
                   'median_lerner_pct': np.nan})
consumer_surplus = pd.concat([consumer_surplus, pd.DataFrame(pooled)], ignore_index=True)
consumer_surplus.to_csv(RES / f'consumer_surplus_ye_{MARKET_YE}.csv', index=False)

print('\nCross-check (should be close): log-sum change / alpha vs. -sum s * d ln p')
print(consumer_surplus.dropna(subset=['dlogsum_over_alpha'])[
    ['market', 'model', 'dlogsum_over_alpha', 'first_order_check']].to_string(index=False))

cs_pretty = consumer_surplus[['market', 'model', 'dCS_per_household_eur', 'dCS_total_mill_eur',
                              'dCS_pct_of_spending', 'median_lerner_pct']].rename(columns={
    'market': 'Market', 'model': 'Model',
    'dCS_per_household_eur': 'dCS per\nhousehold (EUR)',
    'dCS_total_mill_eur': 'dCS total\n(mill. EUR)',
    'dCS_pct_of_spending': 'dCS, % of\ncar spending',
    'median_lerner_pct': 'Median\nLerner (%)'})

cs_pretty.to_latex(RES / f'consumer_surplus_ye_{MARKET_YE}.tex', index=False, na_rep='',
                   float_format='%.2f', escape=True,
                   caption=f'Change in consumer surplus from the merger, {1900 + MARKET_YE}. '
                           r'First-order money measure $-\sum_j \bar s_j \Delta p_j$ per household '
                           r'($\bar s_j$: average of pre- and post-merger shares), times market size; '
                           'relative to baseline spending on cars.',
                   label='tab:consumer_surplus')
save_table_png(cs_pretty.fillna(''), RES / f'consumer_surplus_ye_{MARKET_YE}.png',
               f'Change in consumer surplus, {1900 + MARKET_YE}',
               col_widths=[0.13, 0.25, 0.15, 0.15, 0.16, 0.16])

print('\nConsumer surplus:')
print(cs_pretty.round(3).to_string(index=False))


# %% Break-even efficiency: by market and model, plus one common kappa for all markets

breakeven = pd.DataFrame(breakeven_rows)

pooled = []
for model in MODELS:
    fs = [f for (m, mod), f in dCS_functions.items() if mod == model]
    kappa_all = brentq(lambda k: sum(f(k) for f in fs), 0.0, KAPPA_MAX, xtol=1e-8)
    g = breakeven[breakeven['model'] == model]
    mc_total = g['mc_merging_mill_eur'].sum()
    pooled.append({'market': 'All markets', 'model': model, 'kappa_pct': 100 * kappa_all,
                   'savings_mill_eur': kappa_all * mc_total, 'dp_merging_at_kappa_pct': np.nan,
                   'mc_merging_mill_eur': mc_total})
breakeven = pd.concat([breakeven, pd.DataFrame(pooled)], ignore_index=True)
breakeven.to_csv(RES / f'breakeven_ye_{MARKET_YE}.csv', index=False)

print('\nBreak-even efficiency (cut in merging firms\' marginal costs that leaves dCS = 0):')
print(breakeven.round(3).to_string(index=False))

# LaTeX table: one column pair (kappa*, savings) per model, markets as rows
order = ['All markets'] + sorted(m for m in breakeven['market'].unique() if m != 'All markets')
models = list(MODELS)
headers = [r'(1) Simple logit', r'\shortstack{(2) Extended logit\\(nested by segment)}',
           r'\shortstack{(3) Simple\\(robustness)}']
tex = [r'\begin{table}[!htbp]', r'\centering',
       rf'\caption{{Break-even efficiencies by market, year {1900 + MARKET_YE}}}', r'\label{tab:merger_breakeven}',
       r'\small', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{0.95}', '',
       r'\begin{tabular}{l*{6}{r}}', r'\toprule',
       '& ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{h}}}' for h in headers) + r' \\',
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
       'Market' + r' & \shortstack{$\kappa^*$\\(\%)} & \shortstack{Savings\\(mill.\ EUR)}' * 3 + r' \\',
       r'\midrule']
for m in order:
    cells = []
    for model in models:
        r = breakeven[(breakeven['market'] == m) & (breakeven['model'] == model)].iloc[0]
        cells += [f'{r.kappa_pct:.2f}', f'{r.savings_mill_eur:.1f}']
    tex.append(f'{m:<11} & ' + ' & '.join(cells) + r' \\')
tex += [r'\bottomrule', r'\end{tabular}', '', r'\vspace{3pt}', r'\begin{minipage}{0.9\textwidth}', r'\footnotesize',
        r'\emph{Notes:} $\kappa^*$ is the proportional cut in the marginal costs of all VW-group and',
        r"Fiat-group products, $c' = c(1-\kappa)$, at which the merger leaves consumer surplus",
        r'unchanged ($\Delta CS = 0$), with post-merger prices re-solved at the lower costs. Savings are',
        r"$\kappa^*$ times the merging products' marginal costs times pre-merger quantities, per year, converted",
        r'from SDR to EUR at 1 SDR = 1.2956 EUR. The all-markets row uses one common $\kappa^*$ that sets the',
        r'sum of $\Delta CS$ over the five markets to zero. Column (3) re-estimates the simple logit with all',
        r'instruments.', r'\end{minipage}', r'\end{table}']
(RES / f'breakeven_ye_{MARKET_YE}.tex').write_text('\n'.join(tex) + '\n')


# %% HHI before and after the merger (firm level, sales-based)

hhi_rows = []
for market_name, g in cars[cars['ye'].eq(MARKET_YE)].groupby('market', sort=True):
    owner_pre = g['frm']
    owner_post = g['frm'].where(~g['frm'].isin([VW_FRM, FIAT_FRM]), -1)
    total = g['qu'].sum()
    hhi_pre = 10_000 * ((g.groupby(owner_pre)['qu'].sum() / total) ** 2).sum()
    hhi_post = 10_000 * ((g.groupby(owner_post)['qu'].sum() / total) ** 2).sum()
    hhi_rows.append({'Market': market_name, 'HHI before merger': hhi_pre,
                     'HHI after merger': hhi_post, 'Change in HHI': hhi_post - hhi_pre,
                     'total_sales': total})

hhi = pd.DataFrame(hhi_rows)
pooled_hhi = {'Market': 'All markets (sales-weighted)'}
for col in ['HHI before merger', 'HHI after merger', 'Change in HHI']:
    pooled_hhi[col] = np.average(hhi[col], weights=hhi['total_sales'])
hhi = pd.concat([hhi, pd.DataFrame([pooled_hhi])], ignore_index=True).drop(columns='total_sales')

(RES / f'hhi_merger_ye_{MARKET_YE}.tex').write_text(
    hhi.to_latex(index=False, column_format='lrrr', float_format=lambda x: f'{x:.1f}', escape=True,
                 position='!htbp', label='tab:hhi_merger',
                 caption=f'Firm-level HHI (inside sales) before and after the VW--Fiat merger, '
                         f'{1900 + MARKET_YE}'),
    encoding='utf-8')
print('\nHHI:')
print(hhi.round(1).to_string(index=False))
