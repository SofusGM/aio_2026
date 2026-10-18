import pandas as pd
import numpy as np
from pathlib import Path
import parts
import matplotlib.pyplot as plt
from scipy.special import logsumexp


DATA = Path('../data')
RES = Path('../res/merger')
RES.mkdir(parents=True, exist_ok=True)

MARKET_SIZE = 'hh'   # Keep this consistent with the demand estimation
PRICE_VAR, LOG_PRICE = 'eurpr', True
SEGMENT = 'cla'
MARKET_YE = 99

# Load data and labels
cars = pd.read_csv(DATA / 'cars.csv')
lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels()
cars['market'] = cars['ma'].map(lbl_vals['market'])

# Calculate shares on the full dataset, before selecting markets
cars['market_size'], cars['s'], cars['s0'] = parts.market_size_function(
    data=cars,
    type=MARKET_SIZE
)
cars['segment_share'] = cars.groupby(
    ['ma', 'ye', SEGMENT]
)['s'].transform('sum')






# Load parameters. Rerun the updated estimation export first so that the
# robustness model is present in this CSV.
params = pd.read_csv(DATA / 'demand_model_parameters.csv', index_col=0)

alpha_simple = params.loc['alpha', 'Simple logit']
alpha_extended = params.loc['alpha', 'Extended logit']
rho_extended = params.loc[
    'log_within_segment_share', 'Extended logit'
]
alpha_robust = params.loc[
    'alpha', 'Simple logit (robustness)'
]


def elasticity(pairs, alpha, rho):
    """Price elasticities: rows are responding shares; columns are changed prices."""
    r = 1 / (1 - rho)
    own = pairs['co_j'] == pairs['co_k']
    same_segment = pairs[f'{SEGMENT}_j'] == pairs[f'{SEGMENT}_k']
    dp = 1 if LOG_PRICE else pairs[PRICE_VAR]

    return -alpha * dp * (
        r * own
        - (r - 1) * same_segment
          * pairs['s'] / pairs['segment_share']
        - pairs['s']
    )


def share_jacobian(E, shares, prices):
    """Jacobian with rows = prices changed and columns = shares responding."""
    return E.to_numpy().T * (shares[None, :] / prices[:, None])


def nested_logsum(delta, nests, rho):
    """Log-sum welfare index for simple or one-level nested logit."""
    delta = np.asarray(delta, dtype=float)
    nests = np.asarray(nests)

    log_inclusive_values = []
    for h in np.unique(nests):
        in_nest = nests == h
        log_iv = (1 - rho) * logsumexp(delta[in_nest] / (1 - rho))
        log_inclusive_values.append(log_iv)

    # The outside option has mean utility zero.
    return float(logsumexp(np.r_[0.0, log_inclusive_values]))


def weighted_average(values, weights):
    """Weighted average, ignoring missing values and nonpositive weights."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan

    return np.average(values[valid], weights=weights[valid])


results_list = []
welfare_rows = []

market_codes = (
    cars.loc[cars['ye'].eq(MARKET_YE), 'ma']
    .drop_duplicates()
    .tolist()
)

for MARKET_MA in market_codes:
    market = (
        cars.loc[
            cars['ma'].eq(MARKET_MA) & cars['ye'].eq(MARKET_YE)
        ]
        .sort_values('brand')
        .reset_index(drop=True)
    )

    if market.empty:
        continue

    # Build product pairs within this market. The share and segment share
    # in each pair belong to product k.
    cols = ['ma', 'ye', 'co', SEGMENT]
    pairs = market[cols].merge(
        market[cols + ['s', 'segment_share', PRICE_VAR]],
        on=['ma', 'ye'],
        suffixes=('_j', '_k')
    )

    pairs['simple'] = elasticity(pairs, alpha_simple, rho=0)
    pairs['extended'] = elasticity(
        pairs, alpha_extended, rho_extended
    )
    pairs['robust'] = elasticity(pairs, alpha_robust, rho=0)

    # Keep matrices, shares, prices, and ownership in the same product order.
    product_order = market['co'].tolist()

    E_simple = (
        pairs.pivot(index='co_j', columns='co_k', values='simple')
        .reindex(index=product_order, columns=product_order)
    )
    E_extended = (
        pairs.pivot(index='co_j', columns='co_k', values='extended')
        .reindex(index=product_order, columns=product_order)
    )
    E_robust = (
        pairs.pivot(index='co_j', columns='co_k', values='robust')
        .reindex(index=product_order, columns=product_order)
    )

    market_ordered = market.set_index('co').loc[product_order]
    shares = market_ordered['s'].to_numpy()
    prices = market_ordered[PRICE_VAR].to_numpy()

    J_simple = share_jacobian(E_simple, shares, prices)
    J_extended = share_jacobian(E_extended, shares, prices)
    J_robust = share_jacobian(E_robust, shares, prices)

    # Ownership matrices before and after the Fiat–VW merger
    j = len(market_ordered)
    brands = market_ordered['brand'].astype(str).str.casefold().to_numpy()

    same_brand = brands[:, None] == brands[None, :]
    is_vw = brands == 'vw'
    is_fiat = brands == 'fiat'

    H = same_brand.astype(float)
    H_merge = (
        same_brand
        | (is_vw[:, None] & is_fiat[None, :])
        | (is_fiat[:, None] & is_vw[None, :])
    ).astype(float)

    # Infer model-specific marginal costs from observed prices.
    markup_simple = np.linalg.solve(H * J_simple, -shares)
    mc_simple = prices - markup_simple

    markup_extended = np.linalg.solve(H * J_extended, -shares)
    mc_extended = prices - markup_extended

    markup_robust = np.linalg.solve(H * J_robust, -shares)
    mc_robust = prices - markup_robust

    # Recover mean utility from observed shares.
    nest_id = market_ordered[SEGMENT].to_numpy().reshape(1, -1)
    nest_data = {'nest_id': nest_id}

    s0_obs = float(market_ordered['s0'].iloc[0])
    S_obs = np.r_[s0_obs, shares].reshape(1, -1)

    # Simple and robustness models are both simple logit, so they use the
    # same delta inversion. Their estimated alpha values can differ.
    delta_simple = parts.invert_demand(
        S_obs, rho=0, nest_id=nest_id
    )[0]
    delta_extended = parts.invert_demand(
        S_obs, rho=rho_extended, nest_id=nest_id
    )[0]

    # Solve pre- and post-merger equilibrium prices.
    p_simple_pre = parts.solve_nash(
        H, mc_simple, prices, delta_simple,
        nest_data, alpha_simple, rho=0
    )
    p_simple_post = parts.solve_nash(
        H_merge, mc_simple, prices, delta_simple,
        nest_data, alpha_simple, rho=0
    )

    p_extended_pre = parts.solve_nash(
        H, mc_extended, prices, delta_extended,
        nest_data, alpha_extended, rho=rho_extended
    )
    p_extended_post = parts.solve_nash(
        H_merge, mc_extended, prices, delta_extended,
        nest_data, alpha_extended, rho=rho_extended
    )

    p_robust_pre = parts.solve_nash(
        H, mc_robust, prices, delta_simple,
        nest_data, alpha_robust, rho=0
    )
    p_robust_post = parts.solve_nash(
        H_merge, mc_robust, prices, delta_simple,
        nest_data, alpha_robust, rho=0
    )

    # Verify each model reproduces observed prices before the merger.
    print(
        f"ma={MARKET_MA}: max pre-merger price difference — "
        f"simple {np.max(np.abs(p_simple_pre - prices)):.6g}; "
        f"extended {np.max(np.abs(p_extended_pre - prices)):.6g}; "
        f"robustness {np.max(np.abs(p_robust_pre - prices)):.6g}"
    )

    # Save product-level results for this market.
    result_market = (
        market_ordered.reset_index()[
            [
                'ma', 'market', 'ye', 'co', 'type', 'brand',
                's', 'market_size'
            ]
        ]
        .rename(columns={'s': 'share_pre'})
    )

    result_market['price_observed'] = prices

    model_prices = {
        'simple': (p_simple_pre, p_simple_post),
        'extended': (p_extended_pre, p_extended_post),
        'robust': (p_robust_pre, p_robust_post),
    }

    for model, (p_pre, p_post) in model_prices.items():
        result_market[f'{model}_pre'] = p_pre
        result_market[f'{model}_post'] = p_post
        result_market[f'{model}_change'] = p_post - p_pre
        result_market[f'{model}_change_pct'] = 100 * (
            p_post / p_pre - 1
        )

    results_list.append(result_market)

    welfare_models = [
        (
            'Simple logit',
            alpha_simple,
            0.0,
            delta_simple,
            p_simple_pre,
            p_simple_post,
        ),
        (
            'Extended nested logit',
            alpha_extended,
            rho_extended,
            delta_extended,
            p_extended_pre,
            p_extended_post,
        ),
        (
            'Simple logit (robustness)',
            alpha_robust,
            0.0,
            delta_simple,
            p_robust_pre,
            p_robust_post,
        ),
    ]

    nests = market_ordered[SEGMENT].to_numpy()
    market_size = float(market_ordered['market_size'].iloc[0])
    market_label = market_ordered['market'].iloc[0]

    for model, alpha, rho, delta_obs, p_pre, p_post in welfare_models:
        # Mean utilities at the pre- and post-merger equilibrium prices.
        delta_pre = delta_obs - alpha * np.log(p_pre / prices)
        delta_post = delta_obs - alpha * np.log(p_post / prices)

        W_pre = nested_logsum(delta_pre, nests, rho)
        W_post = nested_logsum(delta_post, nests, rho)

        welfare_rows.append({
            'market': market_label,
            'ma': MARKET_MA,
            'model': model,
            'change_logsum': W_post - W_pre,
            'change_CS_logprice_units': (W_post - W_pre) / alpha,
            'market_size': market_size,
        })


# Combine and save product-level results across markets.
results = pd.concat(results_list, ignore_index=True)
results.to_csv(
    RES / f'merger_prices_ye_{MARKET_YE}.csv',
    index=False
)

# Summary rows: each market and one pooled row across all markets.
model_specs = [
    ('Simple logit', 'simple_change', 'simple_change_pct'),
    (
        'Extended nested logit',
        'extended_change',
        'extended_change_pct'
    ),
    (
        'Simple logit (robustness)',
        'robust_change',
        'robust_change_pct'
    ),
]

groups = [(ma, g) for ma, g in results.groupby('ma', sort=True)]
groups.append(('All ma', results))

summary_rows = []

for ma, g in groups:
    brand = g['brand'].astype(str).str.casefold()
    fiat_mask = brand.eq('fiat').to_numpy()
    vw_mask = brand.eq('vw').to_numpy()

    # Within a market, weight by product shares.
    # Across all markets, weight by quantities: market size × product share.
    weights = g['share_pre'].to_numpy(dtype=float)
    if ma == 'All ma':
        weights = weights * g['market_size'].to_numpy(dtype=float)

    for model, change_col, pct_col in model_specs:
        max_index = g[change_col].idxmax()
        max_product = g.loc[max_index]

        summary_rows.append({
            'market': ma,
            'model': model,
            'average_pct_change_all_products': weighted_average(
                g[pct_col], weights
            ),
            'largest_price_increase_euros': max_product[change_col],
            'pct_change_for_largest_increase': max_product[pct_col],
            'product_with_largest_increase': max_product['type'],
            'average_pct_change_fiat': weighted_average(
                g.loc[fiat_mask, pct_col], weights[fiat_mask]
            ),
            'average_pct_change_vw': weighted_average(
                g.loc[vw_mask, pct_col], weights[vw_mask]
            ),
        })

merger_overview = pd.DataFrame(summary_rows).round(2)

ma_labels = (
    cars.drop_duplicates('ma')
    .set_index('ma')['market']
)
merger_overview['market'] = (
    merger_overview['market']
    .map(ma_labels)
    .fillna(merger_overview['market'])
)

# Save the overview as CSV and LaTeX.
merger_overview.to_csv(
    RES / f'merger_overview_ye_{MARKET_YE}.csv',
    index=False
)
merger_overview.to_latex(
    RES / f'merger_overview_ye_{MARKET_YE}.tex',
    index=False,
    float_format='%.2f',
    escape=True,
    caption=f'Merger price effects by market and demand model, year {MARKET_YE}',
    label='tab:merger_overview'
)

# Prepare a readable table image.
pretty_table = merger_overview.rename(columns={
    'market': 'Market',
    'model': 'Model',
    'average_pct_change_all_products': 'Average price\nchange (%)',
    'largest_price_increase_euros': 'Largest increase\n(€)',
    'pct_change_for_largest_increase': 'Largest increase\n(%)',
    'product_with_largest_increase': 'Product with\nlargest increase',
    'average_pct_change_fiat': 'Fiat average\nchange (%)',
    'average_pct_change_vw': 'VW average\nchange (%)',
}).copy()

pretty_table['Market'] = pretty_table['Market'].replace(
    {'All ma': 'All markets'}
)
pretty_table['Product with\nlargest increase'] = (
    pretty_table['Product with\nlargest increase']
    .astype(str)
    .str.title()
)

for col in pretty_table.columns:
    if pd.api.types.is_numeric_dtype(pretty_table[col]):
        pretty_table[col] = pretty_table[col].map(
            lambda value: f'{value:.2f}'
        )

fig_height = max(4.5, 0.34 * (len(pretty_table) + 2))
fig, ax = plt.subplots(figsize=(16, fig_height))
ax.axis('off')

col_widths = [0.10, 0.18, 0.13, 0.12, 0.11, 0.18, 0.09, 0.09]
table = ax.table(
    cellText=pretty_table.values,
    colLabels=pretty_table.columns,
    colWidths=col_widths,
    cellLoc='center',
    bbox=[0, 0, 1, 0.92]
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.3)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#D5DDE5')
    cell.set_linewidth(0.6)

    if row == 0:
        cell.set_facecolor('#243B53')
        cell.set_text_props(color='white', weight='bold', va='center')
    else:
        cell.set_facecolor('#F1F5F9' if row % 2 == 0 else 'white')

        if pretty_table.iloc[row - 1, 0] == 'All markets':
            cell.set_facecolor('#DCE8F2')
            cell.set_text_props(weight='bold')

for row in range(1, len(pretty_table) + 1):
    for col in [0, 1, 5]:
        table[row, col].get_text().set_ha('left')
        table[row, col].PAD = 0.08

ax.set_title(
    f'Merger price effects by market, year {MARKET_YE}',
    fontsize=15,
    weight='bold',
    color='#243B53',
    pad=14
)

fig.savefig(
    RES / f'merger_overview_ye_{MARKET_YE}.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close(fig)


# Keep only the three requested measures
measure_labels = {
    'average_pct_change_all_products': 'Average price change (%)',
    'average_pct_change_fiat': 'Fiat average change (%)',
    'average_pct_change_vw': 'VW average change (%)',
}

model_order = [
    'Simple logit',
    'Extended nested logit',
    'Simple logit (robustness)',
]

merger_overview_pivot = (
    merger_overview
    .pivot(
        index='market',
        columns='model',
        values=list(measure_labels)
    )
    .reindex(
        columns=pd.MultiIndex.from_product(
            [list(measure_labels), model_order]
        )
    )
    .rename(columns=measure_labels, level=0)
    .round(2)
)

merger_overview_pivot.columns.names = ['Measure', 'Model']
merger_overview_pivot.index.name = 'Market'

# Save this pivoted overview
merger_overview_pivot.to_csv(
    RES / f'merger_overview_pivot_ye_{MARKET_YE}.csv'
)

# Use a display label for the pooled row
pivot_for_export = merger_overview_pivot.rename(
    index={'All ma': 'All markets'}
)

# Save as LaTeX, keeping the measure/model column groups
latex = pivot_for_export.to_latex(
    multicolumn=True,
    multicolumn_format='c',
    float_format='%.2f',
    caption=f'Merger price changes by market and demand model, year {MARKET_YE}',
    label='tab:merger_overview_pivot',
    escape=True
)

(RES / f'merger_overview_pivot_ye_{MARKET_YE}.tex').write_text(latex)

# Prepare the PNG table with measure and model on separate lines per column
column_labels = [
    f'{measure}\n{model}'
    for measure, model in pivot_for_export.columns
]

cell_data = []
for market_name, row in pivot_for_export.iterrows():
    formatted_values = [
        f'{value:.2f}' if pd.notna(value) else ''
        for value in row.to_numpy()
    ]
    cell_data.append([str(market_name)] + formatted_values)

fig_height = max(3.5, 0.42 * (len(cell_data) + 2))
fig, ax = plt.subplots(figsize=(19, fig_height))
ax.axis('off')

table = ax.table(
    cellText=cell_data,
    colLabels=['Market'] + column_labels,
    colWidths=[0.09] + [0.101] * len(column_labels),
    cellLoc='center',
    bbox=[0, 0, 1, 0.90]
)

table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.4)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#D5DDE5')
    cell.set_linewidth(0.6)

    if row == 0:
        cell.set_facecolor('#243B53')
        cell.set_text_props(color='white', weight='bold', va='center')
    else:
        cell.set_facecolor('#F1F5F9' if row % 2 == 0 else 'white')

        if cell_data[row - 1][0] == 'All markets':
            cell.set_facecolor('#DCE8F2')
            cell.set_text_props(weight='bold')

for row in range(1, len(cell_data) + 1):
    table[row, 0].get_text().set_ha('left')
    table[row, 0].PAD = 0.08

ax.set_title(
    f'Merger price changes by market and model, year {MARKET_YE}',
    fontsize=14,
    weight='bold',
    color='#243B53',
    pad=14
)

fig.savefig(
    RES / f'merger_overview_pivot_ye_{MARKET_YE}.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close(fig)


consumer_surplus = pd.DataFrame(welfare_rows)

# Add a market-size-weighted mean across markets for each model.
pooled_rows = []
for model, g in consumer_surplus.groupby('model', sort=False):
    pooled_rows.append({
        'market': 'All markets (weighted mean)',
        'ma': 'All',
        'model': model,
        'change_logsum': np.average(
            g['change_logsum'], weights=g['market_size']
        ),
        'change_CS_logprice_units': np.average(
            g['change_CS_logprice_units'], weights=g['market_size']
        ),
        'market_size': np.nan,
    })

consumer_surplus_table = pd.concat(
    [consumer_surplus, pd.DataFrame(pooled_rows)],
    ignore_index=True
)

consumer_surplus_table = consumer_surplus_table[
    ['market', 'ma', 'model', 'change_logsum', 'change_CS_logprice_units']
]

consumer_surplus_table.to_latex(
    RES / f'consumer_surplus_ye_{MARKET_YE}.tex',
    index=False,
    float_format='%.8f',
    escape=True,
    caption=(
        f'Merger welfare changes by market and demand model, year {MARKET_YE}. '
        r'$\Delta CS/\alpha$ is in log-price units.'
    ),
    label='tab:consumer_surplus'
)

print(
    f"Saved LaTeX table to: "
    f"{(RES / f'consumer_surplus_ye_{MARKET_YE}.tex').resolve()}"
)


# Calculate market-level HHI before and after the Fiat–VW merger
hhi_rows = []

market_data = cars.loc[cars['ye'].eq(MARKET_YE)].copy()

for ma, g in market_data.groupby('ma', sort=True):
    g = g.copy()
    g['owner_pre'] = g['brand'].astype(str).str.strip().str.casefold()

    # Treat Fiat and VW as one firm after the merger
    g['owner_post'] = g['owner_pre'].replace({
        'fiat': 'fiat_vw_merged',
        'vw': 'fiat_vw_merged'
    })

    # Total cars sold in this market
    total_sales = g['qu'].sum()

    if total_sales <= 0:
        raise ValueError(f'Total sales are not positive for ma={ma}, ye={MARKET_YE}')

    # Sum model sales to get each firm's sales
    firm_sales_pre = g.groupby('owner_pre')['qu'].sum()
    firm_sales_post = g.groupby('owner_post')['qu'].sum()

    # Firm shares of total car sales
    firm_shares_pre = firm_sales_pre / total_sales
    firm_shares_post = firm_sales_post / total_sales

    # HHI, on the 0–10,000 scale
    hhi_pre = 10_000 * (firm_shares_pre ** 2).sum()
    hhi_post = 10_000 * (firm_shares_post ** 2).sum()

    hhi_rows.append({
        'Market': g['market'].iloc[0],
        'HHI before merger': hhi_pre,
        'HHI after merger': hhi_post,
        'Change in HHI': hhi_post - hhi_pre,
        'total_sales': total_sales
    })

hhi_overview = pd.DataFrame(hhi_rows)

# Add the sales-weighted average across markets
weighted_row = {
    'Market': 'All markets',
    'HHI before merger': np.average(
        hhi_overview['HHI before merger'],
        weights=hhi_overview['total_sales']
    ),
    'HHI after merger': np.average(
        hhi_overview['HHI after merger'],
        weights=hhi_overview['total_sales']
    ),
    'Change in HHI': np.average(
        hhi_overview['Change in HHI'],
        weights=hhi_overview['total_sales']
    )
}

hhi_overview = pd.concat(
    [hhi_overview, pd.DataFrame([weighted_row])],
    ignore_index=True
).drop(columns='total_sales')

# Save the HHI table as LaTeX
hhi_path = RES / f'hhi_merger_ye_{MARKET_YE}.tex'

hhi_latex = hhi_overview.to_latex(
    index=False,
    caption=f'HHI before and after the Fiat--VW merger, year {MARKET_YE}',
    label='tab:hhi_merger',
    column_format='lrrr',
    float_format=lambda x: f'{x:.1f}',
    escape=True,
    position='!htbp'
)

hhi_path.write_text(hhi_latex, encoding='utf-8')
print(f'Saved HHI table to: {hhi_path.resolve()}')