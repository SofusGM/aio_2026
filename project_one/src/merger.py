import pandas as pd
import numpy as np
from pathlib import Path
import parts
import matplotlib.pyplot as plt

DATA = Path('../data')
RES = Path('../res/merger')
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


def elasticity(pairs, alpha, rho):
    r = 1 / (1 - rho)
    own = pairs['co_j'] == pairs['co_k']
    same_segment = pairs[f'{SEGMENT}_j'] == pairs[f'{SEGMENT}_k']
    dp = 1 if LOG_PRICE else pairs[PRICE_VAR]

    return -alpha * dp * (r * own - (r - 1) * same_segment * pairs['s'] / pairs['segment_share']  - pairs['s'])

results_list = []

market_codes = (cars.loc[cars['ye'].eq(MARKET_YE), 'ma'].drop_duplicates().tolist())

for MARKET_MA in market_codes:

    market = (cars.loc[cars['ma'].eq(MARKET_MA) & cars['ye'].eq(MARKET_YE)].sort_values('brand').reset_index(drop=True))

    if market.empty:
        continue

    # Every product pair in this market; s and segment_share refer to product k
    cols = ['ma', 'ye', 'co', SEGMENT]
    pairs = market[cols].merge(market[cols + ['s', 'segment_share', PRICE_VAR]], on=['ma', 'ye'],suffixes=('_j', '_k'))

    pairs['simple'] = elasticity(pairs, alpha_simple, rho=0)
    pairs['extended'] = elasticity(pairs, alpha_extended, rho_extended)

    # Rows: share responding; columns: price changing
    product_order = market['co'].tolist()

    E_simple = (pairs.pivot(index='co_j', columns='co_k', values='simple').reindex(index=product_order, columns=product_order))
    E_extended = (pairs.pivot(index='co_j', columns='co_k', values='extended').reindex(index=product_order, columns=product_order))

    # Align shares and prices with the elasticity matrices
    market_ordered = market.set_index('co').loc[product_order]
    shares = market_ordered['s'].to_numpy()
    prices = market_ordered[PRICE_VAR].to_numpy()

    def share_jacobian(E):
        values = E.to_numpy().T * (shares[None, :] / prices[:, None])
        return values

    J_simple = share_jacobian(E_simple)
    J_extended = share_jacobian(E_extended)

    # Ownership matrices before and after the merger
    j = len(market)
    H = np.zeros((j, j))
    H_merge = np.zeros((j, j))

    for i in range(j):
        for k in range(j):
            if market.loc[i, 'brand'] == market.loc[k, 'brand']:
                H[i, k] = 1

            if (
                market.loc[i, 'brand'] == market.loc[k, 'brand']
                or (market.loc[i, 'brand'] == 'VW'
                    and market.loc[k, 'brand'] == 'fiat')
                or (market.loc[i, 'brand'] == 'fiat'
                    and market.loc[k, 'brand'] == 'VW')
            ):
                H_merge[i, k] = 1

    # Marginal costs, calculated separately for each demand model
    markup_simple = np.linalg.solve(H * J_simple, -shares)
    mc_simple = prices - markup_simple

    markup_extended = np.linalg.solve(H * J_extended, -shares)
    mc_extended = prices - markup_extended

    # Recover mean utility from observed shares
    nests = market_ordered[SEGMENT].to_numpy()
    nest_id = nests.reshape(1, -1)
    nest_data = {'nest_id': nest_id}

    s0_obs = float(market_ordered['s0'].iloc[0])
    S_obs = np.r_[s0_obs, shares].reshape(1, -1)

    delta_simple = parts.invert_demand(S_obs, rho=0, nest_id=nest_id)[0]
    delta_extended = parts.invert_demand(S_obs, rho=rho_extended, nest_id=nest_id)[0]

    # Solve pre- and post-merger prices
    p_simple_pre = parts.solve_nash(H, mc_simple, prices, delta_simple, nest_data, alpha_simple, rho=0)
    p_simple_post = parts.solve_nash(H_merge, mc_simple, prices, delta_simple, nest_data, alpha_simple, rho=0)

    p_extended_pre = parts.solve_nash(H, mc_extended, prices, delta_extended, nest_data, alpha_extended, rho=rho_extended)
    p_extended_post = parts.solve_nash(H_merge, mc_extended, prices, delta_extended, nest_data, alpha_extended, rho=rho_extended)

    # Keep product identifiers and prices for this market
    result_market = market_ordered.reset_index()[['ma', 'ye', 'co', 'type', 'brand', 's']].rename(columns={'s': 'share_pre'})

    result_market['price_observed'] = prices

    result_market['simple_pre'] = p_simple_pre
    result_market['simple_post'] = p_simple_post
    result_market['simple_change'] = p_simple_post - p_simple_pre
    result_market['simple_change_pct'] = (100 * (p_simple_post / p_simple_pre - 1)
    )

    result_market['extended_pre'] = p_extended_pre
    result_market['extended_post'] = p_extended_post
    result_market['extended_change'] = p_extended_post - p_extended_pre
    result_market['extended_change_pct'] = (100 * (p_extended_post / p_extended_pre - 1))

    # Check pre-merger prices reproduce observed prices
    print(
        f"ma={MARKET_MA}: max pre-merger price difference — "
        f"simple {np.max(np.abs(p_simple_pre - prices)):.6g}; "
        f"extended {np.max(np.abs(p_extended_pre - prices)):.6g}"
    )

    results_list.append(result_market)


# Combine results from all ma markets
results = pd.concat(results_list, ignore_index=True)


results.to_csv(RES / f'merger_prices_ye_{MARKET_YE}.csv', index=False)

model_specs = [
    ('Simple logit', 'simple_change', 'simple_change_pct'),
    ('Extended nested logit', 'extended_change', 'extended_change_pct'),
]

# Summaries for each ma, plus one pooled summary across all ma
groups = [(ma, g) for ma, g in results.groupby('ma', sort=True)]
groups.append(('All ma', results))

summary_rows = []

for ma, g in groups:
    brand = g['brand'].astype(str).str.casefold()

    for model, change_col, pct_col in model_specs:
        # Product with the largest price increase
        max_index = g[change_col].idxmax()
        max_product = g.loc[max_index]

        summary_rows.append({
            'market': ma,
            'model': model,
            'average_pct_change_all_products': g[pct_col].mean(),
            'largest_price_increase_euros': max_product[change_col],
            'pct_change_for_largest_increase': max_product[pct_col],
            'product_with_largest_increase': max_product['type'],
            'average_pct_change_fiat': g.loc[brand.eq('fiat'), pct_col].mean(),
            'average_pct_change_vw': g.loc[brand.eq('vw'), pct_col].mean(),
        })


merger_overview = pd.DataFrame(summary_rows).round(2)
ma_labels = cars.drop_duplicates('ma').set_index('ma')['market']

merger_overview['market'] = merger_overview['market'].map(ma_labels).fillna(merger_overview['market'])

# Shorter labels for the display
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

pretty_table['Market'] = pretty_table['Market'].replace({'All ma': 'All markets'})
pretty_table['Product with\nlargest increase'] = (
    pretty_table['Product with\nlargest increase'].astype(str).str.title()
)

# Format numeric cells consistently
for col in pretty_table.columns:
    if pd.api.types.is_numeric_dtype(pretty_table[col]):
        pretty_table[col] = pretty_table[col].map(lambda x: f'{x:.2f}')

fig, ax = plt.subplots(figsize=(16, 5.5))
ax.axis('off')

col_widths = [0.09, 0.14, 0.14, 0.12, 0.11, 0.20, 0.10, 0.10]

table = ax.table(
    cellText=pretty_table.values,
    colLabels=pretty_table.columns,
    colWidths=col_widths,
    cellLoc='center',
    bbox=[0, 0, 1, 0.92]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.35)

# Style the header, rows, and borders
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

# Left-align text columns
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
    RES / 'merger_overview.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.close(fig)