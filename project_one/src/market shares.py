"""Market structure table: VW- and Fiat-group sales shares and the outside share, 1970-1999 average."""

import pandas as pd
from pathlib import Path
import parts

DATA = parts.DATA
RES = parts.RES / 'market size'
RES.mkdir(parents=True, exist_ok=True)

MARKET_SIZE = 'hh'                 # same market size as in the demand estimation
VW_FRM, FIAT_FRM = 26, 4           # merging firms (frm codes)

cars = pd.read_csv(DATA / 'cars.csv')
if 'market' not in cars.columns:
    lbl_vals = pd.read_stata(DATA / 'cars.dta', iterator=True).value_labels()
    cars['market'] = cars['ma'].map(lbl_vals['market'])

# outside share s0
cars['market_size'], cars['s'], cars['s0'] = parts.market_size_function(data=cars, type=MARKET_SIZE)

# shares in each market-year ...
annual = []
for (market, ye), g in cars.groupby(['market', 'ye'], sort=True):
    total = g['qu'].sum()
    annual.append({
        'market': market,
        'VW group share (\\%)': 100 * g.loc[g['frm'] == VW_FRM, 'qu'].sum() / total,
        'Fiat group share (\\%)': 100 * g.loc[g['frm'] == FIAT_FRM, 'qu'].sum() / total,
        'Outside share $s_0$ (\\%)': 100 * g['s0'].iloc[0],
    })

# ... averaged over the years for each market
table = pd.DataFrame(annual).groupby('market').mean().T
print(table.round(1))

# ---------- LaTeX ----------
cols = list(table.columns)
tex = [r'\begin{table}[!htbp]', r'\centering',
       r'\caption{Market shares of the merging firms, 1970--1999 average}',
       r'\label{tab:market_structure}', r'\small', r'\setlength{\tabcolsep}{5pt}',
       r'\renewcommand{\arraystretch}{0.95}', '',
       r'\begin{tabular}{l' + 'r' * len(cols) + '}', r'\toprule',
       '& ' + ' & '.join(cols) + r' \\', r'\midrule']
for name in table.index:
    tex.append(f'{name:<28} & ' + ' & '.join(f'{v:.1f}' for v in table.loc[name, cols]) + r' \\')
tex += [r'\bottomrule', r'\end{tabular}', '', r'\vspace{2pt}',
        r'\begin{minipage}{0.85\textwidth}', r'\footnotesize',
        r'\emph{Notes:} Firm shares are shares of car sales in the market (top-40 models).',
        r'The outside share is the share of households not buying a new car.',
        r'Averages of the annual shares over 1970--1999.',
        r'\end{minipage}', r'\end{table}']

(RES / 'market_structure.tex').write_text('\n'.join(tex) + '\n')
print(f'Saved {RES / "market_structure.tex"}')