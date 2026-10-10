""" Storing functions that may be used in multiple py-files
"""

from pandas import DataFrame
import numpy as np
from linearmodels.iv import IV2SLS


def market_size_function(data: DataFrame, type: str = 'pop'
                         ):
    """Return (market_size, s, s0): market size, product share and outside share per row."""

    if type == 'pop':
        market_size = data['pop'] / 3

    elif type == 'sales': # replace this with some new type
        market_size = None

    else:
        raise ValueError(f"Unknown market size type: {type}")

    # product share and outside share within each market-year (ma, ye)
    s = data['qu'] / market_size
    s0 = 1.0 - s.groupby([data['ma'], data['ye']]).transform('sum')

    return market_size, s, s0


def invert_demand_nest_2levels(data: DataFrame, rho, nest: str, subgroup: str
                               ):
    """delta for the two-level nested logit (based on demand.pdf). Returns a (T*J,) array.
    rho = (rho_1, rho_2): rho_1 is the correlation within subgroup, rho_2 within nest.
    delta_jt = ln(s_jt) - ln(s_0t) - rho_1 * ln(s_{j|hg,t}) - rho_2 * ln(s_{h|g,t})
    """
    rho_1, rho_2 = rho

    # total share of each nest g and each subgroup h within g, per market
    s_g = data['s'].groupby([data['ma'], data['ye'], data[nest]]).transform('sum')
    s_hg = data['s'].groupby([data['ma'], data['ye'], data[nest], data[subgroup]]).transform('sum')

    s_j_hg = data['s'] / s_hg   # product share within its subgroup
    s_h_g = s_hg / s_g          # subgroup share within its nest

    return (np.log(data['s']) - np.log(data['s0'])
            - rho_1 * np.log(s_j_hg) - rho_2 * np.log(s_h_g)).values
    
def invert_demand_nest_1levels(data: DataFrame, rho, nest: str
                               ):
    """delta(rho) for the nested logit. Returns a (T*J,) array."""
    rho = np.atleast_1d(rho)[0]

    s_g = data['s'].groupby([data['ma'], data['ye'], data[nest]]).transform('sum')

    return (np.log(data['s']) - np.log(data['s0'])
            - rho * np.log(data['s'] / s_g)).values


def Q_nested(theta2, data: DataFrame, X, Z, W, nest: str, subgroup: str, sample
             ):
    """GMM criterion for the nested logit (two levels if len(theta2) == 2, else one).
    Demand is inverted on the full data (correct within-nest shares); moments use the rows in `sample`.
    The linear parameters are concentrated out by GMM with the same Z and W.
    """
    delta = (invert_demand_nest_2levels(data, theta2, nest, subgroup) if len(theta2) == 2
             else invert_demand_nest_1levels(data, theta2, nest))[sample]

    ZX, Zd = Z.T @ X, Z.T @ delta
    beta = np.linalg.solve(ZX.T @ W @ ZX, ZX.T @ W @ Zd)
    g = Zd - ZX @ beta      # = Z'xi
    return g @ W @ g


def instruments_function(data: DataFrame, var_type: str, nest: str = 'home', subgroup: str = 'cla'
                         ):
    """Return one instrument (Series named var_type):
    'log_xexr' -> log real exchange rate, log(avdexr/avexr)
    'hausman'  -> avg. log eurpr of the same model in the other markets, same year (NaN if sold in one market)
    'we_rival' -> sum of weight of the other models in the same nest/subgroup
    'n_sub'    -> log number of models in the same nest/subgroup
    'n_nest'   -> log number of models in the same nest
    """

    sub = data.groupby(['ma', 'ye', nest, subgroup])

    if var_type == 'log_xexr':
        z = np.log(data['xexr'])

    elif var_type == 'hausman':
        lp = np.log(data['eurpr'])
        g = lp.groupby([data['co'], data['ye']])
        z = (g.transform('sum') - lp) / (g.transform('size') - 1)

    elif var_type == 'we_rival':
        z = sub['we'].transform('sum') - data['we']

    elif var_type == 'n_sub':
        z = np.log(sub['qu'].transform('size'))

    elif var_type == 'n_nest':
        z = np.log(data.groupby(['ma', 'ye', nest])['qu'].transform('size'))

    else:
        raise ValueError(f"Unknown instrument type: {var_type}")

    return z.rename(var_type)
