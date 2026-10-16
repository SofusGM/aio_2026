""" Storing functions that may be used in multiple py-files
"""

from pandas import DataFrame
import numpy as np
from scipy import stats
from linearmodels.iv import IV2SLS


def market_size_function(data: DataFrame, type: str = 'pop'
                         ):
    """Return (market_size, s, s0): market size, product share and outside share per row."""

    if type == 'pop':
        market_size = data['pop'] / 3
    elif type == 'hh': # Predicted number of households in each market each year
        market_size = data['households_pred']
    elif type == 'hh_adjust': # Predicted number of households in each market each year adjusted for the number of cars per household and replacement rate of cars
        market_size = data['households_pred'] / 2.5 # this number is taking from a paper, but may need to be adjustet
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


def instruments_function(data: DataFrame, name: str, segment: str = 'cla', subgroup: str = 'home'
                         ):
    """Return one instrument (Series named `name`). All groupings are within market-year (ma, ye).
    Cost shifters (move the price):
    'log_exchange_rate'       -> log(xexr) = log(avdexr/avexr), destination currency per exporter currency
    'log_producer_prices'     -> log(avppr), producer price index of the exporter country (NaN 1970-80 for some loc)
    'log_producer_prices_sdr' -> log(avppr/avexr), exporter producer prices in common currency (SDR)
    Competition instruments (move the within-segment share):
    'sum_weight_subgroup'     -> sum of weight (we) of the other models in the same segment x subgroup
    'log_count_subgroup'      -> log number of models in the same segment x subgroup (incl. j)
    'log_count_segment'       -> log number of models in the same segment (incl. j)
    'distance_segment_<x>'    -> Gandhi-Houde IV as in PS4: mean |x_j - x_k| over the other models in j's segment,
                                 one characteristic x per instrument; 0 if alone in the segment (n = max(len - 1, 1))
    """
    by_segment = [data['ma'], data['ye'], data[segment]]
    by_subgroup = by_segment + [data[subgroup]]

    if name == 'log_exchange_rate':
        z = np.log(data['xexr'])

    elif name == 'log_producer_prices':
        z = np.log(data['avppr'])

    elif name == 'log_producer_prices_sdr':
        z = np.log(data['avppr'] / data['avexr'])

    elif name == 'sum_weight_subgroup':
        z = data['we'].groupby(by_subgroup).transform('sum') - data['we']

    elif name == 'log_count_subgroup':
        z = np.log(data['qu'].groupby(by_subgroup).transform('size'))

    elif name == 'log_count_segment':
        z = np.log(data['qu'].groupby(by_segment).transform('size'))

    elif name.startswith('distance_segment_'):
        x = data[name[len('distance_segment_'):]]
        # sum_k |x_j - x_k| over the segment (the k = j term is zero), divided by the number of other models
        z = x.groupby(by_segment).transform(
            lambda v: np.abs(v.values[:, None] - v.values[None, :]).sum(axis=1) / max(len(v) - 1, 1))

    else:
        raise ValueError(f"Unknown instrument: {name}")

    return z.rename(name)


def first_stage_F(X, Z, kw):
    """First-stage F per endogenous regressor, with the exogenous regressors already partialled out of X and Z.
    One endogenous regressor: robust F (robust Wald / # instruments).
    Several: Sanderson-Windmeijer conditional F (non-robust) for each. kw = # partialled-out exogenous regressors.
    """
    PZ = lambda A: Z @ np.linalg.lstsq(Z, A, rcond=None)[0]
    if X.shape[1] == 1:
        ZZi = np.linalg.inv(Z.T @ Z)
        g = ZZi @ Z.T @ X[:, 0]
        u = X[:, 0] - Z @ g
        return [g @ np.linalg.solve(ZZi @ ((Z * u[:, None]).T @ (Z * u[:, None])) @ ZZi, g) / Z.shape[1]]
    out = []
    for j in range(X.shape[1]):
        Xo = np.delete(X, j, axis=1)
        e = X[:, j] - Xo @ np.linalg.lstsq(PZ(Xo), X[:, j], rcond=None)[0]
        pe = PZ(e)
        out.append((pe @ pe / (Z.shape[1] - X.shape[1] + 1)) / ((e - pe) @ (e - pe) / (len(e) - Z.shape[1] - kw)))
    return out


def overid_pval(y, X, Z, e):
    """Robust two-step GMM (Hansen J) p-value of the overidentifying restrictions; NaN if just identified.
    y, X, Z with the exogenous regressors partialled out; e = 2SLS residuals.
    """
    if Z.shape[1] == X.shape[1]:
        return np.nan
    Si = np.linalg.inv((Z * e[:, None]).T @ (Z * e[:, None]))
    g = Z.T @ (y - X @ np.linalg.solve(X.T @ Z @ Si @ Z.T @ X, X.T @ Z @ Si @ Z.T @ y))
    return stats.chi2.sf(g @ Si @ g, Z.shape[1] - X.shape[1])
