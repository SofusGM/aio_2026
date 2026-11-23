'''What you built in Problem Set 8, packaged so that Problem Set 9 can stand on it.

Nothing here is new. make_fundamentals() is PS8's "Simulate Fundamentals" section, with the same
seed, so you get the same consumers and the same numbers as in PS8. simulate_search() is
PS8's Q3, Q4 and Q5.a in one function.

Series are indexed by (i, j), and j = 0 is the outside option, exactly as in PS8.
'''
import warnings
import numpy as np
import pandas as pd
from scipy import special
from scipy.interpolate import interp1d

beta_true = np.array([-.2867, 0.3753, -0.2465, -0.0089, -0.1477, 0.1584, -0.05, 1.5])
beta_labels = ['price', 'stars', 'review', 'chain', 'location', 'promotion', 'position', 'constant']
gamma_true = np.array([0.25, 1.0])
gamma_labels = ['position', 'constant']

# the numerical inverse from PS8 Q2
z_grid = np.linspace(-10, 20, 10_000)
H_grid = special.exp1(np.exp(-z_grid)) - z_grid + np.euler_gamma
H_inv_func = interp1d(H_grid, z_grid, fill_value='extrapolate')


def make_fundamentals(N=100_000, J=5, seed=1):
    '''PS8's covariates and match values. Returns df_utility, df_search, epsilon.'''
    np.random.seed(seed)
    idx = pd.MultiIndex.from_product([range(N), range(J+1)], names=['i', 'j'])
    df_utility = pd.DataFrame({
        col: np.random.normal(0, 1, size=(N, J+1)).flatten() for col in beta_labels[:6]
    }, idx)
    df_utility['position'] = np.tile(np.arange(J+1), N).astype(float)
    df_utility['constant'] = 1.0
    df_utility.loc[(slice(None), 0), :] = 0.0 # delta_i0 = 0
    df_search = df_utility[gamma_labels].copy() # mu_i0 = 0
    epsilon = pd.Series(np.random.gumbel(0, 1, size=(N*(J+1),)), idx)
    return df_utility, df_search, epsilon


def draw_search_costs(mu, seed=2):
    '''PS8's sampler: draws c_ij from F^c(c|mu_ij). See PS8 for how it works.'''
    rng = np.random.default_rng(seed)
    m = mu.values
    I_negative = m < 0 # F^c(.|mu) is not a distribution when mu < 0
    if I_negative.any():
        warnings.warn(f'{I_negative.sum()} values of mu are negative, where F^c is not a distribution. '
                      'Setting c = 0 (free search) for them: pragmatic but crude, see ps9.pdf.')
    U = rng.uniform(size=m.shape)
    c = np.zeros(m.shape) # the mass point, c = 0, has probability exp(-mu)
    I_bisect = U > np.exp(-m) # off the mass point: only these need the bisection (never where mu < 0)
    m_, U_ = m[I_bisect], U[I_bisect]
    survival = lambda z: np.expm1(-np.exp(-(z + m_))) / np.expm1(-np.exp(-z))
    lo, hi = np.full(m_.shape, -10.0), np.full(m_.shape, 30.0)
    for _ in range(40): # the bracket is 40 wide, so 40 halvings leave an error below 1e-10
        mid = (lo + hi) / 2
        above = survival(mid) > U_
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    z = (lo + hi) / 2
    c[I_bisect] = special.exp1(np.exp(-z)) - z + np.euler_gamma
    return pd.Series(c, index=mu.index)


def simulate_search(delta, mu, epsilon, seed=2):
    '''PS8 Q3 to Q5.a. Returns the option bought, d (by i), and the search dummy, searched (by i, j).'''
    c = draw_search_costs(mu, seed)
    r = (delta + H_inv_func(c)).where(c > 0, np.inf)
    u = delta + epsilon
    w = np.fmin(r, u)
    wide = w.unstack(level='j')
    d = pd.Series(wide.values.argmax(axis=1), wide.index, name='d')
    searched = r >= w.groupby('i').transform('max')
    return d, searched
