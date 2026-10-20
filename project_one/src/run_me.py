"""Run every analysis script in order."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg') # save figures without opening windows, so plt.show() does not block
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent

# in dependency order
SCRIPTS = [
    'market_size.py',           # adds households_pred to data/cars.csv (market size 'hh')
    'market shares.py',
    'demand_models.py',         # writes data/demand_model_parameters.csv, read by the two below
    'characteristics_test.py',
    'diversionratios.py',
    'merger.py',
]

for name in SCRIPTS:
    print(f'\n{"=" * 80}\n{name}\n{"=" * 80}', flush=True)
    exec((SRC / name).read_text(), {'__name__': '__main__'}) # fresh namespace per script
    plt.close('all')
