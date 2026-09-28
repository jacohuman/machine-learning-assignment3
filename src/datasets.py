"""
Processing of the statsmodels datasets
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]

DATA_SPECS = {
    'Nile': dict(period=1, transform='level', unit='10^8 m^3', frequency='annual'),
    'Sunspots': dict(period=1, transform='level', unit='sunspot number', frequency='annual'),
    'CO2': dict(period=12, transform='seasonal_difference', unit='ppm', frequency='monthly'),
    'ElNino': dict(period=12, transform='level', unit='degrees C', frequency='monthly'),
    'GDP': dict(period=1, transform='log_difference', unit='billion 2005 USD', frequency='quarterly'),
}


def export_data():
    """Save original-scale series data"""
    target = ROOT / 'data'
    target.mkdir(exist_ok=True)
    n = sm.datasets.nile.load_pandas().data
    s = sm.datasets.sunspots.load_pandas().data
    co2 = sm.datasets.co2.load_pandas().data['co2'].resample('MS').mean()
    en = sm.datasets.elnino.load_pandas().data
    gdp = sm.datasets.macrodata.load_pandas().data
    series = {
        'Nile': (n['year'].astype(int).astype(str), n['volume'].to_numpy()),
        'Sunspots': (s['YEAR'].astype(int).astype(str), s['SUNACTIVITY'].to_numpy()),
        'CO2': (co2.index.strftime('%Y-%m'), co2.to_numpy()),
        'ElNino': ([f'{int(row.YEAR)}-{m:02d}' for row in en.itertuples() for m in range(1, 13)],
                   en.drop(columns='YEAR').to_numpy().ravel()),
        'GDP': ([f'{int(r.year)}Q{int(r.quarter)}' for r in gdp.itertuples()], gdp['realgdp'].to_numpy()),
    }
    for name, (dates, values) in series.items():
        pd.DataFrame({'date': list(dates), 'value': values}).to_csv(target / f'{name}.csv', index=False)
    (target / 'metadata.json').write_text(json.dumps(DATA_SPECS, indent=2) + '\n')


def load_data():
    """
    Reads the local CSVs. If any of the expected files are missing,
    it calls `export_data()` to regenerate the five files from statsmodels.
    """
    if not all((ROOT / 'data' / f'{name}.csv').exists() for name in DATA_SPECS):
        export_data()
    return {name: pd.read_csv(ROOT / 'data' / f'{name}.csv', dtype={'date': str}) for name in DATA_SPECS}


def transform(values, method):
    """
    Forward fill uses past only. Missing actual targets are excluded later.
    """
    raw = pd.Series(values).ffill().to_numpy(dtype=float)
    z = raw.copy()
    offset = 0
    if method == 'seasonal_difference':
        z, offset = raw[12:] - raw[:-12], 12
    elif method == 'log_difference':
        z, offset = np.diff(np.log(raw)), 1
    return raw, z, offset


def windows(z, indices, offset, mean, scale, length=12):
    """
    Target raw index t uses transformed observations strictly before t.
    """
    j = np.asarray(indices) - offset
    x = np.stack([z[t-length:t] for t in j])
    return (x - mean) / scale, (z[j] - mean) / scale


def inverse(pred, indices, raw, method):
    """
    Undo the data-specific transformation.
    """
    if method == 'seasonal_difference':
        return pred + raw[np.asarray(indices) - 12]
    if method == 'log_difference':
        return np.exp(pred) * raw[np.asarray(indices) - 1]
    return pred
