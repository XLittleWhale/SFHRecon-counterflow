"""Portable locations. Never search outside the configured data/work directories."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('COUNTERFLOW_HOME', Path(__file__).resolve().parents[2])).resolve()
DATA = Path(os.environ.get('COUNTERFLOW_DATA', ROOT/'data')).resolve()
RUN = Path(os.environ.get('COUNTERFLOW_RUN', ROOT/'runs/current')).resolve()
RESULTS = RUN/'Results'
ASSETS = ROOT/'artifacts'
CONFIG = ROOT/'config'


def data_path(relative):
    path=(DATA/relative).resolve()
    if not path.is_relative_to(DATA):
        raise ValueError('Data paths must stay within COUNTERFLOW_DATA')
    if not path.is_file():
        raise FileNotFoundError(f'Missing data asset: {relative}. See docs/DATA.md.')
    return path
