"""Attach the existing causal 1 ms PCHIP/pilot features to native CF intervals.

This module never fits a model or generates additional observation targets.
Each native interval contains one history per internal query midpoint.  The
historical source uses only camera positions through that interval's start;
later history samples follow its previous measured secant, not its endpoint.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import torch

BASE = Path(__file__).resolve().parents[3]
CASES = ('cf_113_camera100_f0184', 'cf_217_camera100_f0333')
HISTORY_CONVENTION = (
    'midpoint+[-5,-4,-3,-2,-1,0]ms; causal PCHIP through native interval start; '
    'past-secant linear pilot afterwards; channels dx/(U*1ms), offset/(5*U*1ms)'
)


def source_paths(case):
    if case not in CASES:
        raise ValueError('Only the two fixed CF windows are supported')
    return dict(history_input=BASE/'github/data/inputs'/f'{case}.pt',
                source_rows=BASE/'github/data/experimental'/('_'.join(case.split('_')[:2]))/'source_rows.json')


def source_metadata(case):
    result = {}
    for key, path in source_paths(case).items():
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        result[key] = dict(path=str(path.resolve()), sha256=digest)
    return result


@lru_cache(maxsize=2)
def _mapping(case):
    old = torch.load(source_paths(case)['history_input'], map_location='cpu', weights_only=False)[0]
    fps = float(old['camera_fps'])
    origin = float(old['start_s'])
    rows = {}
    for kind, state in (('free', 0), ('cap', 1)):
        for pid, packs in old['camera'][kind].items():
            for pack in packs:
                for j, endpoints in enumerate(pack['native_times']):
                    key = (int(pid), state, round(float(endpoints[0])*fps),
                           round(float(endpoints[1])*fps))
                    if key in rows:
                        raise RuntimeError(f'Duplicate native interval in causal history source: {key}')
                    cols = torch.where(pack['weights'][j] > 0)[0]
                    zz = pack['z']
                    rows[key] = dict(
                        h=zz['h'][cols].detach().clone(),
                        time=(zz['frame'][cols]+zz['time_fraction'][cols])*.001,
                        dt=pack['weights'][j, cols]*.001,
                        x=zz['x'][cols].detach().clone(),
                        w=zz['w'][cols].detach().clone(),
                        target=pack['target'][j].detach().clone(),
                    )
    return fps, origin, float(old['physics']['vn_mean_m_s']), rows


def attach_causal_history(c, z):
    """Return a shallow copy of z with h[N,M,5,4], keeping native targets fixed.

    Time/pilot/normalization assertions ensure the stored history is attached to
    its original midpoint.  Zero-duration padding copies the last valid history
    solely to keep network inputs finite; the probability propagator must mask
    it.  A different integration partition needs newly evaluated causal history
    and is rejected here rather than silently using the wrong midpoint.
    """
    case = c['name']
    fps, origin, U, lookup = _mapping(case)
    if abs(float(c['camera_fps'])-fps) > 1e-10 or abs(float(c['start_s'])-origin) > 1e-12:
        raise ValueError('CF history source sampling or window origin changed')
    if abs(float(c['physics']['vn_mean_m_s'])-U) > 1e-15:
        raise ValueError('CF history source velocity normalization changed')
    if z['time'].ndim != 2 or z['dt'].shape != z['time'].shape:
        raise ValueError('Expected native interval quadrature arrays [N,M]')
    history = torch.empty((*z['time'].shape, 5, 4), dtype=z['x'].dtype, device=z['x'].device)
    for i in range(len(z['pid'])):
        key = (int(z['pid'][i]), int(z['state'][i]),
               round((float(z['start'][i])*.001+origin)*fps),
               round((float(z['end'][i])*.001+origin)*fps))
        if key not in lookup:
            raise ValueError(f'Native interval has no frozen causal history: {key}')
        row = lookup[key]
        active = torch.where(z['dt'][i] > 0)[0]
        if len(active) != len(row['h']) or len(active) == 0:
            raise ValueError('Changed integration partition: stored histories do not match')
        checks = (
            ('time', z['time'][i, active]*.001, row['time'], 1e-12),
            ('dt', z['dt'][i, active], row['dt'], 1e-12),
            ('pilot', z['x'][i, active], row['x'], 1e-14),
            ('velocity scale', z['w'][i].expand(len(active), 2), row['w'], 1e-12),
            ('target scale', z['target'][i], row['target'], 1e-12),
        )
        for label, actual, expected, tolerance in checks:
            if actual.shape != expected.shape or not torch.allclose(
                    actual.detach().cpu().double(), expected.double(), rtol=0., atol=tolerance):
                raise ValueError(f'Native {label} disagrees with causal history source: {key}')
        hh = row['h'].to(history)
        history[i] = hh[-1]
        history[i, active] = hh
    if not bool(torch.isfinite(history).all()):
        raise RuntimeError('Nonfinite causal history')
    return dict(z, h=history)

