"""Batch independent PIDs and checkpoint whole camera intervals.

The original propagator, quadrature, covariance algebra and posterior are
unchanged. Checkpoint recomputation preserves derivatives through the entire
captured sequence; it is NOT truncated BPTT. Only the probability TRAINING
branch retains the original four-interval truncation.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def score(c, z, net, cal, force, jets, group, degree=12, metrics=False,
          backward=False, batch=128, batch_pids=32, gradient_weight=.08):
    import native_probability as P
    R = P._runtime
    if backward and any(p.requires_grad for p in net.parameters()):
        raise RuntimeError('Field backward requires frozen probability weights')
    w = P.weights(z, group)
    out = dict(total=0., native_nll=dict(total=0.), units=dict(
        total_free_cap='NLL/ms; each native row divided by its duration, then PID/state averaged',
        native_nll='NLL per native endpoint; equal PID/state weight',
        spread='micrometre; sqrt(trace predictive displacement covariance)',
        rmse_um='micrometre; sqrt(mean squared 2D predictive-mean displacement error)',
        target='dimensionless displacement divided by U*0.001s'))
    allparts = {}
    # Build chronology on CPU once per score, avoiding GPU scalar indexing.
    chronology = {k:z[k].detach().cpu() for k in ('pid','state','start','end','segment')}
    for kind, name in ((0, 'free'), (1, 'cap')):
        selected = (w > 0) & (z['state'] == kind)
        total, native, mass, parts = 0., 0., float(w[selected].sum()), {}
        if kind == 0:
            for ix in torch.where(selected)[0].split(batch):
                zz = P.subset(z, ix)
                pred, cov, lw = P.free_distribution(c, zz, jets(zz), net, cal, force)
                nll = P.core.scores(pred, cov, lw, zz['target'])
                loss = (w[ix]*nll/(zz['end']-zz['start'])).sum()
                if backward:
                    (gradient_weight*loss).backward()
                total += float(loss.detach())
                native += float((w[ix]*nll).detach().sum())
                if metrics:
                    R._record(parts, zz, R._metrics(c, zz, pred, cov, lw, nll))
        else:
            packs = P.sequence_packs(chronology, selected.detach().cpu(), batch_pids)
            for index, valid, reset in packs:
                index, valid, reset = (a.to(z['pid'].device) for a in (index,valid,reset))
                previous, losses, native_parts = None, [], []
                for t in range(index.shape[1]):
                    zz = P.subset(z, index[:, t])
                    # Explicit arguments are crucial: late-bound loop closures
                    # would recompute the last interval during backward.
                    def interval(zz, previous, reset):
                        pred, cov, lw, posterior = P.captured_distribution(
                            c, zz, jets(zz), net, cal, force, degree,
                            previous, reset, True)
                        if metrics:
                            vals = R._metrics(c, zz, pred, cov, lw, -posterior['log_evidence'])
                            return posterior, vals
                        return posterior, None
                    if backward:
                        previous, vals = checkpoint(interval, zz, previous, reset[:,t],
                            use_reentrant=False, preserve_rng_state=False)
                    else:
                        previous, vals = interval(zz, previous, reset[:,t])
                    nll = -previous['log_evidence']
                    ww = w[index[:, t]]*valid[:, t]
                    losses.append((ww*nll/(zz['end']-zz['start'])).sum())
                    native_parts.append((ww*nll).detach().sum())
                    if metrics:
                        R._record(parts, zz, vals, valid[:, t])
                loss = torch.stack(losses).sum()
                if backward:
                    (gradient_weight*loss).backward()
                total += float(loss.detach())
                native += float(torch.stack(native_parts).sum())
        if not math.isfinite(total) or not math.isfinite(native):
            raise FloatingPointError('Nonfinite endpoint likelihood')
        out[name] = total/mass
        out['total'] += total
        out['native_nll'][name] = native/mass
        out['native_nll']['total'] += native
        if metrics:
            for row in parts.values():
                for key in ('nll','native_nll','coverage','spread','squared_error'):
                    row[key] /= row['count']
                row['rmse_um'] = math.sqrt(row['squared_error'])
            allparts[name] = parts
    if metrics:
        out['per_pid'] = allparts
        for key in ('coverage','spread'):
            out[key] = {name:float(np.mean([r[key] for r in parts.values()]))
                        for name,parts in allparts.items()}
        out['rmse_um'] = {name:math.sqrt(float(np.mean([r['squared_error'] for r in parts.values()])))
                          for name,parts in allparts.items()}
    return out
