"""Independent non-target multi-case inner-gradient and frozen-law regression."""
from pathlib import Path
import time
import torch
import training as N
from common import *
import data_training as D
import probability as P
from closure_model import ClosureMomentum


def parity():
    case = D.load_case(D.VALIDATION_CASE)
    w = D.window(case, 30)
    z = w['z']
    candidates = z['pid'][z['state'] == 1].unique().tolist()
    rows = None
    for pid in candidates:
        match = {(int(z['frame'][i])): i for i in torch.where((z['pid'] == pid) & (z['state'] == 1))[0].tolist()}
        if all(frame in match for frame in (0, 1, 2)):
            rows = torch.tensor([match[frame] for frame in (0, 1, 2)])
            break
    assert rows is not None
    with torch.no_grad():
        residual = D.reference_residual(w['op'], w['truth'])
        jets = tuple(v[rows] for v in D.query(w, w['truth'], residual))
    obs = P.core.subset(z, rows)
    net, _ = baseline_network(TRAIN_SEEDS[0])
    oldhead = LM.LatentTransitionHead(net.base, state=net.head.state_dict()).double()
    for parameter in oldhead.parameters():
        parameter.requires_grad_(False)
    outputs, gradients, losses = [], [], []
    for step, head in ((LM.step, oldhead), (P.step, net.head)):
        h = obs['h'].detach().clone().requires_grad_()
        v, g, acc = [item.detach().clone().requires_grad_() for item in jets]
        state = context = None
        branch, factors = [], []
        for frame in range(3):
            zz = P.core.subset(obs, torch.tensor([frame]))
            zz['h'] = h[frame:frame+1]
            pred, cov, lw, state = step(w['c'], zz, v[frame:frame+1], g[frame:frame+1], acc[frame:frame+1],
                net.base, head, w['cal'], w['force'], 12, state, context, 'temporal', None, True)
            context = state['context']
            branch.extend([value.detach() for value in (pred, cov, lw, state['qlogp'], state['emean'], state['ecov'])])
            factors.append(-state['log_evidence'].sum())
        loss = torch.stack(factors).sum()
        derivative = torch.autograd.grad(loss, (h, v, g, acc))
        assert all(bool(torch.isfinite(value).all()) for value in derivative)
        outputs.append(branch); gradients.append(derivative); losses.append(float(loss.detach()))
    max_value = max(float((x-y).abs().max()) for x, y in zip(*outputs))
    gradient_deltas = {name: float((x-y).abs().max()) for name, x, y in zip(('h', 'v', 'g', 'acc'), *gradients)}
    assert max_value < 1e-12 and max(gradient_deltas.values()) < 1e-9
    return dict(passed=True, case=case['case'], physical_pid=int(obs['pid'][0]), original_frames=obs['original_frame'].tolist(),
        degree=12, continuous_endpoints=3, maximum_predictive_and_posterior_difference=max_value,
        input_gradient_max_differences=gradient_deltas, summed_endpoint_nll=losses,
        endpoint_factor_count=3, probability_parameters_frozen=True)


def main():
    setup(); torch.manual_seed(TRAIN_SEEDS[0]); begin = time.perf_counter()
    source_shas = {str(HERE/name): sha(HERE/name) for name in
        ('closure_model.py', 'probability.py', 'data_training.py', 'training.py', 'common.py', 'preflight_independent.py')}
    settings = dict(N.DEFAULT_SETTINGS)
    settings['validation_degree'] = settings['train_degree']
    net, closure, _ = N.build('joint', TRAIN_SEEDS[0], dict(N.CLOSURE_SETTINGS), pilot=True)
    rows = []
    for name in D.TRAIN_CASES:
        if name == 'vn0_0025m_s':
            continue  # Separate training pilot already covers this case.
        case = D.load_case(name)
        w = D.window(case, 30)
        op = ClosureMomentum(case['op'], closure)
        a, history = N.infer_initial(w, op, net, settings, create_graph=False)
        assert bool(torch.isfinite(a).all())
        assert history[-1]['objective'] < history[0]['objective']
        record = dict(case=name, counts=w['counts'], inner=history,
            maximum_absolute_coefficient=float(a.detach().abs().max()),
            gradient_clip_active_count=sum(item.get('gradient_norm', 0.) > settings['inner_gradient_clip'] for item in history),
            minimum_distance_to_gradient_clip_kink=min(abs(item['gradient_norm']-settings['inner_gradient_clip']) for item in history if 'gradient_norm' in item))
        if name == 'vn0_0085m_s' or record['gradient_clip_active_count']:
            result, _ = N.infer_initial(w, op, net, settings, create_graph=True)
            initial_error = N.field_errors(result, w['truth'])['initial']
            probabilities, closures = tuple(net.parameters()), tuple(closure.parameters())
            derivatives = torch.autograd.grad(initial_error, probabilities+closures, allow_unused=True)
            assert all(bool(torch.isfinite(value).all()) for value in derivatives if value is not None)
            pm = max(float(value.detach().abs().max()) for value in derivatives[:len(probabilities)] if value is not None)
            cm = max(float(value.detach().abs().max()) for value in derivatives[len(probabilities):] if value is not None)
            assert pm > 0 and cm > 0
            record.update(initial_field_probability_hypergradient_max=pm, initial_field_closure_hypergradient_max=cm,
                          second_order_path_finite=True)
        rows.append(record)
        print('CHECKED', name, history[0]['objective'], history[-1]['objective'], flush=True)
    regression = parity()
    assert all(sha(path) == digest for path, digest in source_shas.items()), 'Source changed during preflight'
    report = dict(passed=True, target_files_opened=False, formal_training_or_inversion_started=False,
        cases=rows, frozen_probability_parity=regression, settings=settings,
        source_shas=source_shas,
        wall_seconds=time.perf_counter()-begin,
        limits=['Six fixed non-target windows supplement the separate 2.5 mm/s and 6.5 mm/s pilot.',
                'A finite nonzero hypergradient is a numerical capability check, not proof of reconstruction improvement.',
                'Gradient clipping is differentiable away from its threshold; no global smoothness claim is made.'])
    dump(OUT/'preflight/independent_numerics.json', report)
    print('INDEPENDENT_NUMERICS_PASSED', report['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
