"""Real inversion-interface checks on I only; no target reference/H scoring."""
import copy
import time
import torch
import experiment as E


def leaves(fields):
    return [v.detach().clone().requires_grad_() for v in fields]


def main():
    E.M.R.setup('cpu')
    E.P.verify_protocol()
    reports = []
    for U in E.CASES:
        d, cfg, model, base, cal, force = E.P.make_context(U, E.SEEDS[0], 'new_frozen')
        mask = E.group_mask(d['z']['pid'], 'I')
        ids = torch.cat([torch.where(mask & (d['z']['state'] == s))[0][:24] for s in (0, 1)])
        small = dict(d, z=E.EW.subset(d['z'], ids, 'cpu'))
        with torch.no_grad():
            fields = model()
            old = E.P.score(*fields, small, base, cal, force, 'I', degree=6)
        for arm in E.ARMS:
            net = E.TM.load_model(arm, base).double().eval()
            for p in net.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                current = E.score(*fields, small, net, cal, force, 'I', degree=6)
            delta = max(abs(current[k]-old[k]) for k in old)
            assert delta < 1e-11, (arm, delta)
            reports.append(dict(U=U, arm=arm, initial_legacy_score_max_difference=delta))
        # Use a perturbed COPY to exercise observation-dependent covariances.
        # This is numerical preflight, never a training or inference initial state.
        net = E.TM.load_model('shared_gru', base).double().eval()
        with torch.no_grad():
            generator = torch.Generator().manual_seed(411104)
            net.free_output.weight.copy_(torch.randn(net.free_output.weight.shape, generator=generator,
                                                      dtype=torch.float64)*.02)
            net.free_output.bias.copy_(torch.tensor([.1, -.1, .3, .2, -.3], dtype=torch.float64))
        for p in net.parameters():
            p.requires_grad_(False)
        outputs, gradients = [], []
        for batch in (8, 64):
            ll = leaves(fields)
            value = E.score(*ll, small, net, cal, force, 'I', degree=6, backward=True, batch=batch)
            outputs.append(value)
            gradients.append([v.grad.clone() if v.grad is not None else torch.zeros_like(v) for v in ll])
        score_gap = max(abs(outputs[0][k]-outputs[1][k]) for k in outputs[0])
        grad_gap = max(float((a-b).abs().max()) for a, b in zip(*gradients))
        assert score_gap < 1e-11 and grad_gap < 1e-9, (score_gap, grad_gap)
        generator = torch.Generator().manual_seed(411105)
        direction = [torch.randn(v.shape, generator=generator, dtype=v.dtype) for v in fields]
        norm = sum(float(v.square().sum()) for v in direction)**.5
        direction = [v/norm for v in direction]
        derivative = sum(float((g*v).sum()) for g, v in zip(gradients[0], direction))
        eps = 1e-5
        with torch.no_grad():
            plus = E.score(*[v+eps*q for v, q in zip(fields, direction)], small, net, cal, force, 'I', degree=6)['total']
            minus = E.score(*[v-eps*q for v, q in zip(fields, direction)], small, net, cal, force, 'I', degree=6)['total']
        finite_difference = .08*(plus-minus)/(2*eps)
        relative = abs(derivative-finite_difference)/max(1., abs(derivative), abs(finite_difference))
        assert relative < 1e-5, (derivative, finite_difference, relative)
        # One complete full-window field step uses ALL inverse-fit I records.
        groups = model.parameter_groups()
        for g in groups:
            g['lr'] = .0003 if g['name'] == 'theta' else .009
        optimizer = torch.optim.Adam(groups)
        before = E.plain_state(net)
        started = time.perf_counter()
        row = E.field_step(model, optimizer, d, cfg, net, cal, force)
        elapsed = time.perf_counter()-started
        assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())
        with torch.no_grad():
            after = model()
            assert all(torch.isfinite(v).all() for v in after)
        reports.append(dict(U=U, arm='shared_gru_perturbed_preflight_copy',
            heteroscedastic_batch_score_gap=score_gap, heteroscedastic_batch_gradient_gap=grad_gap,
            field_directional_derivative=derivative, finite_difference=finite_difference,
            relative_directional_difference=relative, full_I_field_step_seconds=elapsed,
            full_I_score=row['inversion'], net_bitwise_unchanged=True, all96frames_finite=True))
        print('INVERSION PREFLIGHT', U, 'passed', round(elapsed, 3), 's', flush=True)
    result = dict(passed=True, target_reference_read=False, H_scored=False,
        I_responses_only=True, preflight_networks_discarded=True, checks=reports)
    E.dump(E.OUT/'preflight/inversion_checks.json', result)
    return result


if __name__ == '__main__':
    main()
