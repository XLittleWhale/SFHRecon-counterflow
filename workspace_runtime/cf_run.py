"""Run the accepted channel kernels with explicit case and stage dependencies."""
import importlib
import sys

from . import paths
from .paths import REPO
from .release import FULL, configure, verify_bundle
from .runner import read, run_path


def locations(q):
    case = f"cf_2K_Q{q}"
    return (case, FULL / "outputs/CFSimEachPINN_20261109/prepared" / case,
            FULL / "prepared_supervision" / case)


def with_inputs(q, prepared, supervision):
    _, logical_p, logical_s = locations(q)
    paths.add_routes([(logical_p, prepared), (logical_s, supervision)])


def prepare_source(source, prepared, supervision, q, device):
    if read(source / "configuration.json")["heat_flux_mW_cm2"] != q:
        raise ValueError("Requested heat flux differs from the input case")
    sys.path.insert(0, str(REPO / "experiments/cf_sim_each_pinn_20261109"))
    D = importlib.import_module("data_adapter")
    sys.path.pop(0)
    D.prepare_case(source, prepared)
    import labels, force_context
    labels.prepare(source, prepared, FULL / "reference_solver", supervision)
    force_context.prepare(source, prepared, supervision, device)


def setup():
    verify_bundle(FULL)
    rt = configure(FULL, [REPO])
    rt.torch.set_num_threads(2)
    rt.torch.set_default_dtype(rt.torch.float64)
    return rt


def probability(args, out):
    rt = setup()
    if args.input:
        prepared, supervision = out / "prepared", out / "supervision"
        ready = out / "prepared_inputs.json"
        if not ready.exists():
            prepare_source(args.input, prepared, supervision, args.q, args.device)
            rt.dump(ready,{"case":args.q,"source":args.input.relative_to(paths.ROOT).as_posix()})
        with_inputs(args.q, prepared, supervision)
        if args.action == "prepare":
            return False
    elif args.action == "prepare":
        raise ValueError("Provide --input for preparation; the released cases are already prepared")
    import workflow, method_contract, acceptance
    defaults = workflow.defaults

    def case_defaults(profile="baseline_retrain_all"):
        p = defaults(profile)
        p["cases"] = [args.q]
        if profile != "smoke":
            p["field"]["budgets"] = [args.field_budget]
        return p

    workflow.defaults = case_defaults
    engine = out / "engine"
    profile = "smoke" if args.smoke else args.profile
    with rt.lock(engine):
        p = workflow.seal(engine, profile)
        receipt = engine / "acceptance.json"
        if not receipt.exists():
            result = acceptance.run(args.device, (args.q,))
            result["protocol_fingerprint"] = p["fingerprint"]
            rt.dump(receipt, result)
        method_contract.require_training(engine,args.device)
        return workflow.train_probabilities(engine,args.device,args.limit)["complete"]


def reconstruct(args, out):
    rt = setup()
    parent = run_path(3,args.probability_run)
    status = read(parent / "status.json")
    if not status.get("complete") or bool(status.get("diagnostic_only")) != args.smoke:
        raise ValueError("The probability run must be completed and use the same diagnostic/formal mode")
    work = parent / "engine"
    p = read(work / "protocol.json")
    if p["config"]["cases"] != [args.q]:
        raise ValueError("The frozen probability model is for another heat flux")
    if not args.smoke and p["config"]["field"]["budgets"] != [args.budget]:
        raise ValueError("Field budget differs from the selected probability run contract")
    if (parent / "prepared_inputs.json").exists():
        with_inputs(args.q,parent/"prepared",parent/"supervision")
    dest = out / "engine"
    paths.add_routes([(dest / x,work / x) for x in
        ("protocol.json","probability_frozen.json","probability","datasets","acceptance.json")])
    import workflow, fields, method_contract
    with rt.lock(dest):
        p = workflow.verify(dest)
        method_contract.require_training(dest,args.device)
        for row in p["targets"]:
            for seed in p["config"]["field_seeds"]:
                if not fields.run(dest,row,seed,args.device,args.limit)["complete"]:
                    return False
        fields.freeze(dest)
        fields.evaluate(dest,args.device)
        rt.dump(dest/"complete.json",{"complete":True,"protocol_fingerprint":p["fingerprint"],
            "diagnostic_only":args.smoke,"probability_run":args.probability_run,
            "evaluation_sha256":rt.sha(dest/"evaluation/summary.json")})
        return True
