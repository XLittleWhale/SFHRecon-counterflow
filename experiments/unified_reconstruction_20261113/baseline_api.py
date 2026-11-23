"""Import the authoritative baseline loss kernels without its old controllers.

These kernels are shared by geometry adapters. Availability of these functions
alone is explicitly not acceptance of a native-camera joint implementation.
"""
import ast
import math
import numpy as np
from runtime import *

TM=P._runtime.TM
LM=P._runtime.LM
SOURCES={
 'A':HERE.parent/'shared_transition_20261104/training.py',
 'B':HERE.parent/'latent_transition_20261104_v2/train_latent.py',
 'C':HERE.parent/'neural_closure_20261104/training.py',
 'joint_probability':HERE.parent/'neural_closure_20261104/probability.py',
 'closure':HERE.parent/'neural_closure_20261104/closure_model.py',
}

def functions(path,names,namespace):
    """Use original AST bodies, avoiding imports that start legacy controllers."""
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in selected}!=set(names):raise RuntimeError('Baseline API changed')
    for node in selected:node.decorator_list=[]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def marginal_kernel():
    ns=dict(torch=torch,model_module=lambda:TM,scores=P.core.scores)
    return functions(SOURCES['A'],['loss_parts'],ns)['loss_parts']

def joint_kernels(data_adapter,probability_adapter,initial_scale):
    ns=dict(torch=torch,math=math,np=np,D=data_adapter,P=probability_adapter,initial_scale=initial_scale)
    names=['field_errors','likelihood','infer_initial','teacher_loss','anchors','train_window','composite']
    functions(SOURCES['C'],names,ns)
    original=ns['infer_initial']
    def infer_initial(w,op,net,settings,*,create_graph):
        # Shared physical-time preconditioner. The original five-ms window is
        # exactly unchanged; a longer window scales its displacement Jacobian
        # approximately with duration, so its gradient step scales inversely
        # with duration squared. This rule never branches on geometry/type.
        duration=w.get('dt',.001)*w['length']
        scale=min(1.,.005/duration)**2
        adjusted=dict(settings,inner_lr=settings['inner_lr']*scale)
        w['effective_inner_lr']=adjusted['inner_lr']
        return original(w,op,net,adjusted,create_graph=create_graph)
    ns['infer_initial']=infer_initial
    return ns

def assert_stage_freezing(net,stage):
    base=list(net.base.parameters());head=list(net.head.parameters())
    if stage=='A':
        ok=all(p.requires_grad for p in base) and not any(p.requires_grad for p in head)
    elif stage=='B':
        ok=not any(p.requires_grad for p in base) and all(p.requires_grad for p in head)
    elif stage=='closure':ok=not any(p.requires_grad for p in [*base,*head])
    elif stage=='joint':ok=all(p.requires_grad for p in [*base,*head])
    elif stage=='PINN':ok=not any(p.requires_grad for p in [*base,*head])
    else:raise ValueError(stage)
    if not ok:raise RuntimeError('Stage parameter-freezing contract violated: '+stage)

def configure_stage(net,stage):
    net.enable_training() if stage in ('A','B','joint') else net.freeze()
    for p in net.base.parameters():p.requires_grad_(stage in ('A','joint'))
    for p in net.head.parameters():p.requires_grad_(stage in ('B','joint'))
    assert_stage_freezing(net,stage)
    return net
