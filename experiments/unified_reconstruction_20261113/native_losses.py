"""Shared supervised objectives with an explicit native-clock adapter.

The 1 ms branch uses the original kernels. Longer intervals compose the
original Markov kernel on the actual context clock; fractional advances use
(1-f)I+fK and the same sign-conditional Gaussian moment matching as inference.
Only native y/q labels and one real camera endpoint contribute likelihoods.
"""
import math
import ast
import inspect
from types import SimpleNamespace
from runtime import *
from baseline_api import TM, LM
import clock_contract

R=P._runtime
core=R.core
_runtime=R
weights=P.weights
sequence_packs=P.sequence_packs


def _validate(c,z,net,jets):
    trainable=any(p.requires_grad for p in net.parameters())
    if trainable and not (net.training and getattr(net,'_endpoint_training_authorized',False)):
        raise RuntimeError('Training branch must be explicit')
    v,g,acc=jets;n,m=z['time'].shape
    if v.shape!=(n,m,2,2) or g.shape!=(n,m,2,2,2) or acc.shape!=v.shape:raise ValueError('Native query shape mismatch')
    h=z['dt'];eps=torch.finfo(h.dtype).eps
    if bool((h<0).any()) or bool((h>.001+max(1e-10,8*eps*.001)).any()):raise ValueError('Invalid physical subinterval')
    tolerance=max(2e-7,8*eps*max(1.,float(z['end'].abs().max())))
    if not torch.allclose(h.sum(-1)/.001,z['end']-z['start'],atol=tolerance,rtol=0):
        raise ValueError('Internal pieces do not tile the native interval')
    if bool((h[:,0]<=0).any()) or bool((z['end']<=z['start']).any()):raise ValueError('Empty camera interval')
    if not bool(torch.isfinite(v).all()&torch.isfinite(g).all()&torch.isfinite(acc).all()):raise ValueError('Nonfinite queries')
    R.history(z)

# The original numerical propagation is retained. Only the latent clock is
# changed from query-midpoints to integration-piece starts, where y/q live.
# This private function does not mutate the historical module or its globals.
class _BoundaryClock(ast.NodeTransformer):
    def visit_Subscript(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='z' and isinstance(node.slice,ast.Constant) and node.slice.value=='time':
            node.slice=ast.Constant(value='latent_time')
        return node
_tree=_BoundaryClock().visit(ast.parse(inspect.getsource(R.captured_distribution)))
ast.fix_missing_locations(_tree)
_namespace=dict(R.__dict__)
_namespace['_validate']=_validate
def _latent_predict_clock(q,mean,cov,logq,hist,ctx,prevctx,head,fraction):
    tolerance=1e-3 if hist.dtype==torch.float32 else 1e-8
    if bool(((fraction < -tolerance)|(fraction > 1+tolerance)).any()):raise ValueError('Latent step exceeds 1 ms')
    return R._latent_predict(q,mean,cov,logq,hist,ctx,prevctx,head,fraction.clamp(0,1).to(hist))
_namespace['_latent_predict']=_latent_predict_clock
exec(compile(_tree,'native_boundary_clock','exec'),_namespace)
_captured_boundary=_namespace['captured_distribution']
exec(compile(ast.parse(inspect.getsource(R.free_distribution)),'native_free_distribution','exec'),_namespace)
free_distribution=_namespace['free_distribution']


def captured_distribution(c,z,jets,net,cal,force,degree=12,previous=None,reset=None,update=True):
    dt=z['dt'].double()
    clock=z['start'].double()[:,None]+(dt.cumsum(-1)-dt)/.001
    return _captured_boundary(c,dict(z,latent_time=clock),jets,net,cal,force,degree,previous,reset,update)


def subset(z,ix,dtype=None):
    return {k:(v[ix].to(dtype=dtype) if dtype is not None and v.is_floating_point() else v[ix])
            for k,v in z.items() if torch.is_tensor(v)}


def pid_weights(pid):
    if not len(pid):raise ValueError('Empty state/split')
    _,inverse,count=pid.unique(return_inverse=True,return_counts=True)
    return 1/(len(count)*count[inverse].double())


class IIDHead:
    def parameters_at(self,h,ctx,previous):
        n=len(h)
        return h.new_zeros(n),h.new_zeros(n,2,2),h.new_zeros(n,2),torch.eye(2).to(h).expand(n,2,2)


def marginal_proxy(net):
    return SimpleNamespace(base=net.base,head=IIDHead(),parameters=net.parameters,
        training=net.training,_endpoint_training_authorized=getattr(net,'_endpoint_training_authorized',False))


def contexts(c,z,jets):
    v,g,_=jets
    n,m=v.shape[:2]
    return P.core.context(c,dict(w=z['w'][:,None].expand(n,m,2).reshape(-1,2)),
        v.reshape(-1,2,2),g.reshape(-1,2,2,2)).reshape(n,m,-1)


def marginal(c,z,jets,labels,force_labels,net,cal,force,kind,degree=6):
    """Original label + .1 real-endpoint NLL; no temporal-head gradient in A.

    Each native row has a single native-start marginal target. Its context is
    the first integration midpoint, matching the original start-label/midpoint
    context convention. Intermediate force targets are not counted as rows.
    """
    ctx=contexts(c,z,jets)[:,0];h=R.history(z,0)
    if kind=='free':
        label=TM.free_force_nll(net.base,h,ctx,force_labels[:,0],force)
        pred,cov,lw=free_distribution(c,z,jets,net,cal,force)
    else:
        label=-net.base.log_prob(h,ctx,labels['y'],labels['q'])
        pred,cov,lw,_=captured_distribution(c,z,jets,marginal_proxy(net),cal,force,degree,update=False)
    endpoint=P.core.scores(pred,cov,lw,z['target'])
    return label+.1*endpoint,label,endpoint


def pair_from_path(base,head,prev_h,prev_ctx,prev_y,prev_q,h,ctx,y,q,cal,
                   path_h,path_ctx,fractions):
    """Conditional native-pair NLL, including the terminal calibrated Jacobian.

    path_* list the intermediate/current contexts AFTER the starting context.
    fractions are elapsed milliseconds on that clock, each in (0,1]. Neither
    previous nor intermediate native endpoint observations enter this loss.
    Conditional Gaussian moments are propagated, not pseudo y/q labels.
    """
    n=len(q)
    if path_h.shape[:2]!=path_ctx.shape[:2] or fractions.shape!=path_h.shape[:2]:
        raise ValueError('Transition path dimensions differ')
    if bool(((fractions<=0)|(fractions>1+1e-7)).any()):raise ValueError('Invalid latent-clock increment')
    if bool((fractions.sum(-1)<1-1e-7).any()):raise ValueError('Sub-ms labeled pairs are unsupported')
    # Preserve original values and gradients exactly at the periodic cadence.
    if fractions.shape[1]==1 and torch.allclose(fractions,torch.ones_like(fractions),atol=1e-7,rtol=0):
        return LM.supervised_pair_nll(base,head,prev_h,prev_ctx,prev_y,prev_q,h,ctx,y,q,cal)[0]
    _,_,_,e0,_=LM.whiten_latent(base,prev_h,prev_ctx,prev_y,prev_q,cal)
    _,_,_,e1,L1=LM.whiten_latent(base,h,ctx,y,q,cal)
    qlog=torch.full((n,2),-torch.inf,device=h.device,dtype=h.dtype)
    qlog=qlog.scatter(1,prev_q[:,None],0.)
    mean=torch.cat((e0,h.new_zeros(n,4)),-1)[:,None].expand(n,2,6)
    cov=h.new_zeros(n,2,6,6);last=prev_ctx
    for j in range(fractions.shape[1]):
        hh,cc=path_h[:,j],path_ctx[:,j]
        logq,_,_=R.ob.latent(base,hh,cc,cal)
        qlog,mean,cov=R._latent_predict(qlog,mean,cov,logq,hh,cc,last,head,fractions[:,j])
        last=cc
    ids=torch.arange(n,device=h.device)
    L=LM._cholesky(cov[ids,q,:2,:2])
    white=torch.linalg.solve_triangular(L,(e1-mean[ids,q,:2])[...,None],upper=False)[...,0]
    return (-qlog[ids,q]+.5*white.square().sum(-1)+math.log(2*math.pi)
        +L.diagonal(dim1=-2,dim2=-1).log().sum(-1)
        +L1.diagonal(dim1=-2,dim2=-1).log().sum(-1))


def pair_loss(c,z,jets,labels,net,cal,previous,current):
    # Rows for a case share its native clock. Mixed clocks must be separate jobs.
    ctx=contexts(c,z,jets)
    prev_h=R.history(z,0)[previous];h=R.history(z,0)[current]
    active=z['dt'][previous]>0
    if not bool((active==active[0]).all()):
        result=ctx.new_zeros(len(current))
        for count in active.sum(-1).unique():
            ids=torch.where(active.sum(-1)==count)[0]
            result=result.index_copy(0,ids,pair_loss(c,z,jets,labels,net,cal,previous[ids],current[ids]))
        return result
    count=int(active[0].sum())
    path_h=torch.cat((z['h'][previous,1:count],h[:,None]),1)
    path_ctx=torch.cat((ctx[previous,1:count],ctx[current,0,None]),1)
    fractions=z['dt'][previous,:count]/.001
    duration=z['start'][current]-z['start'][previous]
    tolerance=max(2e-6,8*torch.finfo(duration.dtype).eps*max(1.,float(z['end'].abs().max())))
    if not torch.allclose(fractions.sum(-1),duration,atol=tolerance,rtol=0):raise RuntimeError('Pair clock does not tile native span')
    return pair_from_path(net.base,net.head,prev_h,ctx[previous,0],labels['y'][previous],labels['q'][previous],
        h,ctx[current,0],labels['y'][current],labels['q'][current],cal,path_h,path_ctx,fractions)


def window_nll(c,z,jets,net,cal,force,mask,degree,frames=None,marginals=False):
    """Shared equal-state/PID objective, in NLL per real observed endpoint.

    C and B retain the complete within-window causal filtering graph. No
    truncation/detach, synthetic endpoint or NLL-per-ms rescaling is applied.
    """
    totals={};v=jets[0]
    for code,name in ((0,'free'),(1,'cap')):
        selected=mask&(z['state']==code);ix=torch.where(selected)[0]
        weights=v.new_zeros(len(v));weights[ix]=pid_weights(z['pid'][ix]).to(v)
        total=v.new_zeros(())
        if code==0 or marginals:
            for ids in ix.split(64):
                zz=subset(z,ids);jj=tuple(j[ids] for j in jets)
                if code==0:pred,cov,lw=free_distribution(c,zz,jj,net,cal,force)
                else:pred,cov,lw,_=captured_distribution(c,zz,jj,marginal_proxy(net),cal,force,degree,update=False)
                total=total+(weights[ids]*P.core.scores(pred,cov,lw,zz['target'])).sum()
        else:
            chronology={k:z[k].detach().cpu() for k in ('pid','state','start','end','segment')}
            for index,valid,reset in P.sequence_packs(chronology,selected.detach().cpu(),16):
                index,valid,reset=(a.to(v.device) for a in (index,valid,reset))
                previous=None
                for t in range(index.shape[1]):
                    ids=index[:,t];zz=subset(z,ids)
                    _,_,_,previous=captured_distribution(c,zz,tuple(j[ids] for j in jets),net,
                        cal,force,degree,previous,reset[:,t],True)
                    total=total+(-previous['log_evidence']*weights[ids]*valid[:,t]).sum()
        totals[name]=total
    return dict(**totals,total=.5*(totals['free']+totals['cap']))


def supervised_anchors(c,z,jets,labels,force_labels,net,cal,force,mask,frames=None):
    ctx=contexts(c,z,jets)[:,0];h=R.history(z,0)
    fi=torch.where(mask&(z['state']==0))[0];ci=torch.where(mask&(z['state']==1))[0]
    free=(TM.free_force_nll(net.base,h[fi],ctx[fi],force_labels[fi,0],force)*pid_weights(z['pid'][fi]).to(h)).sum()
    logq,_,_,e,L=LM.whiten_latent(net.base,h[ci],ctx[ci],labels['y'][ci],labels['q'][ci],cal)
    ids=torch.arange(len(ci),device=h.device)
    marginal=-logq[ids,labels['q'][ci]]+.5*e.square().sum(-1)+L.diagonal(dim1=-2,dim2=-1).log().sum(-1)+math.log(2*math.pi)
    cap=(marginal*pid_weights(z['pid'][ci]).to(h)).sum()
    before,after,_=clock_contract.pairs({k:v.detach().cpu() for k,v in z.items()},mask.detach().cpu())
    before,after=before.to(h.device),after.to(h.device)
    if not len(after):raise ValueError('No native captured pair for anchor')
    pair=(pair_loss(c,z,jets,labels,net,cal,before,after)*pid_weights(z['pid'][after]).to(h)).sum()
    return dict(free=free,cap=cap,pair=pair,total=(free+cap+pair)/3)
