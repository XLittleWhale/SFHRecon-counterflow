import math
import torch
def normal_quantile(logcdf,logsf):
    # Evaluate the smaller tail to avoid 1-CDF cancellation. Float64 supports
    # probabilities to 1e-300 here; fail rather than silently clip rare tails.
    tail=torch.minimum(logcdf,logsf)
    if not bool((tail.detach()>-680).all()):
        raise FloatingPointError('Gaussianized residual exceeds supported tail; no silent clipping')
    value=-torch.special.ndtri(tail.exp())
    return torch.where(logcdf<=logsf,-value,value)

def transform(pred,lw,target,sigma,reverse=False):
    """Exact Rosenblatt map of the finite quadrature Gaussian mixture.

    Returns Gaussianized residual [N,2] and ORIGINAL one-step negative logpdf.
    Network and physical Gaussian-kernel mixture weights remain differentiable.
    """
    pred=pred.flatten(1,2);lw=lw.flatten(1,2)
    # decoded() is normalized up to roundoff. Subtracting logsumexp preserves
    # exact normalization of the finite quadrature mixture.
    lw=lw-torch.logsumexp(lw,1,keepdim=True)
    if reverse:pred=pred.flip(-1);target=target.flip(-1)
    r=(target[:,None]-pred)/sigma
    lpx=-.5*r[...,0].square()-math.log(sigma)-.5*math.log(2*math.pi)
    lpy=-.5*r[...,1].square()-math.log(sigma)-.5*math.log(2*math.pi)
    conditional=lw+lpx-torch.logsumexp(lw+lpx,1,keepdim=True)
    z=[]
    for axis,weights in ((0,lw),(1,conditional)):
        lc=torch.logsumexp(weights+torch.special.log_ndtr(r[...,axis]),1)
        ls=torch.logsumexp(weights+torch.special.log_ndtr(-r[...,axis]),1)
        z.append(normal_quantile(lc,ls))
    score=-torch.logsumexp(lw+lpx+lpy,1)
    return torch.stack(z,-1),score
