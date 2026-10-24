import math
import torch
import numpy as np
from .observation import particle_parts,decoded,predict_free,law
BETA=torch.zeros(2,dtype=torch.float64)
def cap_terms(c,a,z,net,cal,degree=16):
    cc=dict(c);cc['cap']=z
    ctx,base,gain,hr=particle_parts(cc,a)
    pred,lw=decoded(net,z['h'],ctx,base,gain,hr,cal,degree)
    logq=torch.logsumexp(lw,-1)
    sigma=law(c['fixed_noise']['captured'],c['physics']['vn_mean_m_s'],.015,1.5)
    lp=-.5*((z['target'][:,None,None]-pred)/sigma).square().sum(-1)-math.log(2*math.pi*sigma**2)
    conditional=torch.logsumexp(lw-logq[:,:,None]+lp,-1)
    return logq,conditional

def captured_nll(c,a,z,net,cal,degree=16):
    T=c.get('observation_steps',c['intervals'])
    logq,conditional=cap_terms(c,a,z,net,cal,degree)
    return -torch.logsumexp(logq+conditional,-1).reshape(-1,T).mean(1)

def covariance(T,beta,dtype=torch.float64):
    eye=torch.eye(T,dtype=dtype)
    return eye[None]+beta[:,None,None]*torch.ones(T,T,dtype=dtype)[None]

def gaussian_nll(residual,cov,sigma,normalizer):
    """Residual [P,T,2], normalized covariance [2,T,T], per-ms score."""
    L=torch.linalg.cholesky(cov)
    white=torch.linalg.solve_triangular(L,(residual/sigma).permute(2,1,0),upper=False)
    T=residual.shape[1]
    return (.5*white.square().sum((0,1))+L.diagonal(dim1=-2,dim2=-1).log().sum()
            +2*T*math.log(sigma)+T*math.log(2*math.pi))/normalizer

def free_nll(c,a,z):
    if c['kind']=='channel':
        return native_nll(c,a,z)
    T=c['observation_steps'];cc=dict(c,free=z)
    r=(z['target']-predict_free(cc,a)).reshape(-1,T,2)
    sigma=law(c['fixed_noise']['free'],c['physics']['vn_mean_m_s'],.015,1.5)
    return gaussian_nll(r,covariance(T,BETA),sigma,T)

def native_predictions(c,a,z):
    """Compose the SAME free drag response over a native camera interval.

    Forces use a causal fixed-path pilot. Velocity propagates within the raw
    interval instead of being reset at unobserved intermediate camera times.
    No new force or learnable parameter, and exactly the old one-step limit.
    """
    cc=dict(c,free=z['z'])
    m=predict_free(cc,a);H,B=c['free_response']
    acceleration=(m-H*z['z']['w'])/(B*.001)
    rate=c['constants']['drag']/c['constants']['mass']
    return integrate_free(acceleration,z['z']['w'],z['weights'],rate)

def integrate_free(acceleration,w,W,rate):
    result=[]
    for row in W:
        columns=torch.where(row>0)[0].tolist();vp=w[columns[0]];displacement=torch.zeros_like(vp)
        for k in columns:
            fraction=float(row[k]);h=.001*fraction;decay=math.exp(-rate*h)
            H=-math.expm1(-rate*h)/(rate*h)
            displacement=displacement+fraction*(H*vp+(1-H)*acceleration[k]/rate)
            vp=decay*vp+(1-decay)*acceleration[k]/rate
        result.append(displacement)
    return torch.stack(result)

def native_nll(c,a,z):
    pids=z['pid'][::c['observation_steps']].tolist();values=[]
    sigma=law(c['fixed_noise']['free'],c['physics']['vn_mean_m_s'],.015,1.5)
    for pid in pids:
        p=c['free_native'][pid];W=p['weights'];J=W.shape[0]
        r=p['target']-native_predictions(c,a,p)
        # Independent disjoint microincrements and a shared segment bias.
        D=W@W.T;duration=W.sum(1)
        cov=D[None]+BETA[:,None,None]*torch.outer(duration,duration)[None]
        values.append(gaussian_nll(r[None],cov,sigma,float(duration.sum()))[0])
    return torch.stack(values)
