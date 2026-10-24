import numpy as np
from scipy.special import logsumexp
def density(points,means,weights,sigma):
    out=[]
    for lo in range(0,len(points),128):
        d=(points[lo:lo+128,None,:]-means[None])/sigma
        out.append(logsumexp(np.log(weights+1e-300)[None]-.5*np.sum(d*d,-1),axis=1)-np.log(2*np.pi*sigma**2))
    return np.concatenate(out)
