import math
import torch
from counterflow import likelihood
from counterflow.probability import transform

def test_gaussian_random_intercept_matches_dense_density_and_gradient():
    torch.manual_seed(7)
    r=torch.randn(3,4,2,dtype=torch.float64,requires_grad=True)
    beta=torch.tensor([.13,1.42],dtype=torch.float64)
    C=likelihood.covariance(4,beta);sigma=.4
    actual=likelihood.gaussian_nll(r,C,sigma,4)
    exact=sum(-torch.distributions.MultivariateNormal(torch.zeros(4,dtype=r.dtype),sigma**2*C[j]).log_prob(r[:,:,j]) for j in range(2))/4
    torch.testing.assert_close(actual,exact)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),r,retain_graph=True)[0],torch.autograd.grad(exact.sum(),r)[0])

def test_native_identity_matches_one_ms_likelihood():
    torch.manual_seed(8)
    W=torch.eye(4,dtype=torch.float64);beta=torch.tensor([.13,1.42],dtype=torch.float64)
    r=torch.randn(2,4,2,dtype=torch.float64)
    native=(W@W.T)[None]+beta[:,None,None]*torch.outer(W.sum(1),W.sum(1))[None]
    torch.testing.assert_close(likelihood.gaussian_nll(r,native,.3,float(W.sum())),
                               likelihood.gaussian_nll(r,likelihood.covariance(4,beta),.3,4))

def test_rosenblatt_single_gaussian():
    target=torch.tensor([[.1,-.2],[1.,2.]],dtype=torch.float64)
    pred=torch.zeros(2,1,1,2,dtype=torch.float64);lw=torch.zeros(2,1,1,dtype=torch.float64)
    z,nll=transform(pred,lw,target,.5)
    torch.testing.assert_close(z,target/.5)
    torch.testing.assert_close(nll,.5*(target/.5).square().sum(-1)+math.log(2*math.pi*.5**2))
