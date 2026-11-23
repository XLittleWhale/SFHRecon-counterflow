"""A single continuous neural representation for every time, including t=0.

Spatial velocity is the existing solenoidal Fourier synthesis of a_theta(t).
No separate initial-state parameter, endpoint increment, or learned RHS closure.
The tangent pass is the exact chain-rule time derivative of this smooth MLP;
ordinary autograd differentiates both values and derivatives with respect to weights.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class NeuralField(nn.Module):
    def __init__(self,cold,ring,seed=0,scale_power=1.,width=128,depth=3,frequencies=8):
        super().__init__()
        self.register_buffer('cold',cold.detach().clone())
        self.register_buffer('scale',ring.to(cold).pow(-scale_power))
        self.register_buffer('frequency',torch.arange(1,frequencies+1).to(cold)*math.pi)
        self.duration=.095
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.hidden=nn.ModuleList([nn.Linear(1+2*frequencies if i==0 else width,width).to(cold) for i in range(depth)])
            self.output=nn.Linear(width,cold.numel()).to(cold)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def coefficients_and_dt(self,t_seconds):
        t=t_seconds.to(self.cold).reshape(-1,1)
        tau=2*t/self.duration-1
        rate=2/self.duration
        phase=tau*self.frequency
        x=torch.cat((tau,phase.sin(),phase.cos()),1)
        dx=torch.cat((torch.ones_like(tau)*rate,phase.cos()*self.frequency*rate,-phase.sin()*self.frequency*rate),1)
        for layer in self.hidden:
            dx=F.linear(dx,layer.weight,None)
            x=layer(x).tanh()
            dx=dx*(1-x.square())
        value=self.output(x).reshape(-1,*self.cold.shape)
        tangent=F.linear(dx,self.output.weight,None).reshape_as(value)
        return self.cold+value*self.scale,tangent*self.scale

    def states(self):
        return self.coefficients_and_dt(torch.arange(96).to(self.cold)*.001)[0]

    def forward(self,t_seconds):
        return self.coefficients_and_dt(t_seconds)[0]

    def parameter_groups(self,lr=.001):
        return [dict(params=list(self.parameters()),lr=lr,name='entire_field')]

def symmetric_regularization(a,ring):
    """Historical spatial/temporal coefficients, with the spectral term uniform in time."""
    ring=ring.to(a);hi=ring>2
    energy=.003*a.square().sum(-1).mean()/24
    temporal=.03*(a[1:]-a[:-1]).square().sum(-1).mean()/24
    spectral=.03*(a[...,hi]*(ring[hi]/2)**2).square().sum(-1).mean()/24
    return energy+temporal+spectral,dict(energy=energy,temporal=temporal,spectral_uniform=spectral)

def residual_loss(a,adot,op,std,prior,nfit):
    """Physical-second strong residual on 191 half-ms nodes, integrated with Simpson weights.

Each of the three shifted 95-node paths is whitened before squaring. B is a
fixed residual tolerance/correlation, not an additional trainable process.
"""
    from pinn_common import CURRENT
    residual=(adot-CURRENT.batched_rhs(op,a))*.001
    grids=(residual[0:-1:2],residual[1::2],residual[2::2])
    qn=qs=a.new_zeros(())
    for weight,r in zip((1/6,4/6,1/6),grids):
        qn=qn+weight*prior.whiten(r[:,0]).square().sum()
        qs=qs+weight*(r[:,1]/std[1]).square().sum()
    loss=.08/(2*nfit)*(qn+qs)
    return loss,dict(normal_whitened_energy=qn,super_whitened_energy=qs,residual_velocity_step_rms=((residual.square().sum(-1)/2).mean(0).sqrt()*op.U*1000))
