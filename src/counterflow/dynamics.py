"""CUDA-capable exact exported momentum/Vinen operators, double precision."""
import torch
from .observation import move,DT
from torch.utils.checkpoint import checkpoint

class Periodic:
    def __init__(self,data,device):
        self.__dict__.update(move(data,device));self.device=device
    def inv(self,z,dim=(0,1)):return torch.fft.ifft2(z,dim=dim).real*self.n**2
    def rhs(self,s):
        c=self.c;a=s[:2*self.d].reshape(2,self.d);q=s[2*self.d:]
        vh=torch.zeros(2,self.n,self.n,2,dtype=torch.complex128,device=s.device)
        z=.5*self.U*(a[:,1::2]+1j*a[:,0::2])[...,None]*self.perp
        vh[:,self.y,self.x]=z;vh[:,self.my,self.mx]=z.conj();vh[:,0,0]=self.mean.to(vh.dtype)
        qh=torch.zeros(self.n,self.n,dtype=torch.complex128,device=s.device);zq=.5*(q[1::2]-1j*q[2::2])
        qh[self.y,self.x]=zq;qh[self.my,self.mx]=zq.conj();qh[0,0]=q[0]
        v=self.inv(vh,(1,2));vx=self.inv(vh*(1j*self.kx)[None,...,None],(1,2));vy=self.inv(vh*(1j*self.ky)[None,...,None],(1,2))
        L=c['line_density_m2']*self.inv(qh).clamp(-12,8).exp();slip=v[0]-v[1]
        F=-(2/3)*c['alpha']*c['rho_s_kg_m3']*c['kappa_m2_s']*L[...,None]*slip
        r=-v[...,0,None]*vx-v[...,1,None]*vy
        rn=r[0]+F/c['rho_n_kg_m3']+c['mu_n_Pa_s']/c['rho_n_kg_m3']*self.inv(-self.k2[...,None]*vh[0])
        rs=r[1]-F/c['rho_s_kg_m3'];rv=torch.stack((rn,rs))
        zz=torch.fft.fft2(rv,dim=(1,2))[:,self.y,self.x]/self.n**2;zz=(zz*self.perp).sum(-1)/self.U
        da=torch.stack((2*zz.imag,2*zz.real),-1).flatten(-2);da=da+torch.stack((torch.zeros_like(da[0]),-self.nu_s*self.lap*a[1]))
        speed=slip.norm(dim=-1).clamp_min(1e-12)
        rq=-v[1,...,0]*self.inv(1j*self.kx*qh)-v[1,...,1]*self.inv(1j*self.ky*qh)
        rq=rq+c['alpha_v']*speed*L.sqrt()-c['beta_v']*L+self.gamma*speed.pow(2.5)/L
        zq=torch.fft.fft2(rq)/self.n**2;zp=zq[self.y,self.x]
        dq=torch.cat((zq[0,0].real[None],torch.stack((2*zp.real,-2*zp.imag),-1).flatten()))
        return torch.cat((da.flatten(),dq))

class Channel:
    def __init__(self,data,device):self.__dict__.update(move(data,device));self.d=self.m
    def rhs(self,s):
        a=s[:2*self.m].reshape(2,self.m);q=s[2*self.m:]
        v=self.v0+torch.einsum('fpm,fm->fp',self.V,a).reshape(self.v0.shape)
        g=self.g0+torch.einsum('fpm,fm->fp',self.G,a).reshape(self.g0.shape)
        log=self.B@q;grad=(self.Gq@q).reshape(-1,2);L=self.ref*log.clamp(-12,8).exp();u=v[0]-v[1]
        speed=u.norm(dim=-1).clamp_min(1e-12);F=self.ck*L[:,None]*u
        # Channel policy is isotropic; no macro reactive force.
        assert self.cr==0
        conv=torch.einsum('fpj,fpij->fpi',v,g);stress=(v[..., :,None]*v[...,None,:]).flatten(1)
        loads=-.5*torch.einsum('fpm,fp->fm',self.Pv,conv.flatten(1))+.5*torch.einsum('fpm,fp->fm',self.Pg,stress)
        ff=torch.einsum('fpm,p->fm',self.Pv,F.flatten())
        dn=loads[0]+ff[0]/self.p['rho_n_kg_m3']-self.D@a[0]-self.d0
        ds=loads[1]-ff[1]/self.p['rho_s_kg_m3']-self.Ds@a[1]-self.ds0
        reaction=self.p['alpha_v']*speed*L.sqrt()-self.p['beta_v']*L+self.gamma*speed.pow(2.5)/L
        dq=self.Pq@(-(v[1]*grad).sum(-1)+reaction)
        return torch.cat((dn,ds,dq))

def make(data,device):return (Periodic if data['kind']=='periodic' else Channel)(data,device)

def rollout(op,a0,intervals,substeps=2,checkpointed=True):
    # Prescribed uniform initial L: q0=0. No latent or field truth enters.
    nq=op.d+1 if isinstance(op,Periodic) else op.B.shape[1]
    s=torch.cat((a0.flatten(),a0.new_zeros(nq)));states=[s];h=DT/substeps
    def step(s):
        for _ in range(substeps):
            k1=op.rhs(s);k2=op.rhs(s+h*k1/2);k3=op.rhs(s+h*k2/2);k4=op.rhs(s+h*k3)
            s=s+h*(k1+2*k2+2*k3+k4)/6
        return s
    for _ in range(intervals):
        s=checkpoint(step,s,use_reentrant=False) if checkpointed and s.requires_grad else step(s)
        states.append(s)
    ss=torch.stack(states);return ss[:,:2*op.d].reshape(intervals+1,2,op.d),ss[:,2*op.d:]
