"""Publication figures: 180-mm pages, no titles, fixed fonts and embedded PDF text."""
from pathlib import Path
import numpy as np
from .figure_math import density
import json,string,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse

AVAILABLE={f.name for f in font_manager.fontManager.ttflist}
FONT_NAME='Times New Roman' if 'Times New Roman' in AVAILABLE else 'STIXGeneral'
plt.rcParams.update({'font.family':'serif','font.serif':[FONT_NAME],'mathtext.fontset':'stix',
    'font.size':8,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'axes.linewidth':.65,'lines.linewidth':1.15,'lines.markersize':3.5,'xtick.major.width':.6,'ytick.major.width':.6,
    'xtick.major.size':3,'ytick.major.size':3,'xtick.direction':'out','ytick.direction':'out',
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,
    'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':True,
    'figure.constrained_layout.h_pad':.07,'figure.constrained_layout.w_pad':.06,
    'legend.frameon':False,'legend.handlelength':1.6,'legend.borderaxespad':.3})

C={'ink':'#233744','teal':'#117C80','orange':'#CC6B37','purple':'#8064A2','gray':'#8A939D','gold':'#C29B36'}
CMAP=colors.LinearSegmentedColormap.from_list('counterflow_balance',['#203C64','#437F9B','#9BC8CA','#F4F1E8','#E6B678','#BF6845','#773D49'],N=256)
ERRMAP=colors.LinearSegmentedColormap.from_list('error_ember',['#FFF6E6','#E8C789','#CB8858','#965065','#432F53'],N=256)
WIDTH=180/25.4
AUDIT=[]

def panel(ax,i,inside=False):
    x,y=(.03,.96) if inside else (-.15,1.04)
    ax.text(x,y,f'({string.ascii_lowercase[i]})',transform=ax.transAxes,fontsize=10,fontweight='bold',
        va='top' if inside else 'bottom',ha='left',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.8) if inside else None)

def clean(ax):
    ax.grid(False);ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))

def density_panel(ax,z,index,extent=None,legend=False):
    means=z['cap_nodes'][index];w=z['cap_weights'][index];sig=float(z['cap_sigma_um'])
    mu=z['cap_mean'][index];obs=z['cap_target'][index]
    std=np.sqrt(((means-mu)**2*w[:,None]).sum(0)+sig**2)
    if extent is None:
        lo=np.minimum(mu-5*std,obs-1.5*sig);hi=np.maximum(mu+5*std,obs+1.5*sig)
    else:lo,hi=extent
    xx=np.linspace(lo[0],hi[0],181);yy=np.linspace(lo[1],hi[1],181);X,Y=np.meshgrid(xx,yy)
    pdf=np.exp(density(np.c_[X.ravel(),Y.ravel()],means,w,sig)).reshape(X.shape)
    dx=(xx[1]-xx[0])*(yy[1]-yy[0]);order=np.sort(pdf.ravel())[::-1];cum=np.cumsum(order)*dx
    assert cum[-1]>.99,(cum[-1],index)
    levels=[order[min(np.searchsorted(cum,p),len(order)-1)] for p in (.95,.8,.5)]
    ax.contourf(X,Y,pdf,levels=[0]+sorted(levels)+[pdf.max()*1.01],colors=['#FFFFFF','#DCEBEE','#A2C5CD','#4D8E9E','#26586C'],alpha=.95)
    ax.contour(X,Y,pdf,levels=sorted(levels),colors=[C['teal']],linewidths=.7)
    ax.scatter(*obs,marker='*',s=48,color=C['orange'],edgecolors='white',linewidths=.45,label='Observed',zorder=5)
    ax.scatter(*mu,marker='+',s=34,color=C['ink'],linewidths=1,label='Mean',zorder=5)
    ax.set(xlabel=r'$\Delta x_p\ (\mu\mathrm{m})$',ylabel=r'$\Delta y_p\ (\mu\mathrm{m})$');ax.set_aspect('equal',adjustable='box');clean(ax)
    if legend:ax.legend(loc='best',fontsize=8)
    return [lo,hi]

def paired_extent(zs,i):
    los=[];his=[]
    for z in zs:
        mu=z['cap_mean'][i];sig=float(z['cap_sigma_um']);w=z['cap_weights'][i]
        std=np.sqrt(((z['cap_nodes'][i]-mu)**2*w[:,None]).sum(0)+sig**2)
        los.append(np.minimum(mu-5*std,z['cap_target'][i]-1.5*sig));his.append(np.maximum(mu+5*std,z['cap_target'][i]+1.5*sig))
    lo,hi=np.min(los,0),np.max(his,0);center=(lo+hi)/2;half=(hi-lo).max()/2
    return center-half,center+half

def spectrum(v):
    v=v-v.mean((2,3),keepdims=True);ft=np.fft.fft2(v,axes=(2,3))/1600
    energy=.5*np.mean(np.sum(abs(ft)**2,-1),axis=0)*1e6
    k=np.fft.fftfreq(40)*40;rr=np.sqrt(k[:,None]**2+k[None,:]**2);shell=np.floor(rr+.5).astype(int)
    E=np.array([[energy[f][shell==i].sum() for i in range(29)] for f in (0,1)])
    assert np.allclose(E.sum(-1),.5*np.mean(np.sum(v*v,-1),axis=(0,2,3))*1e6)
    return E

def periodic_maps():
    z=load('periodic_fields.npz');height=124;fig=plt.figure(figsize=(WIDTH,height/25.4))
    # Fixed physical rectangles: all eight maps are exactly 35.5 mm square.
    x0,side,gap=10,35.5,7
    rect=lambda x,y,w,h:[x/180,y/height,w/180,h/height]
    for f in (0,1):
        bottom,bar=(84,71) if f==0 else (25,12)
        values=[z[key][-1,f]*1000 for key in ('reference','projected_K12','v_K12')]
        lo=min(v[...,1].min() for v in values);hi=max(v[...,1].max() for v in values)
        norm=colors.TwoSlopeNorm(vmin=lo,vcenter=float(z['reference'][:,f,...,1].mean()*1000),vmax=hi)
        for j,v in enumerate(values):
            ax=fig.add_axes(rect(x0+j*(side+gap),bottom,side,side));im=ax.imshow(v[...,1],origin='lower',extent=[0,10,0,10],cmap=CMAP,norm=norm,interpolation='nearest')
            xx=(np.arange(40)+.5)*.25;Y,X=np.meshgrid(xx,xx,indexing='ij');u=v-v.mean((0,1))
            ax.quiver(X[::8,::8],Y[::8,::8],u[::8,::8,0],u[::8,::8,1],color=C['ink'],scale_units='xy',angles='xy',scale=2,width=.005,headwidth=3,alpha=.7)
            ax.set(xlabel=r'$x\ (\mathrm{mm})$',xticks=[0,5,10],yticks=[0,5,10])
            if j==0:ax.set_ylabel(r'$y\ (\mathrm{mm})$')
            else:ax.set_yticklabels([])
            panel(ax,f*4+j,True)
        cb=fig.colorbar(im,cax=fig.add_axes(rect(x0,bar,3*side+2*gap,1.5)),orientation='horizontal');cb.set_label(r'$v_{'+('n' if f==0 else 's')+r',y}\ (\mathrm{mm\,s^{-1}})$',labelpad=1);cb.locator=MaxNLocator(4);cb.update_ticks()
        ax=fig.add_axes(rect(x0+3*(side+gap),bottom,side,side));err=np.linalg.norm(values[2]-values[0],axis=-1)
        im=ax.imshow(err,origin='lower',extent=[0,10,0,10],cmap=ERRMAP,vmin=0,vmax=err.max(),interpolation='nearest')
        ax.set(xlabel=r'$x\ (\mathrm{mm})$',xticks=[0,5,10],yticks=[0,5,10]);ax.set_yticklabels([]);panel(ax,f*4+3,True)
        cb=fig.colorbar(im,cax=fig.add_axes(rect(x0+3*(side+gap),bar,side,1.5)),orientation='horizontal');cb.set_label(r'Error $(\mathrm{mm\,s^{-1}})$',labelpad=1);cb.locator=MaxNLocator(3);cb.update_ticks()
    save(fig,'periodic','P2_reference_reconstruction')
