import numpy as np
import h5py
from .io import dump
def vtk_series(path,values,x,y,times):
    """VTKHDF ImageData on identical reference/reconstruction sample coordinates."""
    path.mkdir(parents=True,exist_ok=True);series=[];nx=len(x);ny=len(y)
    for k,v in enumerate(values):
        file=f'field_{k:04d}.vtkhdf'
        with h5py.File(path/file,'w') as h:
            r=h.create_group('VTKHDF');r.attrs['Version']=np.array([2,0],np.int64);r.attrs['Type']=np.bytes_('ImageData')
            r.attrs['Direction']=np.eye(3).ravel();r.attrs['Origin']=[float(x[0]),float(y[0]),0.];r.attrs['Spacing']=[float(x[1]-x[0]),float(y[1]-y[0]),1.]
            r.attrs['WholeExtent']=np.array([0,nx-1,0,ny-1,0,0],np.int32);pd=r.create_group('PointData');r.create_group('CellData')
            for f,name in enumerate(('vn','vs')):
                vv=np.zeros((ny,nx,3));vv[...,:2]=v[f];pd.create_dataset(name,data=vv,compression='gzip',compression_opts=1)
        series.append(dict(name=file,time=float(times[k])))
    dump(path/'fields.vtkhdf.series',{'file-series-version':'1.0','files':series})
