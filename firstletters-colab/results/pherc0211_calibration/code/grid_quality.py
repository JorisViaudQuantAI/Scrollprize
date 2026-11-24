import sys, glob, os, numpy as np, tifffile
from scipy.ndimage import uniform_filter
def diag(d):
    x,y,z=[tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz']
    P=np.stack([x,y,z],-1); ok=(z>0)&np.isfinite(z)
    a=P[:-1,1:]-P[:-1,:-1]; b=P[1:,:-1]-P[:-1,:-1]
    n=np.cross(a,b); m=ok[:-1,:-1]&ok[:-1,1:]&ok[1:,:-1]
    area=np.linalg.norm(n,axis=-1); nn=n/np.maximum(area,1e-9)[...,None]
    nn[~m]=0
    k=9
    sm=np.stack([uniform_filter(nn[...,i],k,mode='nearest') for i in range(3)],-1)
    w=uniform_filter(m.astype(float),k,mode='nearest')
    sm=sm/np.maximum(np.linalg.norm(sm,axis=-1),1e-9)[...,None]
    c=(nn*sm).sum(-1)[m&(w>0.6)]
    ang=np.degrees(np.arccos(np.clip(c,-1,1)))
    # shear: angle between grid axes
    ca=(a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-9)
    sh=np.abs(90-np.degrees(np.arccos(np.clip(ca[m],-1,1))))
    return dict(n=int(m.sum()),flip=float((c<0).mean()),a45=float((ang>45).mean()),a20=float((ang>20).mean()),angp50=float(np.median(ang)),angp95=float(np.percentile(ang,95)),shear50=float(np.median(sh)),shear95=float(np.percentile(sh,95)))
for root in sys.argv[1:]:
    ds=sorted(glob.glob(root+'/*/x.tif')); R=[diag(os.path.dirname(p)) for p in ds]
    keys=['flip','a45','a20','angp50','angp95','shear50','shear95']
    A=np.array([[r[k] for k in keys] for r in R]); N=np.array([r['n'] for r in R])
    print(root.split('/')[-2:], 'meshes',len(R),'cells',N.sum())
    print('  keys',keys)
    print('  median',np.round(np.median(A,0),4)); print('  max   ',np.round(A.max(0),4))
    print('  pooled flip',round(float((A[:,0]*N).sum()/N.sum()),5),'pooled >45deg',round(float((A[:,1]*N).sum()/N.sum()),4),'pooled >20deg',round(float((A[:,2]*N).sum()/N.sum()),4))
