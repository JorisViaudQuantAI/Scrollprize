import sys, glob, os, numpy as np, tifffile
root=sys.argv[1]; rows=[]
allu=[];allv=[]
for d in sorted(glob.glob(root+'/w*')):
    x,y,z=[tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz']
    P=np.stack([x,y,z],-1); ok=(z>0)&np.isfinite(z)
    du=np.linalg.norm(P[:,1:]-P[:,:-1],axis=-1); mu=ok[:,1:]&ok[:,:-1]
    dv=np.linalg.norm(P[1:]-P[:-1],axis=-1); mv=ok[1:]&ok[:-1]
    su=du[mu]/20.0; sv=dv[mv]/20.0
    # true area vs nominal
    a=np.cross(P[:-1,1:]-P[:-1,:-1],P[1:,:-1]-P[:-1,:-1]); m=ok[:-1,:-1]&ok[:-1,1:]&ok[1:,:-1]
    ar=np.linalg.norm(a,axis=-1)[m].sum()/(400.0*m.sum())
    rows.append((os.path.basename(d),x.shape,*np.percentile(su,[5,50,95]),*np.percentile(sv,[5,50,95]),np.mean((su>1.5)|(su<1/1.5)),np.mean((sv>1.5)|(sv<1/1.5)),ar))
    allu.append(su[::7]);allv.append(sv[::7])
r=np.array([t[2:] for t in rows])
print('n',len(rows),'shape example',rows[0][1])
print('cols: u5 u50 u95 v5 v50 v95 fracU>1.5x fracV>1.5x area_ratio')
print('median over windings',np.round(np.median(r,0),3))
print('min',np.round(r.min(0),3)); print('max',np.round(r.max(0),3))
U=np.concatenate(allu);V=np.concatenate(allv)
print('pooled axis1 p5/50/95',np.round(np.percentile(U,[5,50,95]),3),'outside 1.5x',round(float(np.mean((U>1.5)|(U<1/1.5))),3),'outside 1.25x',round(float(np.mean((U>1.25)|(U<0.8))),3))
print('pooled axis0 p5/50/95',np.round(np.percentile(V,[5,50,95]),3),'outside 1.5x',round(float(np.mean((V>1.5)|(V<1/1.5))),3),'outside 1.25x',round(float(np.mean((V>1.25)|(V<0.8))),3))
np.save(sys.argv[2],r)
for t in rows[:4]: print(t[0],t[1],np.round(t[2:],2))
