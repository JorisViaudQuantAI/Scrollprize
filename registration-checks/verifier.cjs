#!/usr/bin/env node
// Public CT validation, no fit. Node 20+. Run: node verifier.cjs [cache-directory] [--offline]
// Coordinates and comparisons were frozen before reading intensity values. Matrices come from fca6b046ecc97c56227bb13e664db21b7a83df60.
'use strict';
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const spec={
 "PHerc1203": {
  "fine": "20260319130212-2.403um-0.2m-77keV-masked.zarr",
  "coarse": "20250820131727-9.362um-1.2m-113keV-masked.zarr",
  "matrix": [
   [
    0.25663306918676065,
    0.00017894732122507914,
    0.0001696866262822777,
    -22.81487822236022
   ],
   [
    -0.00017874863318873374,
    0.2566329496451328,
    -0.0003003685677108955,
    25.008428005198766
   ],
   [
    -0.00016989591238696818,
    0.00030025023990335587,
    0.2566329557969593,
    7937.553211899798
   ]
  ],
  "regions": [
   [
    11196,
    11708,
    5440
   ],
   [
    15036,
    14780,
    9696
   ],
   [
    10940,
    15292,
    10912
   ]
  ]
 },
 "PHerc0846A": {
  "fine": "20260319102732-2.403um-0.2m-77keV-masked.zarr",
  "coarse": "20250728152254-9.362um-1.2m-113keV-masked.zarr",
  "matrix": [
   [
    0.256626072442538,
    0.00005181308524487002,
    0.0004949555022714758,
    -6.756074166513372
   ],
   [
    -0.000054289967734558166,
    0.25662333450735036,
    0.0012845083712418222,
    -12.850033244674705
   ],
   [
    -0.0004946899477802886,
    -0.0012846106649815568,
    0.25662286293337705,
    1892.0785176912368
   ]
  ],
  "regions": [
   [
    12744,
    13256,
    5440
   ],
   [
    16584,
    16328,
    9696
   ],
   [
    12488,
    16840,
    10912
   ]
  ]
 }
};

function calcJobs(spec,metas){const jobs=[];
function cover(s,type,r,lo,hi){const [,k,u,m]=metas.find(a=>a[0]===s&&a[1]===type);if(m.dtype!=='|u1'||m.order!=='C'||m.compressor!==null)throw Error('format');
const [cz,cy,cx]=m.chunks;
for(let kz=Math.floor(lo[2]/cz);kz<=Math.floor(hi[2]/cz);kz++)for(let ky=Math.floor(lo[1]/cy);ky<=Math.floor(hi[1]/cy);ky++)for(let kx=Math.floor(lo[0]/cx);kx<=Math.floor(hi[0]/cx);kx++){const z0=Math.max(lo[2],kz*cz),z1=Math.min(hi[2],(kz+1)*cz-1);const beg=(z0-kz*cz)*cy*cx,len=(z1-z0+1)*cy*cx;
const file=[s,type,r,kz,ky,kx,z0,z1].join('_')+'.bin';
jobs.push({s,type,r,url:u+'/'+[kz,ky,kx].join(m.dimension_separator??'.'),kz,ky,kx,z0,z1,beg,len,file});}}
for(const [s,sc] of Object.entries(spec)){for(let r=0;r<sc.regions.length;r++){const c=sc.regions[r].map(i=>i/4);for(let dz of [-12,0,12])cover(s,'fine',r,[c[0]-40,c[1]-40,c[2]+dz],[c[0]+39,c[1]+39,c[2]+dz]);
const b=[];for(let dz of [-12,12])for(let dy of [-40,39])for(let dx of [-40,39]){const fine=[c[0]+dx,c[1]+dy,c[2]+dz].map(x=>x*4);b.push(sc.matrix.map(row=>row[3]+fine.reduce((a,t,i)=>a+row[i]*t,0)));}
cover(s,'coarse',r,[0,1,2].map(i=>Math.floor(Math.min(...b.map(x=>x[i])))-12),[0,1,2].map(i=>Math.floor(Math.max(...b.map(x=>x[i])))+13));}}
return jobs;
}


function shiftList(){
 const out=[{label:'nominal',d:[0,0,0]}];
 for(const k of [2,12])for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){
  const d=[0,0,0];d[axis]=sign*k;out.push({label:'xyz'[axis]+(sign<0?'-':'+')+k,d});
 }
 return out;
}
function corr(a,b,keep){
 let sa=0,sb=0,aa=0,bb=0,ab=0,n=0;
 for(const i of keep){const u=a[i],v=b[i];n++;sa+=u;sb+=v;aa+=u*u;bb+=v*v;ab+=u*v;}
 const den=Math.sqrt((aa-sa*sa/n)*(bb-sb*sb/n));return den>0?(ab-sa*sb/n)/den:null;
}
function blur(src,N,sig){
 const out=new Float64Array(N*N),tmp=new Float64Array(N*N),rad=4*sig;
 const w=Array.from({length:2*rad+1},(_,j)=>Math.exp(-.5*((j-rad)/sig)**2));const sum=w.reduce((a,b)=>a+b,0);for(let j=0;j<w.length;j++)w[j]/=sum;
 function refl(i){return i<0?-i-1:i>=N?2*N-i-1:i;}
 for(let y=0;y<N;y++)for(let x=0;x<N;x++){let v=0;for(let k=-rad;k<=rad;k++)v+=w[k+rad]*src[y*N+refl(x+k)];tmp[y*N+x]=v;}
 for(let y=0;y<N;y++)for(let x=0;x<N;x++){let v=0;for(let k=-rad;k<=rad;k++)v+=w[k+rad]*tmp[refl(y+k)*N+x];out[y*N+x]=v;}
 return out;
}
function band(a,n){const g1=blur(a,n,1),g5=blur(a,n,5);for(let i=0;i<g1.length;i++)g1[i]-=g5[i];return g1;}
function samples(jobs,dir){
 const all=new Map();
 for(const j of jobs){const key=j.kx+128*j.ky+16384*j.kz,buf=fs.readFileSync(path.join(dir,j.file));if(buf.length!==j.len)throw Error('incorrect bytes '+j.file);let v=all.get(key);if(!v)all.set(key,v=[]);v.push({z0:j.z0,z1:j.z1,b:buf});}
 const get=(x,y,z)=>{const ls=all.get((x>>7)+128*(y>>7)+16384*(z>>7));if(ls)for(let j=0;j<ls.length;j++){const c=ls[j];if(z>=c.z0&&z<=c.z1)return c.b[(z-c.z0)*16384+(y&127)*128+(x&127)];}throw Error('unacquired coordinate '+[x,y,z]);};return get;
}
function trilinear(g,x,y,z){const X=Math.floor(x),Y=Math.floor(y),Z=Math.floor(z),a=x-X,b=y-Y,c=z-Z;
 const v00=g(X,Y,Z)*(1-a)+g(X+1,Y,Z)*a, v10=g(X,Y+1,Z)*(1-a)+g(X+1,Y+1,Z)*a;
 const v01=g(X,Y,Z+1)*(1-a)+g(X+1,Y,Z+1)*a, v11=g(X,Y+1,Z+1)*(1-a)+g(X+1,Y+1,Z+1)*a;
 return (v00*(1-b)+v10*b)*(1-c)+(v01*(1-b)+v11*b)*c;
}
function readScore(spec,jobs,dir){
 const off=shiftList(),result={protocol:'eed747724073e9f29e3f637e309b9e55b27791462780802c629aa5aa161ef7af',order:off.map(v=>v.label),sites:[]};
 for(const [s,cfg] of Object.entries(spec))for(const [r,rawcenter] of cfg.regions.entries()){
  const c=rawcenter.map(x=>x/4),fine=samples(jobs.filter(j=>j.s===s&&j.r===r&&j.type==='fine'),dir),coarse=samples(jobs.filter(j=>j.s===s&&j.r===r&&j.type==='coarse'),dir);
  const site={sample:s,region:'ABC'[r],fine_center:rawcenter,planes:[]};
  for(const dz of [-12,0,12]){const N=80,P=N*N,F=new Float64Array(P),C=off.map(()=>new Float64Array(P)),mask=new Uint8Array(P).fill(1),z=c[2]+dz;
   for(let yy=0;yy<N;yy++)for(let xx=0;xx<N;xx++){const x=c[0]-40+xx,y=c[1]-40+yy,i=yy*N+xx;F[i]=fine(x,y,z);if(F[i]<=3)mask[i]=0;
    const [a,b,t]=cfg.matrix.map(row=>row[3]+4*(row[0]*x+row[1]*y+row[2]*z));
    for(let k=0;k<off.length;k++){const [dx,dy,dd]=off[k].d;C[k][i]=trilinear(coarse,a+dx,b+dy,t+dd);if(C[k][i]<=3)mask[i]=0;}}
   const valid=[];for(let y=5;y<N-5;y++)for(let x=5;x<N-5;x++){let good=true;for(let yy=y-5;yy<=y+5&&good;yy++)for(let xx=x-5;xx<=x+5;xx++)if(!mask[yy*N+xx]){good=false;break;}if(good)valid.push(y*N+x);}
   const d={dz_fine_l2:dz,common_valid:valid.length,inner_size:(N-10)**2};if(valid.length>=0.5*d.inner_size){const bF=band(F,N);d.raw=C.map(q=>corr(F,q,valid));d.dog=C.map(q=>corr(bF,band(q,N),valid));}site.planes.push(d);
  }
  site.usable_planes=site.planes.filter(p=>p.raw).length;
  for(const key of ['raw','dog']){
   site[key]=off.map((_,i)=>{let sum=0,n=0;for(const p of site.planes)if(p[key]&&p[key][i]!==null){sum+=p.common_valid*p[key][i];n+=p.common_valid;}return n?sum/n:null;});
  }result.sites.push(site);
 }return result;
}


async function get(url,options={}){
 let last;
 for(let a=0;a<3;a++){try{const res=await fetch(url,{...options,signal:AbortSignal.timeout(40000)});if(res.status===429||res.status>=500)throw Error('HTTP '+res.status+' '+url);return res;}catch(e){last=e;}}
 throw last;
}
async function main(){
 const argv=process.argv.slice(2),offline=argv.includes('--offline'),dir=path.resolve(argv.find(x=>x!=='--offline')||'registration-holdout'),base='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com';
 fs.mkdirSync(dir,{recursive:true});
 const metap=path.join(dir,'metadata.json');
 let metas;
 if(offline){metas=JSON.parse(fs.readFileSync(metap));}else{
  const pairs=[];for(const [s,sc] of Object.entries(spec))for(const [t,l] of [['fine',2],['coarse',0]])pairs.push([s,t,base+'/'+s+'/volumes/'+sc[t]+'/'+l]);
  metas=await Promise.all(pairs.map(async([s,t,u])=>{const r=await get(u+'/.zarray');if(r.status!==200)throw Error('Metadata '+r.status+' '+u);return [s,t,u,await r.json()];}));fs.writeFileSync(metap,JSON.stringify(metas));
 }
 for(const [,t,u,m] of metas){if(m.dtype!=='|u1'||m.order!=='C'||m.compressor!==null||JSON.stringify(m.chunks)!=='[128,128,128]')throw Error('Unexpected layout: '+u);}
 const jobs=calcJobs(spec,metas);fs.writeFileSync(path.join(dir,'jobs.json'),JSON.stringify(jobs));
 let cursor=0;async function work(){while(cursor<jobs.length){const j=jobs[cursor++],dst=path.join(dir,j.file);if(fs.existsSync(dst)&&fs.statSync(dst).size===j.len)continue;if(offline)throw Error('Missing '+j.file);
   const res=await get(j.url,{headers:{Range:'bytes='+j.beg+'-'+(j.beg+j.len-1)}});let b=Buffer.from(await res.arrayBuffer());
   if(res.status===200&&b.length===128**3)b=b.subarray(j.beg,j.beg+j.len);
   else if(res.status!==206||!res.headers.get('content-range')?.startsWith('bytes '+j.beg+'-'+(j.beg+j.len-1)+'/'))throw Error('Missing or wrong range '+res.status+' '+j.url);
   if(b.length!==j.len)throw Error('Range length mismatch '+j.url);fs.writeFileSync(dst,b);
  }}
 await Promise.all(Array.from({length:6},()=>work()));
 fs.writeFileSync(path.join(dir,'spec.json'),JSON.stringify(spec,null,2)+'\n');
 const hashed=jobs.map(j=>({file:j.file,url:j.url,byte_start:j.beg,byte_size:j.len,sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(dir,j.file))).digest('hex')}));
 fs.writeFileSync(path.join(dir,'input-hashes.json'),JSON.stringify(hashed,null,2)+'\n');
 const results=readScore(spec,jobs,dir);fs.writeFileSync(path.join(dir,'scores.json'),JSON.stringify(results,null,2)+'\n');
 const lines=['sample\tregion\tmoving_XYZ\tn_planes\tmask_pixels_per_plane\traw_nominal\traw_best_2vox\traw_best_12vox\tdog_nominal\tdog_best_2vox\tdog_best_12vox'];
 for(const v of results.sites){const best=(a,n,m)=>Math.max(...a.slice(n,m));const vals=[v.raw[0],best(v.raw,1,7),best(v.raw,7,13),v.dog[0],best(v.dog,1,7),best(v.dog,7,13)];lines.push([v.sample,v.region,v.fine_center.join(','),v.usable_planes,v.planes.map(x=>x.common_valid).join(','),...vals.map(x=>Number.isFinite(x)?x.toFixed(6):'NA')].join('\t'));}
 fs.writeFileSync(path.join(dir,'summary.tsv'),lines.join('\n')+'\n');console.log(lines.join('\n'));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
