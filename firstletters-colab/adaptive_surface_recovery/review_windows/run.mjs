// Node 18+; frozen archived JSON inputs.
import fs from 'node:fs';
import crypto from 'node:crypto';
function evaluate(archive, protocol) {
  const side=20, step=.25;
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const rect=(x,y)=>[x,y,x+side,y+side];
  function intersection(a,b){return [Math.max(a[0],b[0]),Math.max(a[1],b[1]),Math.min(a[2],b[2]),Math.min(a[3],b[3])]}
  const area=r=>Math.max(0,r[2]-r[0])*Math.max(0,r[3]-r[1]);
  function unionArea(rs){if(!rs.length)return 0;if(rs.length===1)return area(rs[0]);return area(rs[0])+area(rs[1])-area(intersection(rs[0],rs[1]));}
  function unionLength(intervals){const a=intervals.filter(i=>i[1]>i[0]).sort((a,b)=>a[0]-b[0]);let sum=0,end=-Infinity;for(const [l,r] of a){sum+=Math.max(0,r-Math.max(l,end));end=Math.max(end,r)}return sum;}
  function inside(x,y,poly){let hit=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const [a,b]=poly[i],[c,d]=poly[j];if((b>y)!==(d>y)&&x<(c-a)*(y-b)/(d-b)+a)hit=!hit}return hit}
  function select(W,H,props){
    const n=Math.min(2,props.length);
    if(!n||W<side||H<side)return {baseline:[],adaptive:[],gains:[]};
    const baseline=props.slice(0,n).map(p=>rect(clamp(p.poly.reduce((s,q)=>s+q[0],0)/p.poly.length-side/2,0,W-side),clamp(p.poly.reduce((s,q)=>s+q[1],0)/p.poly.length-side/2,0,H-side)));
    const unique=(v)=>[...new Set(v.map(x=>Math.round(x*1e9)/1e9))].sort((a,b)=>a-b);
    function starts(max){const a=[];for(let x=0;x<=max+1e-9;x+=2)a.push(Math.min(x,max));a.push(max);return a}
    const xs=unique([...starts(W-side),...baseline.map(r=>r[0])]),ys=unique([...starts(H-side),...baseline.map(r=>r[1])]);
    const nx=Math.ceil(W/step-.5),ny=Math.ceil(H/step-.5),weights=new Float64Array(nx*ny),active=[];
    for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){let s=0;for(const p of props)if(inside((x+.5)*step,(y+.5)*step,p.poly))s=Math.max(s,p.score);if(s>0){weights[y*nx+x]=s;active.push([x,y])}}
    const adaptive=[],gains=[];
    for(let k=0;k<n;k++){
      const stride=nx+1,integral=new Float64Array((ny+1)*stride);
      for(let y=0;y<ny;y++){let row=0;for(let x=0;x<nx;x++){row+=weights[y*nx+x];integral[(y+1)*stride+x+1]=integral[y*stride+x+1]+row}}
      function bounds(r){return [clamp(Math.ceil(r[0]/step-.5-1e-10),0,nx),clamp(Math.ceil(r[1]/step-.5-1e-10),0,ny),clamp(Math.ceil(r[2]/step-.5-1e-10),0,nx),clamp(Math.ceil(r[3]/step-.5-1e-10),0,ny)]}
      let best=null,gain=-Infinity;
      for(const y of ys)for(const x of xs){const r=rect(clamp(x,0,W-side),clamp(y,0,H-side));if(adaptive.some(a=>Math.abs(a[0]-r[0])<1e-8&&Math.abs(a[1]-r[1])<1e-8))continue;const [l,t,rr,b]=bounds(r),s=integral[b*stride+rr]-integral[t*stride+rr]-integral[b*stride+l]+integral[t*stride+l];if(s>gain+1e-9){best=r;gain=s}}
      if(!best)break;
      adaptive.push(best);gains.push(Math.max(0,gain)*step*step);
      const [l,t,rr,b]=bounds(best);for(let y=t;y<b;y++)for(let x=l;x<rr;x++)weights[y*nx+x]=0;
    }
    return {baseline,adaptive,gains};
  }
  function metric(rs,targets){
    const target=targets.map(t=>{const box=t.box,center=(box[1]+box[3])/2,clipped=rs.map(r=>intersection(r,box)),intervals=rs.filter(r=>r[1]<=center&&center<r[3]).map(r=>[Math.max(r[0],box[0]),Math.min(r[2],box[2])]);return {id:t.id,box_fraction:unionArea(clipped)/area(box),centerline_fraction:unionLength(intervals)/(box[2]-box[0])}});
    return {windows_mm:rs,display_area_cm2:rs.length*4,union_chart_area_cm2:unionArea(rs)/100,targets:target,mean_box_fraction:target.length?target.reduce((s,t)=>s+t.box_fraction,0)/target.length:null,minimum_centerline_fraction:target.length?Math.min(...target.map(t=>t.centerline_fraction)):null,mean_centerline_fraction:target.length?target.reduce((s,t)=>s+t.centerline_fraction,0)/target.length:null,boxes_at_least_half:target.filter(t=>t.box_fraction>=.5-1e-12).length};
  }
  if(archive.results.length!==5||Object.keys(protocol.cases).length!==5)throw Error("source count");
  if(Math.abs(unionArea([[0,0,2,2],[1,0,3,2]])-6)>1e-12||unionLength([[0,3],[2,4],[8,9]])!==5||!inside(.5,.5,[[0,0],[1,0],[1,1],[0,1]])||inside(2,2,[[0,0],[1,0],[1,1],[0,1]]))throw Error("geometry test");
  const cases=archive.results.map(c=>{
    const conf=protocol.cases[c.id], [h,w]=conf.shape,[sy,sx]=conf.spacing_um_yx.map(x=>x/1000), W=w*sx,H=h*sy;
    const props=c.top10_proposals.map(p=>({score:p.score,poly:p.band_polygon_xy.map(([x,y])=>[x*sx,y*sy])}));
    const picked=select(W,H,props);
    const targets=conf.targets.map(t=>({id:t.id,box:t.box_xyxy.map((x,i)=>x*(i%2?sy:sx))}));
    const baseline=metric(picked.baseline,targets),adaptive=metric(picked.adaptive,targets);
    if([...picked.baseline,...picked.adaptive].some(r=>r[0]<-1e-8||r[1]<-1e-8||r[2]>W+1e-8||r[3]>H+1e-8||Math.abs(area(r)-400)>1e-6))throw Error("bad window");
    if([...baseline.targets,...adaptive.targets].some(t=>[t.box_fraction,t.centerline_fraction].some(x=>x<0||x>1+1e-9)))throw Error("bad fraction");
    return {id:c.id,chart_mm:[W,H],proposal_count:props.length,baseline,adaptive,adaptive_marginal_score_area:picked.gains};
  });
  const a=cases.find(c=>c.id==="pherc306b_context_raw"),b=cases.find(c=>c.id==="pherc343_context_display");
  const gate={full306b_minrow_gain10pp:a.adaptive.minimum_centerline_fraction-a.baseline.minimum_centerline_fraction>=.1-1e-12,full306b_no_row_loss5pp:a.adaptive.targets.every((t,i)=>t.centerline_fraction>=a.baseline.targets[i].centerline_fraction-.05-1e-12),display343_count_not_lower:b.adaptive.boxes_at_least_half>=b.baseline.boxes_at_least_half,display343_mean_box_no_loss5pp:b.adaptive.mean_box_fraction>=b.baseline.mean_box_fraction-.05-1e-12,same_budget:cases.every(c=>c.baseline.display_area_cm2===c.adaptive.display_area_cm2),empty_controls:cases.filter(c=>c.proposal_count===0).every(c=>c.baseline.windows_mm.length===0&&c.adaptive.windows_mm.length===0)};
  return {experiment:"fixed_review_windows_v1",units:"nominal chart millimetres, no support mask",self_tests:"pass",gate,passed:Object.values(gate).every(Boolean),cases,limitations:"retrospective geometry using top ten published bands and known coarse target boxes, not actual reader recall, ink recognition or valid-area coverage"};
}
function read(path,expected){const bytes=fs.readFileSync(path);const hash=crypto.createHash('sha1').update('blob '+bytes.length+'\0').update(bytes).digest('hex');if(hash!==expected)throw Error('wrong source blob');return JSON.parse(bytes)}
console.log(JSON.stringify(evaluate(read(process.argv[2],'a614144401fefec6ca49fd124de2a28b2a28dc07'),read(process.argv[3],'a36c130e8d2d00c660ff2e5d6fbd8449d970645a')),null,2));
