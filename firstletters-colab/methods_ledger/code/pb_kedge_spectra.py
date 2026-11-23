# ---------------------------------------------------------------------------------------------------
# methods_ledger/code/pb_kedge_spectra.py
# Provided for REVIEW of our arithmetic, not as a packaged tool. No support, no tests, no stable interface.
# Sanitised copy of a private lane script (no paths in the original). Numerical part unchanged.
# Ledger entry: 05_label_free_and_physics.md L8 (dual energy / Pb K-edge: our two studies disagree)
# ---------------------------------------------------------------------------------------------------
"""Effective lead-to-papyrus attenuation ratio k_Pb for the two filtered beams of the 9.36 um / "113 keV" and
2.4 um / "77 keV" scans. This is the calculation behind the claim "the K-edge pair keeps only ~1/25 of the
single-energy lead signal". Needs numpy, scipy and xraylib.

Model, with every assumption a reviewer should attack:
  source     on-axis bending-magnet-like spectrum  y * K_{2/3}(y/2)^2,  y = E / E_c   (three-pole wiggler)
             E_c scanned over 25, 30 and 37.3 keV; 26.3 keV = side pole 1.1 T at 6 GeV, 37.3 keV = central pole
  filters    taken from the operator COMMENT field of each volume's public metadata record
             high beam: Mo 1.3 mm + Ag 0.1 mm          low beam: sapphire 1 mm + Au 0.1 mm + Ag 0.1 mm + C 10 mm
             (the 'attenuators' dictionary in the same records contradicts the comment; we used the comment)
  extra      0 or 2 mm of carbon for unknown fixed beamline windows
  detector   GAGG scintillator, 100 um (high beam) or 50 um (low beam), energy-integrating (weight ~ absorbed x E)
  papyrus    C 76.6, O 10, H 3, Si 8.8, Al 0.5, Na 0.65, P 0.4 wt % (a published carbonised-papyrus composition)
  k_X        = sum_E w(E) mu/rho_X(E)  /  sum_E w(E) mu/rho_papyrus(E),   w = normalised detected spectrum
The other study in our notes used tabulated MONOCHROMATIC values at the nominal energies (ratio of the lead
excess between 77 and 113 keV about 1.7). The last lines print that monochromatic comparison for reference.
The spectra themselves are not published, so neither number has been checked against a measurement.
"""
import numpy as np, xraylib as x
from scipy.special import kv
E=np.arange(15.,400.,0.5)
def mr(m): return np.array([x.CS_Total_CP(m,float(v)) for v in E])
MU={m:mr(m) for m in ['Mo','Ag','Al2O3','Au','C','Be','Gd3Al2Ga3O12','Pb','Fe','Cu','SiO2']}
pap={'C':0.766,'O':0.10,'H':0.03,'Si':0.088,'Al':0.005,'Na':0.0065,'P':0.004};s=sum(pap.values())
MU['pap']=sum((w/s)*np.array([x.CS_Total(x.SymbolToAtomicNumber(k),float(v)) for v in E]) for k,w in pap.items())
def spec(Ec,filt,th,extraC):
    y=E/Ec; f=y*kv(2/3,y/2)**2
    for m,r,t in filt+[('C',2.2,extraC)]: f=f*np.exp(-MU[m]*r*t/10)
    f*= (1-np.exp(-MU['Gd3Al2Ga3O12']*6.63*th/10))*E
    return f/f.sum()
hi=[('Mo',10.22,1.3),('Ag',10.5,0.1)]
lo=[('Al2O3',3.98,1.0),('Au',19.3,0.1),('Ag',10.5,0.1),('C',1.5,10.0)]
print('Ec  extraC | Emean hi lo | kPb hi lo | ratio(low/high): pap Pb Fe Cu')
for Ec in [25,30,37.3]:
  for extraC in [0,2]:
    w1=spec(Ec,hi,0.1,extraC); w2=spec(Ec,lo,0.05,extraC)
    e1=(w1*E).sum(); e2=(w2*E).sum()
    m1={k:(w1*MU[k]).sum() for k in ['pap','Pb','Fe','Cu']}; m2={k:(w2*MU[k]).sum() for k in ['pap','Pb','Fe','Cu']}
    r={k:m2[k]/m1[k] for k in m1}
    print(f"{Ec:5.1f} {extraC:3.0f} | {e1:6.1f} {e2:6.1f} | {m1['Pb']/m1['pap']:5.1f} {m2['Pb']/m2['pap']:5.1f} | {r['pap']:.3f} {r['Pb']:.3f} {r['Fe']:.3f} {r['Cu']:.3f}")
for e1,e2 in [(113,77),(113,80)]:
    i1=np.argmin(abs(E-e1));i2=np.argmin(abs(E-e2))
    print('mono',e1,e2,'ratio pap %.3f Pb %.3f Fe %.3f'%(MU['pap'][i2]/MU['pap'][i1],MU['Pb'][i2]/MU['Pb'][i1],MU['Fe'][i2]/MU['Fe'][i1]))
print('--- side pole (Ec 26.3 keV) and central pole (37.3 keV), with the share of detected weight below the Pb K-edge ---')
lo_dec=[('Al2O3',3.98,2.5),('Ag',10.5,0.2)]       # the later "78 keV" filter set used on some READ scrolls
for nm,Ec,filt,th in [('high beam, Ec 26.3',26.3,hi,0.1),('high beam, Ec 37.3',37.3,hi,0.1),('low beam, Ec 26.3',26.3,lo,0.05),('later low beam (no Au), Ec 26.3',26.3,lo_dec,0.05)]:
    w=spec(Ec,filt,th,0); e=(w*E).sum()
    m={k:(w*MU[k]).sum() for k in ['pap','Pb','Fe','Cu','SiO2']}
    print(f"{nm:34s} Emean {e:6.1f}  frac<88 {w[E<88].sum():.2f} frac 50-81 {w[(E>50.2)&(E<80.7)].sum():.2f}  mu/rho pap {m['pap']:.4f} Pb {m['Pb']:.3f}  kPb {m['Pb']/m['pap']:.1f} kFe {m['Fe']/m['pap']:.2f} kCu {m['Cu']/m['pap']:.2f}")
