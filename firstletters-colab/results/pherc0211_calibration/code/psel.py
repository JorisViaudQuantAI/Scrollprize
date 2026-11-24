"""psel.py WINDIR OUT.json : rank all st96desc windows by line finder, pick top 25 + 15 random from the rest (rng 2110)"""
import sys, json, glob, numpy as np
W = []
for f in glob.glob(sys.argv[1] + '/*_desc.json'): W += json.load(open(f))
for w in W: w['lr'] = w['line'] if w['line'] is not None else -99
W.sort(key=lambda w: -w['lr'])
for i, w in enumerate(W): w['rank'] = i + 1
top = W[:25]; rest = W[25:]
rng = np.random.default_rng(2110); rnd = [rest[i] for i in sorted(rng.choice(len(rest), 15, replace=False))]
json.dump(dict(n_windows=len(W), all=W, top=top, random=rnd), open(sys.argv[2], 'w'), indent=1)
print('windows', len(W), 'valid cm2 sum', round(sum((w['valid_cm2'] or 0) for w in W), 1))
for w in top[:25]: print('T', w['rank'], w['window'], w['lr'], w['row'], w['hot'])
for w in rnd: print('R', w['rank'], w['window'], w['lr'])
