"""Verify B21/B26 immutable panels and aligned scores."""
import hashlib
from pathlib import Path
import numpy as np
PINS={'panel1447/panel.npz':'0f529972da360305aaad80d3400f209230427aa4ad9ad45b8df0f5807e4c9fb4','panel841/panel.npz':'926a284be12043aff107107c97b04a0af472e4e37c27a5277131891bad7d6357','pred841/predictions.npz':'2fb79c53bc83c4d008a9bcffaad7fd788691140e24c7012440b61553ffb8c691','b21/predictions.npz':'baf48b9301c9697c0dd8f4efbb7ffde9c2f891305f08d39194323eb860bd2d81'}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(d):
    for n,h in PINS.items():assert sha(d/n)==h,n
    a,b,q,p=[dict(np.load(d/n,allow_pickle=False)) for n in PINS]
    for x,y in ((a,p),(b,q)):
        assert np.array_equal(x['states'],y['states']) and np.array_equal(x['origins_yx'],y['origins_yx'])
        assert x['raw'].dtype==np.uint8 and x['raw'].shape[1:]==(21,384,384)
    assert np.array_equal(a['names'],p['windings']) and p['arms'].tolist()==['reverse','forward','minus2','plus2','bf16'] and q['directions'].tolist()==['ascending','descending']
    return a,b,p,q
