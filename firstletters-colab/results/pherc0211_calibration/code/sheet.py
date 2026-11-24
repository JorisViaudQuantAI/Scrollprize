"""sheet.py OUT RED_DIR TAG... : per winding, rv3 forward over reverse (z20 maps, 20 px/mm), stacked; label each"""
import sys, cv2, numpy as np
out, red = sys.argv[1], sys.argv[2]; rows = []
for t in sys.argv[3:]:
    ims = []
    for d in ('rv3_forward', 'rv3_reverse'):
        im = cv2.imread(f'{red}/red__PHerc0211_{t}_{d}_z20.png', 0)
        if im is None: continue
        c = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR); cv2.putText(c, f'{t} {d[4:7]}', (4, 20), 0, 0.6, (0, 0, 255), 2)
        for x in range(0, c.shape[1], 200): cv2.line(c, (x, 0), (x, 6), (0, 255, 255), 1)  # 10 mm ticks
        ims.append(c)
    if ims: rows += ims + [np.full((6, ims[0].shape[1], 3), (0, 120, 0), np.uint8)]
W = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(out, W, sum(r.shape[0] for r in rows))
