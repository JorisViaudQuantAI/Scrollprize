"""Generated selection-invariant tests; no real source arrays."""
import unittest

import numpy as np

from selector import rectangle_gap_squared, select


class SelectionTests(unittest.TestCase):
    def test_gaps_and_half_open_edges(self):
        self.assertEqual(rectangle_gap_squared((0, 2, 0, 2), (2, 4, 2, 4)), 0)
        self.assertEqual(rectangle_gap_squared((0, 2, 0, 2), (5, 7, 6, 8)), 25)

    def test_class_counts_ties_distance_and_no_fallback(self):
        ref = np.tile(np.array([255, 1], np.uint8), (8, 4))
        kwargs = dict(count=3, alignment=2, shape=(2, 2), min_gap_px=2, border_guard_px=0)
        r = select(ref, (0, 2, 0, 2), **kwargs)
        self.assertEqual(r['selected'][0]['bounds_y0_y1_x0_x1'], [6, 8, 6, 8])
        self.assertEqual(r['selected'][1]['bounds_y0_y1_x0_x1'], [0, 2, 6, 8])
        for c in r['eligible_candidates']:
            y0, y1, x0, x1 = c['bounds_y0_y1_x0_x1']
            p = ref[y0:y1, x0:x1]
            self.assertEqual(c['full_counts_high_low_nonzero'],
                             [int((p >= 179).sum()), int(((p > 0) & (p <= 64)).sum()), int((p > 0).sum())])
        self.assertEqual(r, select(ref, (0, 2, 0, 2), **kwargs))
        kwargs['min_gap_px'] = 20
        self.assertEqual(select(ref, (0, 2, 0, 2), **kwargs)['status'], 'unassessable')

    def test_zero_middle_and_discarded_border_are_not_core_classes(self):
        ref = np.array([[255, 0], [255, 100]], dtype=np.uint8)
        r = select(ref, (4, 6, 4, 6), count=1, alignment=2,
                   shape=(2, 2), min_gap_px=0, border_guard_px=0)
        self.assertEqual(r['eligible_count'], 0)
        ref = np.ones((8, 8), dtype=np.uint8)
        ref[:2] = 255
        r = select(ref, (20, 28, 20, 28), count=1, alignment=8,
                   shape=(8, 8), min_gap_px=0, border_guard_px=2)
        self.assertEqual(r['eligible_count'], 0)


if __name__ == '__main__':
    unittest.main()
