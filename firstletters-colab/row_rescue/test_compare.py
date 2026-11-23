#!/usr/bin/env python3
"""Purely synthetic invariants; real published-data comparison is separate."""
import importlib.util
from pathlib import Path
import unittest

s = importlib.util.spec_from_file_location("rescue", Path(__file__).with_name("compare.py"))
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)

def p(y_mm, score=2, length=10):
    v = y_mm * 100
    return {"score":score,"length_mm":length,
            "band_polygon_xy":[[0,v-10],[100,v-10],[100,v+10],[0,v+10]]}

class Invariants(unittest.TestCase):
    def test_second_row_selected(self):
        a=[p(0,3,20),p(.1,2.9,20),p(-.1,2.8,20),
           p(8,2.7,8),p(4.5,2.6,25),p(4.6,2.5,25)]
        self.assertEqual(m.rescue_indices(a,10),[0,1,2,4])
    def test_existing_diversity_unchanged(self):
        self.assertEqual(m.rescue_indices([p(0),p(4),p(.3),p(6)],10),[0,1,2,3])
    def test_no_proposals_invented(self):
        self.assertEqual(m.rescue_indices([],10),[])
        self.assertEqual(m.rescue_indices([p(0)],10),[0])
    def test_no_second_row_retains_baseline(self):
        self.assertEqual(m.rescue_indices([p(0),p(.1),p(.2),p(1),p(2)],10),[0,1,2,3])
    def test_deterministic_ties(self):
        a=[p(0),p(.1),p(-.1),p(4.5),p(-4.5)]
        self.assertEqual(m.rescue_indices(a,10),m.rescue_indices(a,10))
        self.assertEqual(m.rescue_indices(a,10)[:3],[0,1,2])

if __name__ == "__main__":
    unittest.main()
