import copy
import json
import tempfile
import unittest
from pathlib import Path
from paris4_chunk_budget import chunk_span, load_spec, min_xy_chunks, max_xy_chunks, evaluate

ROOT=Path(__file__).resolve().parent

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.specs=[load_spec(ROOT / (name+"keV-L0.zarray.json")) for name in ("78","137")]
        self.protocol=json.loads((ROOT / "frozen_protocol.json").read_text())

    def test_exact_span_boundary(self):
        self.assertEqual(chunk_span(128,128,0),1)
        self.assertEqual(chunk_span(128,128,1),2)
        self.assertEqual(chunk_span(129,128,127),2)
        self.assertEqual(chunk_span(834,128,0),7)
        self.assertEqual(chunk_span(834,128,127),8)

    def test_real_metadata_ceil_834(self):
        out=evaluate(self.protocol,self.specs)
        self.assertEqual(out["side_pixels_ceil"],834)
        self.assertEqual(out["min_chunk_objects_one_pair_each_source"],[49,49])
        self.assertEqual(out["one_pair_best_aligned_no_halo_mib"],196.0)
        self.assertEqual(out["three_disjoint_pairs_best_aligned_no_halo_mib"],588.0)
        self.assertTrue(out["single_pair_exceeds_budget"])
        self.assertEqual(out["one_pair_interior_misalignment_ceiling_mib"],256.0)
        self.assertAlmostEqual(out["max_square_side_for_three_disjoint_pairs_aligned_um"],921.6)

    def test_ceil_validation(self):
        for s in self.specs:
            self.assertEqual(min_xy_chunks(s,384),9)
            self.assertEqual(min_xy_chunks(s,385),16)
            self.assertEqual(max_xy_chunks(s,384),16)
        s=copy.deepcopy(self.specs[0])
        s["shape"][1]=200
        with self.assertRaises(ValueError):min_xy_chunks(s,834)

    def test_invalid_compression_refused(self):
        spec=copy.deepcopy(self.specs[0]);spec["compressor"]={"id":"zlib"}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"spec.json";p.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,"Compressed"):
                load_spec(p)

if __name__=="__main__": unittest.main()
