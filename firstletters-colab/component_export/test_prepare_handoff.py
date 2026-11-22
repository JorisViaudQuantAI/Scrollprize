import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('prepare_handoff', Path(__file__).with_name('prepare_handoff.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

def emit(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2, sort_keys=True)+'\n')
    return h.sha(path)

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.export = self.base/'export'; self.export.mkdir()
        self.normal = 'b'*64; self.mesh = 'a'*64; self.source='c'*64
        self.frame = dict(acquisition='known-public-control',axes='XYZ',units='native_voxels',level=0,
                          voxel_um_xyz=[8.64]*3,origin_native_xyz=[0,0,0])
        self.producer=dict(schema='scrollprize.producer-context-support.v1', support_kind='acquisition-and-model-context',
            producer='native render with verified tile context',definition='Synthetic fixture: every necessary input plane present',
            mesh_sha256=self.mesh,normal_sha256=self.normal,source_content_sha256=self.source,
            normal_convention='synthetic +z',plane_order='ascending native z',frame=self.frame, shift_to_normal_sign=1)
        self.context=np.ones((5,7), dtype=np.bool_)
        self.context[0,0]=False
        self.mask = self.base/'context.npy';np.save(self.mask,self.context)
        self.producer['mask_sha256']=h.sha(self.mask)
        self.producer_file=self.base/'producer.json'; emit(self.producer_file,self.producer)
        self.references=[]
        for i,shift in enumerate((-1,0,1)):
            d=self.export / f'component-{i:04d}'; d.mkdir()
            p=np.full((5,7), 0.3 + i*.1, dtype=np.float32)
            supported=np.ones((5,7),dtype=np.bool_)
            supported[1,1]=False
            p[1,1]=np.nan
            for name,array in [('probability.npy',p),('prediction_support.npy',supported),('complete_depth_support.npy',supported)]: np.save(d/name,array)
            sampling=dict(blend='hann',stride=32,tta=False,resampled=True,model_patch_zyx=[16,64,64],
                model_roi_y0_x0_y1_x1=[0,0,64,64],source_roi_y0_x0_y1_x1=[0,0,5,7],
                model_grid_shape_yx=[64,64], output_shape_yx=[5,7],source_shape_zyx=[24,5,7],
                source_layer_pitch_um=8.64,source_layers_per_model_plane=1.1,xy_pitch_um=[8.64,8.64],
                model_plane_order='increasing source z',input_resampling_border_mode='cv2.BORDER_REPLICATE',
                output_interpolation='cv2.INTER_LINEAR',output_grid='native ROI',positive_shift_convention='positive z',
                full_depth_context=True,physical_pitch_known=True,xy_tile_padding_present=False,
                model_plane_source_weight_masses=[1.0]*16,effective_depth_pitch_um=9.504,
                model_plane_source_positions=list(range(16)),shift_um=shift*9.504)
            m=dict(schema='scrollprize.inference-component.v1',status='complete',model_index=0,direction='forward',
                   shift_model_planes=shift,model={'name':'synthetic'},checkpoint_sha256='d'*64,shape_yx=[5,7],
                   full_depth_context=True, sampling=sampling,declared_geometry={k:self.producer[k] for k in h.MUST},
                   files_sha256={name:h.sha(d/name) for name in ('probability.npy','prediction_support.npy','complete_depth_support.npy')})
            self.references.append(dict(directory=d.name, model_index=0,direction='forward',shift_model_planes=shift,
                                        component_json_sha256=emit(d/'component.json',m)))
        emit(self.export/'receipt.json',dict(schema='scrollprize.inference-components.v1',status='complete',components=self.references))

    def run_good(self):
        return h.prepare(self.export,self.mask,self.producer_file,self.base/'handoff',0,'forward')
    def test_positive_handoff_and_no_inference(self):
        result=self.run_good()
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['valid_pixels_per_depth'],[33]*3)
        self.assertEqual(result['common_valid_pixels'],33)
        manifest=json.loads((self.base/'handoff/inputs.json').read_text())
        self.assertEqual([r['offset_planes'] for r in manifest['maps']],[-1,0,1])
        np.testing.assert_array_equal(np.load(self.base/'handoff/valid-depth-0.npy'),self.context & (np.load(self.export/'component-0000/complete_depth_support.npy')))
    def test_wrong_mask_bytes(self):
        self.context[2,2]=False;np.save(self.mask,self.context)
        with self.assertRaisesRegex(ValueError,'SHA256'):self.run_good()
    def test_missing_geometry_is_rejected(self):
        self.producer['normal_sha256']=None;emit(self.producer_file,self.producer)
        with self.assertRaisesRegex(ValueError,'producer hash'):self.run_good()
    def test_incomplete_receipt(self):
        (self.export/'receipt.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_good()
    def test_incomplete_depth(self):
        c=self.export/'component-0000/component.json'
        m=json.loads(c.read_text());m['full_depth_context']=False
        self.references[0]['component_json_sha256']=emit(c,m)
        emit(self.export/'receipt.json',dict(schema='scrollprize.inference-components.v1',status='complete',components=self.references))
        with self.assertRaisesRegex(ValueError,'Incomplete depth'):self.run_good()
    def test_no_xy_padding(self):
        c=self.export/'component-0001/component.json'
        m=json.loads(c.read_text());m['sampling']['xy_tile_padding_present']=True
        self.references[1]['component_json_sha256']=emit(c,m)
        emit(self.export/'receipt.json',dict(schema='scrollprize.inference-components.v1',status='complete',components=self.references))
        with self.assertRaisesRegex(ValueError,'XY padding'):self.run_good()
    def test_producer_source_mismatch(self):
        self.producer['source_content_sha256']='e'*64;emit(self.producer_file,self.producer)
        with self.assertRaisesRegex(ValueError,'geometry'):self.run_good()
    def test_reject_worktree(self):
        (self.base/'.git').mkdir()
        with self.assertRaisesRegex(ValueError,'worktree'):self.run_good()
    def test_reverse_sign(self):
        self.producer['shift_to_normal_sign']=-1;emit(self.producer_file,self.producer)
        self.run_good()
        m=load=json.loads((self.base/'handoff/inputs.json').read_text())
        self.assertEqual([r['offset_planes'] for r in m['maps']],[1,0,-1])
    def test_reject_untruthful_mask_kind(self):
        self.producer['support_kind']='prediction-only';emit(self.producer_file,self.producer)
        with self.assertRaisesRegex(ValueError,'acquisition'):self.run_good()
    def test_no_overwrite(self):
        self.run_good()
        with self.assertRaises(FileExistsError):self.run_good()
    def test_tampered_component_hash(self):
        path=self.export/'component-0000/probability.npy'
        with path.open('ab') as f:f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'SHA256'):self.run_good()

if __name__=='__main__':unittest.main()
