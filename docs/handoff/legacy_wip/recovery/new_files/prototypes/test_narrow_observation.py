import unittest
from types import SimpleNamespace
import numpy as np
from prototypes.narrow_observation import corridor_features,masks,COMMANDS,DIM
from prototypes.narrow_alignment import AlignmentNarrowCore

class ObservationTests(unittest.TestCase):
    def test_action_prefix_and_masks_preserved(self):
        from neural_navigation.masked_actions import COMMANDS as old
        np.testing.assert_array_equal(COMMANDS[:12],old)
        self.assertEqual(COMMANDS.shape,(20,2));self.assertTrue((COMMANDS[:,0]>=0).all())
        o=np.zeros(DIM);o[300:308]=1;mask=masks(o)
        self.assertTrue(mask[0]);self.assertFalse(mask[1:12].any());self.assertTrue(mask[12:].all())
        with self.assertRaises(ValueError):masks(np.zeros(300))
    def test_scan_corridor_and_absence(self):
        angles=np.linspace(-1.89,1.94,220);laser=SimpleNamespace(range_min=.1,range_max=12.,angles=angles,mount=np.zeros(4))
        ranges=np.minimum(5/np.maximum(np.cos(angles),.001),.24/np.maximum(np.abs(np.sin(angles)),.001))
        f=corridor_features(ranges,laser)
        self.assertEqual(f[0],1);self.assertAlmostEqual(f[1]*2,.48,places=5);self.assertAlmostEqual(f[2],0,places=5)
        np.testing.assert_array_equal(corridor_features(np.full(220,np.nan),laser),np.zeros(6))
    def test_unaligned_route_is_not_reported_complete(self):
        # Exercise the diagnostic extension against the real parent contract.
        from unittest.mock import patch
        fake=object.__new__(AlignmentNarrowCore)
        fake.motion_route_valid=False;fake.narrow_route_state='initial_alignment_required';fake.alignment_required=True
        from prototypes.anchored_narrow_runtime import AnchoredNarrowCore
        with patch.object(AnchoredNarrowCore,'diagnostics',return_value={'route_complete':True}):
            result=AlignmentNarrowCore.diagnostics(fake)
        self.assertFalse(result['route_complete']);self.assertTrue(result['alignment_required'])

    def test_entrance_face_does_not_hide_longitudinal_walls(self):
        angles=np.linspace(-1.89,1.94,220)
        laser=SimpleNamespace(range_min=.1,range_max=12.,angles=angles,mount=np.zeros(4))
        side=.24/np.maximum(np.abs(np.sin(angles)),.001)
        face=.25/np.maximum(np.cos(angles),.001)
        scan=np.where((np.cos(angles)>0)&(np.abs(.25*np.tan(angles))>.24),face,side)
        scan=np.where(np.cos(angles)<=0,4.,scan)
        feature=corridor_features(scan,laser)
        self.assertEqual(feature[0],1)
        self.assertAlmostEqual(feature[1]*2,.48,delta=.01)

if __name__=='__main__':unittest.main()
