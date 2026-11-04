"""Check the exported policy's external state/action/language contract."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_io.preprocessing import StateActionNormalizer, read_language_assets


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = StateActionNormalizer(ROOT/'robot_assets/stats.json')

    def test_exported_state_mean_maps_to_zero(self):
        raw = np.asarray([-6.563590049743652,-1.089989423751831,-7.835640907287598,
                          76.23908233642578,-90.87577819824219,68.13605499267578],np.float32)
        state = self.normalizer.normalize_state(raw)
        self.assertEqual(state.shape,(1,6))
        self.assertEqual(state.dtype,np.float32)
        np.testing.assert_array_equal(state,np.zeros((1,6),np.float32))

    def test_normalized_zero_actions_restore_exported_units(self):
        actions = self.normalizer.denormalize_actions(np.zeros((1,50,6),np.float32))
        expected = [-6.393108367919922,-2.330458402633667,-9.206443786621094,
                    76.06784057617188,-90.70893859863281,62.991172790527344]
        np.testing.assert_array_equal(actions[0,0],np.asarray(expected,np.float32))
        self.assertEqual(actions.shape,(1,50,6))
        self.assertTrue(actions.flags.c_contiguous)

    def test_reference_task_tokens_keep_newline_and_mask(self):
        language = read_language_assets(ROOT/'robot_assets/language.npz')
        # The pretokenized task ends with token 198 (newline); no chat role prefix.
        np.testing.assert_array_equal(language['language_tokens'][0,:11],
            [8979,9399,260,4461,20636,284,1830,618,260,3985,198])
        self.assertEqual(int(language['language_mask'].sum()),11)
        self.assertEqual(language['language_tokens'].dtype,np.int64)
        self.assertEqual(language['language_mask'].dtype,np.bool_)
        self.assertTrue((language['language_tokens'][0,11:]==2).all())

    def test_invalid_state_or_action_is_rejected(self):
        for state in ([0]*5,[float('nan')]*6):
            with self.assertRaises(ValueError):
                self.normalizer.normalize_state(state)
        with self.assertRaises(ValueError):
            self.normalizer.denormalize_actions(np.zeros((50,6),np.float32))


if __name__ == '__main__':
    unittest.main()
