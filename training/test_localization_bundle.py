"""A bundle must opt in explicitly and pin its native executable contents."""
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import unittest

from neural_navigation.fast_static_localizer import temporal_contract
from neural_navigation.localization_bundle import localization_configuration


class BundleTests(unittest.TestCase):
    def test_legacy_without_contract_keeps_original_localizer(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(localization_configuration(directory))

    def test_native_requires_hashed_kernel_and_target_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / 'localization_contract.json').write_text(json.dumps(temporal_contract(True)))
            (bundle / 'localization_kernel.so').write_bytes(b'not executed in this metadata test')
            with self.assertRaises(ValueError):
                localization_configuration(bundle)
            files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in bundle.iterdir()}
            manifest = dict(files=files, native_kernel=dict(architecture='incompatible-platform'))
            (bundle / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'architecture'):
                localization_configuration(bundle)
            manifest['native_kernel']['architecture'] = platform.machine()
            (bundle / 'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(localization_configuration(bundle), temporal_contract(True))
            (bundle / 'localization_kernel.so').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                localization_configuration(bundle)

    def test_unknown_contract_and_orphan_components_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / 'uncertainty_contract.json').write_text('{}')
            with self.assertRaises(ValueError):
                localization_configuration(bundle)
            (bundle / 'localization_contract.json').write_text('{"schema": "unsupported"}')
            with self.assertRaises(ValueError):
                localization_configuration(bundle)


if __name__ == '__main__':
    unittest.main()
