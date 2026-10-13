"""Validate provenance without launching an environment or PPO rollout."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from prototypes import train_continued_recovery as entry


class EntryTests(unittest.TestCase):
    def test_capture_precedes_trainer_and_keeps_output_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'trial'
            provenance = Path(temp)/'trial_prototype_provenance'
            def fake_main():
                self.assertFalse(output.exists())
                data = json.loads((provenance/'PROTOTYPE_SOURCE.json').read_text())
                self.assertTrue(data['captured_before_training'])
                self.assertEqual(len(data['source_sha256']),3)
            with patch.object(entry.sys, 'argv', ['entry','--output',str(output),
                    '--resume-quality','--scheduled-recovery']), \
                 patch.object(entry.trainer,'main',side_effect=fake_main), \
                 patch.object(entry.trainer,'make_environment'), \
                 patch.object(entry.trainer,'resume_quality'), \
                 patch.object(entry.trainer,'RecoveryCoverageProgress'):
                entry.main()
            data = json.loads((provenance/'PROTOTYPE_SOURCE.json').read_text())
            self.assertTrue(data['source_hashes_unchanged'])
            self.assertFalse(data['deployment_approved'])

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(entry.sys,'argv',['entry','--output',temp,
                    '--resume-quality','--scheduled-recovery']):
                with self.assertRaises(FileExistsError): entry.main()


if __name__ == '__main__': unittest.main()
