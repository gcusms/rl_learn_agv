import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from recover_optimization_pilot import checkpoint,verify_source
from run_optimization_pilot import completed_artifact


class RecoveryTests(unittest.TestCase):
    def test_truncated_latest_checkpoint_falls_back_to_intact_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            with zipfile.ZipFile(p/'policy_60000.zip','w') as z:z.writestr('data',json.dumps({'num_timesteps':60000}))
            (p/'policy_70008.zip').write_bytes(b'truncated zip')
            (p/'policy_initial.zip').write_bytes(b'not a numbered checkpoint')
            with zipfile.ZipFile(p/'policy_80004.zip','w') as z:z.writestr('data',json.dumps({'num_timesteps':90000}))
            f,steps=checkpoint(p);self.assertEqual(f.name,'policy_60000.zip');self.assertEqual(steps,60000)

    def test_changed_reward_or_runtime_source_refuses_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);candidate=p/'candidate';candidate.mkdir();source=p/'reward.py';source.write_text('original')
            (candidate/'config.json').write_text(json.dumps({'launch_source_sha256':{'reward.py':hashlib.sha256(source.read_bytes()).hexdigest()}}))
            verify_source(candidate,p);source.write_text('changed')
            with self.assertRaises(RuntimeError):verify_source(candidate,p)

    def test_partial_wrong_policy_or_changed_case_report_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);candidate=p/'candidate';candidate.mkdir();policy=candidate/'policy.zip';policy.write_bytes(b'policy1')
            cases=p/'cases.json';cases.write_text('{}');report=p/'report.json';command=['python','eval','--manifest',str(cases),'--output',str(report)]
            report.write_text('{broken');self.assertFalse(completed_artifact(command,candidate))
            x=dict(completed=True,manifest_sha256=hashlib.sha256(cases.read_bytes()).hexdigest(),policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
            report.write_text(json.dumps(x));self.assertTrue(completed_artifact(command,candidate))
            policy.write_bytes(b'policy2');self.assertFalse(completed_artifact(command,candidate))
            policy.write_bytes(b'policy1');cases.write_text('{"changed":true}');self.assertFalse(completed_artifact(command,candidate))


if __name__=='__main__':unittest.main()
