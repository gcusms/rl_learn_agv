"""Check the published model package, with optional native/TorchScript parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', action='store_true')
    args = parser.parse_args()
    release = ROOT/'models/recovery_20261012'
    manifest = json.loads((release/'MODEL_MANIFEST.json').read_text())
    for relative, expected in manifest['sha256'].items():
        path = release/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Package hash mismatch: {relative}')
    print(f"Verified {len(manifest['sha256'])} packaged files")
    if args.inference:
        sys.path[:0] = [str(ROOT/'training'), str(ROOT)]
        import numpy as np
        import torch
        from sb3_contrib import MaskablePPO
        from neural_navigation.masked_actions import masks_from_observation, ACTIONS
        torch.set_num_threads(1)
        model = MaskablePPO.load(release/'candidate/policy.zip', device='cpu')
        actor = torch.jit.load(str(release/'portable/actor.pt'), map_location='cpu')
        observations = np.zeros((8, 300), np.float32)
        observations[:,235:244] = 1
        observations[:,267:269] = 1
        observations[:,216] = .2
        observations[:,229] = 1
        observations[:,294] = 1
        observations[:,250] = 1
        observations[:4,232] = -.3
        observations[:4,248] = 1
        observations[4:,232] = .8
        mask = masks_from_observation(observations[:,:282])
        choices, _ = model.predict(observations, action_masks=mask, deterministic=True)
        with torch.no_grad():
            commands = actor(torch.from_numpy(observations)).numpy()
        np.testing.assert_array_equal(commands, ACTIONS[choices])
        print('Native checkpoint and exported actor agree on 8 inputs; no rollout or hardware commands')


if __name__ == '__main__':
    main()
