"""Finish the offline candidate's standard continuation metadata, no weights.

The separate offline fitter has no physical environment. This postprocessing
adds the top-level recovery contract expected by the existing reference-BC
training CLI, and labels the model card as supervised-only initialization.
"""
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    plan = json.loads((OUT / 'BC_PLAN.json').read_text())
    model = ROOT / plan['model_output']
    if (model / 'METADATA_FINALIZATION.json').exists():
        raise ValueError('Metadata is already finalized; do not rewrite a reviewed candidate')
    status = json.loads((model / 'status.json').read_text())
    result = json.loads((model / 'RESULTS.json').read_text())
    if (status['status'] != 'offline_bc_complete' or not result['completed']
            or result['additional_ppo_steps'] or result['environment_steps']):
        raise ValueError('Only a completed, zero-PPO offline fit can be finalized')
    bundle = model / 'bc_portable'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if (digest(model / 'policy_bc.zip') != result['policy_sha256']
            or digest(bundle / 'manifest.json') != result['export_manifest_sha256']):
        raise ValueError('The completed checkpoint or exported manifest changed')
    if 'MODEL_CARD.json' in manifest['files'] or manifest['additional_ppo_steps']:
        raise ValueError('Unexpected card hash or nonzero-PPO export contract')
    for name, expected in manifest['files'].items():
        if digest(bundle / name) != expected:
            raise ValueError('Exported component changed: ' + name)
    with zipfile.ZipFile(model / 'policy_bc.zip') as checkpoint:
        data = json.loads(checkpoint.read('data'))
    contract = manifest['navigation_preservation_contract']['recovery_contract']
    if data['recovery_contract'] != contract or data['num_timesteps'] != 0:
        raise ValueError('Saved PPO object and exported recovery contract disagree')
    protected = [model / 'policy_bc.zip', model / 'demonstrations.npz', model / 'random_state.pt']
    protected += [bundle / name for name in manifest['files']]
    before = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    config_path = model / 'config.json'
    card_path = bundle / 'MODEL_CARD.json'
    old_hashes = dict(config_sha256=digest(config_path), card_sha256=digest(card_path))
    config = json.loads(config_path.read_text())
    config.update(seed=data['seed'], recovery_contract=contract,
        maximum_bc_updates=plan['maximum_supervised_updates'],
        demonstration_input_frames=result['total_examples'],
        frozen_base_sha256=result['frozen_base_sha256'],
        initialization_not_counted_as_ppo_steps=True, additional_ppo_steps=0,
        no_physical_environment=True, reward_changed=False,
        continuation_reference_contract_ready=True)
    write(config_path, config)
    card = json.loads(card_path.read_text())
    card.update(method='Offline BC initialization from complete safe PPO trajectories; no new PPO steps',
        supervised_updates=result['supervised_updates'], training_examples=result['total_examples'],
        training_data_audit=str(OUT.relative_to(ROOT) / 'RESULTS.json'),
        old_holdout00_and01_now_training_tasks=True,
        physical_validation_completed=False, deployment_approved=False)
    write(card_path, card)
    after = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    if before != after:
        raise ValueError('Metadata postprocessing changed model or data artifacts')
    final = dict(completed=True, weights_optimizer_data_and_rng_unchanged=True,
        top_level_recovery_contract_matches_checkpoint_and_manifest=True,
        reference_bc_metadata_ready=True, prior_metadata=old_hashes,
        final_config_sha256=digest(config_path), final_card_sha256=digest(card_path),
        protected_artifact_sha256=after, finalizer_sha256=digest(Path(__file__)),
        physical_validation_completed=False, new_ppo_steps=0,
        goal_achieved=False, deployment_approved=False)
    write(model / 'METADATA_FINALIZATION.json', final)
    print(json.dumps(dict(completed=True, reference_bc_metadata_ready=True,
        protected_artifacts_unchanged=len(after), new_ppo_steps=0), indent=2))


if __name__ == '__main__':
    main()
