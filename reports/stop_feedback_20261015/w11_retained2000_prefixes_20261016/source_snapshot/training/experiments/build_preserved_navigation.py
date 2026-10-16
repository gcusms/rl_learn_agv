"""Compose two retained learned policies; no steering teacher at inference.

Use the real deployed 300D navigation actor outside a directional approach;
inside that approach retain the validated 303D approach/terminal actor. Both
see the same live sensor prefix and action masks. Save the two complete PPO
parents explicitly: the composite is not falsely labelled a new PPO run.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch


class PreservedNavigationActor(torch.nn.Module):
    def __init__(self,navigation,directional,radius=.30):
        super().__init__();self.navigation=navigation;self.directional=directional;self.radius=radius
    def forward(self,x):
        approach=(torch.linalg.vector_norm(x[:,218:220]*12,dim=1)<=self.radius)&(x[:,302]>.5)
        return torch.where(approach[:,None],self.directional(x),self.navigation(x[:,:300]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--navigation',type=Path,required=True)
    p.add_argument('--directional',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--navigation-checkpoint',type=Path,required=True);p.add_argument('--terminal-checkpoint',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise ValueError('Preserve existing model outputs')
    old_manifest=json.loads((a.navigation/'manifest.json').read_text());manifest=json.loads((a.directional/'manifest.json').read_text())
    if old_manifest['action_contract']!=manifest['action_contract']:raise ValueError('Action contracts differ')
    if old_manifest['observation_contract']!=manifest['quality_prefix_contract']:raise ValueError('300D sensor prefix differs')
    # Verify parent files before composition; do not accept unidentified weights.
    for bundle,m in [(a.navigation,old_manifest),(a.directional,manifest)]:
        for name,sha in m['files'].items():
            if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=sha:raise ValueError('Parent hash mismatch: '+name)
    a.output.mkdir();portable=a.output/'portable';shutil.copytree(a.directional,portable)
    archived=torch.jit.load(str(a.navigation/'actor.pt')).eval();terminal=torch.jit.load(str(a.directional/'actor.pt')).eval()
    composite=PreservedNavigationActor(archived,terminal).eval()
    rng=np.random.default_rng(117610);obs=rng.uniform(-1,1,(1024,303)).astype(np.float32)
    obs[:256,302]=0.;obs[256:512,218:220]=rng.uniform(.04,.08,(256,2));obs[256:512,302]=1.
    obs[512:,218:220]=rng.uniform(-.015,.015,(512,2));obs[512:,302]=1.
    x=torch.from_numpy(obs)
    with torch.no_grad():
        gate=(torch.linalg.vector_norm(x[:,218:220]*12,dim=1)<=.30)&(x[:,302]>.5)
        expected=torch.where(gate[:,None],terminal(x),archived(x[:,:300]))
        exported=torch.jit.trace(composite,x);torch.testing.assert_close(exported(x),expected,rtol=0,atol=0)
        torch.testing.assert_close(exported(x)[~gate],archived(x[:,:300])[~gate],rtol=0,atol=0)
    exported.save(str(portable/'actor.pt'))
    preservation=dict(schema='archived_navigation_directional_approach_v1',approach_radius_m=.30,
        requires_directional_goal=True,archived_navigation_dimensions=300,composite_dimensions=303,
        unchanged_actions=True,comparison_inputs=len(obs),navigation_parity_inputs=int((~gate).sum()),
        runtime_teacher=False,scope='exact parent parity on identical observations; closed-loop inputs may still differ')
    (portable/'navigation_preservation.json').write_text(json.dumps(preservation,indent=2))
    manifest['navigation_preservation_contract']=preservation
    manifest['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in portable.iterdir() if p.name!='manifest.json'}
    (portable/'manifest.json').write_text(json.dumps(manifest,indent=2))
    parents=a.output/'training_parents';parents.mkdir()
    for checkpoint,name in [(a.navigation_checkpoint,'navigation_policy.zip'),(a.terminal_checkpoint,'terminal_policy.zip')]:
        shutil.copy2(checkpoint,parents/name)
    card=dict(deployment_approved=False,additional_ppo_steps=0,
        composition='archived navigation outside 0.30m directional approach; trained approach/terminal inside',
        navigation_checkpoint_sha256=hashlib.sha256(a.navigation_checkpoint.read_bytes()).hexdigest(),
        terminal_checkpoint_sha256=hashlib.sha256(a.terminal_checkpoint.read_bytes()).hexdigest(),
        composite_actor_sha256=manifest['files']['actor.pt'],
        resumption='Use the two original PPO checkpoints; a single composite PPO optimizer has not been constructed',
        localization='v5 from directional bundle; subject to paired command-input validation')
    (a.output/'MODEL_CARD.json').write_text(json.dumps(card,ensure_ascii=False,indent=2));print(json.dumps(card,indent=2))


if __name__=='__main__':main()
