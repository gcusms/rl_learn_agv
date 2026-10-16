"""Check whether a new stateless actor changes the recorded failure decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    z=np.load(a.trace);model=torch.jit.load(str(a.actor),map_location='cpu').eval()
    with torch.no_grad():commands=model(torch.from_numpy(z['observations'])).numpy()*[.2,.5]
    changed=np.max(abs(commands-z['commands']),axis=1)>2e-8
    report=dict(scope='Decisions on the original recorded sensor history; not a new closed-loop result',
        actor_sha256=hashlib.sha256(a.actor.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
        frames=len(changed),changed_frames=int(changed.sum()),
        changed=[dict(time_s=float(z['labels'][i,0]),before=z['commands'][i].tolist(),after=commands[i].tolist())
            for i in np.flatnonzero(changed)])
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
