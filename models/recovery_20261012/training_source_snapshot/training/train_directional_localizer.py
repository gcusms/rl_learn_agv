"""Train scan alignment on independent train maps; truth is label-only.

Warm-start the existing scorer and select epochs on validation maps, never on
navigation acceptance cases. Outputs are experimental components, not deployed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from navrl.geometry import ROOT, wrap
from navrl.physical_env import PhysicalNavigationEnv
from neural_navigation.models import PoseScorer, alignment_features


class CollectionPrior:
    """Inert reset prior for ray dataset collection; forbidden as runtime pose."""
    def __init__(self,*args):self.std=np.ones(3)
    def reset(self,nominal,rng):self.pose=np.array(nominal)
    def update(self,odom,scan):return self.pose.copy()
    def close(self):pass


def collect(entries,count,seed):
    rng=np.random.default_rng(seed);features=[];targets=[];truths=[];hypotheses=[];wide=[]
    for entry in entries:
        env=PhysicalNavigationEnv(scene_dir=entry['directory'],dynamic=False,localizer_factory=CollectionPrior)
        try:
            env.reset(seed=seed);cells=np.argwhere(env.map.clearance>.45)
            for i in range(count):
                true=np.r_[env.map.xy(cells[int(rng.integers(len(cells)))]),rng.uniform(-np.pi,np.pi)]
                env.truth=true;env._sync();scan=env._scan()
                broad=rng.random()<.2;spans=(.48,.48,.64) if broad else (.24,.24,.16)
                noise=(.15,.15,.18) if broad else ((.035,.035,.025) if rng.random()<.65 else (.10,.10,.065))
                nominal=true+rng.normal(0,noise);nominal[2]=wrap(nominal[2])
                x,h=alignment_features(env.map,env.lidar,nominal,scan,spans=spans,directional=True)
                error=h-true;error[:,2]=wrap(error[:,2]);logits=-.5*np.sum((error/[.055,.055,.035])**2,axis=1)
                prob=np.exp(logits-logits.max());prob/=prob.sum()
                features.append(x);targets.append(prob.astype(np.float32));truths.append(true);hypotheses.append(h);wide.append(broad)
            print('collected',entry['id'],count,flush=True)
        finally:env.close()
    return np.array(features),np.array(targets),np.array(truths),np.array(hypotheses),np.array(wide)


def metrics(net,x,truth,hypotheses,wide,device):
    with torch.no_grad():
        weights=torch.softmax(net(torch.as_tensor(x,device=device)),dim=1).cpu().numpy()
    xy=np.sum(weights[:,:,None]*hypotheses[:,:,:2],axis=1)
    yaw=np.arctan2(np.sum(weights*np.sin(hypotheses[:,:,2]),axis=1),np.sum(weights*np.cos(hypotheses[:,:,2]),axis=1))
    err=np.linalg.norm(xy-truth[:,:2],axis=1);ae=np.abs(wrap(yaw-truth[:,2]))
    result={}
    for name,selected in [('all',np.ones(len(x),bool)),('normal',~wide),('recovery',wide)]:
        result[name]=dict(samples=int(selected.sum()),xy_mean_m=float(err[selected].mean()),xy_p95_m=float(np.quantile(err[selected],.95)),yaw_mean_rad=float(ae[selected].mean()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples-per-map',type=int,default=400);p.add_argument('--epochs',type=int,default=30);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(63001);rng=np.random.default_rng(63001)
    device='cuda' if torch.cuda.is_available() else 'cpu';print('device',device,flush=True)
    manifest=json.loads((ROOT/'assets/navigation_stage2_dataset/manifest.json').read_text())
    entries=manifest['splits']['train']+[dict(id='room_train',directory=str(ROOT/'assets/room_navigation_scripted'))]
    x,y,truth,h,wide=collect(entries,a.samples_per_map,63001)
    vx,vy,vt,vh,vw=collect(manifest['splits']['validation'],150,64001)
    np.savez_compressed(a.output/'training_labels.npz',features=x,targets=y)
    np.savez_compressed(a.output/'validation_labels.npz',features=vx,targets=vy,truth=vt,hypotheses=vh,wide=vw)
    parent=torch.load(a.parent/'localizer.pt',map_location='cpu',weights_only=True)
    old=PoseScorer();old.load_state_dict(parent['state_dict']);old.to(device)
    baseline=metrics(old,vx[:,:,:11],vt,vh,vw,device)
    net=PoseScorer(15);state=net.state_dict()
    for key,value in parent['state_dict'].items():
        if key=='net.0.weight':state[key].zero_();state[key][:,:11]=value
        else:state[key]=value.clone()
    net.load_state_dict(state);net.to(device);opt=torch.optim.Adam(net.parameters(),lr=.0003)
    tx=torch.as_tensor(x,device=device);ty=torch.as_tensor(y,device=device)
    best=float('inf');history=[];best_state=None;best_metrics=None
    for epoch in range(a.epochs):
        net.train();losses=[]
        for index in np.array_split(rng.permutation(len(x)),max(1,len(x)//96)):
            pred=net(tx[index]);loss=-(ty[index]*torch.log_softmax(pred,dim=1)).sum(dim=1).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        net.eval();val=metrics(net,vx,vt,vh,vw,device)
        score=val['normal']['xy_mean_m']+val['normal']['xy_p95_m']+.2*val['recovery']['xy_mean_m']
        if score<best:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};best_metrics=val
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=val));print('epoch',epoch,val,flush=True)
    saved={**parent,'state_dict':best_state,'feature_dim':15,'hypothesis_points':5,'metrics':best_metrics,'training_seed':63001,'validation_seed':64001,'experimental_not_approved':True}
    torch.save(saved,a.output/'localizer.pt')
    report=dict(device=device,training_samples=len(x),validation_samples=len(vx),train_maps=[e['id'] for e in entries],validation_maps=[e['id'] for e in manifest['splits']['validation']],baseline=baseline,candidate=best_metrics,history=history,limits='Single scan supervised pose labels; no dynamic clutter, rolling-scan deskew or global relocation validation. Acceptance set untouched.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print('saved',a.output,flush=True)


if __name__=='__main__':main()
