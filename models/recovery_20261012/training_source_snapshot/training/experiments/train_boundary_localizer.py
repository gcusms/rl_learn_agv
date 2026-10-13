"""Train learned scan alignment near physical boundaries; no navigation eval labels."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from navrl.geometry import ROOT,wrap
from navrl.physical_env import PhysicalNavigationEnv
from neural_navigation.models import PoseScorer,alignment_features
from neural_navigation.static_footprint import static_footprint_clear
from train_directional_localizer import CollectionPrior
from experiments.train_quality_navigation import digest


def collect(entries,count,seed,footprint,validation=False):
    rng=np.random.default_rng(seed);data={k:[] for k in ('features','targets','truth','hypotheses','near','clutter','dropout')}
    if validation:data.update(coarse_features=[],coarse_hypotheses=[])
    for entry in entries:
        env=PhysicalNavigationEnv(scene_dir=entry['directory'],dynamic=False,localizer_factory=CollectionPrior)
        try:
            env.reset(seed=seed+int(entry.get('seed',0)))
            near=np.argwhere((env.map.clearance>.27)&(env.map.clearance<=.5));far=np.argwhere(env.map.clearance>.6)
            if not len(near) or not len(far):raise ValueError('Both boundary and open samples required')
            for i in range(count):
                boundary=i%2==0;cells=near if boundary else far
                for _ in range(1000):
                    truth=np.r_[env.map.xy(cells[int(rng.integers(len(cells)))])+rng.uniform(-.02,.02,2),rng.uniform(-np.pi,np.pi)]
                    if static_footprint_clear(env.map,truth,footprint,.04):break
                else:raise ValueError('Unable to sample physically clear true body')
                env.truth=truth;env._sync();scan=env._scan().copy()
                clutter=bool(rng.random()<.3);dropout=bool(rng.random()<.25)
                if clutter:
                    width=int(rng.integers(8,25));begin=int(rng.integers(len(scan)-width+1))
                    returns=rng.uniform(.4,2.)+rng.normal(0,.02,width)
                    scan[begin:begin+width]=np.minimum(scan[begin:begin+width],returns)
                if dropout:scan[rng.random(len(scan))<.1]=np.nan
                nominal=truth+rng.normal(0,[.08,.08,.055]);nominal[2]=wrap(nominal[2])
                x,h=alignment_features(env.map,env.lidar,nominal,scan,points=7,directional=True)
                error=h-truth;error[:,2]=wrap(error[:,2]);logits=-.5*np.sum((error/[.045,.045,.03])**2,axis=1)
                target=np.exp(logits-logits.max());target/=target.sum()
                for key,value in [('features',x),('targets',target.astype(np.float32)),('truth',truth),('hypotheses',h),('near',boundary),('clutter',clutter),('dropout',dropout)]:data[key].append(value)
                if validation:
                    cx,ch=alignment_features(env.map,env.lidar,nominal,scan,points=5,directional=True)
                    data['coarse_features'].append(cx);data['coarse_hypotheses'].append(ch)
            print('collected',entry['id'],count,flush=True)
        finally:env.close()
    return {k:np.asarray(v) for k,v in data.items()}


def metrics(net,data,coarse=False):
    feature='coarse_features' if coarse else 'features';hyp='coarse_hypotheses' if coarse else 'hypotheses'
    xy=[];yaw=[]
    with torch.no_grad():
        for start in range(0,len(data['truth']),64):
            x=data[feature][start:start+64,:,:net.net[0].in_features];h=data[hyp][start:start+64]
            weights=torch.softmax(net(torch.from_numpy(x)),dim=1).numpy()
            xy.extend(np.sum(weights[:,:,None]*h[:,:,:2],axis=1));yaw.extend(np.arctan2(np.sum(weights*np.sin(h[:,:,2]),axis=1),np.sum(weights*np.cos(h[:,:,2]),axis=1)))
    error=np.linalg.norm(np.asarray(xy)-data['truth'][:,:2],axis=1);angle=np.abs(wrap(np.asarray(yaw)-data['truth'][:,2]));result={}
    for name,keep in [('all',np.ones(len(error),bool)),('near_boundary',data['near']),('open',~data['near']),('clutter',data['clutter']),('dropout',data['dropout'])]:
        if keep.any():result[name]=dict(samples=int(keep.sum()),xy_mean_m=float(error[keep].mean()),xy_p95_m=float(np.quantile(error[keep],.95)),yaw_p95_rad=float(np.quantile(angle[keep],.95)))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples-per-map',type=int,default=240);parser.add_argument('--epochs',type=int,default=12);args=parser.parse_args()
    if args.output.exists() or args.samples_per_map<40 or not 1<=args.epochs<=50:parser.error('Fresh output and bounded pilot required')
    args.output.mkdir();torch.set_num_threads(1);torch.manual_seed(93501);rng=np.random.default_rng(93501)
    hardware=json.loads((args.parent.parent/'hardware.json').read_text());footprint=np.asarray(hardware['robot_footprint_m'])
    manifest=json.loads((ROOT/'assets/navigation_stage2_dataset/manifest.json').read_text())
    train_entries=manifest['splits']['train']+[dict(id='room_train',directory=str(ROOT/'assets/room_navigation_scripted'))]
    val_entries=manifest['splits']['validation'];project=ROOT.parent;sources={}
    for path in ROOT.rglob('*.py'):
        relative=path.relative_to(project)
        if 'runs' in relative.parts:continue
        sources[str(relative)]=digest(path);target=args.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    config=dict(parent_sha256=digest(args.parent),source_sha256=sources,training_seed=93501,validation_seed=93502,
        train_maps=[x['id'] for x in train_entries],validation_maps=[x['id'] for x in val_entries],footprint_m=footprint.tolist(),
        true_body_margin_m=.04,hypothesis_points=7,clutter_probability=.3,dropout_probability=.25,deployment_approved=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2));status=dict(status='running',phase='collection',started_unix=time.time())
    def save():(args.output/'status.json').write_text(json.dumps(status,indent=2))
    save()
    try:
        train=collect(train_entries,args.samples_per_map,93501,footprint);validation=collect(val_entries,120,93502,footprint,True)
        parent=torch.load(args.parent,map_location='cpu',weights_only=True);old=PoseScorer(int(parent.get('feature_dim',11)));old.load_state_dict(parent['state_dict']);old.eval()
        if int(parent.get('hypothesis_points',5))!=5:raise ValueError('Pilot baseline requires native 5-point parent')
        baseline=metrics(old,validation,True);grid_control=metrics(old,validation)
        net=PoseScorer(15);state=net.state_dict()
        for name,value in old.state_dict().items():
            if name=='net.0.weight':state[name].zero_();state[name][:,:value.shape[1]]=value
            else:state[name]=value.clone()
        net.load_state_dict(state);opt=torch.optim.Adam(net.parameters(),lr=3e-4)
        best_state=copy.deepcopy(net.state_dict());best_metrics=grid_control;best_score=grid_control['near_boundary']['xy_p95_m']+grid_control['open']['xy_p95_m'];history=[]
        status.update(phase='training',training_samples=len(train['truth']),validation_samples=len(validation['truth']));save()
        for epoch in range(args.epochs):
            net.train();losses=[]
            for indices in np.array_split(rng.permutation(len(train['truth'])),max(1,len(train['truth'])//64)):
                x=torch.from_numpy(train['features'][indices]);target=torch.from_numpy(train['targets'][indices]);logits=net(x)
                loss=-(target*torch.log_softmax(logits,dim=1)).sum(dim=1).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();losses.append(float(loss.detach()))
            net.eval();validation_metrics=metrics(net,validation);score=validation_metrics['near_boundary']['xy_p95_m']+validation_metrics['open']['xy_p95_m']
            if score<best_score and validation_metrics['open']['xy_p95_m']<=baseline['open']['xy_p95_m']*1.1:
                best_score=score;best_state=copy.deepcopy(net.state_dict());best_metrics=validation_metrics
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=validation_metrics));status['completed_epochs']=epoch+1;save();print('epoch',epoch,validation_metrics,flush=True)
        saved={**parent,'state_dict':best_state,'feature_dim':15,'hypothesis_points':7,'boundary_training':config,'experimental_not_approved':True}
        torch.save(saved,args.output/'localizer.pt')
        report=dict(baseline_native_5=baseline,unchanged_scorer_grid_7=grid_control,candidate=best_metrics,history=history,
            training_samples=len(train['truth']),validation_samples=len(validation['truth']),deployment_approved=False,
            limitation='Independent validation maps with simulated single scans and synthetic occlusion/dropout. Truth is offline target only; does not validate temporal recovery, navigation, real noise, pose covariance or Orin timing.')
        (args.output/'REPORT.json').write_text(json.dumps(report,indent=2));status.update(status='complete',finished_unix=time.time());save()
    except Exception as error:status.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
