"""Supervised model training. Simulator truth and graph solvers create TRAINING LABELS only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from navrl.geometry import ROOT,wrap
from navrl.physical_env import PhysicalNavigationEnv
from neural_navigation.models import PoseScorer,NeuralRouter,alignment_features


def collect_pose(split,count,seed):
    manifest=json.loads((ROOT/'assets/navigation_dataset/manifest.json').read_text());rng=np.random.default_rng(seed)
    x=[];target=[];poses=[];nominals=[]
    entries=manifest['splits'][split]
    for entry in entries:
        env=PhysicalNavigationEnv(scene_dir=entry['directory'],localizer='particle',dynamic=True)
        try:
            env.reset(seed=seed);cells=np.argwhere(env.map.clearance>.40)
            for i in range(count):
                true=np.r_[env.map.xy(cells[int(rng.integers(len(cells)))]),rng.uniform(-np.pi,np.pi)]
                env.truth=true;env._sync()
                if i%10==0:
                    for name in env.names[1:]:
                        q,d=env.addresses[name]
                        for _ in range(50):
                            point=env.map.xy(cells[int(rng.integers(len(cells)))])
                            if np.linalg.norm(point-true[:2])>.8:break
                        env.data.qpos[q]=point;env.data.qvel[d]=0.
                    import mujoco
                    mujoco.mj_forward(env.model,env.data)
                scan=env._scan();noise=[.035,.035,.025] if rng.random()<.65 else [.10,.10,.065];nominal=true+rng.normal(0,noise);nominal[2]=wrap(nominal[2])
                features,hypotheses=alignment_features(env.map,env.lidar,nominal,scan)
                error=hypotheses-true;error[:,2]=wrap(error[:,2])
                logits=-.5*np.sum((error/[.055,.055,.035])**2,axis=1);logits-=logits.max();prob=np.exp(logits);prob/=prob.sum()
                x.append(features);target.append(prob.astype(np.float32));poses.append(true);nominals.append(nominal)
            print('pose dataset',split,entry['id'],count,flush=True)
        finally:env.close()
    return np.array(x),np.array(target),np.array(poses),np.array(nominals)


def coarse_free(m):
    # Sensor/map representation, not route search: 0.20 m center samples of the robot-inflated occupancy grid.
    h,w=m.free.shape;return m.free[2:h//4*4:4,2:w//4*4:4].astype(np.float32)


def router_data(goals_per_map,seed):
    from navrl.map import StaticMap
    manifest=json.loads((ROOT/'assets/navigation_dataset/manifest.json').read_text());rng=np.random.default_rng(seed)
    features=[];targets=[];reachable=[]
    for entry in manifest['splits']['train']:
        m=StaticMap(Path(entry['directory'])/'decision_map.yaml',radius=.30);base=coarse_free(m)
        for j in range(goals_per_map):
            free=base.copy();dynamic=np.zeros_like(free)
            if j%3==1:
                for _ in range(2):
                    r,c=np.argwhere(free)[int(rng.integers(np.count_nonzero(free)))];dynamic[max(0,r-1):r+2,max(0,c-1):c+2]=1.
                free[dynamic>0]=0.
            if j%3==2:free[:,free.shape[1]//2]=0.
            cells=np.argwhere(free);goal=cells[int(rng.integers(len(cells)))];h,w=free.shape
            rows=[];cols=[]
            for r,c in cells:
                for dr,dc in NeuralRouter.directions:
                    rr,cc=r+dr,c+dc
                    if 0<=rr<h and 0<=cc<w and free[rr,cc]:rows.append(r*w+c);cols.append(rr*w+cc)
            graph=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(h*w,h*w))
            distance=dijkstra(graph,directed=False,indices=int(goal[0]*w+goal[1])).reshape(h,w)
            distance[free==0]=np.inf;value=np.exp(-.02*distance).astype(np.float32)
            gm=np.zeros_like(free);gm[tuple(goal)]=1.;pose=np.zeros_like(free)
            # Closed barriers are observations, provided to the network just like map occupancy.
            features.append(np.stack([free,dynamic,gm,pose]));targets.append(value[None]);reachable.append((np.isfinite(distance)&(free>0))[None].astype(np.float32))
    return np.array(features),np.array(targets),np.array(reachable)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/neural_components');p.add_argument('--pose-samples-per-map',type=int,default=1500);p.add_argument('--pose-epochs',type=int,default=25);p.add_argument('--router-epochs',type=int,default=12)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(15100);rng=np.random.default_rng(15100)
    x,y,truth,nominal=collect_pose('train',a.pose_samples_per_map,15100)
    vx,vy,vtruth,vnominal=collect_pose('validation',300,15200)
    np.savez_compressed(a.output/'pose_training.npz',features=x,targets=y)
    net=PoseScorer();opt=torch.optim.Adam(net.parameters(),lr=.001);tx=torch.from_numpy(x);ty=torch.from_numpy(y)
    for epoch in range(a.pose_epochs):
        losses=[]
        for index in np.array_split(rng.permutation(len(tx)),max(1,len(tx)//128)):
            pred=net(tx[index]);loss=-(ty[index]*torch.log_softmax(pred,dim=1)).sum(dim=1).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        if epoch%5==0:print('pose epoch',epoch,'cross_entropy',float(np.mean(losses)),flush=True)
    with torch.no_grad():weights=torch.softmax(net(torch.from_numpy(vx)),dim=1).numpy()
    from neural_navigation.models import pose_hypotheses
    estimates=[]
    for prior,prob in zip(vnominal,weights):
        hypotheses,_=pose_hypotheses(prior);estimates.append(np.r_[prob@hypotheses[:,:2],np.arctan2(prob@np.sin(hypotheses[:,2]),prob@np.cos(hypotheses[:,2]))])
    estimates=np.array(estimates);errors=estimates-vtruth;errors[:,2]=wrap(errors[:,2]);pos=np.linalg.norm(errors[:,:2],axis=1)
    metrics=dict(validation_maps=2,validation_samples=len(vx),mean_position_error_m=float(pos.mean()),p95_position_error_m=float(np.quantile(pos,.95)),mean_heading_error_rad=float(np.abs(errors[:,2]).mean()),nominal_mean_position_error_m=float(np.linalg.norm(vnominal[:,:2]-vtruth[:,:2],axis=1).mean()))
    torch.save(dict(state_dict=net.state_dict(),architecture='trained local pose-hypothesis attention; no AMCL/particle filter',metrics=metrics),a.output/'localizer.pt');print('pose validation',metrics,flush=True)
    rx,ry,rr=router_data(36,15300);model=NeuralRouter();opt=torch.optim.Adam(model.parameters(),lr=.025)
    rx=torch.from_numpy(rx);ry=torch.from_numpy(ry);rr=torch.from_numpy(rr)
    for epoch in range(a.router_epochs):
        losses=[]
        for index in np.array_split(rng.permutation(len(rx)),max(1,len(rx)//12)):
            value,_,reach=model(rx[index]);loss=torch.nn.functional.mse_loss(value,ry[index])+ .02*torch.nn.functional.binary_cross_entropy(reach,rr[index])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        print('neural router epoch',epoch,'loss',float(np.mean(losses)),flush=True)
    torch.save(dict(state_dict=model.state_dict(),iterations=model.iterations,architecture='learned convolutional goal propagation; no runtime graph search'),a.output/'router.pt')
    config=dict(initial_pose='coarse supplied prior',runtime_amcl=False,runtime_particle_filter=False,runtime_robot_astar=False,training_truth_labels_only=True,localizer_validation=metrics,pose_training_samples=len(x),map_split='train',router_examples=len(rx))
    (a.output/'config.json').write_text(json.dumps(config,indent=2));print('Saved learned components',a.output,flush=True)


if __name__=='__main__':main()
