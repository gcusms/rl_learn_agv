"""Learned map/laser alignment and an unrolled, trainable neural routing model."""
import numpy as np
import torch
from torch import nn
from scipy.ndimage import distance_transform_edt
from navrl.geometry import wrap


class PoseScorer(nn.Module):
    def __init__(self,feature_dim=11):
        super().__init__();self.net=nn.Sequential(nn.Linear(feature_dim,48),nn.Tanh(),nn.Linear(48,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def pose_hypotheses(pose,spans=(.24,.24,.16),points=5):
    if points<3 or points>11 or points%2!=1:raise ValueError('Pose grid must have 3 to 11 odd-numbered samples per axis')
    dx,dy,da=np.meshgrid(*(np.linspace(-span,span,points) for span in spans),indexing='ij')
    delta=np.c_[dx.ravel(),dy.ravel(),da.ravel()]
    c,s=np.cos(pose[2]),np.sin(pose[2]);world=delta[:,:2]@np.array([[c,s],[-s,c]])
    poses=np.c_[pose[:2]+world,wrap(pose[2]+delta[:,2])]
    return poses,delta


def alignment_features(static_map,lidar,pose,scan,spans=(.24,.24,.16),points=5,directional=False):
    hypotheses,delta=pose_hypotheses(pose,spans,points)
    indices=np.linspace(0,len(scan)-1,60).astype(int)
    valid=np.isfinite(scan[indices])&(scan[indices]>=lidar.range_min)&(scan[indices]<lidar.range_max)
    indices=indices[valid]
    if not len(indices):
        empty=np.c_[np.zeros((len(delta),7)),np.ones(len(delta)),delta/spans]
        if directional:empty=np.c_[empty,np.ones((len(delta),2)),np.zeros((len(delta),2))]
        return empty.astype(np.float32),hypotheses
    yaw=hypotheses[:,2];mount=lidar.mount
    origin=hypotheses[:,:2]+np.c_[np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1]]
    angles=yaw[:,None]+mount[3]+lidar.angles[indices]
    ends=origin[:,None,:]+scan[indices][None,:,None]*np.stack([np.cos(angles),np.sin(angles)],axis=-1)
    d=static_map.lookup(ends,static_map.distance,outside=1.)
    features=np.c_[np.minimum(d,.6).mean(axis=1)/.6,np.median(np.minimum(d,.6),axis=1)/.6,
                    (d<.06).mean(axis=1),(d<.12).mean(axis=1),(d<.25).mean(axis=1),
                    np.minimum(d,.2).mean(axis=1)/.2,(d>.5).mean(axis=1),
                    ~static_map.lookup(hypotheses[:,:2],~static_map.blocked,outside=False),delta/spans]
    if directional:
        # End walls can cover fewer angles than side walls. Supply directional
        # evidence explicitly instead of letting the all-ray average hide it.
        # Geometry is immutable; cache field normals, never sensor returns.
        if not hasattr(static_map,'_alignment_normals'):
            signed=static_map.distance-distance_transform_edt(static_map.occupied)*static_map.res
            gy,gx=np.gradient(signed,static_map.res);norm=np.maximum(np.hypot(gx,gy),1e-8)
            static_map._alignment_normals=(gx/norm,-gy/norm)
        nx=static_map.lookup(ends,static_map._alignment_normals[0],outside=0.)
        ny=static_map.lookup(ends,static_map._alignment_normals[1],outside=0.)
        forward=np.abs(nx*np.cos(yaw[:,None])+ny*np.sin(yaw[:,None]))
        side=np.abs(-nx*np.sin(yaw[:,None])+ny*np.cos(yaw[:,None]))
        costs=[];inliers=[]
        for weight in [forward,side]:
            total=weight.sum(axis=1);den=np.maximum(total,1e-8)
            costs.append(np.where(total>1e-6,(np.minimum(d,.2)*weight).sum(axis=1)/den/.2,1.))
            inliers.append(np.where(total>1e-6,((d<.06)*weight).sum(axis=1)/den,0.))
        features=np.column_stack([features,*costs,*inliers])
    return features.astype(np.float32),hypotheses


def scan_map_agreement(static_map,lidar,pose,scan):
    """Fraction of measured endpoints within 12cm of known static occupancy.

    Low support can mean wrong pose, unmapped obstacles or a changed map. This
    is an observable diagnostic, not ground-truth error or calibrated confidence.
    """
    indices=np.linspace(0,len(scan)-1,60).astype(int);scan=np.asarray(scan)
    valid=np.isfinite(scan[indices])&(scan[indices]>=lidar.range_min)&(scan[indices]<lidar.range_max)
    indices=indices[valid]
    if not len(indices):return 0.
    yaw=pose[2];mount=lidar.mount
    origin=pose[:2]+[np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1]]
    angles=yaw+mount[3]+lidar.angles[indices]
    ends=origin+scan[indices,None]*np.c_[np.cos(angles),np.sin(angles)]
    return float(np.mean(static_map.lookup(ends,static_map.distance,outside=1.)<.12))


class LearnedLocalizer:
    """Odometry propagates state; trained attention over local pose hypotheses corrects it.
    No particle filter, AMCL process, map search planner or simulator truth is used.
    """
    def __init__(self,static_map,lidar,checkpoint):
        self.map=static_map;self.lidar=lidar
        checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.feature_dim=int(checkpoint.get('feature_dim',11))
        if self.feature_dim not in (11,15):raise ValueError('Unsupported localization feature dimension')
        self.network=PoseScorer(self.feature_dim);self.network.load_state_dict(checkpoint['state_dict']);self.network.eval()
        self.recovery_config=checkpoint.get('localization_recovery')
        self.hypothesis_points=int(checkpoint.get('hypothesis_points',5))
        if self.hypothesis_points<3 or self.hypothesis_points>11 or self.hypothesis_points%2!=1:
            raise ValueError('Invalid localizer hypothesis_points')
        self.pose=None;self.std=np.ones(3);self.last_odom=None
        self.valid_scan_fraction=0.;self.scan_update_accepted=False;self.missing_scan_updates=0
        self.map_agreement=0.;self.posterior_entropy=1.;self.recovery_attempts=0;self.recovery_accepted=0
    def reset(self,nominal,rng):
        self.pose=np.array(nominal,float);self.std=np.array([.12,.12,.08]);self.last_odom=None
        self.valid_scan_fraction=0.;self.scan_update_accepted=False;self.missing_scan_updates=0
        self.map_agreement=0.;self.posterior_entropy=1.;self.recovery_attempts=0;self.recovery_accepted=0
    def update(self,odom,scan):
        if self.last_odom is not None:
            movement=np.asarray(odom[:2])-self.last_odom[:2];yaw=self.last_odom[2]
            local=np.array([np.cos(yaw)*movement[0]+np.sin(yaw)*movement[1],-np.sin(yaw)*movement[0]+np.cos(yaw)*movement[1]])
            c,s=np.cos(self.pose[2]),np.sin(self.pose[2]);self.pose[:2]+=np.array([c*local[0]-s*local[1],s*local[0]+c*local[1]])
            self.pose[2]=wrap(self.pose[2]+wrap(odom[2]-self.last_odom[2]))
        self.last_odom=np.array(odom,copy=True)
        # Missing returns carry no map-alignment evidence. Preserve odometry
        # propagation instead of allowing the scorer to invent a correction.
        scan=np.asarray(scan)
        indices=np.linspace(0,len(scan)-1,60).astype(int)
        valid=np.isfinite(scan[indices])&(scan[indices]>=self.lidar.range_min)&(scan[indices]<self.lidar.range_max)
        self.valid_scan_fraction=float(valid.mean());self.scan_update_accepted=bool(valid.any())
        if not self.scan_update_accepted:
            self.map_agreement=0.;self.posterior_entropy=1.
            self.missing_scan_updates+=1
            self.std=np.sqrt(self.std**2+np.array([.015,.015,.01])**2)
            return self.pose.copy()
        self.missing_scan_updates=0
        x,hypotheses=alignment_features(self.map,self.lidar,self.pose,scan,points=self.hypothesis_points,directional=self.feature_dim==15)
        with torch.no_grad():weights=torch.softmax(self.network(torch.from_numpy(x)),dim=0).numpy()
        def estimate(prob,poses):
            return np.r_[np.sum(prob[:,None]*poses[:,:2],axis=0),np.arctan2(prob@np.sin(poses[:,2]),prob@np.cos(poses[:,2]))]
        proposed=estimate(weights,hypotheses)
        agreement=scan_map_agreement(self.map,self.lidar,proposed,scan)
        # Optional experimental broader learned hypothesis attention. No graph
        # search or simulator pose enters this recovery; checkpoint opt-in only.
        if self.recovery_config and agreement<float(self.recovery_config['minimum_agreement']):
            self.recovery_attempts+=1
            rx,rposes=alignment_features(self.map,self.lidar,self.pose,scan,tuple(self.recovery_config['spans']),self.hypothesis_points,self.feature_dim==15)
            with torch.no_grad():rw=torch.softmax(self.network(torch.from_numpy(rx)),dim=0).numpy()
            recovered=estimate(rw,rposes);support=scan_map_agreement(self.map,self.lidar,recovered,scan)
            if support>=float(self.recovery_config['minimum_agreement']) and support>agreement+float(self.recovery_config['minimum_improvement']):
                proposed=recovered;agreement=support;weights=rw;hypotheses=rposes;self.recovery_accepted+=1
        self.pose=proposed;self.map_agreement=agreement
        self.posterior_entropy=float(-np.sum(weights*np.log(np.maximum(weights,1e-12)))/np.log(len(weights)))
        error=hypotheses-self.pose;error[:,2]=wrap(error[:,2])
        self.std=np.maximum(np.sqrt(np.sum(weights[:,None]*error**2,axis=0)),[.015,.015,.01])
        return self.pose.copy()
    def close(self):pass


class NeuralRouter(nn.Module):
    """Trainable convolutional goal propagation, with no runtime graph search.
    Direction kernels and occupancy costs are learned from offline route supervision.
    """
    directions=[(-1,0),(1,0),(0,-1),(0,1)]
    def __init__(self,iterations=96):
        super().__init__();self.iterations=iterations
        self.cost=nn.Conv2d(4,1,1);self.transition_logits=nn.Parameter(torch.full((4,1,3,3),-4.))
        self.reachability=nn.Linear(1,1)
        with torch.no_grad():
            self.cost.weight.zero_();self.cost.weight[0,0]=12.;self.cost.weight[0,1]=-12.;self.cost.bias.fill_(-7.)
            for k,(r,c) in enumerate(self.directions):self.transition_logits[k,0,1+r,1+c]=4.
            self.reachability.weight.fill_(200.);self.reachability.bias.fill_(-1.)
    def forward(self,x):
        discount=torch.sigmoid(self.cost(x));goal=x[:,2:3];value=goal
        kernels=torch.softmax(self.transition_logits.flatten(2),dim=-1).reshape(4,1,3,3)
        for _ in range(self.iterations):
            q=torch.nn.functional.conv2d(value,kernels,padding=1)
            value=torch.maximum(goal,discount*q.max(dim=1,keepdim=True).values)
        q=torch.nn.functional.conv2d(value,kernels,padding=1)
        return value,q,torch.sigmoid(self.reachability(value.permute(0,2,3,1))).permute(0,3,1,2)
