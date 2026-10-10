"""Learned map/laser alignment and an unrolled, trainable neural routing model."""
import numpy as np
import torch
from torch import nn
from navrl.geometry import wrap


class PoseScorer(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(11,48),nn.Tanh(),nn.Linear(48,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def pose_hypotheses(pose):
    dx,dy,da=np.meshgrid(np.linspace(-.24,.24,5),np.linspace(-.24,.24,5),np.linspace(-.16,.16,5),indexing='ij')
    delta=np.c_[dx.ravel(),dy.ravel(),da.ravel()]
    c,s=np.cos(pose[2]),np.sin(pose[2]);world=delta[:,:2]@np.array([[c,s],[-s,c]])
    poses=np.c_[pose[:2]+world,wrap(pose[2]+delta[:,2])]
    return poses,delta


def alignment_features(static_map,lidar,pose,scan):
    hypotheses,delta=pose_hypotheses(pose)
    indices=np.linspace(0,len(scan)-1,60).astype(int)
    valid=np.isfinite(scan[indices])&(scan[indices]>=lidar.range_min)&(scan[indices]<lidar.range_max)
    indices=indices[valid]
    if not len(indices):return np.c_[np.zeros((125,7)),np.ones(125),delta/[.24,.24,.16]].astype(np.float32),hypotheses
    yaw=hypotheses[:,2];mount=lidar.mount
    origin=hypotheses[:,:2]+np.c_[np.cos(yaw)*mount[0]-np.sin(yaw)*mount[1],np.sin(yaw)*mount[0]+np.cos(yaw)*mount[1]]
    angles=yaw[:,None]+mount[3]+lidar.angles[indices]
    ends=origin[:,None,:]+scan[indices][None,:,None]*np.stack([np.cos(angles),np.sin(angles)],axis=-1)
    d=static_map.lookup(ends,static_map.distance,outside=1.)
    features=np.c_[np.minimum(d,.6).mean(axis=1)/.6,np.median(np.minimum(d,.6),axis=1)/.6,
                    (d<.06).mean(axis=1),(d<.12).mean(axis=1),(d<.25).mean(axis=1),
                    np.minimum(d,.2).mean(axis=1)/.2,(d>.5).mean(axis=1),
                    ~static_map.lookup(hypotheses[:,:2],~static_map.blocked,outside=False),delta/[.24,.24,.16]]
    return features.astype(np.float32),hypotheses


class LearnedLocalizer:
    """Odometry propagates state; trained attention over local pose hypotheses corrects it.
    No particle filter, AMCL process, map search planner or simulator truth is used.
    """
    def __init__(self,static_map,lidar,checkpoint):
        self.map=static_map;self.lidar=lidar
        self.network=PoseScorer();self.network.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']);self.network.eval()
        self.pose=None;self.std=np.ones(3);self.last_odom=None
    def reset(self,nominal,rng):
        self.pose=np.array(nominal,float);self.std=np.array([.12,.12,.08]);self.last_odom=None
    def update(self,odom,scan):
        if self.last_odom is not None:
            movement=np.asarray(odom[:2])-self.last_odom[:2];yaw=self.last_odom[2]
            local=np.array([np.cos(yaw)*movement[0]+np.sin(yaw)*movement[1],-np.sin(yaw)*movement[0]+np.cos(yaw)*movement[1]])
            c,s=np.cos(self.pose[2]),np.sin(self.pose[2]);self.pose[:2]+=np.array([c*local[0]-s*local[1],s*local[0]+c*local[1]])
            self.pose[2]=wrap(self.pose[2]+wrap(odom[2]-self.last_odom[2]))
        self.last_odom=np.array(odom,copy=True)
        x,hypotheses=alignment_features(self.map,self.lidar,self.pose,scan)
        with torch.no_grad():weights=torch.softmax(self.network(torch.from_numpy(x)),dim=0).numpy()
        xy=np.sum(weights[:,None]*hypotheses[:,:2],axis=0)
        angle=np.arctan2(np.dot(weights,np.sin(hypotheses[:,2])),np.dot(weights,np.cos(hypotheses[:,2])))
        self.pose=np.r_[xy,angle];error=hypotheses-self.pose;error[:,2]=wrap(error[:,2])
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
