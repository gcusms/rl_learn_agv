"""Evaluation-only figures: true bodies are labels, never policy inputs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
from navrl.geometry import ROOT
from navrl.map import StaticMap
from neural_navigation.experiment_budget import ExperimentBudget


def main():
    root=Path('reports/localization_mapping_20261015/blind_repair_20261015')
    old=root.parent/'anticipatory_ppo_20261015_132151/paired_12'
    z=np.load(old/'motion_profile_015_constraints_before_ppo.npz')
    hardware=json.loads(Path('models/localization_mapping_20261015/contour_mask_candidate/portable/hardware.json').read_text())
    lidar=hardware['lidar'];angles=np.linspace(lidar['angle_min'],lidar['angle_max'],lidar['beams'])
    m=StaticMap(ROOT/'assets/room_navigation_scripted/decision_map.yaml',radius=.3)
    budget=ExperimentBudget(.1);fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    for axis,stamp in zip(axes.ravel(),(7.4,8.2,8.8,11.4)):
        i=int(np.argmin(abs(z['labels'][:,0]-stamp)));label=z['labels'][i];pose=label[1:4]
        axis.imshow(m.blocked,cmap='Greys',origin='upper',extent=[m.origin[0],m.origin[0]+m.width*m.res,m.origin[1],m.origin[1]+m.height*m.res],alpha=.4)
        c,s=np.cos(pose[2]),np.sin(pose[2]);rotation=np.array([[c,-s],[s,c]])
        half=np.array(hardware['robot_footprint_m'])/2
        corners=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*half
        axis.add_patch(Polygon(corners@rotation.T+pose[:2],fill=False,ec='#1e63ae',lw=2,label='Robot truth (evaluation)'))
        scan=z['scans'][i];valid=np.isfinite(scan)&(scan>=lidar['range_min'])&(scan<lidar['range_max'])
        body=scan[valid,None]*np.c_[np.cos(angles[valid]),np.sin(angles[valid])]+lidar['mount'][:2]
        points=body@rotation.T+pose[:2]
        axis.scatter(*points.T,s=8,c='#18a0ac',label='Actual laser returns')
        center=z['human_positions'][i,0]
        axis.add_patch(Circle(center,.25,fill=False,color='#db5a34',lw=2,label='Obstacle truth (evaluation)'))
        trail=z['human_positions'][:i+1,0]
        trail=trail[np.linalg.norm(trail-pose[:2],axis=1)<3.]
        axis.plot(*trail.T,color='#db5a34',alpha=.6)
        axis.plot(*z['labels'][:i+1,1:3].T,color='#1e63ae',alpha=.6)
        for angle in (lidar['angle_min'],lidar['angle_max']):
            ray=pose[:2]+np.array(lidar['mount'][:2])@rotation.T
            end=ray+1.3*np.array([np.cos(angle+pose[2]),np.sin(angle+pose[2])])
            axis.plot([ray[0],end[0]],[ray[1],end[1]],'--',color='#557999',lw=.7)
        axis.set(xlim=(6.9,8.7),ylim=(1.9,3.6),aspect='equal',xlabel='World x (m)',ylabel='World y (m)',
            title=f"t={label[0]:.1f}s; action={z['choices'][i]}; legal moving={z['masks'][i,1:].sum()}")
        budget.tick()
    axes[0,0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Remaining stop-go collision: measured scans versus evaluation truth')
    fig.savefig(root/'stop_go_diagnosis.png',dpi=150);plt.close(fig);budget.tick()
    print(root/'stop_go_diagnosis.png')


if __name__=='__main__':main()
