"""Put the first learned-route corner in a checked approach area.

Only reconnects to points on the learned proposal, with bounded candidate
staging offsets. All translations and rotations must fit the same footprint
and map/scan guard. No graph search, command, or simulator state is used.
"""
import numpy as np
from prototypes.narrow_passage import rotation_clear
from neural_navigation.static_footprint import static_footprint_clear


def connect_proposal(core,proposed,prefix=None,index=0,yaw=None):
    path=[proposed[0]] if prefix is None else list(prefix)
    while index<len(proposed)-1:
        chosen=None
        for end in range(len(proposed)-1,index,-1):
            point=proposed[end];delta=point-path[-1];length=np.linalg.norm(delta)
            if length<.05 and end<len(proposed)-1:continue
            if length<1e-8:continue
            heading=float(np.arctan2(delta[1],delta[0]))
            if not core._segment_clear(path[-1],point):continue
            if yaw is not None and not rotation_clear(core.static_map,path[-1],yaw,heading,core.footprint,core.margin):continue
            chosen=(end,point,heading);break
        if chosen is None:break
        index,point,yaw=chosen;path.append(point)
    return np.asarray(path),index


def entrance_connection(core,proposed,baseline,baseline_index):
    core.entrance_staging_point=None
    if not hasattr(core,'scan') or len(proposed)<3:return baseline
    from prototypes.narrow_observation import corridor_features
    feature=corridor_features(core.scan,core.lidar)
    if feature[0]<.5 or not (.40<feature[1]*2<.80):return baseline
    heading=core.estimate[2]+float(feature[3])*.35
    direction=np.array([np.cos(heading),np.sin(heading)])
    pos=core.estimate[:2]
    # A narrow point must come from the learned proposal, not a guessed goal
    # through unseen space. Its centerline determines the staging placement.
    first=None
    for index,point in enumerate(proposed[1:],1):
        along=(point-pos)@direction
        if (.15<along<1.2 and core.static_map.lookup(point)<min(core.footprint)/2+core.margin+.04
                and static_footprint_clear(core.static_map,np.r_[point,heading],core.footprint,core.margin)):
            first=(index,point);break
    if first is None:return baseline
    index,point=first
    # Prefer the already connected complete route if all its future turns fit.
    if baseline_index==len(proposed)-1:
        return baseline
    for reserve in (.35,.45,.55):
        stage=point-direction*reserve;delta=stage-pos
        if not (.05<delta@direction<.8) or np.linalg.norm(delta)<.08:continue
        before=float(np.arctan2(delta[1],delta[0]))
        if not core._segment_clear(pos,stage):continue
        if not rotation_clear(core.static_map,pos,float(core.estimate[2]),before,core.footprint,core.margin):continue
        path,end=connect_proposal(core,proposed,[pos.copy(),stage],index-1,before)
        if end>baseline_index and len(path)>2:
            core.entrance_staging_point=stage.copy()
            return path
    return baseline
