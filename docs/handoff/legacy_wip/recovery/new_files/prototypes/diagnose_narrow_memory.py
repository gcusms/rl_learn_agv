"""Sensor-memory ablation of read-only guard probes; no altered command executes."""
import copy
import numpy as np
import prototypes.diagnose_narrow_wait as replay

original_probes=replay.probes

def point_summary(points,margins,half):
    if not len(points):return dict(count=0)
    gap=np.linalg.norm(points-np.clip(points,-half,half),axis=1)
    order=np.argsort(gap-margins)[:3]
    return dict(count=len(points),minimum_body_gap_m=float(gap.min()),
        below_margin=int(np.count_nonzero(gap<margins)),
        worst=[dict(point=points[i].tolist(),gap=float(gap[i]),margin=float(margins[i])) for i in order])

def audited_probes(core,guard,commands,force_route_complete=False):
    results=original_probes(core,guard,commands,force_route_complete)
    if not force_route_complete and len(commands)==20:
        valid=np.isfinite(core.scan)&(core.scan>=core.lidar.range_min)&(core.scan<core.lidar.range_max)
        angle=core.lidar.angles[valid]+core.lidar.mount[3]
        fresh=core.scan[valid,None]*np.c_[np.cos(angle),np.sin(angle)]+core.lidar.mount[:2]
        memory=copy.deepcopy(guard.memory);remembered,uncertainty=memory.update(core)
        together=np.vstack([fresh,remembered]);margins=np.r_[np.full(len(fresh),.04),.04+uncertainty]
        parts=dict(fresh=(fresh,np.full(len(fresh),.04)),remembered=(remembered,.04+uncertainty),
            combined=(together,margins),combined_without_age_inflation=(together,np.full(len(together),.04)))
        audit={name:point_summary(points,margin,guard.half) for name,(points,margin) in parts.items()}
        for name,(points,margin) in parts.items():
            audit[name]['trajectory_reasons']=[dict(command=command,reason=guard._trajectory_check(core,np.array(command),points,margin,0.,allow_escape=True))
                for command in [[0.,0.],[.05,0.],[0.,.1],[0.,-.1]]]
        results[0]['point_sources']=audit
    return results

if __name__=='__main__':
    replay.probes=audited_probes
    replay.CASES=replay.CASES[2:4]
    replay.main()
