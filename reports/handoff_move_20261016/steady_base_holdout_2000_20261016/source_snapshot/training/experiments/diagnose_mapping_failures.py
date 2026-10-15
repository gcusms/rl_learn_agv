"""Post-hoc failure attribution; physical labels never enter the controller."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.ndimage import label
from navrl.map import StaticMap
from neural_navigation.online_mapping_runtime import clear_body_unknown


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.evaluation/'RESULTS.json').read_text())
    if not report.get('completed'):raise ValueError('Evaluate the complete frozen set first')
    rows=[];plots=[]
    for result in report['rows']:
        if result['safe_success']:continue
        folder=a.evaluation/result['case_id'];trace=json.loads((folder/'trace.json').read_text());last=trace[-1]
        diagnostic=last['localization'];reasons=Counter(str(t['stop_reason']) for t in trace)
        observed=StaticMap(folder/'slam/map.yaml');pose=np.asarray(last['estimate'])
        planning=clear_body_unknown(observed,pose)
        fine,_=label(planning.free);coarse,_=label(planning.free[2::4,2::4])
        endpoints=[planning.cells(pose[:2]),planning.cells(last['subgoal'])]
        def component(grid,cell,scale=1):
            r,c=np.asarray(cell)//scale
            return int(grid[r,c]) if 0<=r<grid.shape[0] and 0<=c<grid.shape[1] else 0
        f=[component(fine,x) for x in endpoints];c=[component(coarse,x,4) for x in endpoints]
        true_error=float(np.linalg.norm(np.asarray(last['truth'])[:2]-pose[:2]))
        row=dict(case_id=result['case_id'],failure='contact' if result['collisions'] else 'false_arrival' if result['false_arrival'] else 'timeout',
            waiting_observed_seconds=.2*reasons['waiting_observed_free_space'],
            incomplete_route_seconds=.2*reasons['learned_route_incomplete'],
            final_localization_error_m=true_error,reported_std_xy_m=diagnostic['localization_std'][:2],
            final_physical_goal_distance_m=result['final_xy_error_m'],
            final_estimated_subgoal_distance_m=diagnostic.get('estimated_goal_distance_m'),
            fine_component_ids=f,coarse_component_ids=c,
            fine_connected=bool(f[0] and f[0]==f[1]),coarse_connected=bool(c[0] and c[0]==c[1]),
            geometry_scope='final saved map with current-body unknown clearance; not exact per-frame historical map')
        rows.append(row)
        if result['case_id'] in ('online_map_04','online_map_05','online_map_11'):plots.append((row,planning,pose,last['subgoal']))
    packet=dict(completed=True,scope='offline diagnosis only; no simulator truth supplied to policy',rows=rows)
    (a.evaluation/'FAILURE_DIAGNOSIS.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(plots),2,figsize=(10,3*len(plots)),squeeze=False,constrained_layout=True)
    for pair,(row,m,pose,subgoal) in zip(axes,plots):
        extent=[m.origin[0],m.origin[0]+m.width*m.res,m.origin[1],m.origin[1]+m.height*m.res]
        for axis,grid,title in zip(pair,[m.free,m.free[2::4,2::4]],['Fine planning grid','Actual neural grid samples']):
            axis.imshow(grid,cmap='gray',vmin=0,vmax=1,extent=extent,origin='upper',interpolation='nearest')
            axis.plot(pose[0],pose[1],'o',color='#e39121',label='Estimated robot')
            axis.plot(subgoal[0],subgoal[1],'x',color='#388dde',label='Selected subgoal')
            axis.set_title(row['case_id']+' / '+title);axis.set_xlabel('Map x (m)');axis.set_ylabel('Map y (m)');axis.legend(fontsize=7)
    fig.suptitle('Final observed-map snapshots; white = traversable, black = blocked')
    fig.savefig(a.evaluation/'planning_grid_diagnosis.png',dpi=150);plt.close(fig)
    print(json.dumps(packet,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
