"""Build portable review artifacts from completed evaluations, never estimates."""
import argparse,base64,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from navrl.geometry import ROOT
from navrl.map import StaticMap


def read_completed(path):
    data=json.loads(path.read_text())
    if not data.get('completed'):raise ValueError('Incomplete evaluation: '+str(path))
    return data


def map_packet(m):
    pixels=np.where(m.occupied,35,np.where(~m.blocked,242,150)).astype(np.uint8)
    buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG')
    return dict(image=base64.b64encode(buffer.getvalue()).decode(),origin=m.origin.tolist(),
        resolution=m.res,width=m.width,height=m.height)


def clean(values,decimals=5):
    a=np.asarray(values,float)
    return [[round(float(v),decimals) if np.isfinite(v) else None for v in row] for row in a]


def build_replay(root,output):
    known=root/'navigation_48_validation';post=root/'navigation_48_post_ppo_validation'
    online=root/'online_final_validation_12';reports=[(known,read_completed(known/'RESULTS.json')),
        (post,read_completed(post/'RESULTS.json'))]
    for name in ['preserved_navigation_48','preserved_turn_navigation_48']:
        folder=root/name
        if (folder/'RESULTS.json').exists():reports.append((folder,read_completed(folder/'RESULTS.json')))
    cases=json.loads((known/'cases.json').read_text());maps={};items=[]
    for case in cases:
        scene=case['scene']
        if scene not in maps:maps[scene]=map_packet(StaticMap(ROOT/scene/'decision_map.yaml'))
        variants={}
        for folder,report in reports:
            for row in report['rows']:
                if row['case_id']!=case['id']:continue
                if folder==known and row['variant']!='retained_5k':continue
                with np.load(folder/f"{case['id']}_{row['variant']}.npz") as archive:
                    variants[row['variant']]=dict(labels=clean(archive['labels']),result=row,
                        scans=clean(archive['scans'],3) if 'scans' in archive else [],
                        humans=np.round(archive['human_positions'],5).tolist() if 'human_positions' in archive else [],
                        commands=clean(archive['commands']))
        items.append(dict(id=case['id'],mode='known',category=case['pilot_category'],map=scene,goal=case['goal'],variants=variants))
    extra_cases={}
    for name in ['command_route_challenge_8','command_position_pair_8','unseen_command_maps_8','original_release_regression_12',
        'preserved_command_position_8','preserved_turn_command_8','imu_command_position_pair_8',
        'preserved_turn_imu_position_8','preserved_turn_historical_12','imu_directional_pair_8',
        'imu_unseen_position_pair_8']:
        folder=root/name
        if not (folder/'RESULTS.json').exists():continue
        report=read_completed(folder/'RESULTS.json')
        for case in json.loads((folder/'cases.json').read_text()):
            scene=case['scene']
            if scene not in maps:maps[scene]=map_packet(StaticMap(ROOT/scene/'decision_map.yaml'))
            variants={}
            for row in report['rows']:
                if row['case_id']!=case['id']:continue
                with np.load(folder/f"{case['id']}_{row['variant']}.npz") as archive:
                    variants[row['variant']]=dict(labels=clean(archive['labels']),result=row,
                        scans=clean(archive['scans'],3),humans=np.round(archive['human_positions'],5).tolist(),
                        commands=clean(archive['commands']))
            input_label={'command_integrated':'纯指令积分退化对照',
                'command_linear_imu_yaw':'指令平移＋IMU 转向近似'}.get(
                    case.get('input_odometry'),'运动反馈／历史回归')
            identity=json.dumps(case,sort_keys=True)
            if identity not in extra_cases:
                extra_cases[identity]=dict(id=case['id'],mode='known',category=input_label+' · '+case['pilot_category'],
                    map=scene,goal=case['goal'],variants={})
            extra_cases[identity]['variants'].update(variants)
    items.extend(extra_cases.values())
    report=read_completed(online/'RESULTS.json')
    for row in report['rows']:
        folder=online/row['case_id'];traces=json.loads((folder/'trace.json').read_text())
        key='online/'+row['case_id'];maps[key]=map_packet(StaticMap(folder/'slam/map.yaml'))
        labels=[[t['seconds'],*t['truth'],*t['estimate'],t['xy_error'],t['yaw_error']] for t in traces]
        variants=dict(online_ppo=dict(labels=clean(labels),result=row,scans=clean([t['scan'] for t in traces],3),
            humans=[t['humans'] for t in traces],commands=[t['command'] for t in traces],
            subgoals=[t['subgoal'] for t in traces],stops=[t['stop_reason'] for t in traces],
            map_versions=[t['map_version'] for t in traces]))
        items.append(dict(id=row['case_id'],mode='online',category=row['task_kind'],map=key,
            goal=traces[0]['goal'],variants=variants))
    for report_name,input_label in [('online_command_challenge_4','纯指令积分退化对照'),
        ('imu_online_4','指令平移＋IMU 转向近似'),
        ('preserved_online_command_4','组合策略＋纯指令积分'),
        ('imu_known_free_dynamics_4','IMU＋已知空闲动态层试验')]:
        command_online=root/report_name
        if not (command_online/'RESULTS.json').exists():continue
        report=read_completed(command_online/'RESULTS.json')
        for row in report['rows']:
            folder=command_online/row['case_id'];traces=json.loads((folder/'trace.json').read_text())
            key=report_name+'/'+row['case_id'];maps[key]=map_packet(StaticMap(folder/'slam/map.yaml'))
            labels=[[t['seconds'],*t['truth'],*t['estimate'],t['xy_error'],t['yaw_error']] for t in traces]
            variants=dict(online_ppo=dict(labels=clean(labels),result=row,scans=clean([t['scan'] for t in traces],3),
                humans=[t['humans'] for t in traces],commands=[t['command'] for t in traces],
                subgoals=[t['subgoal'] for t in traces],stops=[t['stop_reason'] for t in traces],
                map_versions=[t['map_version'] for t in traces]))
            items.append(dict(id=report_name+'_'+row['case_id'],mode='online',category=input_label+' · '+row['task_kind'],
                map=key,goal=traces[0]['goal'],variants=variants))
    hardware=json.loads((Path(__file__).resolve().parents[2]/'models/localization_mapping_20261015/boundary_assisted_terminal/portable_8000/hardware.json').read_text())
    packet=dict(maps=maps,cases=items,lidar=hardware['lidar'])
    template=Path(__file__).with_name('localization_review.html').read_text()
    output.write_text(template.replace('__DATA__',json.dumps(packet,separators=(',',':'),allow_nan=False).replace('</','<\\/')))


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    paths=[p for p in sorted((root/'long_localization_probe').glob('*.json')) if p.name not in ('RESULTS.json','PROVENANCE.json')]
    for axis,path in zip(axes.ravel(),paths):
        d=json.loads(path.read_text());samples=d['samples'];times=[t['seconds'] for t in samples]
        for i,name,color in [(0,'Legacy','#a56a17'),(1,'Anchored v4','#6980aa'),(2,'Anchored v5','#128b70')]:
            error=[np.linalg.norm(np.array(t['estimates'][i])[:2]-np.array(t['truth'])[:2]) for t in samples]
            axis.plot(times,error,label=name,color=color,linewidth=1.5)
        axis.set_title(d['result']['scene']);axis.set_xlabel('Physical time (s)');axis.set_ylabel('Position error (m)')
        axis.set_yscale('log');axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle('12 m out-and-back, biased odometry; truth used for scoring only')
    fig.savefig(root/'long_localization_errors.png',dpi=170);fig.savefig(root/'long_localization_errors.pdf');plt.close(fig)


def plot_online_long(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=root/'long_online_slam_2';trial=root/'long_online_slam_distance_prior_2'
    report=read_completed(original/'RESULTS.json');read_completed(trial/'RESULTS.json')
    fig,axes=plt.subplots(2,len(report['rows']),figsize=(11,8),squeeze=False,constrained_layout=True)
    for column,row in enumerate(report['rows']):
        name=row['scene'];before=json.loads((original/name/'trace.json').read_text())
        after=json.loads((trial/name/'trace.json').read_text());truth=np.array([t['truth'] for t in before])
        np.testing.assert_allclose(truth,np.array([t['truth'] for t in after]),atol=1e-10)
        times=np.array([t['seconds'] for t in before]);top,bottom=axes[:,column]
        top.plot(truth[:,0],truth[:,1],color='#222',label='Physical reference')
        for source,key,label,color in [(before,'odometry','Command + gyro input','#c78420'),
            (before,'slam','SLAM default','#286dd1'),(after,'slam','Distance-prior trial (rejected)','#c14356')]:
            values=np.array([t[key] if t[key] is not None else [np.nan]*3 for t in source])
            top.plot(values[:,0],values[:,1],label=label,color=color,linestyle='--')
            error=np.linalg.norm(values[:,:2]-truth[:,:2],axis=1)
            bottom.plot(times,np.maximum(error,1e-5),label=label,color=color)
        top.set_title(name);top.set_xlabel('World x relative to start (m)');top.set_ylabel('World y (m)')
        top.axis('equal');top.grid(alpha=.2);top.legend(fontsize=8)
        bottom.set_xlabel('Physical time (s)');bottom.set_ylabel('Position error (m)')
        bottom.set_yscale('log');bottom.grid(alpha=.2);bottom.legend(fontsize=8)
    fig.suptitle('Initially unknown map: manual 12m sensor probes, not autonomous policy success')
    fig.savefig(root/'long_online_slam_comparison.png',dpi=170)
    fig.savefig(root/'long_online_slam_comparison.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plots-only',action='store_true');a=p.parse_args()
    plot(a.root)
    if (a.root/'long_online_slam_distance_prior_2/RESULTS.json').exists():plot_online_long(a.root)
    if not a.plots_only:build_replay(a.root,a.root/'index.html')


if __name__=='__main__':main()
