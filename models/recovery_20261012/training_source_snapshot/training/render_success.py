"""Render an actual successful evaluation trace in MuJoCo as MP4 and interactive replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from navrl.geometry import ROOT, LidarProfile
from navrl.visualization import NavigationOverlay


def main():
    p=argparse.ArgumentParser(); p.add_argument('trace',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'runs/success_demo')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--playback-speed',type=float,default=1.,help='Video speed multiplier; replay timestamps stay original')
    p.add_argument('--render-scale',type=float,default=1.)
    p.add_argument('--max-time',type=float,help='Render a prefix of the trace; interactive replay remains complete')
    p.add_argument('--allow-failure',action='store_true')
    p.add_argument('--html-only',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    trace=json.loads(args.trace.read_text())
    if not trace['outcome']['verified_success'] and not args.allow_failure: raise ValueError('Refusing to label an unsuccessful trace as a successful replay')
    model=mujoco.MjModel.from_xml_path(str(Path(trace.get('scene_directory',ROOT/'assets'))/trace.get('scene_file','decision_scene.xml'))); data=mujoco.MjData(model)
    model.vis.global_.offwidth=960; model.vis.global_.offheight=720
    lidar=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'lidar')
    model.site_pos[lidar]=[.118389,0,-.039910]
    recorded_samples=trace['samples'] if args.max_time is None else [s for s in trace['samples'] if s['time_s']<=args.max_time]
    if not recorded_samples:raise ValueError('No samples within selected time window')
    clipped=len(recorded_samples)<len(trace['samples'])
    samples=recorded_samples[::max(1,args.stride)]
    if samples[-1] is not recorded_samples[-1]:samples.append(recorded_samples[-1])
    mission=bool(trace.get('waypoints'))
    video_name='navigation.mp4' if mission else 'successful_navigation.mp4'
    if not samples[0].get('scan'): raise ValueError('Re-record evaluation with --record-success to include real laser scans')
    overlay=NavigationOverlay(model,LidarProfile(),trace['path'],trace['goal'])
    detail_option=mujoco.MjvOption(); detail_option.geomgroup[0]=0
    detail_camera=mujoco.MjvCamera(); detail_camera.distance=1.05; detail_camera.azimuth=135; detail_camera.elevation=-25
    def place(sample):
        if model.nq and sample.get('qpos') is not None:
            data.qpos[:]=sample['qpos'];data.qvel[:]=sample['qvel']
        else:
            data.mocap_pos[:]=sample['mocap_pos']; data.mocap_quat[:]=sample['mocap_quat']
        mujoco.mj_forward(model,data)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    columns=trace.get('scene_config',{}).get('dynamic_obstacle_kind')=='columns'
    obstacle_label='Moving solid columns' if columns else 'Solid pedestrians'
    start=np.array(samples[0]['truth'][:2]);distance=float(np.linalg.norm(np.asarray(trace['goal'])-start))
    # Non-overlapping panels: map below header, robot detail and status in a sidebar.
    if args.playback_speed<=0:raise ValueError('Playback speed must be positive')
    if not .25<=args.render_scale<=1.:raise ValueError('Render scale must be between .25 and 1')
    scale=args.render_scale
    if not args.html_only:
        with imageio.get_writer(args.output/video_name,fps=5/max(1,args.stride)*args.playback_speed,codec='libx264',quality=8,macro_block_size=16) as video:
            with mujoco.Renderer(model,height=round(720*scale),width=round(960*scale)) as renderer, mujoco.Renderer(model,height=round(240*scale),width=round(320*scale)) as detail:
                for i,sample in enumerate(samples):
                    place(sample);renderer.update_scene(data,camera='overview')
                    scan=np.array([np.inf if x is None else x for x in sample['scan']])
                    if sample.get('planned_path'):overlay.path=np.asarray(sample['planned_path'])
                    overlay.goal=np.asarray(sample.get('active_goal',trace['goal']))
                    overlay.draw(renderer.scene,data,sample['truth'],scan)
                    for k,point in enumerate(trace.get('waypoints',[])):
                        if renderer.scene.ngeom>=renderer.scene.maxgeom:break
                        g=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
                        color=[.1,.7,.25,1] if k<sample.get('waypoints_completed',sample.get('waypoint_index',0)) else [.6,.25,.85,1]
                        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CYLINDER,np.array([.12,.03,0]),np.array([*point,.05]),np.eye(3).reshape(-1),np.array(color,dtype=np.float32));g.label=f'{k+1}';g.category=mujoco.mjtCatBit.mjCAT_DECOR
                    frame=Image.new('RGB',(1280,864),(20,25,30))
                    frame.paste(Image.fromarray(renderer.render()).resize((960,720),Image.Resampling.LANCZOS),(0,128))
                    detail_camera.lookat[:]=[*sample['truth'][:2],.23]
                    detail.update_scene(data,camera=detail_camera,scene_option=detail_option)
                    frame.paste(Image.fromarray(detail.render()).resize((320,240),Image.Resampling.LANCZOS),(960,160))
                    draw=ImageDraw.Draw(frame)
                    draw.text((12,8),f"PPO + {trace['localizer']} | t={sample['time_s']:.1f}s | contacts={sample['collisions']}",font=font,fill='white')
                    goal=sample.get('active_goal',trace['goal'])
                    target_label=f"Target {sample.get('waypoint_index',0)+1}/{len(trace['waypoints'])}" if mission else 'Goal'
                    draw.text((12,31),f"Start ({start[0]:.2f}, {start[1]:.2f}) -> {target_label} ({goal[0]:.2f}, {goal[1]:.2f}) | SCLIDAR FOV 219.44 deg",font=font,fill='white')
                    draw.text((12,54),'Cyan: neural route | Orange: actual trajectory | Green rays: laser | Red points: returns',font=font,fill='white')
                    cv=sample.get('commanded_velocity',[0,0]);av=sample.get('actual_velocity',[0,0])
                    draw.text((12,77),f"Policy v={cv[0]:.3f} m/s w={cv[1]:.3f} rad/s | Actual v={av[0]:.3f} m/s | {sample.get('motion_source','')}",font=font,fill='white')
                    d=sample.get('decision') or {}
                    draw.text((12,100),f"{d.get('state','')} | route updates {d.get('plan_revision',0)} | robot navigation supervisor override: {sample.get('supervisor_override',False)}",font=font,fill='white')
                    draw.text((976,137),'URDF ROBOT DETAIL',font=font,fill='white')
                    lines=([f"Target {sample.get('waypoint_index',0)+1}/{len(trace['waypoints'])}",f"Completed: {sample.get('waypoints_completed',0)}"] if mission else [])+[obstacle_label, 'Native MuJoCo solid contacts',f"Laser pose error: {sample['error_m']*100:.1f} cm",f"Robot: {sample['truth'][0]:.2f}, {sample['truth'][1]:.2f}"]
                    states=sample.get('pedestrian_states',[]);vels=sample.get('pedestrian_velocities',[])
                    for j,state in enumerate(states):
                        speed=float(np.linalg.norm(vels[j])) if j<len(vels) else 0.
                        lines.extend([f"{'C' if columns else 'P'}{j+1}: {state}",f"    speed {speed:.2f} m/s"])
                    for event in sample.get('column_block_events',[]):
                        lines.extend([f"Block C{event['column_index']+1}: {event['phase']}",f"    hold {event['hold_seconds']:.0f}s"])
                    for j,line in enumerate(lines):draw.text((976,420+j*23),line,font=font,fill='white')
                    if i==len(samples)-1:
                        success=trace['outcome']['verified_success']
                        draw.rectangle([970,805,1272,843],fill=(20,125,60) if success else (160,70,30));draw.text((982,816),'CLIP END - SEE FULL REPLAY' if clipped else 'ALL TARGETS REACHED' if mission and success else 'FINAL GOAL REACHED' if success else 'MISSION INCOMPLETE',font=font,fill='white')
                        frame.save(args.output/('clip_end.png' if clipped else 'goal_reached.png' if success else 'mission_incomplete.png'))
                    if i==0:frame.save(args.output/'first_frame.png')
                    if i==len(samples)//2:frame.save(args.output/'middle_frame.png')
                    video.append_data(np.asarray(frame))
                    if i==len(samples)-1:
                        for _ in range(15):video.append_data(np.asarray(frame))
    if not args.html_only:
        (args.output/'video_metadata.json').write_text(json.dumps(dict(is_clip=clipped,simulation_last_s=samples[-1]['time_s'],playback_speed=args.playback_speed,stride=args.stride),indent=2))
    scene=trace.get('scene_config') or json.loads((ROOT/'assets/scene.json').read_text())
    html="""<!doctype html><meta charset="utf-8"><title>MuJoCo navigation replay</title>
<style>body{font:16px sans-serif;background:#eef2f6;margin:20px;color:#202b39}main{max-width:1280px;margin:auto}canvas{background:white;border-radius:10px;width:100%;max-width:900px}button,input{margin:8px}video{width:100%;max-width:1280px}</style>
<main><h2>成功导航回放：训练后的 PPO 策略</h2><p id="report"></p><button id="play">播放 / 暂停</button><input id="time" type="range" min="0" style="width:65%"><p id="state"></p><canvas id="map" width="1000" height="800"></canvas><p>蓝色：机器人；橙色：未知动态障碍；绿色：最终目标；青色虚线：全局规划路径；绿色射线 / 红色返回点：实时激光扫描。W：门旁等待位置；B：临时堵路位置（仅在定点堵路任务显示）。这里回放的是实际评估记录。</p><video controls src="successful_navigation.mp4"></video></main>
<script>
const trace=TRACE_DATA, scene=SCENE_DATA, cv=document.getElementById('map'),ctx=cv.getContext('2d'),slider=document.getElementById('time');
slider.max=trace.samples.length-1;let playing=false;const obstacleLabel=scene.dynamic_obstacle_kind==='columns'?'移动柱子':'行人';
const decisionNames={following_path:'沿规划路线行驶',replanning:'规划器重新选路',waiting_for_clearance:'等待障碍让开',localization_uncertain:'定位不确定',goal_reached:'已到达目标',static_unreachable:'已知地图上不可达',timeout:'导航超时',model_following_route:'模型自主选路行驶',model_predicts_blockage:'模型预测当前可能堵路',model_goal_reached:'模型定位已到达目标'};
const reasonNames={known_map_route:'已知地图上有可行路线',laser_obstacle_on_route:'激光发现路线上的未知障碍',lack_of_progress:'近期没有前进',temporary_laser_costmap_blocks_route:'激光观测区域暂时无可行路线',position_standard_deviation_exceeds_0_35_m:'定位位置标准差过大',localized_goal_tolerance:'定位已达到目标容差',no_collision_free_route_in_known_map:'已知地图中找不到通路',episode_time_budget_exhausted:'超过本回合时间限制',learned_map_scan_pose_and_neural_route_values:'学习定位和神经路线输出'};
const columnStateNames={waiting_beside_door:'在门旁等待',moving_into_door:'正在走入堵路点',temporary_block:'临时堵路',returning_beside_door:'正在退回门旁',parked_beside_door:'已退回原位',waiting_for_partner:'等待另一根柱子到位',yielding:'靠近小车，停下让行',blocked_on_script:'移动路径暂时受阻',parking:'正在停稳'};
const motionNames={policy_reverse:'模型低速倒退',policy_stop:'模型主动低速/停车',policy_drive:'模型请求行驶',collision_blocked:'接触导致运动阻挡',supervisor_wait:'监督器要求等待'};
const dims=scene.size_m??[10,8],originMap=scene.origin??[0,0,0],scale=1000/dims[0];cv.height=dims[1]*scale;
const xy=p=>[(p[0]-originMap[0])*scale,cv.height-(p[1]-originMap[1])*scale];
function line(points,color,dashed=false){ctx.beginPath();ctx.strokeStyle=color;ctx.setLineDash(dashed?[7,6]:[]);points.forEach((p,i)=>{let [x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.setLineDash([])}
function draw(){let n=+slider.value,s=trace.samples[n];ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle='#465569';scene.walls.forEach(w=>{let [x,y]=xy([w[0],w[3]]);ctx.fillRect(x,y,(w[2]-w[0])*scale,(w[3]-w[1])*scale)});
const blocks=scene.tasks?.[trace.task??0]?.column_task?.blocks??[];
blocks.filter(b=>b.mode==='approach_cross_return').forEach((b,i)=>{for(const [p,label,color] of [[b.position,'W'+(i+1),'#3d73a8'],[b.block_position,'B'+(i+1),'#c95d76']]){const [x,y]=xy(p);ctx.strokeStyle=color;ctx.beginPath();ctx.arc(x,y,10,0,Math.PI*2);ctx.stroke();ctx.fillStyle=color;ctx.font='13px sans-serif';ctx.fillText(label,x+12,y+4)}});
line(s.planned_path??trace.path,'#13b7d7',true);
let mount=trace.lidar.mount,yaw=s.truth[2],origin=[s.truth[0]+Math.cos(yaw)*mount[0]-Math.sin(yaw)*mount[1],s.truth[1]+Math.sin(yaw)*mount[0]+Math.cos(yaw)*mount[1]];
s.scan.forEach((d,i)=>{if(d===null)return;let a=trace.lidar.angle_min+i*(trace.lidar.angle_max-trace.lidar.angle_min)/(s.scan.length-1)+yaw+mount[3],p=[origin[0]+Math.cos(a)*d,origin[1]+Math.sin(a)*d];line([origin,p],'#50b57755');let [x,y]=xy(p);ctx.fillStyle='#ef3333';ctx.fillRect(x-1.5,y-1.5,3,3)});
line(trace.samples.slice(0,n+1).map(p=>p.truth),'#3087cf');
(trace.waypoints??[]).forEach((p,k)=>{let [x,y]=xy(p);ctx.fillStyle=k<(s.waypoints_completed??s.waypoint_index)?'#269650':'#9250bb';ctx.beginPath();ctx.arc(x,y,18,0,Math.PI*2);ctx.fill();ctx.fillStyle='white';ctx.font='bold 16px sans-serif';ctx.fillText(k+1,x-5,y+6)});
ctx.fillStyle='#269650';let [gx,gy]=xy(s.active_goal??trace.goal);ctx.beginPath();ctx.arc(gx,gy,15,0,Math.PI*2);if(trace.waypoints){ctx.strokeStyle='#269650';ctx.lineWidth=4;ctx.stroke();ctx.lineWidth=1}else ctx.fill();
const humans=s.pedestrian_positions??s.mocap_pos.slice(1);for(let i=0;i<humans.length;i++){let [x,y]=xy(humans[i]);ctx.fillStyle=i===0?'#e87835':'#3e83c7';ctx.beginPath();ctx.arc(x,y,.25*scale,0,Math.PI*2);ctx.fill();ctx.fillStyle='white';ctx.font='14px sans-serif';ctx.fillText((scene.dynamic_obstacle_kind==='columns'?'C':'P')+(i+1),x-9,y+5)}
let [x,y]=xy(s.truth);ctx.save();ctx.translate(x,y);ctx.rotate(-s.truth[2]);ctx.fillStyle='#2385d0';ctx.fillRect(-.197*scale,-.165*scale,.394*scale,.33*scale);ctx.strokeStyle='white';ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(28,0);ctx.stroke();ctx.restore();
line(trace.samples.slice(0,n+1).map(p=>p.estimate),'#e5a726',true);document.getElementById('state').textContent=`${trace.waypoints?'当前导航点 '+(s.waypoint_index+1)+' / '+trace.waypoints.length+' | ':''}模拟时间 ${s.time_s.toFixed(1)} 秒 | 接触次数 ${s.collisions} | 激光定位误差 ${(s.error_m*100).toFixed(1)} 厘米 | 策略前进速度 ${(s.commanded_velocity?.[0]??0).toFixed(3)} m/s | 实际速度 ${(s.actual_velocity?.[0]??0).toFixed(3)} m/s | ${motionNames[s.motion_source]??s.motion_source??''} | 决策：${decisionNames[s.decision?.state]??''} (${reasonNames[s.decision?.reason]??s.decision?.reason??''}) | 路线更新次数：${s.decision?.plan_revision??0}${s.decision?.current_reachability_probability!==undefined?' | 当前通行预测 '+s.decision.current_reachability_probability.toFixed(2)+'（非抵达成功率）':''} | 控制覆盖：${s.supervisor_override?'监督器等待':'模型动作'} | ${obstacleLabel}：${(s.pedestrian_states??[]).map(x=>columnStateNames[x]??x).join(', ')}`}
slider.oninput=draw;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){if(+slider.value<+slider.max)slider.value=+slider.value+1;else playing=false;draw()}},200);
document.getElementById('report').textContent=`${trace.waypoints?'四房间连续任务：'+trace.outcome.targets_reached+'/'+trace.outcome.targets_total+' 个导航点抵达；切换目标时不重置位置或定位。':''}定位后端：${trace.localizer}；实际终点距离：${trace.outcome.true_goal_distance_m.toFixed(3)} 米；${trace.outcome.verified_success?'本次任务成功':'本次任务未完成全部目标'}。整体成功率请查看评估报告，不由这个单次演示推断。`;draw();
</script>"""
    if mission:
        html=html.replace('成功导航回放：训练后的 PPO 策略','四房间连续导航回放：训练后的 PPO 策略').replace('successful_navigation.mp4',video_name).replace('绿色：最终目标','编号紫色/绿色：待访问/已访问导航点；绿色：当前目标')
    if args.html_only and not (args.output/video_name).exists():
        html=html.replace('<video controls src="'+video_name+'"></video>','<p>本任务提供完整交互轨迹回放；MuJoCo 视频见总览页的任务 1。</p>')
    metadata_path=args.output/'video_metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()).get('is_clip'):
        html=html.replace('<video controls', '<p>以下录像为堵路过程和第一导航点片段；完整导航轨迹和超时结果见上方交互回放。</p><video controls')
    html=html.replace('TRACE_DATA',json.dumps(trace)).replace('SCENE_DATA',json.dumps(scene))
    (args.output/'replay.html').write_text(html)
    (args.output/'result.json').write_text(json.dumps(trace['outcome'],indent=2))
    print('Saved interactive replay' if args.html_only else 'Saved MP4, interactive replay and final goal frame',args.output)
    if args.viewer:
        import time
        from mujoco import viewer
        with viewer.launch_passive(model,data) as display:
            display.cam.type=2; display.cam.fixedcamid=0
            for sample in samples:
                if not display.is_running(): break
                place(sample); display.user_scn.ngeom=0
                overlay.draw(display.user_scn,data,sample['truth'],np.array([np.inf if x is None else x for x in sample['scan']]))
                display.sync(); time.sleep(.2)


if __name__=='__main__': main()
