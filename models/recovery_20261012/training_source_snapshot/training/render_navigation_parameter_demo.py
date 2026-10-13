"""Render three actual MuJoCo traces plus a parameter dashboard on one time axis, with unobstructed parameter sidebars."""
import argparse,json,math
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from navrl.geometry import ROOT,LidarProfile
from navrl.visualization import NavigationOverlay

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'runs/navigation_parameter_demo');p.add_argument('--output',type=Path,default=Path('training/runs/导航策略_三窗口参数演示.mp4'));p.add_argument('--speed',type=float,default=5.);p.add_argument('--fps',type=int,default=25);a=p.parse_args()
    names=['ppo_10k','ppo_50k','ppo_100k'];titles=['追加 PPO 1 万步 · 保守参数','追加 PPO 5 万步 · 中等参数','追加 PPO 10 万步 · 优化参数'];colors=['#eaa866','#69b2df','#72d6ac'];traces=[json.loads((a.root/n/'trace.json').read_text()) for n in names]
    metrics=[]
    for t in traces:
        s=[x for x in t['samples'] if x['time_s']<=t['outcome']['simulation_seconds']+1e-6];vel=np.array([x['actual_velocity'] for x in s]);delta=np.abs(np.diff(vel,axis=0));r=t['outcome'];r['linear_velocity_total_variation_m_s']=float(delta[:,0].sum());r['angular_velocity_total_variation_rad_s']=float(delta[:,1].sum());r['mean_angular_velocity_change_rad_s']=float(delta[:,1].mean());r['angular_variation_per_m']=float(delta[:,1].sum()/max(r['travel_m'],1e-6));metrics.append(r)
    (a.root/'comparison_metrics.json').write_text(json.dumps(dict(methods=dict(zip(names,metrics)),playback_speed=a.speed,shared_simulation_time=True,selection_note='UNEQUAL PARAMETERS: lower speed caps in comparison groups; 100k controller settings optimized without retraining. Demonstration does not establish training gains or RL superiority.'),indent=2))
    fonts=[ImageFont.truetype(FONT,z) for z in [15,18,22,28]];models=[];datas=[];renderers=[];overlays=[];indices=[0]*3;cache=[None]*3
    for t in traces:
        m=mujoco.MjModel.from_xml_path(str(Path(t['scene_directory'])/'physical_scene.xml'));m.vis.global_.offwidth=800;m.vis.global_.offheight=640;models.append(m);datas.append(mujoco.MjData(m));renderers.append(mujoco.Renderer(m,height=640,width=800));overlays.append(NavigationOverlay(m,LidarProfile(),t['samples'][0]['planned_path'],t['waypoints'][0]))
    end=max(t['samples'][-1]['time_s'] for t in traces);count=math.ceil(end/a.speed*a.fps);a.output.parent.mkdir(parents=True,exist_ok=True)
    def pane(k,sim_time):
        t=traces[k];ss=t['samples'];i=min(max(0,int(np.searchsorted([v['time_s'] for v in ss],sim_time,side='right'))-1),len(ss)-1);s=ss[i];result=t['outcome'];finished=sim_time>=result['simulation_seconds']
        if cache[k] is None or indices[k]!=i:
            d=datas[k];d.qpos[:]=s['qpos'];d.qvel[:]=s['qvel'];mujoco.mj_forward(models[k],d)
            r=renderers[k];r.update_scene(d,camera='overview');o=overlays[k];o.path=np.asarray(s['planned_path']);o.goal=np.asarray(s['active_goal']);o.draw(r.scene,d,s['truth'],np.array([np.inf if x is None else x for x in s['scan']]))
            for j,point in enumerate(t['waypoints']):
                if r.scene.ngeom>=r.scene.maxgeom:break
                g=r.scene.geoms[r.scene.ngeom];r.scene.ngeom+=1;rgba=[.1,.85,.3,1] if j<s['waypoints_completed'] else [.75,.3,.9,1];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.11,0,0]),np.array([*point,.10]),np.eye(3).reshape(-1),np.array(rgba,np.float32));g.label=str(j+1)
            for j,body in enumerate(['manual_column_1','manual_column_2']):
                g=r.scene.geoms[r.scene.ngeom];r.scene.ngeom+=1;bid=mujoco.mj_name2id(models[k],mujoco.mjtObj.mjOBJ_BODY,body);mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.035,0,0]),d.xpos[bid]+[0,0,1.77],np.eye(3).reshape(-1),np.array([1,1,.1,1],np.float32));g.label=f'C{j+1}'
            cache[k]=Image.fromarray(r.render());indices[k]=i
        im=Image.new('RGB',(960,720),'#18222e');im.paste(cache[k],(0,80));draw=ImageDraw.Draw(im);draw.rectangle((0,0,959,5),fill=colors[k]);draw.text((16,14),titles[k],font=fonts[2],fill=colors[k]);sub='不同参数配置演示 / 激光模型定位 / 神经路线规划';draw.text((16,47),sub,font=fonts[0],fill='#bbc8d7')
        x=813;y=94;v,w=s['actual_velocity'];completed=result['targets_reached'] if finished else s['waypoints_completed'];state='全部抵达' if finished and result['verified_success'] else '超时未完成' if finished else '正在导航';fill='#70dda5' if finished and result['verified_success'] else '#ecaa70' if finished else 'white'
        c=result['configuration'];rows=[('实测状态',state),('抵达目标',f'{completed} / 4'),('当前目标',str(s['active_target'])),('抵达用时' if finished else '模拟时间',f"{result['simulation_seconds'] if finished else s['time_s']:.1f} 秒"),('实际速度',f'{v:.3f} m/s'),('实际转速',f'{w:.3f} rad/s'),('碰撞事件',str(s['collisions'])),('前视 / 滤波',f"{c.get('lookahead_points',2)} 点 / {c['turn_alpha']:.2f}"),('最大前进',f"{c['forward_limit']:.2f} m/s"),('最大转速',f"{c['yaw_limit']:.2f} rad/s"),('激光视野','219.44°'),('激光射线','220 束'),('决策周期','0.20 秒'),('柱子干预','靠近目标触发'),('临时停留','8 秒后离开')]
        for label,value in rows:
            draw.text((x,y),label,font=fonts[0],fill='#899bb0');draw.text((x,y+19),value,font=fonts[1],fill=fill if label=='实测状态' else '#eef4fb');y+=39
        return im
    baseline=json.loads((a.root/'nav2_teb/trace.json').read_text());bs=baseline['samples'];bv=np.array([q['actual_velocity'] for q in bs if q['time_s']<=baseline['outcome']['simulation_seconds']+1e-6]);br=baseline['outcome'];br['mean_angular_velocity_change_rad_s']=float(np.abs(np.diff(bv[:,1])).mean());br['angular_variation_per_m']=float(np.abs(np.diff(bv[:,1])).sum()/max(br['travel_m'],1e-6))
    all_results=metrics+[br];all_titles=titles+['普通导航 · 保守限速（仅参数汇总）'];all_colors=colors+['#a4b0c2']
    (a.root/'comparison_metrics.json').write_text(json.dumps(dict(methods=dict(zip(names+['nav2_teb'],all_results)),playback_speed=a.speed,shared_simulation_time=True,comparison_type='unequal_parameter_demonstration',weights_retrained=False),indent=2))
    def dashboard(sim_time):
        im=Image.new('RGB',(960,720),'#101820');dd=ImageDraw.Draw(im)
        dd.text((30,25),'参数配置与实测进度',font=fonts[3],fill='white')
        dd.text((30,70),'本片为调参演示；对照组有意设置较低限速。',font=fonts[2],fill='#edc18b')
        for k,r in enumerate(metrics):
            c=r['configuration'];y=132+k*110;done=sim_time>=r['simulation_seconds']
            dd.text((30,y),titles[k],font=fonts[2],fill=colors[k])
            dd.text((30,y+34),f"限速 {c['forward_limit']:.2f} m/s  |  转向 {c['yaw_limit']:.2f} rad/s  |  前视 {c.get('lookahead_points',2)} 点 / 滤波 {c['turn_alpha']:.2f}",font=fonts[1],fill='white')
            dd.text((30,y+64),f"{'已抵达 4/4' if done else '正在导航'}  |  {'抵达用时' if done else '当前时间'} {r['simulation_seconds'] if done else sim_time:.1f} 秒",font=fonts[1],fill='#b6c6d9')
        dd.text((30,480),'10 万步组：前视距离、速度响应与转向滤波优化',font=fonts[2],fill=colors[2])
        dd.text((30,522),'本次 PPO 三组：10 万步组用时最少、路程最短。',font=fonts[1],fill='white')
        dd.text((30,556),'其单位路程转向波动也略低；实际指标在片尾。',font=fonts[1],fill='#b6c6d9')
        dd.text((30,588),'普通导航不播放，片尾列出其调参后实测结果。',font=fonts[1],fill='#b6c6d9')
        dd.text((30,638),'检查点权重未重训；差异不能证明步数越多越强。',font=fonts[1],fill='#edc18b')
        return im
    def summary_card():
        im=Image.new('RGB',(1920,1504),'#101820');dd=ImageDraw.Draw(im)
        dd.text((100,100),'四组配置与实际结果',font=fonts[3],fill='white')
        dd.text((100,160),'不同参数配置演示 · 普通导航也采用保守限速 · 结果由真实运动测量',font=fonts[2],fill='#edc18b')
        for k,r in enumerate(all_results):
            c=r['configuration'];y=270+k*225
            dd.text((100,y),all_titles[k],font=fonts[3],fill=all_colors[k])
            dd.text((100,y+48),f"最大前进 {c['forward_limit']:.2f} m/s  |  最大转向 {c['yaw_limit']:.2f} rad/s  |  速度增益 {c['forward_gain']:.2f}  |  转向滤波 {c['turn_alpha']:.2f}  |  前视 {str(c['native_lookahead_m'])+' m' if 'native_lookahead_m' in c else str(c.get('lookahead_points',2))+' 点'}",font=fonts[2],fill='white')
            dd.text((100,y+92),f"抵达 {r['targets_reached']}/4  |  用时 {r['simulation_seconds']:.1f} 秒  |  路程 {r['travel_m']:.2f} 米  |  碰撞 {r['collision_events']} 次",font=fonts[2],fill='white')
            dd.text((100,y+133),f"单位路程转向波动 {r['angular_variation_per_m']:.3f} rad/s/m（Σ|Δω| / 路程，越小越平稳）",font=fonts[1],fill='#b6c6d9')
        dd.text((100,1240),'PPO 策略均有预训练基础；1/5/10 万为追加 PPO 步数，本次仅调整运行参数。',font=fonts[2],fill='#b6c6d9')
        dd.text((100,1300),'本片不能证明训练步数带来的提升，也不能证明所有场景下的最优或成功率。',font=fonts[2],fill='#edc18b')
        return im
    def advantages_card():
        im=Image.new('RGB',(1920,1504),'#101820');dd=ImageDraw.Draw(im)
        dd.text((100,120),'强化学习与普通导航：能力与本次证据',font=fonts[3],fill='white')
        lines=[('强化学习：本片显示的能力','#72d6ac'),('已知地图 + 部分视野激光；神经定位、神经路线和 PPO 速度决策。','white'),('遇到实体柱子后，能够根据传感器数据调整路线或速度，并连续到达目标。','white'),('机器人运行时不使用 AMCL / A*；仍有显式地图融合和运动边界。','#b6c6d9'),('普通导航：方案优势','#a4b0c2'),('定位、全局规划、局部控制分工清楚；行为较容易解释和逐项调参。','white'),('本项目使用实际 AMCL / NavFn / TEB，可通过配置改变速度与安全余量。','white'),('本次演示：如何解读差异','#edc18b'),('10 万步组使用更高限速与优化参数，对照组有意采用较低限速。','white'),('更快或更短只属于本次配置和场景，不能推断强化学习普遍优于普通导航。','white'),('原统一参数实测中，普通导航为 122.8 秒，三个 PPO 为 200.4 / 204.8 / 208.6 秒。','#b6c6d9'),('泛化能力、稳健性和成功率，需要统一参数、多地图、多任务的独立评估。','#b6c6d9')]
        y=245
        for line,color in lines:dd.text((100,y),line,font=fonts[2],fill=color);y+=77
        return im
    try:
        with imageio.get_writer(a.output,fps=a.fps,codec='libx264',quality=8,macro_block_size=16) as video:
            for frame_index in range(count+1):
                sim_time=min(end,frame_index*a.speed/a.fps);frame=Image.new('RGB',(1920,1504),'#101820');draw=ImageDraw.Draw(frame);draw.text((20,10),f'三阶段导航 · 不同参数配置演示  |  统一 {a.speed:g} 倍速  |  同步模拟时间 {sim_time:.1f} 秒',font=fonts[2],fill='white');draw.text((20,39),'青色：规划路线   橙色：实际轨迹   绿色：真实激光   实体柱子按相同规则干预；配置差异已标注，不能归因于训练步数',font=fonts[0],fill='#b6c6d9')
                for k in range(3):frame.paste(pane(k,sim_time),((k%2)*960,64+(k//2)*720))
                frame.paste(dashboard(sim_time),(960,784))
                video.append_data(np.asarray(frame))
                if frame_index==0:frame.save(a.root/'first_frame.png')
                if frame_index==count//2:frame.save(a.root/'middle_frame.png')
                if frame_index%100==0:print('FRAME',frame_index,'/',count,flush=True)
            frame.save(a.root/'last_frame.png')
            for _ in range(2*a.fps):video.append_data(np.asarray(frame))
            card=summary_card()
            card.save(a.root/'results_frame.png')
            for _ in range(8*a.fps):video.append_data(np.asarray(card))
            card=advantages_card()
            card.save(a.root/'advantages_frame.png')
            for _ in range(10*a.fps):video.append_data(np.asarray(card))
        print('SAVED',a.output,flush=True)
    finally:
        for r in renderers:r.close()

if __name__=='__main__':main()
