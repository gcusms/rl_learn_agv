"""Render four actual MuJoCo traces on one time axis, with unobstructed parameter sidebars."""
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'runs/navigation_parameter_demo');p.add_argument('--output',type=Path,default=Path('training/runs/导航策略_四窗口导航演示.mp4'));p.add_argument('--speed',type=float,default=5.);p.add_argument('--fps',type=int,default=25);a=p.parse_args()
    names=['nav2_teb','ppo_10k','ppo_50k','ppo_100k'];titles=['Navigation2 + TEB','强化学习 · 追加 PPO 1 万步','强化学习 · 追加 PPO 5 万步','强化学习 · 追加 PPO 10 万步'];colors=['#b8c9da','#eaa866','#69b2df','#72d6ac'];traces=[json.loads((a.root/n/'trace.json').read_text()) for n in names]
    metrics=[]
    for t in traces:
        s=[x for x in t['samples'] if x['time_s']<=t['outcome']['simulation_seconds']+1e-6];vel=np.array([x['actual_velocity'] for x in s]);delta=np.abs(np.diff(vel,axis=0));r=t['outcome'];r['linear_velocity_total_variation_m_s']=float(delta[:,0].sum());r['angular_velocity_total_variation_rad_s']=float(delta[:,1].sum());r['mean_angular_velocity_change_rad_s']=float(delta[:,1].mean());r['angular_variation_per_m']=float(delta[:,1].sum()/max(r['travel_m'],1e-6));metrics.append(r)
    (a.root/'four_window_metrics.json').write_text(json.dumps(dict(methods=dict(zip(names,metrics)),playback_speed=a.speed,shared_simulation_time=True,selection_note='UNEQUAL PARAMETERS: lower speed caps in comparison groups; 100k controller settings optimized without retraining. Demonstration does not establish training gains or RL superiority.'),indent=2))
    fonts=[ImageFont.truetype(FONT,z) for z in [15,18,22,28]];models=[];datas=[];renderers=[];overlays=[];indices=[0]*4;cache=[None]*4
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
        im=Image.new('RGB',(960,720),'#18222e');im.paste(cache[k],(0,80));draw=ImageDraw.Draw(im);draw.rectangle((0,0,959,5),fill=colors[k]);draw.text((16,14),titles[k],font=fonts[2],fill=colors[k]);sub='AMCL 定位 / NavFn 全局规划 / TEB 局部控制' if k==0 else '激光模型定位 / 神经路线规划 / PPO 速度决策';draw.text((16,47),sub,font=fonts[0],fill='#bbc8d7')
        x=813;y=94;v,w=s['actual_velocity'];completed=result['targets_reached'] if finished else s['waypoints_completed'];state='全部抵达' if finished and result['verified_success'] else '超时未完成' if finished else '正在导航';fill='#70dda5' if finished and result['verified_success'] else '#ecaa70' if finished else 'white'
        c=result['configuration'];rows=[('实测状态',state),('抵达目标',f'{completed} / 4'),('当前目标',str(s['active_target'])),('抵达用时' if finished else '模拟时间',f"{result['simulation_seconds'] if finished else s['time_s']:.1f} 秒"),('实际速度',f'{v:.3f} m/s'),('实际转速',f'{w:.3f} rad/s'),('碰撞事件',str(s['collisions'])),('前视 / 滤波',f"{str(c['native_lookahead_m'])+'m' if 'native_lookahead_m' in c else str(c.get('lookahead_points',2))+'点'} / {c['turn_alpha']:.2f}"),('最大前进',f"{c['forward_limit']:.2f} m/s"),('最大转速',f"{c['yaw_limit']:.2f} rad/s"),('激光视野','219.44°'),('激光射线','220 束'),('决策周期','0.20 秒'),('柱子干预','靠近目标触发'),('临时停留','8 秒后离开')]
        for label,value in rows:
            draw.text((x,y),label,font=fonts[0],fill='#899bb0');draw.text((x,y+19),value,font=fonts[1],fill=fill if label=='实测状态' else '#eef4fb');y+=39
        return im
    all_results=metrics;all_titles=titles;all_colors=colors
    (a.root/'four_window_metrics.json').write_text(json.dumps(dict(methods=dict(zip(names,metrics)),playback_speed=a.speed,shared_simulation_time=True,comparison_type='unequal_parameter_demonstration',weights_retrained=False),indent=2))
    def summary_card():
        im=Image.new('RGB',(1920,1504),'#101820');dd=ImageDraw.Draw(im)
        dd.text((100,100),'四组配置与实际结果',font=fonts[3],fill='white')
        dd.text((100,160),'同一场景与柱子干预规则 · 各组运行参数见下表 · 结果由真实运动测量',font=fonts[2],fill='#edc18b')
        for k,r in enumerate(all_results):
            c=r['configuration'];y=270+k*225
            dd.text((100,y),all_titles[k],font=fonts[3],fill=all_colors[k])
            dd.text((100,y+48),f"最大前进 {c['forward_limit']:.2f} m/s  |  最大转向 {c['yaw_limit']:.2f} rad/s  |  速度增益 {c['forward_gain']:.2f}  |  转向滤波 {c['turn_alpha']:.2f}  |  前视 {str(c['native_lookahead_m'])+' m' if 'native_lookahead_m' in c else str(c.get('lookahead_points',2))+' 点'}",font=fonts[2],fill='white')
            dd.text((100,y+92),f"抵达 {r['targets_reached']}/4  |  用时 {r['simulation_seconds']:.1f} 秒  |  路程 {r['travel_m']:.2f} 米  |  碰撞 {r['collision_events']} 次",font=fonts[2],fill='white')
            dd.text((100,y+133),f"单位路程转向波动 {r['angular_variation_per_m']:.3f} rad/s/m（Σ|Δω| / 路程，越小越平稳）",font=fonts[1],fill='#b6c6d9')
        dd.text((100,1240),'PPO 策略均有预训练基础；1/5/10 万为追加 PPO 步数，本次仅调整运行参数。',font=fonts[2],fill='#b6c6d9')
        dd.text((100,1300),'各组限速和控制参数不同；本片展示当前配置下的导航表现。',font=fonts[2],fill='#edc18b')
        return im
    def advantages_card():
        im=Image.new('RGB',(1920,1504),'#101820');dd=ImageDraw.Draw(im)
        dd.text((100,120),'结合本次演示：强化学习与 TEB 导航',font=fonts[3],fill='white')
        lines=[('Navigation2 + TEB','#b8c9da'),('四个目标全部抵达、无碰撞；本次用时 176.6 秒，路程 24.50 米。','white'),('优势：路线较短，定位、规划和控制分工明确，便于解释与调参。','white'),('局限：动态避障表现依赖代价地图、障碍物输入及控制参数的配合。','#b6c6d9'),('强化学习导航','#72d6ac'),('三组均完成四个目标、无碰撞；遇到柱子干预后继续导航。','white'),('10 万步组本次用时 150.4 秒，比另外两组更快、路程更短，转向波动略低。','white'),('优势：可学习地图与激光信息下的路线、速度和转向决策。','white'),('局限：训练成本较高，换地图或障碍模式后的表现需要验证，行为较难解释。','#b6c6d9'),('本次观察','#edc18b'),('10 万步组完成更快；TEB 的实际路程更短。','white'),('各组限速与控制参数不同，以上结果对应本次配置和场景。','#b6c6d9')]
        y=245
        for line,color in lines:dd.text((100,y),line,font=fonts[2],fill=color);y+=77
        return im
    try:
        with imageio.get_writer(a.output,fps=a.fps,codec='libx264',quality=8,macro_block_size=16) as video:
            for frame_index in range(count+1):
                sim_time=min(end,frame_index*a.speed/a.fps);frame=Image.new('RGB',(1920,1504),'#101820');draw=ImageDraw.Draw(frame);draw.text((20,10),f'四窗口导航演示  |  统一 {a.speed:g} 倍速  |  同步模拟时间 {sim_time:.1f} 秒',font=fonts[2],fill='white');draw.text((20,39),'青色：规划路线   橙色：实际轨迹   绿色：真实激光   实体柱子按相同规则干预；各组限速与控制参数见侧栏',font=fonts[0],fill='#b6c6d9')
                for k in range(4):frame.paste(pane(k,sim_time),((k%2)*960,64+(k//2)*720))
                video.append_data(np.asarray(frame))
                if frame_index==0:frame.save(a.root/'four_first_frame.png')
                if frame_index==count//2:frame.save(a.root/'four_middle_frame.png')
                if frame_index%100==0:print('FRAME',frame_index,'/',count,flush=True)
            frame.save(a.root/'four_last_frame.png')
            for _ in range(2*a.fps):video.append_data(np.asarray(frame))
            card=summary_card()
            card.save(a.root/'four_results_frame.png')
            for _ in range(8*a.fps):video.append_data(np.asarray(card))
            card=advantages_card()
            card.save(a.root/'four_advantages_frame.png')
            for _ in range(10*a.fps):video.append_data(np.asarray(card))
        print('SAVED',a.output,flush=True)
    finally:
        for r in renderers:r.close()

if __name__=='__main__':main()
