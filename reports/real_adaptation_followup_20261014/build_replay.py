"""Standalone offline physical/estimated trajectory comparison, no ROS output."""
from pathlib import Path
import base64
import io
import json
import numpy as np
from PIL import Image
from navrl.geometry import ROOT as ASSETS
from navrl.map import StaticMap

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EVAL=ROOT/'training/runs/goal_pose_paired_evaluation_20261014'


def main():
    report=json.loads((EVAL/'RESULTS.json').read_text());assert report['completed']
    cases=json.loads((EVAL/'cases.json').read_text());data=[]
    for case in cases:
        variants={}
        for r in report['rows']:
            if r['case_id']!=case['id']:continue
            with np.load(EVAL/f"{r['case_id']}_{r['variant']}.npz") as a:
                variants[r['variant']]=dict(labels=np.round(a['labels'],5).tolist(),result=r)
        data.append(dict(case=case,variants=variants))
    m=StaticMap(ASSETS/'assets/room_navigation_scripted/decision_map.yaml',radius=.3)
    # Display original three-valued map, with no additional visible border wall.
    image=np.where(m.occupied,35,np.where(~m.blocked,240,140)).astype(np.uint8)
    buffer=io.BytesIO();Image.fromarray(image).save(buffer,format='PNG')
    packet=dict(cases=data,map=dict(image=base64.b64encode(buffer.getvalue()).decode(),
        origin=m.origin.tolist(),resolution=m.res,width=m.width,height=m.height))
    html=r'''<!doctype html><meta charset="utf-8"><title>目标朝向 PPO 配对回放</title>
<style>body{font:16px sans-serif;margin:24px;background:#edf1f5;color:#182333}main{max-width:1100px;margin:auto}canvas{max-width:100%;background:#fff;border:1px solid #aaa}button,select{padding:8px;margin:5px}input[type=range]{width:70%}pre{white-space:pre-wrap;background:white;padding:12px}small{display:block;line-height:1.6}.note{background:#fff0d2;padding:14px}h1{font-size:24px}</style>
<main><h1>目标朝向 PPO 配对回放 · 开发候选未通过验收</h1>
<p class="note">这是离线 MuJoCo 数据回放。实线为物理真值轨迹，虚线车框为估计位姿；真值没有进入模型输入。此页面仅显示静态地图及车体，未保存动态障碍物动画。不能把本页当成完整动态场景录像。</p>
<select id="case"></select><button id="play">播放</button><button id="reset">回到起点</button>
<p><span style="color:#1965c0">■ 原部署（位置任务）</span>　<span style="color:#bb6818">■ 303 维训练前</span>　<span style="color:#128257">■ 3k 候选</span></p>
<canvas id="canvas" width="850" height="750"></canvas><p><input id="time" type="range" min="0" value="0" step="1"><span id="stamp"></span></p>
<pre id="result"></pre><small>方向任务要求：位置 &lt;0.12 m、方向 &lt;8.6°、低速并稳定 0.4 s。旧位置任务沿用配对验收规则。12 个开发任务不是最终验收；没有证明最短路径或实车定位精度。</small></main>
<script>const data=__DATA__;const colors={legacy:'#1965c0',initial:'#bb6818',trained:'#128257'};
const select=document.getElementById('case'),slider=document.getElementById('time'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const image=new Image();image.src='data:image/png;base64,'+data.map.image;let playing=false,timer=null;
data.cases.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=x.case.id+' · '+x.case.pilot_category;select.appendChild(o)});
const px=800/(data.map.width*data.map.resolution),ox=25,oy=25;
function point(x,y){return [ox+(x-data.map.origin[0])*px,oy+(data.map.origin[1]+data.map.height*data.map.resolution-y)*px]}
function body(p,color,dashed=false){const [x,y]=point(p[0],p[1]);ctx.save();ctx.translate(x,y);ctx.rotate(-p[2]);ctx.strokeStyle=color;ctx.lineWidth=dashed?1:2;ctx.setLineDash(dashed?[4,3]:[]);ctx.strokeRect(-.394*px/2,-.329*px/2,.394*px,.329*px);ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(.25*px,0);ctx.stroke();ctx.restore()}
function stop(){playing=false;clearInterval(timer);document.getElementById('play').textContent='播放'}
function draw(){const item=data.cases[select.value||0],index=Number(slider.value);ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,ox,oy,data.map.width*data.map.resolution*px,data.map.height*data.map.resolution*px);
const goal=item.case.goal,[gx,gy]=point(goal[0],goal[1]);ctx.strokeStyle='#cc1427';ctx.lineWidth=2;ctx.beginPath();ctx.arc(gx,gy,.12*px,0,2*Math.PI);ctx.stroke();if(goal.length===3){ctx.beginPath();ctx.moveTo(gx,gy);ctx.lineTo(gx+.4*px*Math.cos(goal[2]),gy-.4*px*Math.sin(goal[2]));ctx.stroke()}
let text='';for(const [name,variant] of Object.entries(item.variants)){const samples=variant.labels,limit=Math.min(index,samples.length-1),current=samples[limit],r=variant.result;ctx.strokeStyle=colors[name];ctx.lineWidth=2;ctx.beginPath();samples.slice(0,limit+1).forEach((s,i)=>{const p=point(s[1],s[2]);i?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.stroke();body(current.slice(1,4),colors[name]);body(current.slice(4,7),colors[name],true);
text+=name+': '+(r.safe_success?'物理成功':r.collisions?'碰撞':r.false_arrival?'假到达':r.timeout?'超时':'未通过')+'；末端位置 '+r.final_xy_error_m.toFixed(3)+' m'+(r.final_yaw_error_rad===null?'':'，方向 '+(r.final_yaw_error_rad*180/Math.PI).toFixed(1)+'°')+'；路径 '+r.travel_m.toFixed(2)+' m；有可行动作时主动等待 '+r.wait_with_motion_frames+' 帧\n'}
document.getElementById('result').textContent=text;document.getElementById('stamp').textContent=(index*.2).toFixed(1)+' s'}
function change(){stop();const item=data.cases[select.value||0];slider.max=Math.max(...Object.values(item.variants).map(v=>v.labels.length))-1;slider.value=0;draw()}
select.onchange=change;slider.oninput=()=>{stop();draw()};document.getElementById('reset').onclick=change;document.getElementById('play').onclick=()=>{if(playing){stop();return}playing=true;document.getElementById('play').textContent='暂停';timer=setInterval(()=>{if(Number(slider.value)>=Number(slider.max)){stop();return}slider.value=Number(slider.value)+1;draw()},200)};image.onload=change;
</script>'''
    # JSON is supplied by fixed local numerical traces; escape script delimiters.
    (OUT/'index.html').write_text(html.replace('__DATA__',json.dumps(packet,separators=(',',':')).replace('</','<\\/')))


if __name__=='__main__':main()
