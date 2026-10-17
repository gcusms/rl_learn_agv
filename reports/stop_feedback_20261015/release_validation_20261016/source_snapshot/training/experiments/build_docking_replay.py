"""Portable offline replay with the correct map for each physical task."""
import argparse,base64,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from navrl.geometry import ROOT
from navrl.map import StaticMap


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.evaluation/'RESULTS.json').read_text());assert report['completed']
    cases=json.loads((a.evaluation/'cases.json').read_text());maps={};items=[]
    for case in cases:
        scene=case['scene']
        if scene not in maps:
            m=StaticMap(ROOT/scene/'decision_map.yaml',radius=.3)
            pixels=np.where(m.occupied,35,np.where(~m.blocked,240,140)).astype(np.uint8)
            buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG')
            maps[scene]=dict(image=base64.b64encode(buffer.getvalue()).decode(),origin=m.origin.tolist(),
                resolution=m.res,width=m.width,height=m.height)
        variants={}
        for row in report['rows']:
            if row['case_id']!=case['id']:continue
            with np.load(a.evaluation/f"{row['case_id']}_{row['variant']}.npz") as archive:
                variants[row['variant']]=dict(labels=np.round(archive['labels'],5).tolist(),result=row)
        items.append(dict(case=case,variants=variants))
    packet=dict(maps=maps,cases=items)
    html=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>终点方向停靠与定位回放</title>
<style>body{margin:24px;background:#eef2f6;color:#182538;font:16px system-ui}main{max-width:1080px;margin:auto}h1{font-size:24px}canvas{background:white;max-width:100%;border:1px solid #bdc7d2}button,select{padding:9px;margin:4px}pre{white-space:pre-wrap;line-height:1.7;background:white;padding:15px}.note{background:#fff0cf;padding:14px;line-height:1.7}input{width:75%}</style>
<main><h1>终点方向停靠与定位 · 离线配对回放</h1><p class="note">实线车框表示 MuJoCo 物理真值，虚线表示定位估计；真值只用于评估。这里显示静态地图与机器人，没有动态障碍物动画。结果属于固定开发任务，不能视为实车部署验收或最短路径证明。</p>
<select id="case"></select><button id="play">播放</button><button id="reset">回到起点</button>
<div id="visibility"></div><p id="legend"></p><canvas width="850" height="720" id="canvas"></canvas><p><input type="range" id="time" min="0" step="1" value="0"><span id="stamp"></span></p><pre id="result"></pre><p>方向任务验收：位置 &lt;12 cm，方向 &lt;8.6°，实际线速度 &lt;0.03 m/s、角速度 &lt;0.08 rad/s，并连续稳定 0.4 s。旧位置任务沿用 20 cm 判定。</p></main>
<script>const data=__DATA__;const colors={previous_3k:'#a46818',bc_only:'#7256ab',trained:'#158469',trained_v4_localization:'#176ccb',trained_5k:'#a51685',trained_12500_v4:'#1840a0'};const names={previous_3k:'上轮 3k',bc_only:'示范预热',trained:'10k PPO',trained_v4_localization:'10k PPO＋定位 v4',trained_5k:'5k 停靠 PPO',trained_12500_v4:'组合适配 12.5k'};
const select=document.getElementById('case'),slider=document.getElementById('time'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');let activeMap,image,scale,ox=25,oy=25,timer=null;const images={};let visible=new Set();
const categories={alignment:'方向对齐',short_approach:'接近后停靠',new_alignment:'新增方向对齐',new_approach:'新增接近停靠',position_regression:'旧导航回归'};data.cases.forEach((item,i)=>{const option=document.createElement('option');option.value=i;option.textContent=item.case.id+' · '+(categories[item.case.pilot_category]||item.case.pilot_category);select.appendChild(option)});
function point(x,y){return [ox+(x-activeMap.origin[0])*scale,oy+(activeMap.origin[1]+activeMap.height*activeMap.resolution-y)*scale]}
function body(p,color,dashed){const [x,y]=point(p[0],p[1]);ctx.save();ctx.translate(x,y);ctx.rotate(-p[2]);ctx.strokeStyle=color;ctx.lineWidth=dashed?1:2;ctx.setLineDash(dashed?[4,3]:[]);ctx.strokeRect(-.394*scale/2,-.329*scale/2,.394*scale,.329*scale);ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(.23*scale,0);ctx.stroke();ctx.restore()}
function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放'}
function draw(){if(!image||!image.complete)return;const item=data.cases[select.value||0],index=Number(slider.value);ctx.clearRect(0,0,850,720);ctx.drawImage(image,ox,oy,activeMap.width*activeMap.resolution*scale,activeMap.height*activeMap.resolution*scale);const g=item.case.goal,[gx,gy]=point(g[0],g[1]);ctx.strokeStyle='#ce2747';ctx.lineWidth=2;ctx.beginPath();ctx.arc(gx,gy,(g.length===3?.12:.2)*scale,0,2*Math.PI);ctx.stroke();if(g.length===3){ctx.beginPath();ctx.moveTo(gx,gy);ctx.lineTo(gx+.4*scale*Math.cos(g[2]),gy-.4*scale*Math.sin(g[2]));ctx.stroke()}
let text='',legend='';for(const [name,v] of Object.entries(item.variants)){if(!visible.has(name))continue;const s=v.labels,k=Math.min(index,s.length-1),current=s[k],r=v.result,color=colors[name]||'#444';legend+='<span style="color:'+color+'">■ '+(names[name]||name)+'</span>　';ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();s.slice(0,k+1).forEach((row,i)=>{const p=point(row[1],row[2]);i?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.stroke();body(current.slice(1,4),color,false);body(current.slice(4,7),color,true);text+=(names[name]||name)+': '+(r.safe_success?'物理成功':r.collisions?'碰撞':r.false_arrival?'假到达':r.timeout?'超时':'未通过')+'；'+r.seconds.toFixed(1)+' s；末端位置 '+(r.final_xy_error_m*100).toFixed(1)+' cm'+(r.final_yaw_error_rad==null?'':'，方向 '+(r.final_yaw_error_rad*180/Math.PI).toFixed(1)+'°')+'；行驶 '+r.travel_m.toFixed(2)+' m\n'}document.getElementById('legend').innerHTML=legend;document.getElementById('result').textContent=text;document.getElementById('stamp').textContent=(index*.2).toFixed(1)+' s'}
function change(){stop();const item=data.cases[select.value||0];visible=new Set();const controls=document.getElementById('visibility');controls.innerHTML='';for(const name of Object.keys(item.variants)){const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.style.width='auto';check.checked=!(Object.keys(item.variants).includes('trained_5k'))||['previous_3k','trained_5k'].includes(name);if(check.checked)visible.add(name);check.onchange=()=>{check.checked?visible.add(name):visible.delete(name);draw()};label.append(check,document.createTextNode(names[name]||name));label.style.marginRight='14px';controls.appendChild(label)}activeMap=data.maps[item.case.scene];scale=Math.min(800/(activeMap.width*activeMap.resolution),660/(activeMap.height*activeMap.resolution));slider.max=Math.max(...Object.values(item.variants).map(v=>v.labels.length))-1;slider.value=0;if(!images[item.case.scene]){const im=new Image();im.src='data:image/png;base64,'+activeMap.image;images[item.case.scene]=im;im.onload=draw}image=images[item.case.scene];draw()}
select.onchange=change;slider.oninput=()=>{stop();draw()};document.getElementById('reset').onclick=change;document.getElementById('play').onclick=()=>{if(timer){stop();return}document.getElementById('play').textContent='暂停';timer=setInterval(()=>{if(Number(slider.value)>=Number(slider.max)){stop();return}slider.value=Number(slider.value)+1;draw()},200)};const first=data.cases.findIndex(item=>item.case.id==='goal_pose_04');if(first>=0)select.value=first;change();</script></html>'''
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(html.replace('__DATA__',json.dumps(packet,separators=(',',':')).replace('</','<\\/')))


if __name__=='__main__':main()
