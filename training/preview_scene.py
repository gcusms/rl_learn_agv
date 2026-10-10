"""Validate scene/map connectivity and demonstrate moving obstacles (no trained policy)."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time
import numpy as np
import mujoco
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/'assets/scene.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT/'assets/decision_scene.xml'))
    data = mujoco.MjData(model)
    def move(t):
        for item in config['dynamic_obstacles']:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item['name'])
            mid = model.body_mocapid[bid]
            point = np.array([*item['center'], .5])
            point[0 if item['axis']=='x' else 1] += item['amplitude'] * math.sin(2*math.pi*t/item['period_s'])
            data.mocap_pos[mid] = point
        mujoco.mj_step(model, data)
    for i in range(1000):
        move(i*.01)
    assert np.isfinite(data.mocap_pos).all()
    res = config['resolution_m']
    with Image.open(ROOT/'assets/decision_map.pgm') as image:
        occupied = np.asarray(image) < 128
    # Inflate occupancy for a conservative disc footprint using Euclidean offsets.
    ny,nx = occupied.shape
    radius = math.ceil(config['conservative_radius_m']/res)
    inflated = occupied.copy()
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if math.hypot(dx,dy)*res > config['conservative_radius_m'] + res*math.sqrt(2)/2:
                continue
            ya,yb=max(0,dy),min(ny,ny+dy)
            xa,xb=max(0,dx),min(nx,nx+dx)
            inflated[ya:yb,xa:xb] |= occupied[ya-dy:yb-dy,xa-dx:xb-dx]
    def cell(point):
        return ny-1-int(point[1]/res), int(point[0]/res)
    source = cell(config['tasks'][0]['start'])
    queue=deque([source]); reached={source}
    while queue:
        row,col=queue.popleft()
        for rr,cc in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]:
            if 0<=rr<ny and 0<=cc<nx and not inflated[rr,cc] and (rr,cc) not in reached:
                reached.add((rr,cc)); queue.append((rr,cc))
    for task in config['tasks']:
        assert cell(task['start']) in reached and cell(task['goal']) in reached, task
    fig,ax=plt.subplots(figsize=(10,8))
    ax.imshow(occupied, cmap='gray_r', origin='upper', extent=[0,10,0,8], vmin=0, vmax=1)
    for i,task in enumerate(config['tasks']):
        ax.plot(*task['start'][:2], 'o', color='royalblue')
        ax.plot(*task['goal'], '*', color='green', markersize=13)
        ax.annotate(f'S{i+1}',task['start'][:2],xytext=(6,5),textcoords='offset points')
        ax.annotate(f'G{i+1}',task['goal'],xytext=(6,5),textcoords='offset points')
    for item in config['dynamic_obstacles']:
        p=np.array(item['center']); delta=np.array([item['amplitude'],0] if item['axis']=='x' else [0,item['amplitude']])
        ax.plot([*(p-delta)[:1],*(p+delta)[:1]],[(p-delta)[1],(p+delta)[1]],'--',color='orangered')
        ax.add_patch(plt.Circle(p,item['radius_m'],color='orangered'))
    ax.set(xlabel='x (m)',ylabel='y (m)',title='Custom AGV map: static walls and example dynamic trajectories',aspect='equal')
    fig.tight_layout(); fig.savefig(ROOT/'assets/map_preview.png',dpi=140); plt.close(fig)
    with mujoco.Renderer(model,height=640,width=800) as renderer:
        renderer.update_scene(data,camera='overview')
        Image.fromarray(renderer.render()).save(ROOT/'assets/mujoco_preview.png')
    print(json.dumps({'scene_steps':1000,'connected_tasks_after_footprint_inflation':len(config['tasks']),
                      'robot_model':config['robot_model'],'preview':'assets/map_preview.png'},indent=2))
    if args.viewer:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(model,data) as viewer:
            started=time.monotonic()
            while viewer.is_running():
                tick=time.monotonic()
                move(tick-started); viewer.sync()
                time.sleep(max(0,model.opt.timestep-(time.monotonic()-tick)))


if __name__ == '__main__':
    main()
