"""Generate matching ROS occupancy map and simple MuJoCo decision scene."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from navrl.visualization import attach_urdf_visuals

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
W, H, RES = 10., 8., .05
# Bounds in metres, shared by occupancy image and MuJoCo geometry.
WALLS = [
    (0., 0., 10., .2), (0., 7.8, 10., 8.),
    (0., .2, .2, 7.8), (9.8, .2, 10., 7.8),
    (3.9, .2, 4.1, 2.8), (3.9, 4.2, 4.1, 7.8),
    (.2, 3.9, 2.8, 4.1), (5.2, 3.9, 9.8, 4.1),
    (7., 5.7, 7.8, 6.5), (1.5, 6., 2.1, 6.6),
]


def main():
    ASSETS.mkdir(exist_ok=True)
    nx, ny = round(W/RES), round(H/RES)
    pixels = bytearray([254]) * (nx * ny)
    for row in range(ny):
        y = H - (row + .5) * RES
        for col in range(nx):
            x = (col + .5) * RES
            if any(x0 <= x < x1 and y0 <= y < y1 for x0,y0,x1,y1 in WALLS):
                pixels[row*nx + col] = 0
    (ASSETS/'decision_map.pgm').write_bytes(f'P5\n{nx} {ny}\n255\n'.encode() + pixels)
    (ASSETS/'decision_map.yaml').write_text(
        'image: decision_map.pgm\nmode: trinary\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n'
        'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    model = ET.Element('mujoco', model='agv_navigation_decision_demo')
    ET.SubElement(model, 'compiler', angle='radian')
    ET.SubElement(model, 'option', timestep='.01', gravity='0 0 -9.81')
    visual = ET.SubElement(model, 'visual')
    ET.SubElement(visual, 'global', offwidth='800', offheight='640')
    world = ET.SubElement(model, 'worldbody')
    ET.SubElement(world, 'light', pos='5 4 10', dir='0 0 -1', directional='true')
    ET.SubElement(world, 'camera', name='overview', pos='5 4 13', xyaxes='1 0 0 0 1 0', mode='fixed')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='10 8 .1', rgba='.92 .92 .92 1')
    for i,(x0,y0,x1,y1) in enumerate(WALLS):
        ET.SubElement(world, 'geom', name=f'wall_{i}', type='box',
            pos=f'{(x0+x1)/2} {(y0+y1)/2} .5', size=f'{(x1-x0)/2} {(y1-y0)/2} .5', rgba='.35 .4 .45 1')
    robot = ET.SubElement(world, 'body', name='robot', mocap='true', pos='1 1 .156592537')
    ET.SubElement(robot, 'geom', name='robot_footprint', type='box', size='.196962452 .164730011 .156592537', rgba='0 0 0 0', group='3')
    ET.SubElement(robot, 'site', name='lidar', pos='.118389 0 -.039910', size='.025', rgba='1 .8 .1 1')
    visual_count=attach_urdf_visuals(model,robot)
    obstacles = [dict(name='crossing_obstacle', center=[4.,3.5], axis='x', amplitude=1.5, period_s=12., radius_m=.25),
                 dict(name='corridor_obstacle', center=[4.8,4.], axis='y', amplitude=2., period_s=16., radius_m=.25)]
    for item in obstacles:
        body = ET.SubElement(world, 'body', name=item['name'], mocap='true',
                             pos=f"{item['center'][0]} {item['center'][1]} .5")
        ET.SubElement(body, 'geom', name=item['name']+'_geom', type='cylinder',
                      size=f"{item['radius_m']} .5", rgba='.9 .25 .15 1')
    ET.indent(model)
    ET.ElementTree(model).write(ASSETS/'decision_scene.xml', encoding='unicode')
    (ASSETS/'scene.json').write_text(json.dumps(dict(size_m=[W,H], resolution_m=RES, origin=[0,0,0],
        walls=WALLS, robot_model='ideal planar differential drive; no calibrated wheel dynamics',
        robot_collision_box_m=[.393924904,.329460022,.313185074],
        conservative_radius_m=.27, action_limits=dict(vx_m_s=[0.,.2], wz_rad_s=[-.5,.5]),
        tasks=[dict(start=[1,1,0],goal=[8.5,6.8]),dict(start=[1,7,0],goal=[8.5,1]),
               dict(start=[8.5,1,3.14],goal=[1,1])], dynamic_obstacles=obstacles), indent=2)+'\n')
    from generate_physical_scene import main as build_physical_scene
    build_physical_scene()
    print('Imported URDF visual meshes:',visual_count)
    print('Generated 10 x 8 m static map and matching MuJoCo scene in', ASSETS)


if __name__ == '__main__':
    main()
