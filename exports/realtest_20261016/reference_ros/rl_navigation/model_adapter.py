"""Map preparation and conservative command adaptation, without ROS."""
import math
from pathlib import Path
import yaml
from PIL import Image


def resolve_map(value):
    path = Path(value).expanduser()
    if len(path.parts)==1 and not path.is_absolute():
        filename=path.name if path.suffix else path.name+'.yaml'
        path=Path('/home/ubuntu/ros2_ws/src/slam/maps')/filename
    path=path.resolve()
    if not path.is_file():
        raise ValueError('Map YAML not found: '+str(path))
    return path


def prepare_map(source, directory):
    source = resolve_map(source)
    config = yaml.safe_load(source.read_text())
    if abs(float(config['origin'][2])) > 1e-8:
        raise ValueError('Model does not support rotated map origin')
    if not math.isclose(float(config['resolution']), .05, abs_tol=1e-8):
        raise ValueError('Current model requires 0.05m map resolution (0.20m routing grid)')
    image = Image.open(source.parent / config['image']).convert('L')
    width, height = image.size
    top, right = (-height)%4, (-width)%4
    if not (top or right):
        return source
    padded = Image.new('L',(width+right,height+top),255 if config.get('negate',0) else 0)
    padded.paste(image,(0,top))  # Keep lower-left origin and every original world cell.
    directory = Path(directory)
    padded.save(directory/'model_map.pgm')
    config['image'] = 'model_map.pgm'
    result = directory/'model_map.yaml'
    result.write_text(yaml.safe_dump(config))
    return result


def limit_command(command, geometry, max_rps):
    vx,wz = map(float,command)
    if not all(math.isfinite(v) for v in (vx,wz,max_rps)) or max_rps<=0:
        raise ValueError('Invalid model command/limit')
    peak = max(map(abs,geometry.to_wheel_rps(vx,0.,wz)))
    scale = min(1.,max_rps/peak) if peak else 1.
    return (vx*scale,wz*scale),scale
