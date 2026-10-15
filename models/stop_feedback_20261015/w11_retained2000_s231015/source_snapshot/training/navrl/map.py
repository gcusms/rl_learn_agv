import heapq
from math import floor,isfinite
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import yaml


def segment_is_free(a,b,origin_m,resolution,free):
    height,width=free.shape
    # Supercover grid traversal checks every cell touched by the segment,
    # including both neighbors at corner crossings (no sampled-ray gaps).
    # Scalar traversal avoids allocating tiny NumPy arrays at every cell.
    # Keep the same corner tolerance, boundary tests and visited-cell semantics.
    ox=float((a[0]-origin_m[0])/resolution);oy=float((a[1]-origin_m[1])/resolution)
    ex=float((b[0]-origin_m[0])/resolution);ey=float((b[1]-origin_m[1])/resolution)
    if not all(isfinite(v) for v in (ox,oy,ex,ey)):return False
    col,y=floor(ox),floor(oy);target_col,target_y=floor(ex),floor(ey)
    dx,dy=ex-ox,ey-oy
    sx=(1 if dx>0 else -1 if dx<0 else 0);sy=(1 if dy>0 else -1 if dy<0 else 0)
    def clear(col,y):
        row=height-1-y
        return 0<=row<height and 0<=col<width and bool(free[row,col])
    if not clear(col,y) or not clear(target_col,target_y): return False
    tx=(col+(1 if sx>0 else 0)-ox)/dx if sx else float('inf')
    ty=(y+(1 if sy>0 else 0)-oy)/dy if sy else float('inf')
    stride_x=1/abs(dx) if sx else float('inf');stride_y=1/abs(dy) if sy else float('inf')
    for _ in range(width+height+4):
        if col==target_col and y==target_y:return True
        if min(tx,ty)>1.+1e-10: return True
        if abs(tx-ty)<1e-12:
            if not clear(col+sx,y) or not clear(col,y+sy): return False
            col+=sx;y+=sy;tx+=stride_x;ty+=stride_y
        elif tx<ty:
            col+=sx;tx+=stride_x
        else:
            y+=sy;ty+=stride_y
        if not clear(col,y): return False
    return False


class StaticMap:
    @classmethod
    def from_occupancy_grid(cls,data,width,height,resolution,origin,radius=.30):
        """ROS-style bottom-up occupancy, with unknown padding and fixed origin.

        No physical geometry is created here. Padding never becomes laser-visible
        walls; it only prevents the learned router from entering unknown space.
        """
        if width<=0 or height<=0 or not np.isfinite(resolution) or resolution<=0 or len(data)!=width*height:
            raise ValueError('Invalid occupancy grid dimensions')
        origin=np.asarray(origin,float)
        if origin.shape!=(3,) or not np.isfinite(origin).all() or abs(origin[2])>1e-8:
            raise ValueError('Finite axis-aligned map origin required')
        values=np.asarray(data).reshape(height,width)
        if not np.isfinite(values).all() or np.any((values<-1)|(values>100)):
            raise ValueError('Occupancy values must be -1 or 0..100')
        gray=np.flipud(values).copy()
        gray=np.pad(gray,(((-height)%4,0),(0,(-width)%4)),constant_values=-1)
        obj=cls.__new__(cls);obj.path=None;obj.res=float(resolution);obj.origin=origin.copy()
        obj.occupied=gray>65;obj.unknown=(gray<0)|((gray>=25)&~obj.occupied)
        obj.blocked=obj.occupied|obj.unknown
        obj.distance=distance_transform_edt(~obj.occupied)*obj.res if obj.occupied.any() else np.full(obj.occupied.shape,1e3)
        obj.clearance=distance_transform_edt(~obj.blocked)*obj.res
        obj.free=(obj.clearance>radius+obj.res*.71)&~obj.blocked
        obj.height,obj.width=obj.free.shape
        return obj

    def __init__(self, yaml_path, radius=.27):
        self.path=Path(yaml_path).resolve()
        c=yaml.safe_load(self.path.read_text())
        self.res=float(c['resolution']); self.origin=np.array(c['origin'],float)
        if abs(self.origin[2])>1e-8: raise ValueError('Rotated map origins are not supported yet')
        gray=np.asarray(Image.open(self.path.parent/c['image']).convert('L'),dtype=float)/255
        probability=gray if c.get('negate',0) else 1-gray
        self.occupied=probability>c['occupied_thresh']
        self.unknown=(probability>=c['free_thresh']) & ~self.occupied
        self.blocked=self.occupied|self.unknown
        self.distance=distance_transform_edt(~self.occupied)*self.res if self.occupied.any() else np.full(self.occupied.shape,1e3)
        self.clearance=distance_transform_edt(~self.blocked)*self.res
        self.free=(self.clearance>radius+self.res*.71)&~self.blocked
        self.height,self.width=self.free.shape

    def cells(self, xy):
        xy=np.asarray(xy)
        col=np.floor((xy[...,0]-self.origin[0])/self.res).astype(int)
        row=self.height-1-np.floor((xy[...,1]-self.origin[1])/self.res).astype(int)
        return row,col

    def xy(self, cell):
        row,col=cell
        return self.origin[:2]+np.array([(col+.5)*self.res,(self.height-row-.5)*self.res])

    def lookup(self, xy, field=None, outside=0.):
        field=self.clearance if field is None else field
        row,col=self.cells(xy)
        valid=(row>=0)&(row<self.height)&(col>=0)&(col<self.width)
        value=field[np.clip(row,0,self.height-1),np.clip(col,0,self.width-1)]
        return np.where(valid,value,outside)

    def visible(self,a,b):
        return segment_is_free(a,b,self.origin,self.res,self.free)

    def plan(self, start, goal):
        start=tuple(int(v) for v in self.cells(start)); goal=tuple(int(v) for v in self.cells(goal))
        if 0<=start[0]<self.height and 0<=start[1]<self.width and not self.free[start]:
            candidates=np.argwhere(self.free)
            if not len(candidates):raise ValueError('No inflated free space')
            distances=np.linalg.norm((candidates-np.array(start))*self.res,axis=1)
            closest=int(np.argmin(distances))
            if distances[closest]<=.25: start=tuple(candidates[closest])
        for cell in (start,goal):
            if not (0<=cell[0]<self.height and 0<=cell[1]<self.width and self.free[cell]):
                raise ValueError(f'Planning endpoint outside inflated free space: {cell}')
        queue=[(0.,0.,start)]; costs={start:0.}; parent={}
        while queue:
            _,cost,current=heapq.heappop(queue)
            if cost>costs[current]: continue
            if current==goal: break
            r,c=current
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                rr,cc=r+dr,c+dc
                if not (0<=rr<self.height and 0<=cc<self.width and self.free[rr,cc]): continue
                if dr and dc and not (self.free[r+dr,c] and self.free[r,c+dc]): continue
                neighbor=(rr,cc); new=cost+np.hypot(dr,dc)
                if new<costs.get(neighbor,np.inf):
                    costs[neighbor]=new; parent[neighbor]=current
                    heuristic=np.hypot(rr-goal[0],cc-goal[1])
                    heapq.heappush(queue,(new+heuristic,new,neighbor))
        if goal not in costs: raise ValueError('No path in static map')
        path=[goal]
        while path[-1]!=start: path.append(parent[path[-1]])
        dense=np.array([self.xy(p) for p in reversed(path)])
        # Short-cut only through inflated free cells, never through walls/corners.
        sparse=[dense[0]]; index=0
        while index<len(dense)-1:
            next_index=len(dense)-1
            while next_index>index+1 and not self.visible(dense[index],dense[next_index]): next_index-=1
            sparse.append(dense[next_index]); index=next_index
        sampled=[]
        for a,b in zip(sparse[:-1],sparse[1:]):
            sampled.extend(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.1)+1))[:-1])
        sampled.append(sparse[-1]); return np.array(sampled)


def path_features(path, position, static_map=None):
    index=int(np.argmin(np.linalg.norm(path-position,axis=1)))
    remaining=float(np.linalg.norm(np.diff(path[index:],axis=0),axis=1).sum())
    target_index=min(index+8,len(path)-1)
    if static_map is not None:
        while target_index>index and not static_map.visible(position,path[target_index]):
            target_index-=1
    target=path[target_index]
    return target, remaining, float(np.linalg.norm(path[index]-position))


def continuous_remaining(path,position):
    """Continuous path potential avoids rewards from jumping between nearest vertices."""
    path=np.asarray(path); position=np.asarray(position)
    if len(path)<2: return float(np.linalg.norm(path[-1]-position))
    delta=np.diff(path,axis=0); lengths=np.linalg.norm(delta,axis=1)
    t=np.clip(np.sum((position-path[:-1])*delta,axis=1)/np.maximum(lengths**2,1e-12),0,1)
    projections=path[:-1]+t[:,None]*delta
    offsets=np.linalg.norm(projections-position,axis=1); index=int(np.argmin(offsets))
    return float(lengths[index]*(1-t[index])+lengths[index+1:].sum()+offsets[index])
