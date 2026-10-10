import heapq
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import yaml


class StaticMap:
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
        self.distance=distance_transform_edt(~self.occupied)*self.res
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

    def visible(self, a, b):
        # Supercover grid traversal checks every cell touched by the segment,
        # including both neighbors at corner crossings (no sampled-ray gaps).
        origin=(np.asarray(a)-self.origin[:2])/self.res
        end=(np.asarray(b)-self.origin[:2])/self.res
        cell=np.floor(origin).astype(int); target=np.floor(end).astype(int)
        delta=end-origin; direction=np.sign(delta).astype(int)
        def clear(c):
            col,y=c; row=self.height-1-y
            return 0<=row<self.height and 0<=col<self.width and bool(self.free[row,col])
        if not clear(cell) or not clear(target): return False
        next_t=np.full(2,np.inf); stride=np.full(2,np.inf)
        for axis in range(2):
            if direction[axis]:
                boundary=cell[axis]+(1 if direction[axis]>0 else 0)
                next_t[axis]=(boundary-origin[axis])/delta[axis]
                stride[axis]=1/abs(delta[axis])
        for _ in range(self.width+self.height+4):
            if np.array_equal(cell,target): return True
            if min(next_t)>1.+1e-10: return True
            if abs(next_t[0]-next_t[1])<1e-12:
                if not clear(cell+[direction[0],0]) or not clear(cell+[0,direction[1]]): return False
                cell+=direction; next_t+=stride
            else:
                axis=int(np.argmin(next_t)); cell[axis]+=direction[axis]; next_t[axis]+=stride[axis]
            if not clear(cell): return False
        return False

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
