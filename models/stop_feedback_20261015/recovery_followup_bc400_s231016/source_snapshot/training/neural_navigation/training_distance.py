"""Fixed static geodesic potential for TRAINING rewards only.

This module is deliberately absent from the sensor navigation runtime. A sparse
offline graph supplies reward labels; it never supplies a route or an action.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class TrainingDistanceGraph:
    def __init__(self,static_map):
        self.map=static_map;free=static_map.free;h,w=free.shape
        r,c=np.indices((h,w));source=[];target=[];cost=[]
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            rr=r+dr;cc=c+dc
            valid=free&(rr>=0)&(rr<h)&(cc>=0)&(cc<w)
            valid&=free[np.clip(rr,0,h-1),np.clip(cc,0,w-1)]
            if dr and dc:
                valid&=free[np.clip(rr,0,h-1),c]&free[r,np.clip(cc,0,w-1)]
            source.append((r*w+c)[valid]);target.append((rr*w+cc)[valid])
            cost.append(np.full(valid.sum(),np.hypot(dr,dc)*static_map.res))
        self.graph=csr_matrix((np.concatenate(cost),(np.concatenate(source),np.concatenate(target))),shape=(h*w,h*w))

    def field(self,goal):
        r,c=[int(v) for v in self.map.cells(goal)]
        if not(0<=r<self.map.height and 0<=c<self.map.width and self.map.free[r,c]):
            raise ValueError('Reward goal must be inside static inflated free space')
        distance=dijkstra(self.graph,directed=False,indices=r*self.map.width+c).reshape(self.map.free.shape)
        reachable=np.isfinite(distance)
        # Extend finite labels to the thin inflation fringe. Contact remains a
        # separate terminal penalty; no such extension is used for planning.
        offsets,indices=distance_transform_edt(~reachable,return_indices=True)
        filled=distance[indices[0],indices[1]]+offsets*self.map.res
        return TrainingDistanceField(self.map,filled,reachable)


class TrainingDistanceField:
    def __init__(self,static_map,values,reachable):
        self.map=static_map;self.values=values;self.reachable=reachable
    def value(self,xy):
        x,y=(np.asarray(xy)-self.map.origin[:2])/self.map.res
        col=x-.5;row=self.map.height-.5-y
        if not(-.5<=col<self.map.width-.5 and -.5<=row<self.map.height-.5):
            raise ValueError('Training pose outside the map')
        row=np.clip(row,0,self.map.height-1);col=np.clip(col,0,self.map.width-1)
        r,c=int(np.floor(row)),int(np.floor(col));dy,dx=row-r,col-c
        rr=min(r+1,self.map.height-1);cc=min(c+1,self.map.width-1)
        return float((1-dy)*((1-dx)*self.values[r,c]+dx*self.values[r,cc])+dy*((1-dx)*self.values[rr,c]+dx*self.values[rr,cc]))
