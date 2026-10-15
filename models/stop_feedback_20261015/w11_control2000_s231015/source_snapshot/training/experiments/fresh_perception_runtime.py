"""Experimental per-frame laser fusion, retaining the existing routing cadence."""
from neural_navigation.core import NavigationCore
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime


class FreshPerceptionCore(NavigationCore):
    def reset(self,*args,**kwargs):
        self.last_perception_time=None;self.perception_update_count=0
        return super().reset(*args,**kwargs)

    def _update_perception(self):
        if self.last_perception_time!=self.elapsed:
            self.laser_fusion.update(self)
            self.last_perception_time=self.elapsed;self.perception_update_count+=1

    def _routing_dynamic_layer(self):
        self._update_perception()
        return self.laser_fusion.layer

    def _observation(self,scan,initialize=False):
        self.scan=scan
        self._update_perception()
        return super()._observation(scan,initialize)


class FreshPerceptionRuntime(QualitySensorNavigationRuntime):
    def __init__(self,bundle,map_path,initial_pose,goal,route_mode='raw'):
        super().__init__(bundle,map_path,initial_pose,goal,route_mode)
        core=FreshPerceptionCore(self.core.static_map,self.core.lidar,bundle,route_mode)
        core.reset(initial_pose,goal)
        self.core=core;self.policy.core=core

    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        result['perception_update_count']=self.core.perception_update_count
        result['perception_update_time']=self.core.last_perception_time
        return result
