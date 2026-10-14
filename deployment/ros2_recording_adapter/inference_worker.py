"""Persistent, ROS-free process owning all learned model state."""
import importlib
import signal
import sys
import time
from pathlib import Path
import numpy as np
from .model_adapter import limit_command
from .fast_footprint import static_footprint_clear
from .vendor.mecanum_kinematics import MecanumKinematics
from .execution_feedback import restore_published_history

_worker=None


class InferenceWorker:
    def __init__(self,settings):
        self.settings=settings
        sys.dont_write_bytecode=True
        sys.path[:0]=[settings['model_source'],str(Path(settings['model_source'])/'training')]
        import torch
        torch.set_num_threads(1)
        module,name=settings['required_runtime'].rsplit('.',1)
        self.factory=getattr(importlib.import_module(module),name)
        from .fast_guard import install_acceleration
        install_acceleration()
        self.geometry=MecanumKinematics(*settings['geometry'])
        self.runtime=None;self.generation=None

    def infer(self,frame,generation,initial,goal,received_timestamp,capture_observation=False,published_command=None,published_reason='none'):
        started=time.perf_counter()
        s=self.settings
        if self.runtime is None or self.generation!=generation:
            self.runtime=self.factory(s['bundle'],s['map_path'],initial,goal,s['route_mode'])
            self.generation=generation
        scan,odom,velocity,stamp,valid=frame
        if published_command is not None:
            restore_published_history(self.runtime,published_command,published_reason,stamp)
        result=self.runtime.step(scan,odom,velocity,stamp,
            received_timestamp=received_timestamp,scan_valid=valid,capture_observation=capture_observation)
        if capture_observation:
            from neural_navigation.masked_actions import ACTIONS, masks_from_observation
            observation=np.asarray(result['policy_observation'],np.float32)
            result['recording_input']={
                'scan': ['nan' if np.isnan(v) else 'inf' if np.isposinf(v) else '-inf' if np.isneginf(v) else float(v) for v in scan],
                'odom':np.asarray(odom).tolist(),'velocity':np.asarray(velocity).tolist(),
                'scan_valid':bool(valid),'received_timestamp':received_timestamp,
                'action_mask':masks_from_observation(observation[:282]).tolist(),
                'selected_action_index':int(np.argmin(np.max(np.abs(ACTIONS-np.asarray(result['proposed_action'])),axis=1)))}
        guard=self.runtime.guard
        result['current_body_map_clear']=static_footprint_clear(
            self.runtime.core.static_map,self.runtime.core.estimate,guard.half*2,guard.static_margin)
        command,scale=limit_command(result['command'],self.geometry,s['max_rps'])
        executed=[command[0]/(.2 if command[0]>=0 else .08),command[1]/.5]
        self.runtime.core.record_action(executed)
        if self.runtime.context:
            self.runtime.context.last_executed=np.asarray(executed,float)
        result['adapter_inference_ms']=(time.perf_counter()-started)*1000
        return generation,stamp,result,command,scale,self.runtime.core.path.copy(),np.asarray(odom).copy()


def initialize_worker(settings):
    global _worker
    # The ROS parent owns cancellation and pool shutdown. A terminal SIGINT
    # must not interrupt a child midway through inference or its queue read.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker=InferenceWorker(settings)


def worker_ready():
    return _worker is not None


def infer_frame(frame,generation,initial,goal,received_timestamp,capture_observation=False,published_command=None,published_reason='none'):
    return _worker.infer(frame,generation,initial,goal,received_timestamp,capture_observation,published_command,published_reason)
