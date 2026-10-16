"""Persistent odometry-error budget in directions without static constraints.

This envelope is deliberately labelled uncalibrated. It is separate from the
scan information covariance and never changes the pose estimate. A succession
of short odometry steps must not average away systematic scale/bias errors.
"""
import numpy as np


def contract():
    return dict(schema='weak_geometry_odometry_budget_v1',calibrated=False,
        translation_rate_m_s=.004,translation_scale_error=.08,
        yaw_bias_rad_s=.002,yaw_scale_error=.05,minimum_current_hits=12,
        yaw_length_scale_m=3.,pose_corrections_unchanged=True)


class WeakGeometryUncertainty:
    def __init__(self,config):
        if config!=contract():raise ValueError('Weak geometry uncertainty contract mismatch')
        self.config=config;self.metric=np.array([1.,1.,config['yaw_length_scale_m']]);self.reset()
    def reset(self):self.budget=np.zeros((3,3));self.floor=np.zeros(3)
    def update(self,projection,transition,dt,movement,turn,current_hits):
        # Too few current rays may constrain pose, but do not erase a history
        # of systematic drift as though a full independent scan was observed.
        if current_hits<self.config['minimum_current_hits']:projection=np.zeros((3,3))
        weak=np.eye(3)-projection
        translation=self.config['translation_rate_m_s']*dt+self.config['translation_scale_error']*movement
        yaw=self.config['yaw_bias_rad_s']*dt+self.config['yaw_scale_error']*abs(turn)
        transform=self.metric[:,None]*transition/self.metric[None,:]
        self.budget=weak@(transform@self.budget@transform.T+np.diag([translation,translation,yaw*self.metric[2]]))@weak
        self.floor=np.linalg.norm(self.budget,axis=1)/self.metric
        return self.floor.copy()
