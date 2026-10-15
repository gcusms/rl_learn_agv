"""Opt-in slow, non-worsening recovery, with unchanged body/braking margins.

No steering is invented. Full route failure can admit a policy-requested slow
command only inside a currently checked local segment. Side/rear returns already
inside the nominal laser margin may remain at their current distance while
the robot moves slowly; physical/map margins and all other points still apply.
"""
import copy
import numpy as np
from neural_navigation.runtime import EmergencyStop


class NarrowRecoveryGuard(EmergencyStop):
    schema='experimental_narrow_alignment_reserve_v2'
    command_tolerance=2e-8  # Same float32 round-trip tolerance as the trainer.

    def _trajectory_check(self,core,command,points,margins,age,allow_escape=False):
        reason=super()._trajectory_check(core,command,points,margins,age,allow_escape)
        if reason!='escape_not_increasing_clearance' or not allow_escape:return reason
        v,w=command
        slow_translation=.005<v<=.05+self.command_tolerance and abs(w)<=.1+self.command_tolerance
        slow_rotation=abs(v)<1e-8 and .005<abs(w)<=.2+self.command_tolerance
        if not ((slow_translation or slow_rotation) and 0<=age<=.2):return reason
        if self.static_footprint_schema!='oriented_rectangle_v1':return reason
        # Only a fresh, accepted scan/map alignment supports this relaxation of
        # the old final +5 mm condition. Moving or unknown tracks cannot use it.
        if not getattr(core.localizer,'scan_update_accepted',False):return reason
        agreement=getattr(core.localizer,'map_agreement',0.)
        if not np.isfinite(agreement) or agreement<.8:return reason
        if getattr(core,'unmatched_returns',1)>0:return reason
        if getattr(getattr(core,'laser_fusion',None),'forecast_cells',0):return reason
        predictor=getattr(getattr(core,'laser_fusion',None),'predictor',None)
        if predictor is not None and any(t['hits']<3 or not np.isfinite(t['velocity']).all()
            or np.linalg.norm(t['velocity'])>=.04 or core.elapsed-t.get('time',core.elapsed)>.4
            for t in predictor.tracks):return reason
        initial=np.linalg.norm(points-np.clip(points,-self.half,self.half),axis=1)
        violating=initial<margins
        if not violating.any() or np.min(initial[violating])<.02:return reason
        close_points=points[violating]
        # Rear corner returns must not be confused with a forward obstruction.
        # Forward progress can increase their clearance by less than 5 mm;
        # the parent's complete no-worsening sweep has already been satisfied.
        if np.any(close_points[:,0]>=self.half[0]):return reason
        # The parent has already checked the entire braking sweep: no point
        # may get closer, no new margin violation, and map footprint is clear.
        return None

    def check(self,core,command,scan_timestamp,now,scan_valid=True,rear_clearance_known=False):
        command=np.asarray(command,float)
        # Keep the native fast rejection order for invalid sensor states.
        # Do not run local geometry with NaN timestamps or an uncertain pose.
        finite=np.isfinite(np.r_[command,scan_timestamp,now]).all()
        std=np.asarray(core.localizer.std)
        if (not finite or not scan_valid or not (-.05<=now-scan_timestamp<=self.max_scan_age)
            or core.odometry_sanity_failure or not np.isfinite(std).all()
            or max(std[:2])>.2 or std[2]>.25):
            return super().check(core,command,scan_timestamp,now,scan_valid,rear_clearance_known)
        from prototypes.entrance_guard import retains_alignment_space
        if not retains_alignment_space(self,core,command,max(now-scan_timestamp,0.)):
            return np.zeros(2),'entrance_alignment_space'
        diagnostics=core.diagnostics()
        if not diagnostics['route_complete'] and .005<command[0]<=.05+self.command_tolerance and abs(command[1])<=.1+self.command_tolerance:
            path=getattr(core,'predicted_path',None)
            if path is not None and len(path)>1:
                point=path[1];delta=point-core.estimate[:2];length=np.linalg.norm(delta)
                clear=length>.05 and (core._segment_clear(core.estimate[:2],point) or
                    (hasattr(core,'_local_segment_clear') and core._local_segment_clear(core.estimate[:2],point)))
                if clear:
                    unit=delta/length;fits=True
                    for _,world,_ in self._trajectory(core,command,max(now-scan_timestamp,0)):
                        offset=world[:2]-core.estimate[:2];along=offset@unit
                        if along<-.001 or along>length or np.linalg.norm(offset-along*unit)>.015:
                            fits=False;break
                    if fits:
                        view=copy.copy(core);checked=diagnostics.copy();checked['route_complete']=True
                        view.diagnostics=lambda:checked
                        return super().check(view,command,scan_timestamp,now,scan_valid,rear_clearance_known)
        return super().check(core,command,scan_timestamp,now,scan_valid,rear_clearance_known)


def replace_guard(guard):
    result=copy.copy(guard);result.__class__=NarrowRecoveryGuard
    return result
