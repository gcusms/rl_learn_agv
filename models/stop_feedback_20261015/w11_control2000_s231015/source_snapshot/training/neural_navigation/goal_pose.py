"""Goal orientation is a task input, never a localization measurement."""
import numpy as np
from navrl.geometry import wrap


def contract():
    return dict(schema='quality_goal_pose_v1',dimensions=303,prefix_dimensions=300,
                appended_features=['sin_goal_yaw_error','cos_goal_yaw_error','goal_yaw_required'],
                position_tolerance_m=.12,yaw_tolerance_rad=.15,
                stopped_linear_m_s=.03,stopped_angular_rad_s=.08,settle_seconds=.4,
                control_period_s=.2,scan_history_offsets_s=[-.4,-.2,0.],calibrated=False)


def append_goal_features(observation,core):
    required=core.goal_yaw is not None
    error=wrap(core.goal_yaw-core.estimate[2]) if required else 0.
    return np.r_[observation,np.sin(error) if required else 0.,np.cos(error) if required else 0.,float(required)].astype(np.float32)


def goal_status(core):
    c=contract();distance=float(np.linalg.norm(core.estimate[:2]-core.goal))
    error=float(wrap(core.goal_yaw-core.estimate[2])) if core.goal_yaw is not None else None
    position=distance<c['position_tolerance_m']
    aligned=error is None or abs(error)<c['yaw_tolerance_rad']
    stopped=bool(abs(core.measured_velocity[0])<c['stopped_linear_m_s'] and abs(core.measured_velocity[1])<c['stopped_angular_rad_s'])
    settled=(core._goal_settled_since is not None and core.elapsed-core._goal_settled_since>=c['settle_seconds']-1e-9)
    reached=position if error is None else position and aligned and stopped and settled
    return dict(goal_yaw_required=error is not None,goal_yaw_error_rad=error,
                estimated_goal_position_reached=position,estimated_goal_pose_reached=bool(reached))


def update_goal_settling(core):
    c=contract();state=goal_status(core)
    good=(state['estimated_goal_position_reached'] and
          (state['goal_yaw_error_rad'] is None or abs(state['goal_yaw_error_rad'])<c['yaw_tolerance_rad']) and
          abs(core.measured_velocity[0])<c['stopped_linear_m_s'] and abs(core.measured_velocity[1])<c['stopped_angular_rad_s'])
    if not good:core._goal_settled_since=None
    elif core._goal_settled_since is None:core._goal_settled_since=core.elapsed
