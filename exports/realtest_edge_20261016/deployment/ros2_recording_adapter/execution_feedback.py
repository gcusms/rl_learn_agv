"""Only the ROS parent knows whether a worker candidate was published."""
import numpy as np


def restore_published_history(runtime,command,reason,timestamp):
    command=np.asarray(command,float)
    if command.shape!=(2,) or not np.isfinite(np.r_[command,timestamp]).all():raise ValueError('Finite published command/time required')
    executed=command/[.2 if command[0]>=0 else .08,.5]
    runtime.core.record_action(executed)
    if runtime.context:
        runtime.context.record(dict(command=command.tolist(),stop_reason=reason,
            timestamp=float(timestamp),proposed_action=runtime.context.last_proposal.tolist(),
            safety_override=reason not in ('none','estimated_goal_reached')))
    feedback=getattr(getattr(runtime,'policy',None),'feedback',None)
    if feedback is not None:
        from neural_navigation.decision_feedback import FEEDBACK_REASONS, WAIT_SCHEMA
        # The worker's last candidate may never have been published (deadline,
        # sensor loss, cancellation). Correct its history before the next scan.
        # This does not invent an executed policy sample or its likelihood.
        idle=bool(np.max(np.abs(command))<1e-8)
        actual_reason=reason or 'none'
        choices=feedback.frame_mask is not None and bool(feedback.frame_mask[1:].any())
        if actual_reason=='none' and idle:
            actual_reason='policy_wait' if choices else 'only_stop_available'
        feedback.previous_reason=actual_reason if actual_reason in FEEDBACK_REASONS else 'other'
        if idle and actual_reason!='estimated_goal_reached':
            if feedback.idle_since is None:feedback.idle_since=float(timestamp)
        else:feedback.idle_since=None
        with_choices=idle and actual_reason=='policy_wait'
        if feedback.schema==WAIT_SCHEMA:
            with_choices=idle and choices and actual_reason!='estimated_goal_reached'
        if with_choices:
            if feedback.idle_with_choices_since is None:feedback.idle_with_choices_since=float(timestamp)
        else:feedback.idle_with_choices_since=None
        # frame_timestamp and recorded_timestamp still describe the original
        # guard probes; the next runtime.step records its own real candidate.
