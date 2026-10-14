"""Only the ROS parent knows whether a worker candidate was published."""
import numpy as np


def restore_published_history(runtime,command,reason,timestamp):
    command=np.asarray(command,float)
    if command.shape!=(2,) or not np.isfinite(command).all():raise ValueError('Finite published command required')
    executed=command/[.2 if command[0]>=0 else .08,.5]
    runtime.core.record_action(executed)
    if runtime.context:
        runtime.context.record(dict(command=command.tolist(),stop_reason=reason,
            timestamp=float(timestamp),proposed_action=runtime.context.last_proposal.tolist(),
            safety_override=reason not in ('none','estimated_goal_reached')))
