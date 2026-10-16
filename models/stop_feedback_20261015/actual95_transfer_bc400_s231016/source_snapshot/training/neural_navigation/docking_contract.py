"""Terminal actor contract shared by training and simulator-free deployment."""


def docking_contract(radius=.8):
    if radius not in (.8,.10):raise ValueError('Unsupported terminal radius')
    return dict(schema='learned_terminal_actor_v1' if radius==.8 else 'learned_terminal_actor_v2',observation_dimensions=303,
        activation_distance_m=radius,requires_directional_goal=True,actions=12,
        frozen_navigation_actor=True,teacher_at_runtime=False,
        action_space_expanded=False,slow_rotation_calibrated=False)
