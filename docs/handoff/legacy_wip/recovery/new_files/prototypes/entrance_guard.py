"""Keep a rotation reserve before driving an unaligned body into a narrow lane.

This predicate only masks a requested forward action; it never creates a turn.
Normal 4 cm body/braking checks remain authoritative for every action.
"""
import numpy as np
from prototypes.narrow_passage import rotation_clear


def retains_alignment_space(guard,core,command,age):
    if command[0]<=.005:return True
    feature=getattr(core,'corridor_alignment_feature',None)
    if feature is None or feature[0]<.5 or not (.40<feature[1]*2<.80):return True
    heading_error=float(feature[3])*.35;offset=float(feature[2])*.25
    if abs(heading_error)<=.025 and abs(offset)<=.01:return True
    # A one-centimetre extra reserve is applied only while still aligning.
    # It is not subtracted from the nominal margin after entry. This catches
    # commands that are immediately clear but remove the last safe turn at
    # the next sensor update. No reduction of footprint or map occupancy.
    desired_heading=float(core.estimate[2])+heading_error
    _,endpoint,_=list(guard._trajectory(core,command,age))[-1]
    return rotation_clear(core.static_map,endpoint[:2],float(endpoint[2]),desired_heading,
                          guard.half*2,guard.static_margin+.01)
