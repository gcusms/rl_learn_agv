"""Experimental completion from registered past measurements only.

Retain measured surface extent under partial occlusion instead of replacing
all missing extent with an isotropic expansion toward every direction. Fits
that lack support retain the cumulative predictor's original bound.
"""
import numpy as np
from neural_navigation.contour_history import CumulativeContourPrediction
from neural_navigation.contour_translation import contour_translation
from neural_navigation.contour_velocity import ContourAwareVelocity


class CompletedContourPrediction(CumulativeContourPrediction):
    def __init__(self):
        super().__init__(registered_motion=True);self.measured_history={}

    def update(self,tracks,timestamp):
        output=super().update(tracks,timestamp)
        for raw in tracks:
            key=raw['id'];stamp=float(raw['time'])
            history=self.measured_history.get(key,[])
            if not history or stamp>history[-1][0]:
                history.append((stamp,np.asarray(raw['points'],float).copy()))
            self.measured_history[key]=[(t,p) for t,p in history if timestamp-t<=1.2+1e-9]
        self.measured_history={key:history for key,history in self.measured_history.items()
            if key in self.states and history}
        for track in output:
            # An old native track is never a fresh fit. Held/unknown tracks
            # keep their original reachable envelope and measured age.
            if track['source']!='observed' or track['age']>1e-8:continue
            points=np.asarray(track['points'],float);surfaces=[points];errors=[]
            for stamp,reference in self.measured_history.get(track['id'],[]):
                if timestamp-stamp<.15:continue
                fit=contour_translation(reference,points)
                if fit is None:continue
                displacement,error,covariance=fit
                surfaces.append(reference+displacement)
                errors.append(2*max(error,np.sqrt(np.linalg.eigvalsh(covariance)[-1]),.005))
            if not errors:continue
            completed=np.concatenate(surfaces,axis=0)
            diameter,_=ContourAwareVelocity.geometry(completed)
            missing=max(0.,self.states[track['id']]['peak_diameter']-diameter)/2
            # Keep the real footprint/margin and residual missing extent.
            # No guessed shape, simulator radius, or discarded fresh point.
            track.update(points=completed,
                contour_position_uncertainty_m=missing+max(errors),
                completed_from_measured_history=True)
        return output
