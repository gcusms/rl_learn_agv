"""Interactive manual columns using the video's 100k policy configuration."""
import sys
from navigation_parameter_demo import ParameterDemoEnv
from neural_navigation.waypoint_interference import WaypointInterferenceEnv
from navrl.geometry import ROOT
import preview_live_navigation as preview

class Manual100kEnv(ParameterDemoEnv):
    preset='ppo_100k'
    def reset(self,**kwargs):
        self.demo_turn=0.
        return WaypointInterferenceEnv.reset(self,**kwargs)
    _walker_velocity=WaypointInterferenceEnv._walker_velocity
    select_goal=WaypointInterferenceEnv.select_goal

if __name__=='__main__':
    preview.WaypointInterferenceEnv=Manual100kEnv
    sys.argv=[sys.argv[0],'--policy',str(ROOT/'runs/remote_room_100k/policy_100000.zip'),'--max-leg-steps','1800','--output',str(ROOT/'runs/local_manual_100k_demo')]+sys.argv[1:]
    preview.main()
