"""Use the historical viewer with the selected native 300-field runtime."""
from experiments.quality_sensor_runtime import QualitySensorNavigationRuntime
import preview_sensor_navigation

if __name__=='__main__':
    preview_sensor_navigation.SensorNavigationRuntime=QualitySensorNavigationRuntime
    preview_sensor_navigation.main()
