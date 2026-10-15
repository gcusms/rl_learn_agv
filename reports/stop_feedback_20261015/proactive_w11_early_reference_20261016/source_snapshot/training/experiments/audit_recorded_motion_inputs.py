"""Read-only audit of actual recorded odometry/gyro configuration and streams."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
import yaml


def main():
    p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    root=a.recording;db=sqlite3.connect('file:'+str(root/'rosbag/rosbag_0.db3')+'?mode=ro',uri=True)
    series={};rows={};feedback={};status={}
    wanted=['/imu','/imu_corrected','/ros_robot_controller/imu_raw','/command/odom','/odom','/wheel/odom','/rl_navigation/odom_raw']
    for tid,name,typ in db.execute('SELECT id,name,type FROM topics'):
        if name not in wanted and name not in ['/wheel/feedback','/rl_navigation/chassis_status']:continue
        values=[];counts={}
        for receipt,raw in db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp',(tid,)):
            if name=='/wheel/feedback':
                d=json.loads(deserialize_message(raw,String).data)
                for field in ['fresh','measurement_valid']:
                    counts[field]=counts.get(field,0)+int(bool(d.get(field)))
                counts['frames']=counts.get('frames',0)+1
                feedback=counts;continue
            if name=='/rl_navigation/chassis_status':
                d=json.loads(deserialize_message(raw,String).data);source=d.get('odom_source','unknown')
                counts[source]=counts.get(source,0)+1;status=counts;continue
            m=deserialize_message(raw,Imu if 'imu' in name else Odometry)
            stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
            if 'imu' in name:values.append([stamp,m.angular_velocity.z,receipt/1e9-stamp])
            else:
                q=m.pose.pose.orientation;yaw=np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                values.append([stamp,m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.angular.z,
                    m.pose.pose.position.x,m.pose.pose.position.y,yaw])
        if not values:continue
        v=np.asarray(values);series[name]=v;dt=np.diff(v[:,0]);stat=dict(frames=len(v),median_positive_interval_s=float(np.median(dt[dt>0])),backward_timestamps=int((dt<0).sum()))
        if 'imu' in name:stat.update(gyro_z_percentiles=np.percentile(v[:,1],[5,50,95]).tolist(),receive_age_p95_s=float(np.percentile(v[:,2],95)))
        else:stat.update(xy_span_m=np.ptp(v[:,4:6],axis=0).tolist(),twist_percentiles=np.percentile(v[:,1:4],[5,50,95],axis=0).tolist())
        rows[name]=stat
    comparisons={}
    if '/imu' in series:
        gyro=series['/imu']
        for name in ['/command/odom','/odom','/wheel/odom']:
            v=series.get(name)
            if v is None:continue
            keep=(v[:,0]>=gyro[0,0])&(v[:,0]<=gyro[-1,0]);rate=np.interp(v[keep,0],gyro[:,0],gyro[:,1]);error=v[keep,3]-rate
            comparisons[name]=dict(gyro_rate_rmse_rad_s=float(np.sqrt(np.mean(error**2))),gyro_rate_absolute_difference_p95_rad_s=float(np.percentile(abs(error),95)))
    packet=dict(completed=True,scope='recorded input/configuration audit, not ground-truth accuracy or encoder calibration',
        ekf_config=yaml.safe_load((root/'snapshots/package/config/ekf.yaml').read_text()),
        recorded_chassis_odometry_sources=status,wheel_feedback_quality_counts=feedback,streams=rows,gyro_comparisons=comparisons,
        interpretation=['Recorded EKF uses command-source odometry velocity and IMU angular rate.',
            'Wheel feedback exists in this recording, but is not selected as the chassis odometry source.',
            'Fully command-integrated yaw is a degraded ablation, not an exact model of this recorded EKF.',
            'Wheel odometry is a measured-motion estimate, not independent global pose truth; wheel slip remains possible.'])
    a.output.write_text(json.dumps(packet,ensure_ascii=False,indent=2));print(json.dumps(packet,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
