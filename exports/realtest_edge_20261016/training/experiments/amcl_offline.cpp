// Offline adapter to the INSTALLED Nav2 AMCL libraries, not a reimplementation.
// Binary data contains only map, lidar, initial prior, odometry and scans.
// Independent truth is intentionally absent. No ROS communication or sleeps.
#include <nav2_amcl/map/map.hpp>
#include <nav2_amcl/pf/pf.hpp>
#include <nav2_amcl/sensors/laser/laser.hpp>
#include <nav2_amcl/motion_model/omni_motion_model.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/resource.h>
#include <ctime>
#include <thread>
#include <string>

template<class T> T read(std::ifstream &in) {T value;in.read(reinterpret_cast<char*>(&value),sizeof(T));if(!in)throw std::runtime_error("Truncated input");return value;}
pf_vector_t random_pose(void *data) {
  auto map=static_cast<map_t*>(data);pf_vector_t p{};
  for(int attempt=0;attempt<100000;++attempt) {
    int x=lrand48()%map->size_x,y=lrand48()%map->size_y;
    if(map->cells[MAP_INDEX(map,x,y)].occ_state!=-1)continue;
    p.v[0]=MAP_WXGX(map,x);p.v[1]=MAP_WYGY(map,y);p.v[2]=(drand48()*2-1)*M_PI;return p;
  }
  throw std::runtime_error("No free random pose");
}
int main(int argc,char **argv) {
  try {
    if(argc!=8)throw std::runtime_error("input output min_particles max_particles beamskip seed motion_gate");
    std::ifstream in(argv[1],std::ios::binary);if(!in)throw std::runtime_error("Input unavailable");
    int width=read<int32_t>(in),height=read<int32_t>(in),frames=read<int32_t>(in),beams=read<int32_t>(in);
    if(width<=0||height<=0||frames<=0||beams<2||width*height>10000000)throw std::runtime_error("Invalid dimensions");
    auto map=map_alloc();map->size_x=width;map->size_y=height;map->scale=read<double>(in);
    // Match AmclNode::convertMap (including its grid convention) exactly.
    map->origin_x=read<double>(in)+(width/2)*map->scale;
    map->origin_y=read<double>(in)+(height/2)*map->scale;
    double angle_min=read<double>(in),angle_max=read<double>(in),range_min=read<double>(in),range_max=read<double>(in);
    pf_vector_t mount{};mount.v[0]=read<double>(in);mount.v[1]=read<double>(in);double mount_yaw=read<double>(in);
    pf_vector_t prior{};for(double &v:prior.v)v=read<double>(in);
    map->cells=static_cast<map_cell_t*>(calloc(width*height,sizeof(map_cell_t)));
    for(int i=0;i<width*height;++i)map->cells[i].occ_state=read<int8_t>(in);
    int minimum=atoi(argv[3]),maximum=atoi(argv[4]);bool skip=atoi(argv[5]);
    if(minimum<50||maximum<minimum)throw std::runtime_error("Invalid particle count");
    auto pf=pf_alloc(minimum,maximum,0.,0.,random_pose);pf->pop_err=.05;pf->pop_z=.99;
    pf_matrix_t cov=pf_matrix_zero();cov.m[0][0]=cov.m[1][1]=.12*.12;cov.m[2][2]=.08*.08;pf_init(pf,prior,cov);
    // pf_init's Gaussian allocator seeds drand48 internally. Keep its native
    // initial population, then vary motion/resampling across repeat seeds.
    srand48(strtol(argv[6],nullptr,10));
    nav2_amcl::OmniMotionModel motion;motion.initialize(.1,.1,.1,.1,.1);
    auto laser=std::make_unique<nav2_amcl::LikelihoodFieldModelProb>(.9,.1,.12,2.,skip,.5,.3,.9,60,map);
    laser->SetLaserPose(mount);pf_vector_t previous{},last_odom{},best=prior;pf_matrix_t covariance=cov;
    double gate=atof(argv[7]);
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<"x,y,yaw,std_x,std_y,std_yaw,seconds,particles,cpu_seconds\n";
    double quota=getenv("RL_EXPERIMENT_CPU_CORE")?atof(getenv("RL_EXPERIMENT_CPU_CORE")):.35;
    if(!(quota>0&&quota<=1))throw std::runtime_error("Invalid CPU quota");
    auto budget_start=std::chrono::steady_clock::now();auto cpu_start=std::clock();
    for(int frame=0;frame<frames;++frame) {
      read<double>(in);pf_vector_t odom{};for(double &v:odom.v)v=read<double>(in);
      int selected=std::min(60,beams);
      nav2_amcl::LaserData data;data.laser=laser.get();data.range_count=selected;data.range_max=range_max;data.ranges=new double[selected][2];
      // Input parsing/file IO is excluded from both algorithms' timers.
      int next=0;
      for(int b=0;b<beams;++b) {
        double range=read<double>(in);
        if(next<selected&&b==int(double(next)*(beams-1)/(selected-1))) {
          data.ranges[next][0]=range;data.ranges[next][1]=angle_min+(angle_max-angle_min)*b/(beams-1)+mount_yaw;++next;
        }
      }
      auto start=std::chrono::steady_clock::now();
      auto frame_cpu=std::clock();
      for(int b=0;b<selected;++b)if(data.ranges[b][0]<=range_min)data.ranges[b][0]=range_max;
      pf_vector_t delta{};for(int i=0;i<3;++i)delta.v[i]=odom.v[i]-previous.v[i];delta.v[2]=std::atan2(std::sin(delta.v[2]),std::cos(delta.v[2]));
      bool update=!frame||!gate||fabs(delta.v[0])>gate||fabs(delta.v[1])>gate||fabs(delta.v[2])>gate;
      if(update) {
        if(frame)motion.odometryUpdate(pf,odom,delta);
        previous=odom;laser->sensorUpdate(pf,&data);pf_update_resample(pf,map);
        auto set=&pf->sets[pf->current_set];double best_weight=-1;
        for(int c=0;c<set->cluster_count;++c){double weight;pf_vector_t mean;pf_matrix_t pcov;if(pf_get_cluster_stats(pf,c,&weight,&mean,&pcov)&&weight>best_weight){best_weight=weight;best=mean;covariance=pcov;}}
        if(best_weight<0)throw std::runtime_error("No AMCL hypothesis");
      } else {
        // AMCL holds map->odom; propagate its latest estimate using odom.
        double angle=best.v[2]-last_odom.v[2],dx=odom.v[0]-last_odom.v[0],dy=odom.v[1]-last_odom.v[1];
        best.v[0]+=cos(angle)*dx-sin(angle)*dy;best.v[1]+=sin(angle)*dx+cos(angle)*dy;
        best.v[2]+=atan2(sin(odom.v[2]-last_odom.v[2]),cos(odom.v[2]-last_odom.v[2]));
      }
      last_odom=odom;
      double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      for(double v:best.v)out<<v<<',';for(int i=0;i<3;++i)out<<sqrt(std::max(0.,covariance.m[i][i]))<<',';
      double frame_cpu_seconds=double(std::clock()-frame_cpu)/CLOCKS_PER_SEC;
      out<<seconds<<','<<pf->sets[pf->current_set].sample_count<<','<<frame_cpu_seconds<<'\n';
      double cpu=double(std::clock()-cpu_start)/CLOCKS_PER_SEC;
      double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-budget_start).count();
      if(cpu/quota>wall)std::this_thread::sleep_for(std::chrono::duration<double>(cpu/quota-wall));
    }
    // Linux ru_maxrss can inherit the Python parent's pre-exec high water
    // mark. VmHWM describes this executable's own address space instead.
    std::ifstream status("/proc/self/status");std::string key,line;long peak=0;
    while(status>>key){if(key=="VmHWM:"){status>>peak;break;}std::getline(status,line);}
    laser.reset();pf_free(pf);map_free(map);std::cout<<"peak_rss_kib="<<peak<<'\n';
  } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
