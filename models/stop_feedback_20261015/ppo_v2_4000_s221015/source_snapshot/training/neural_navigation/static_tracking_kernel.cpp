// Optional acceleration of fast_static_localizer.py, same four-iteration math.
// Build explicitly on the target architecture; never compile during model load.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
using Vec=Eigen::Vector3d;
using Mat=Eigen::Matrix3d;

struct Field {
  int height,width;double res,ox,oy;const double *sdf,*gx,*gy;
  double sample(const double *field,double x,double y,double outside) const {
    double c=(x-ox)/res-.5,r=height-.5-(y-oy)/res;
    // scipy.ndimage.map_coordinates(order=1, mode='constant') convention.
    if(c<0||r<0||c>width-1||r>height-1)return outside;
    int c0=int(c),r0=int(r),c1=std::min(c0+1,width-1),r1=std::min(r0+1,height-1);
    double u=c-c0,v=r-r0;
    return (1-v)*((1-u)*field[r0*width+c0]+u*field[r0*width+c1])+
           v*((1-u)*field[r1*width+c0]+u*field[r1*width+c1]);
  }
  double linearize(const Vec &pose,const double *points,int n,double *jac,double *weights,Mat &h,Vec &g) const {
    double c=cos(pose[2]),s=sin(pose[2]),cost=0.;h.setZero();g.setZero();
    for(int i=0;i<n;++i) {
      double dx=c*points[2*i]-s*points[2*i+1],dy=s*points[2*i]+c*points[2*i+1];
      double x=pose[0]+dx,y=pose[1]+dy,r=sample(sdf,x,y,1.);
      double nx=sample(gx,x,y,0.),ny=sample(gy,x,y,0.);
      Vec j(nx,ny,(-nx*dy+ny*dx)/3.);double w=std::min(1.,.05/std::max(fabs(r),1e-12));
      h.noalias()+=w*(j*j.transpose());g.noalias()+=j*(w*r);
      if(jac){jac[3*i]=nx;jac[3*i+1]=ny;jac[3*i+2]=j[2]*3.;weights[i]=w;}
      cost+=fabs(r)<=.05?.5*r*r:.05*(fabs(r)-.025);
    }
    h/=n;g/=n;return cost/n;
  }
  double cost(const Vec &pose,const double *points,int n) const {
    double c=cos(pose[2]),s=sin(pose[2]),total=0.;
    for(int i=0;i<n;++i){double dx=c*points[2*i]-s*points[2*i+1],dy=s*points[2*i]+c*points[2*i+1];
      double r=fabs(sample(sdf,pose[0]+dx,pose[1]+dy,1.));total+=r<=.05?.5*r*r:.05*(r-.025);}
    return total/n;
  }
};

static int refine(int height,int width,double res,double ox,double oy,
 const double *sdf,const double *gx,const double *gy,const double *points,int n,
 const double *input,double xy_cap,double yaw_cap,double *output,double *jac,double *weights,
 int minimum_hits,double relative_threshold,double absolute_threshold) noexcept {
 try {
  if(n<minimum_hits||n>220||height<2||width<2||res<=0)return 1;
  Field field{height,width,res,ox,oy,sdf,gx,gy};Vec prior(input),pose=prior;
  double cost=field.cost(pose,points,n);Vec metric(1.,1.,3.);
  for(int iteration=0;iteration<4;++iteration){
    Mat h;Vec g;field.linearize(pose,points,n,nullptr,nullptr,h,g);
    Eigen::SelfAdjointEigenSolver<Mat> eig(h);if(eig.info()!=Eigen::Success)return 2;
    Mat projection=Mat::Zero();double threshold=std::max(absolute_threshold,eig.eigenvalues()[2]*relative_threshold);
    for(int i=0;i<3;++i)if(eig.eigenvalues()[i]>threshold)projection.noalias()+=eig.eigenvectors().col(i)*eig.eigenvectors().col(i).transpose();
    Vec delta=(-(projection*(h+Mat::Identity()*.02).ldlt().solve(g))).cwiseQuotient(metric);
    if(!delta.allFinite())return 3;
    Vec trial=pose+delta,innovation=trial-prior;
    innovation[2]=atan2(sin(innovation[2]),cos(innovation[2]));
    innovation*=std::min({1.,xy_cap/std::max(innovation.head<2>().norm(),1e-12),yaw_cap/std::max(fabs(innovation[2]),1e-12)});
    trial=prior+innovation;trial[2]=atan2(sin(trial[2]),cos(trial[2]));
    double after=field.cost(trial,points,n);if(after>=cost-1e-12)break;
    pose=trial;cost=after;if(delta.cwiseProduct(metric).norm()<1e-4)break;
  }
  Mat h;Vec g;field.linearize(pose,points,n,jac,weights,h,g);
  Eigen::Map<Vec> mapped(output);mapped=pose;return 0;
 } catch(...){return 4;}
}

extern "C" int static_refine_v1(int height,int width,double res,double ox,double oy,
 const double *sdf,const double *gx,const double *gy,const double *points,int n,
 const double *input,double xy_cap,double yaw_cap,double *output,double *jac,double *weights) noexcept {
 return refine(height,width,res,ox,oy,sdf,gx,gy,points,n,input,xy_cap,yaw_cap,output,jac,weights,12,.04,.02);
}

extern "C" int static_refine_v2(int height,int width,double res,double ox,double oy,
 const double *sdf,const double *gx,const double *gy,const double *points,int n,
 const double *input,double xy_cap,double yaw_cap,double *output,double *jac,double *weights) noexcept {
 return refine(height,width,res,ox,oy,sdf,gx,gy,points,n,input,xy_cap,yaw_cap,output,jac,weights,3,.01,.01);
}
