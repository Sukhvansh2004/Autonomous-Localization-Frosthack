#include <memory>
#include <sstream>
#include <unistd.h>
#include <stdlib.h>
#include <SDL2/SDL.h>
#include <chrono>
#include <thread>
#include <atomic>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Empty.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_srvs/Empty.h>

#include <arp/cameras/PinholeCamera.hpp>
#include "arp/cameras/RadialTangentialDistortion.hpp"

#include <arp/Autopilot.hpp>
#include <ros/package.h>
#include <arp/VisualInertialTracker.hpp>
#include <arp/StatePublisher.hpp>
#include <arp/Planner.hpp>
#include <visualization_msgs/MarkerArray.h>

#define WINDOW_HEIGHT 360
#define WINDOW_WIDTH 640

class Subscriber
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Subscriber(arp::VisualInertialTracker *tracker) : tracker_(tracker) {}

  Subscriber() = default;

  void imageCallback(const sensor_msgs::ImageConstPtr &msg)
  {
    uint64_t timeMicroseconds = uint64_t(msg->header.stamp.sec) * 1000000ll + msg->header.stamp.nsec / 1000;
    std::lock_guard<std::mutex> l(imageMutex_);
    lastImage_ = cv_bridge::toCvShare(msg, "bgr8")->image;
    if (tracker_)
      tracker_->addImage(timeMicroseconds, lastImage_);
  }

  bool getLastImage(cv::Mat &image)
  {
    std::lock_guard<std::mutex> l(imageMutex_);
    if (lastImage_.empty())
      return false;
    image = lastImage_.clone();
    lastImage_ = cv::Mat(); // clear, only get the same image once.
    return true;
  }

  bool getLastVisualizationImage(cv::Mat &image)
  {
    return tracker_->getLastVisualisationImage(image);
    ;
  }

  void imuCallback(const sensor_msgs::ImuConstPtr &msg)
  {
    uint64_t timeMicroseconds = uint64_t(msg->header.stamp.sec) * 1000000ll + msg->header.stamp.nsec / 1000;
    Eigen::Vector3d acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    Eigen::Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    if (tracker_)
      tracker_->addImuMeasurement(timeMicroseconds, gyro, acc);
  }

private:
  cv::Mat lastImage_;
  std::mutex imageMutex_;
  arp::VisualInertialTracker *tracker_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "arp_node");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  double fu = 0, fv = 0, cu = 0, cv = 0, k1 = 0, k2 = 0, p1 = 0, p2 = 0;
  int uniformityRadius, octaves, maxNumKpt;
  float absoluteThreshold, mappingCamFocalLength;

  bool obtainedCameraParameters = nh.getParam("/arp_node/fu", fu);
  obtainedCameraParameters &= nh.getParam("/arp_node/fv", fv);
  obtainedCameraParameters &= nh.getParam("/arp_node/cu", cu);
  obtainedCameraParameters &= nh.getParam("/arp_node/cv", cv);
  obtainedCameraParameters &= nh.getParam("/arp_node/k1", k1);
  obtainedCameraParameters &= nh.getParam("/arp_node/k2", k2);
  obtainedCameraParameters &= nh.getParam("/arp_node/p1", p1);
  obtainedCameraParameters &= nh.getParam("/arp_node/p2", p2);
  obtainedCameraParameters &= nh.getParam("/arp_node/uniformityRadius", uniformityRadius);
  obtainedCameraParameters &= nh.getParam("/arp_node/octaves", octaves);
  obtainedCameraParameters &= nh.getParam("/arp_node/absoluteThreshold", absoluteThreshold);
  obtainedCameraParameters &= nh.getParam("/arp_node/maxNumKpt", maxNumKpt);
  obtainedCameraParameters &= nh.getParam("/arp_node/mappingCamFocalLength", mappingCamFocalLength);

  bool doUndistortImages = false;

  if (!obtainedCameraParameters)
  {
    std::cerr << "No valid paramter configuration was obtained from the launch file! Make sure to specify fu, fv, cu, cv, k1, k2, p1, p2, uniformityRadius, octaves, absoluteThreshold, maxNumKpt and mappingCamFocalLength" << std::endl;
    return -1;
  }

  // Set up frontend with parameters loaded earlier
  arp::Frontend frontend(640, 360, fu, fv, cu, cv, k1, k2, p1, p2, uniformityRadius, octaves, absoluteThreshold, maxNumKpt, mappingCamFocalLength);

  // Load map
  std::string path = ros::package::getPath("VI_Localization");
  std::string mapFile;
  if (!nh.getParam("arp_node/map", mapFile))
  {
    ROS_FATAL("Error loading parameter for map");
  }
  std::string mapPath = path + "/maps/" + mapFile;
  if (!frontend.loadMap(mapPath))
  {
    ROS_FATAL_STREAM("Could not load map from " << mapPath << " !");
  }

  // load DBoW2 vocabulary
  std::string vocPath = path + "/maps/small_voc.yml.gz";
  if (!frontend.loadDBoW2Voc(vocPath))
  {
    ROS_FATAL_STREAM("could not load DBoW2 voc. from " << vocPath << " !");
  }

  if (!frontend.assignDBoW2Histograms())
  {
    ROS_FATAL_STREAM("could not assign DBoW2 histograms!");
  }

  // State publisher for RViz visualization
  arp::StatePublisher pubState(nh);

  // Set up EKF
  arp::ViEkf viEkf;
  Eigen::Matrix4d T_SC_mat;
  std::vector<double> T_SC_array;
  if (!nh.getParam("arp_node/T_SC", T_SC_array))
  {
    ROS_FATAL("Error loading parameter for T_SC");
  }
  T_SC_mat << T_SC_array[0], T_SC_array[1], T_SC_array[2], T_SC_array[3],
      T_SC_array[4], T_SC_array[5], T_SC_array[6], T_SC_array[7],
      T_SC_array[8], T_SC_array[9], T_SC_array[10], T_SC_array[11],
      T_SC_array[12], T_SC_array[13], T_SC_array[14], T_SC_array[15];
  arp::kinematics::Transformation T_SC(T_SC_mat);
  viEkf.setCameraExtrinsics(T_SC);
  viEkf.setCameraIntrinsics(frontend.camera());

  // Set up Visual-Inertial Tracker
  arp::VisualInertialTracker visualInertialTracker;
  visualInertialTracker.setFrontend(frontend);
  visualInertialTracker.setEstimator(viEkf);

  // Set up visualization: publish poses to topic `ardrone/vi_ekf_pose`
  visualInertialTracker.setVisualisationCallback(
      std::bind(&arp::StatePublisher::publish, &pubState, std::placeholders::_1, std::placeholders::_2));

  // setup inputs
  Subscriber subscriber(&visualInertialTracker);
  image_transport::Subscriber subImage = it.subscribe("/image_raw", 2, &Subscriber::imageCallback, &subscriber);
  ros::Subscriber subImu = nh.subscribe("/imu", 50, &Subscriber::imuCallback, &subscriber);

  // Visualization enabling/disabling
  visualInertialTracker.enableFusion(true);

  // set up autopilot
  arp::Autopilot autopilot(nh);

  // Set up visual inertial tracker callback
  visualInertialTracker.setControllerCallback(
      std::bind(&arp::Autopilot::controllerCallback, &autopilot, std::placeholders::_1, std::placeholders::_2));

  // Setup Pinholecamera model
  arp::cameras::PinholeCamera<arp::cameras::RadialTangentialDistortion> pinholeCamera(WINDOW_WIDTH, WINDOW_HEIGHT, fu, fv, cu, cv, arp::cameras::RadialTangentialDistortion(k1, k2, p1, p2));
  pinholeCamera.initialiseUndistortMaps();

  // setup rendering
  SDL_Event event;
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window *window = SDL_CreateWindow("VI Localization", SDL_WINDOWPOS_UNDEFINED,
                                        SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
  SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
  SDL_RenderClear(renderer);
  SDL_RenderPresent(renderer);
  SDL_Texture *texture;

  // enter main event loop
  std::cout << "===== VI Localization ====" << std::endl;
  cv::Mat image;

  double velocityStepSize{0.5};

  while (ros::ok())
  {
    ros::spinOnce();
    ros::Duration dur(0.04);
    dur.sleep();
    SDL_PollEvent(&event);

    if (event.type == SDL_QUIT)
    {
      break;
    }


    // render image, if there is a new one available
    if (subscriber.getLastVisualizationImage(image))
    {
      if (doUndistortImages)
      {
        pinholeCamera.undistortImage(image, image);
      }

      // https://stackoverflow.com/questions/22702630/converting-cvmat-to-sdl-texture
      // I'm using SDL_TEXTUREACCESS_STREAMING because it's for a video player, you should
      // pick whatever suits you most: https://wiki.libsdl.org/SDL_TextureAccess
      // remember to pick the right SDL_PIXELFORMAT_* !

      texture = SDL_CreateTexture(
          renderer, SDL_PIXELFORMAT_BGR24, SDL_TEXTUREACCESS_STREAMING, image.cols, image.rows);
      SDL_UpdateTexture(texture, NULL, (void *)image.data, image.step1());
      SDL_RenderClear(renderer);
      SDL_RenderCopy(renderer, texture, NULL, NULL);
      SDL_RenderPresent(renderer);
      // cleanup (only after you're done displaying. you can repeatedly call UpdateTexture without destroying it)
      SDL_DestroyTexture(texture);
    }
  }

  // cleanup
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
