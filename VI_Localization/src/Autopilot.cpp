/*
 * Autopilot.cpp
 *
 *  Created on: 10 Jan 2017
 *      Author: sleutene
 */

#include <arp/Autopilot.hpp>
#include <arp/PidController.hpp>
#include <arp/kinematics/operators.hpp>
#include <math.h>

namespace arp
{

  Autopilot::Autopilot(ros::NodeHandle &nh)
      : nh_(&nh)
  {

    pubReset_ = nh_->advertise<std_msgs::Empty>("/ardrone/reset", 1);
    pubTakeoff_ = nh_->advertise<std_msgs::Empty>("/ardrone/takeoff", 1);
    cmd_vel_pub = nh_->advertise<geometry_msgs::Twist>("/cmd_vel", 10);

    srvFlattrim_ = nh_->serviceClient<std_srvs::Empty>(
        nh_->resolveName("ardrone/flattrim"), 1);

  }

  // Turn off all motors and reboot.
  bool Autopilot::estopReset()
  {
    // ARdrone -> Emergency mode
    std_msgs::Empty resetMsg;
    pubReset_.publish(resetMsg);
    return true;
  }

  // Move the drone automatically.
  bool Autopilot::setPoseReference(double x, double y, double z, double yaw)
  {
    std::lock_guard<std::mutex> l(refMutex_);
    ref_x_ = x;
    ref_y_ = y;
    ref_z_ = z;
    ref_yaw_ = yaw;
    return true;
  }

  bool Autopilot::getPoseReference(double &x, double &y, double &z, double &yaw)
  {
    std::lock_guard<std::mutex> l(refMutex_);
    x = ref_x_;
    y = ref_y_;
    z = ref_z_;
    yaw = ref_yaw_;
    return true;
  }

  void Autopilot::controllerCallback(uint64_t timeMicroseconds,
                                     const arp::kinematics::RobotState &x)
  {
    latestState = x;
    {
      // keep resetting this to make sure we use the current state as reference as soon as sent to automatic mode
      const double yaw = kinematics::yawAngle(x.q_WS);
      setPoseReference(x.t_WS[0], x.t_WS[1], x.t_WS[2], yaw);
      return;
    }
  }

} // namespace arp
