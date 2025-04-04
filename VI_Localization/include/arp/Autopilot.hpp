/*
 * Autopilot.hpp
 *
 *  Created on: 10 Jan 2017
 *      Author: sleutene
 */

#ifndef ARDRONE_PRACTICALS_INCLUDE_ARP_AUTOPILOT_HPP_
#define ARDRONE_PRACTICALS_INCLUDE_ARP_AUTOPILOT_HPP_

#include <mutex>
#include <Eigen/Core>
#include <atomic>
#include <deque>

#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Empty.h>
#include <std_srvs/Empty.h>

#include <arp/kinematics/Imu.hpp>
#include <arp/PidController.hpp>
#include <arp/Planner.hpp>

namespace arp {

/// \brief The autopilot highlevel interface for commanding the drone manually or automatically.
class Autopilot {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Autopilot(ros::NodeHandle& nh);

  /// \brief Set to automatic control mode.
  void setManual();

  /// \brief Set to manual control mode.
  void setAutomatic();

  /// \brief Are we currently in automatic mode?;
  bool isAutomatic() { return isAutomatic_; }

  /// \brief Turn off all motors and reboot.
  /// \return True on success.
  /// \warning When in flight, this will let the drone drop down to the ground.
  bool estopReset();

  /// \brief Move the drone automatically.
  /// @param[in] x World x position reference [m].
  /// @param[in] y World y position reference [m].
  /// @param[in] z World z position reference [m].
  /// @param[in] yaw Yaw angle reference [rad].
  /// \return True on success.
  /// \note  This will only do something when in automatic mode and flying.
  bool setPoseReference(double x, double y, double z, double yaw);

  /// \brief Get the pose reference.
  /// @param[out] x World x position reference [m].
  /// @param[out] y World y position reference [m].
  /// @param[out] z World z position reference [m].
  /// @param[out] yaw Yaw angle reference [rad].
  /// \return True on success.
  bool getPoseReference(double& x, double& y, double& z, double& yaw);

  /// \brief The callback from the estimator that sends control outputs to the drone
  /// \note  This will only do something when in automatic mode and flying.
  void controllerCallback(uint64_t timeMicroseconds,
                          const arp::kinematics::RobotState& x);

  /// \brief How many waypoints still have to be flown to?
  /// \return The number of waypoints still not reached.
  int waypointsLeft() {
    std::lock_guard<std::mutex> l(waypointMutex_);
    return waypoints_.size();
  }

  arp::kinematics::RobotState latestState;

 protected:

  ros::NodeHandle * nh_;  ///< ROS node handle.
  ros::Publisher pubReset_;  ///< The reset publisher -- use to reset the drone (e-stop).
  ros::Publisher pubTakeoff_;  ///< Publish to take-off the drone.
  ros::Publisher pubLand_;  ///< Publish to land the drone.
  ros::ServiceClient srvFlattrim_;  ///< To request a flat trim calibration.

  double ref_x_ = 0.0; ///< World frame x position reference [m].
  double ref_y_ = 0.0; ///< World frame y position reference [m].
  double ref_z_ = 0.0; ///< World frame z position reference [m].
  double ref_yaw_ = 0.0; ///< World frame yaw reference [rad].
  std::mutex refMutex_; ///< We need to lock the reference access due to asynchronous arrival.
  std::atomic<bool> isAutomatic_; ///< True, if in automatic control mode.

  std::deque<Waypoint> waypoints_;  ///< A list of waypoints that will be approached, if not empty.
  std::mutex waypointMutex_;  ///< We need to lock the waypoint access due to asynchronous arrival.
  ros::Publisher cmd_vel_pub;

};

} // namespace arp



#endif /* ARDRONE_PRACTICALS_INCLUDE_ARP_AUTOPILOT_HPP_ */
