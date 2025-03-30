#ifndef ARDRONE_PRACTICALS_INCLUDE_ARP_PLANNER_HPP_
#define ARDRONE_PRACTICALS_INCLUDE_ARP_PLANNER_HPP_

#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <chrono>
#include <atomic>
#include <unordered_set>
#include <opencv2/core/core.hpp>

namespace arp
{
  /// \brief A Helper struct to send lists of waypoints.
  struct Waypoint
  {
    double x;            ///< The World frame x coordinate.
    double y;            ///< The World frame y coordinate.
    double z;            ///< The World frame z coordinate.
    double yaw;          ///< The yaw angle of the robot w.r.t. the World frame.
    double posTolerance; ///< The position tolerance: if within, it's considered reached.
  };

  class Planner
  {
  public:
    struct Node
    {
      cv::Vec3i position;
      double distance;
      double totalDistanceEstimate;
      std::vector<cv::Vec3i> childrenCoords;
      cv::Vec3i previousNode;

      Node(const int &x, const int &y, const int &z, const bool &initChildCoords = true);
      Node(const cv::Vec3i &pos, const double &distance, const double &totalDistanceEstimate, const cv::Vec3i &previousNodeCoords, const bool &initChildCoords = true);
      Node(const cv::Vec3i &pos, const bool &initChildCoords = true);
      Node() = default;

      // Overwrite equality operator for Node
      bool operator==(const Node &other) const
      {
        return position == other.position;
      }

      // Overwrite equality operator for Node
      bool operator!=(const Node &other) const
      {
        return !(*this == other);
      }

      bool operator<(const Node &other) const
      {
        return totalDistanceEstimate < other.totalDistanceEstimate;
      }
    };

    Planner(const std::string &filepath, const cv::Vec3d &goal_pos, const cv::Vec3d &start_pos = cv::Vec3d(0, 0, 1.0));
    Planner(const std::string &filepath);

    void conduct_planning(const cv::Vec3d &goal_pos, const cv::Vec3d &start_pos = cv::Vec3d(0, 0, 1.0));
    void generate2DOccupancyGrid();

    std::deque<Waypoint> getWaypoints();

  private:
    std::deque<Waypoint> obtain_waypoints(const Node &goalNode, const double& yaw);

    cv::Vec3i drone_to_map_coordinates(const cv::Vec3d &droneCoordinates);
    cv::Vec3d map_to_drone_coordinates(const cv::Vec3i &mapCoordinates);

    double get_probability(const cv::Vec3i &position) const;

    double heuristic(const Node &curr, const Node &goal) const;
    double distance(const Node &n1, const Node &n2) const;
    double distance(const cv::Vec2d &n1, const cv::Vec2d &n2) const;

    bool coords_in_map(const cv::Vec3i &coords) const;

    // Custom hash for Node to use in std::unordered_set
    struct Hash_Node
    {
      // Combining hashes according to https://stackoverflow.com/questions/16792751/hashmap-for-2d3d-coordinates-i-e-vector-of-doubles
      size_t operator()(const Node &k) const
      {
        size_t h1 = std::hash<int>()(k.position[0]);
        size_t h2 = std::hash<int>()(k.position[1]);
        size_t h3 = std::hash<int>()(k.position[2]);
        return (h1 ^ (h2 << 1)) ^ h3;
      }

      // For vec3i we only look at the 2d indices -> 2d occupancy map
      size_t operator()(const cv::Vec3i &k) const
      {
        size_t h1 = std::hash<int>()(k[0]);
        size_t h2 = std::hash<int>()(k[1]);
        return (h1 ^ (h2 << 1));
      }
    };

    std::unordered_set<Node, Hash_Node> graph;
    std::unordered_set<cv::Vec3i, Hash_Node> occupancy2D;

    bool initialized = false;
    int map_sizes[3];
    std::vector<char> map_data;
    cv::Mat wrappedMapData;
    Node start;
    Node goal;

    // For async planning

    std::deque<Waypoint> waypoints_async_planning;

    std::chrono::system_clock::time_point planning_start_time;
    std::chrono::system_clock::time_point planning_end_time;

    const double probability_threshold = 0;
  };
} // namespace arp

#endif /* ARDRONE_PRACTICALS_INCLUDE_ARP_PLANNER_HPP_ */
