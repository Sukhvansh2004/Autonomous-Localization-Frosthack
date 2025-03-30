#include <arp/Planner.hpp>
#include <ros/ros.h>
#include <fstream>
#include <limits>
#include <cmath>

namespace arp
{
  Planner::Planner(const std::string &filepath, const cv::Vec3d &goal_pos, const cv::Vec3d &start_pos)
  {
    // open the file:
    std::ifstream mapFile(filepath, std::ios::in | std::ios::binary);
    if (!mapFile.is_open())
    {
      ROS_FATAL_STREAM("could not open map file " << filepath);
      return;
    }

    // first read the map size along all the dimensions:
    if (!mapFile.read((char *)map_sizes, 3 * sizeof(int)))
    {
      ROS_FATAL_STREAM("could not read map file " << filepath);
      return;
    }

    // now read the map data: don’t forget to delete[] in the end!
    map_data = std::vector<char>(map_sizes[0] * map_sizes[1] * map_sizes[2]);

    if (!mapFile.read(map_data.data(), map_sizes[0] * map_sizes[1] * map_sizes[2]))
    {
      ROS_FATAL_STREAM("could not read map file " << filepath);
    }
    mapFile.close();
    start = drone_to_map_coordinates(start_pos);
    goal = drone_to_map_coordinates(goal_pos);

    // now wrap it with a cv::Mat for easier access:
    wrappedMapData = cv::Mat(3, map_sizes, CV_8SC1, map_data.data());
    initialized = true;
  }

  Planner::Planner(const std::string &filepath)
  {
    // open the file:
    std::ifstream mapFile(filepath, std::ios::in | std::ios::binary);
    if (!mapFile.is_open())
    {
      ROS_FATAL_STREAM("could not open map file " << filepath);
      return;
    }

    // first read the map size along all the dimensions:
    if (!mapFile.read((char *)map_sizes, 3 * sizeof(int)))
    {
      ROS_FATAL_STREAM("could not read map file " << filepath);
      return;
    }

    // now read the map data: don’t forget to delete[] in the end!
    map_data = std::vector<char>(map_sizes[0] * map_sizes[1] * map_sizes[2]);

    if (!mapFile.read(map_data.data(), map_sizes[0] * map_sizes[1] * map_sizes[2]))
    {
      ROS_FATAL_STREAM("could not read map file " << filepath);
    }
    mapFile.close();

    // now wrap it with a cv::Mat for easier access:
    wrappedMapData = cv::Mat(3, map_sizes, CV_8SC1, map_data.data());
  }

  Planner::Node::Node(const int &x, const int &y, const int &z, const bool &initChildCoords)
      : position(x, y, z),
        distance(std::numeric_limits<double>::infinity()),
        totalDistanceEstimate(std::numeric_limits<double>::infinity())
  {
    if (initChildCoords)
    {
      // All 26 neighbouring nodes
      childrenCoords = {cv::Vec3i(x + 1, y, z),
                        cv::Vec3i(x - 1, y, z),
                        cv::Vec3i(x, y + 1, z),
                        cv::Vec3i(x, y - 1, z),
                        cv::Vec3i(x + 1, y + 1, z),
                        cv::Vec3i(x - 1, y - 1, z),
                        cv::Vec3i(x + 1, y - 1, z),
                        cv::Vec3i(x - 1, y + 1, z),
                        cv::Vec3i(x + 1, y, z + 1),
                        cv::Vec3i(x - 1, y, z + 1),
                        cv::Vec3i(x, y + 1, z + 1),
                        cv::Vec3i(x, y - 1, z + 1),
                        cv::Vec3i(x + 1, y + 1, z + 1),
                        cv::Vec3i(x - 1, y - 1, z + 1),
                        cv::Vec3i(x + 1, y - 1, z + 1),
                        cv::Vec3i(x - 1, y + 1, z + 1),
                        cv::Vec3i(x + 1, y, z - 1),
                        cv::Vec3i(x - 1, y, z - 1),
                        cv::Vec3i(x, y + 1, z - 1),
                        cv::Vec3i(x, y - 1, z - 1),
                        cv::Vec3i(x + 1, y + 1, z - 1),
                        cv::Vec3i(x - 1, y - 1, z - 1),
                        cv::Vec3i(x + 1, y - 1, z - 1),
                        cv::Vec3i(x - 1, y + 1, z - 1),
                        cv::Vec3i(x, y, z + 1),
                        cv::Vec3i(x, y, z - 1)};
    }
  }

  Planner::Node::Node(const cv::Vec3i &pos, const double &distance, const double &totalDistanceEstimate, const cv::Vec3i &previousNodeCoords, const bool &initChildCoords)
      : Node(pos[0], pos[1], pos[2], initChildCoords)
  {
    this->distance = distance;
    this->totalDistanceEstimate = totalDistanceEstimate;
    this->previousNode = previousNodeCoords;
  }

  Planner::Node::Node(const cv::Vec3i &pos, const bool &initChildCoords)
      : Node(pos[0], pos[1], pos[2], initChildCoords)
  {
  }

  cv::Vec3i Planner::drone_to_map_coordinates(const cv::Vec3d &droneCoordinates)
  {
    const int i = std::round(droneCoordinates[0] / 0.1 + double(map_sizes[0] - 1) / 2.0);
    const int j = std::round(droneCoordinates[1] / 0.1 + double(map_sizes[1] - 1) / 2.0);
    const int k = std::round(droneCoordinates[2] / 0.1 + double(map_sizes[2] - 1) / 2.0);

    return cv::Vec3i(i, j, k);
  }

  cv::Vec3d Planner::map_to_drone_coordinates(const cv::Vec3i &mapCoordinates)
  {
    const double i = (mapCoordinates[0] - double(map_sizes[0] - 1) / 2.0) * 0.1;
    const double j = (mapCoordinates[1] - double(map_sizes[1] - 1) / 2.0) * 0.1;
    const double k = (mapCoordinates[2] - double(map_sizes[2] - 1) / 2.0) * 0.1;

    return cv::Vec3d(i, j, k);
  }

  void Planner::generate2DOccupancyGrid()
  {
    int constant_z = std::round(double(map_sizes[2] - 1) / 2.0) + 2;
    for (int x = 0; x < map_sizes[0]; x++)
    {
      for(int y = 0; y < map_sizes[1]; y++)
      {
        cv::Vec3i pos(x, y, constant_z);
        if(get_probability(cv::Vec3i(x, y, constant_z))>=probability_threshold)
        {
          occupancy2D.insert(pos);
        }
      }
    }
    std::cout<<"2d grid size: "<<occupancy2D.size()<<std::endl;
  }

  double Planner::get_probability(const cv::Vec3i &position) const
  {
    if (!coords_in_map(position))
    {
      // Return some probability for a certain collision
      return 10 * probability_threshold;
    }

    int8_t log_odds = wrappedMapData.at<int8_t>(position[0], position[1], position[2]);

    double logOdds = static_cast<double>(log_odds);

    /*double scalingFactor = 127.0; // Scaling factor needs to be determined
    logOdds /= scalingFactor;

    double probability = 1.0 / (1.0 + std::exp(-logOdds));
    logOdds *= scalingFactor;

    double probability_unscaled = 1.0 / (1.0 + std::exp(-logOdds));*/
    return logOdds;
  }

  void Planner::conduct_planning(const cv::Vec3d &goal_pos, const cv::Vec3d &start_pos)
  {
    start = drone_to_map_coordinates(start_pos);
    goal = drone_to_map_coordinates(goal_pos);

    cv::Vec2d directionVec(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]);
    double yawAngle = std::atan2(directionVec[1], directionVec[0]);

    planning_start_time = std::chrono::high_resolution_clock::now();
    std::set<Node> openSet;
    openSet.clear();
    start.distance = 0;
    start.totalDistanceEstimate = distance(start, goal);
    openSet.insert(start);

    graph.clear();
    graph.insert(start);
    graph.insert(goal);

    while (!openSet.empty())
    {
      const Node closestNode = *openSet.begin(); // Get node with smallest distance (copy is mandatory)
      openSet.erase(closestNode);

      if (closestNode == goal)
      {
        this->waypoints_async_planning = obtain_waypoints(closestNode, yawAngle);
        planning_end_time = std::chrono::high_resolution_clock::now();
        return;
      }

      for (const auto &neighbourCoords : closestNode.childrenCoords)
      {
        double alternativeDistance = closestNode.distance + distance(closestNode.position, neighbourCoords);

        const auto &neighbourNodeIterator = graph.find(Node(neighbourCoords, false)); // Runs in O(1)

        if (neighbourNodeIterator == graph.end() && get_probability(neighbourCoords) < probability_threshold)
        {
          // Node does not exist yet in graph but needs to be added
          Node updatedNode = Node(neighbourCoords, alternativeDistance, alternativeDistance + heuristic(updatedNode, goal), closestNode.position, true);
          openSet.insert(updatedNode);
          graph.insert(updatedNode);
        }
        else if (neighbourNodeIterator != graph.end())
        {
          if (alternativeDistance < neighbourNodeIterator->distance && get_probability(neighbourCoords) < probability_threshold)
          {
            Node updatedNode = *neighbourNodeIterator;
            updatedNode.distance = alternativeDistance;
            updatedNode.totalDistanceEstimate = alternativeDistance + heuristic(updatedNode, goal);
            updatedNode.previousNode = closestNode.position;

            openSet.erase(updatedNode); // Not sure if this is necessary
            openSet.insert(updatedNode);

            graph.erase(updatedNode);
            graph.insert(updatedNode);
          }
        }
      }
    }
    this->waypoints_async_planning = {};
    planning_end_time = std::chrono::high_resolution_clock::now();
    std::cerr << "Error! No suitable path found." << std::endl;
  }

  std::deque<Waypoint> Planner::obtain_waypoints(const Node &goalNode, const double &yaw)
  {
    std::deque<Waypoint> path;

    if (goalNode == start && start != goal)
    {
      std::cerr << "Error! No suitable path found." << std::endl;
      return {};
    }

    std::cout << "Found best path with distance: " << goalNode.distance * 10 << "cm" << std::endl;

    int i = 0;
    const Node *currNode = &goalNode;
    while (*currNode != start)
    {
      cv::Vec3d dronePosition = map_to_drone_coordinates(currNode->position);
      Waypoint wp;
      wp.x = dronePosition[0];
      wp.y = dronePosition[1];
      wp.z = dronePosition[2];
      wp.yaw = yaw;
      // Clamping lambda as std::clamp is only available at C++17
      auto clamp = [](double value, double min, double max)
      {
        return std::max(min, std::min(value, max));
      };
      wp.z = clamp(wp.z, 0.5, 10);
      wp.posTolerance = 0.2; // Grid size is 10cm - so an error of 5cm in every direction is safe

      if (i % 2 == 0)
      {
        // wp.yaw = yaw + M_PI/6;          // TODO: Change this to something meaningfull - maybe not even in this function
      }
      else
      {
        // wp.yaw = yaw - M_PI/6;          // TODO: Change this to something meaningfull - maybe not even in this function
      }

      path.push_front(wp);
      const auto &prevNodeIterator = graph.find(currNode->previousNode);

      if (prevNodeIterator == graph.end())
      {
        // Chain was corrupted - this should not happen!
        throw std::runtime_error("Planning chain was corrupted - automatic route planning must not continue!");
      }

      i++;
      currNode = &(*prevNodeIterator);
    }
    return path;
  }

  std::deque<Waypoint> Planner::getWaypoints()
  {
    std::chrono::duration<double> elapsed_seconds = planning_end_time - planning_start_time;
    std::cout << "Total waypoints computed: " << waypoints_async_planning.size() << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    return waypoints_async_planning;
  }

  double Planner::heuristic(const Node &curr, const Node &goal) const
  {
    return distance(curr, goal);
  }

  double Planner::distance(const Node &n1, const Node &n2) const
  {
    return cv::norm(n2.position - n1.position);
  }

  double Planner::distance(const cv::Vec2d &n1, const cv::Vec2d &n2) const
  {
    return cv::norm(n2 - n1);
  }

  bool Planner::coords_in_map(const cv::Vec3i &coords) const
  {
    return coords[0] < map_sizes[0] && coords[0] >= 0 && coords[1] < map_sizes[1] && coords[1] >= 0 && coords[2] < map_sizes[2] && coords[2] >= 0;
  }

} // namespace arp
