# Vision-Based Localization Framework with OKVIS

This repository contains a vision-based localization framework that leverages the [OKVIS (Open Keyframe-based Visual-Inertial SLAM)](https://github.com/ethz-asl/okvis) framework to perform robust localization in pre-mapped, sparse visual environments. The system integrates visual localization with inertial measurements using an Extended Kalman Filter (EKF), ensuring improved robustness and accuracy in pose estimation. Key components include feature extraction with BRISK, bag-of-visual-words for efficient keypoint matching, and a robust initialization using RANSAC combined with a Perspective-n-Point (PnP) algorithm.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Technical Details](#technical-details)
  - [Visual Localization](#visual-localization)
  - [Inertial Integration with EKF](#inertial-integration-with-ekf)
  - [Keypoint Detection and Matching](#keypoint-detection-and-matching)
  - [Bag-of-Visual-Words](#bag-of-visual-words)
  - [Robust Pose Initialization (RANSAC + PnP)](#robust-pose-initialization-ransac--pnp)
- [Results and Demonstration](#results-and-demonstration)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This framework is designed to localize a camera in a previously built sparse visual map. It exploits the synergy between visual information and inertial data to produce a robust and accurate pose estimate. The visual component uses the OKVIS pipeline to extract keyframes and features, while the inertial component is integrated through an Extended Kalman Filter to compensate for rapid motions and temporary visual degradations.

---

## Key Features

- **Pre-built Visual Map:** Localization within an already mapped sparse visual environment.
- **Visual-Inertial Fusion:** Seamless integration of camera and IMU data via an Extended Kalman Filter.
- **BRISK Keypoint Descriptors:** Fast and robust keypoint detection and description.
- **Bag-of-Visual-Words:** Efficient keypoint matching for large-scale localization.
- **Robust Pose Estimation:** Combination of RANSAC and PnP algorithms to reliably initialize the camera pose.
- **Real-Time Performance:** Optimized for on-line operation in dynamic environments.

---

## System Architecture

The framework is organized into the following core modules:

1. **Visual Localization Module:**  
   - Processes incoming camera frames.  
   - Extracts BRISK keypoints and computes descriptors.  
   - Matches features against the pre-built map using a bag-of-visual-words approach.  

2. **Inertial Integration Module:**  
   - Reads and preprocesses IMU data.  
   - Fuses visual and inertial measurements using an Extended Kalman Filter (EKF) for smoothing and drift correction.  

3. **Pose Estimation Module:**  
   - Utilizes RANSAC to robustly identify inliers from feature matches.  
   - Applies the PnP algorithm on the inliers to compute an initial pose estimate.  
   - The initial pose is refined using visual-inertial fusion.  

4. **Data Management and Visualization Module:**  
   - Handles storage and retrieval of map data.  
   - Provides real-time visualization of the localization process (optional integration with tools such as RViz).  

---

## Installation

### Prerequisites
- **Operating System:** Linux (Ubuntu recommended) or macOS.
- **C++ Compiler:** Supporting C++11 or later.
- **ROS:** (Robot Operating System) if using visualization or additional sensor integration.
- **Dependencies:**  
  - OpenCV (>=3.0)  
  - Eigen3  
  - Boost  
  - [OKVIS dependencies](https://github.com/ethz-asl/okvis#dependencies)  

### Build Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/visual-localization-framework.git
   cd visual-localization-framework
Setup Environment (if using ROS):

```bash
source /opt/ros/<ros-distro>/setup.bash
```

2. **Install Dependencies:**

```bash
sudo apt-get update
sudo apt-get install libopencv-dev libeigen3-dev libboost-all-dev
```

3. **Build the Project:**

```bash
mkdir build && cd build
cmake ..
make -j4
```

4. **Configuration**
Visual Map:
Update the configuration file (config/map_config.yaml) to point to the pre-built sparse visual map files.

IMU Parameters:
Adjust the IMU noise and bias parameters in config/imu_config.yaml to suit your sensor characteristics.

Feature Extraction Settings:
Configure BRISK parameters such as threshold and octaves in config/feature_config.yaml.

Bag-of-Words Settings:
Set vocabulary and matching thresholds in config/bovw_config.yaml.

Usage
Running Localization
```bash
Edit
./bin/localization_node --map config/map_config.yaml --imu config/imu_config.yaml --features config/feature_config.yaml --bovw config/bovw_config.yaml
```

5. **Visualization**
If using ROS for visualization:

```bash
roslaunch visual_localization rviz.launch
```

6. **Debugging and Logs**
Logging is integrated using spdlog or ROS logging. Adjust logging levels in config/logging_config.yaml.


### Technical Details

Visual Localization
Keyframe-Based Approach: Uses keyframes to efficiently query the pre-built map.

Feature Extraction: Uses BRISK for rapid keypoint detection and description.

Inertial Integration with EKF
Extended Kalman Filter (EKF): Fuses IMU data with visual pose estimates to maintain accuracy during rapid movements.

State Propagation: IMU data is used to propagate the state estimate between visual updates.

Keypoint Detection and Matching
BRISK Descriptors: Fast binary descriptors robust to scale and moderate rotations.

Bag-of-Visual-Words (BoVW): Encodes visual information into a compact representation, enabling rapid matching.

Robust Pose Initialization (RANSAC + PnP)
RANSAC: Robustly identifies inliers from feature matches.

PnP Algorithm: Computes the initial pose from inliers and refines it through visual-inertial fusion.

Results and Demonstration
The demonstration video showcases:

Localization Accuracy: Accurate pose initialization under challenging conditions.

Real-Time Performance: Visual overlays indicating tracking and pose consistency.
