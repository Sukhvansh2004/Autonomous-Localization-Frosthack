/*
 * Frontend.cpp
 *
 *  Created on: 9 Dec 2020
 *      Author: sleutene
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <ros/ros.h>

#include <brisk/brisk.h>

#include <arp/Frontend.hpp>

#ifndef CV_AA
#define CV_AA cv::LINE_AA // maintains backward compatibility with older OpenCV
#endif

#ifndef CV_BGR2GRAY
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY // maintains backward compatibility with older OpenCV
#endif

namespace arp
{

  Frontend::Frontend(int imageWidth, int imageHeight,
                     double focalLengthU, double focalLengthV,
                     double imageCenterU, double imageCenterV,
                     double k1, double k2, double p1, double p2, int uniformityRadius, int octaves, float absoluteThreshold, int maxNumKpt, float mappingCamFocalLength) : camera_(imageWidth, imageHeight, focalLengthU, focalLengthV, imageCenterU, imageCenterV, arp::cameras::RadialTangentialDistortion(k1, k2, p1, p2))
  {
    camera_.initialiseUndistortMaps();

    // also save for OpenCV RANSAC later
    cameraMatrix_ = cv::Mat::zeros(3, 3, CV_64FC1);
    cameraMatrix_.at<double>(0, 0) = focalLengthU;
    cameraMatrix_.at<double>(1, 1) = focalLengthV;
    cameraMatrix_.at<double>(0, 2) = imageCenterU;
    cameraMatrix_.at<double>(1, 2) = imageCenterV;
    cameraMatrix_.at<double>(2, 2) = 1.0;
    distCoeffs_ = cv::Mat::zeros(1, 4, CV_64FC1);
    distCoeffs_.at<double>(0) = k1;
    distCoeffs_.at<double>(1) = k2;
    distCoeffs_.at<double>(2) = p1;
    distCoeffs_.at<double>(3) = p2;

    // BRISK detector and descriptor
    detector_.reset(new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(uniformityRadius, octaves, absoluteThreshold, maxNumKpt));
    extractor_.reset(new brisk::BriskDescriptorExtractor(true, false));

    // leverage camera-aware BRISK (caution: needs the *_new* maps...)
    cv::Mat rays = cv::Mat(imageHeight, imageWidth, CV_32FC3);
    cv::Mat imageJacobians = cv::Mat(imageHeight, imageWidth, CV_32FC(6));
    for (int v = 0; v < imageHeight; ++v)
    {
      for (int u = 0; u < imageWidth; ++u)
      {
        Eigen::Vector3d ray;
        Eigen::Matrix<double, 2, 3> jacobian;
        if (camera_.backProject(Eigen::Vector2d(u, v), &ray))
        {
          ray.normalize();
        }
        else
        {
          ray.setZero();
        }
        rays.at<cv::Vec3f>(v, u) = cv::Vec3f(ray[0], ray[1], ray[2]);
        Eigen::Vector2d pt;
        if (camera_.project(ray, &pt, &jacobian) == cameras::ProjectionStatus::Successful)
        {
          cv::Vec6f j;
          j[0] = jacobian(0, 0);
          j[1] = jacobian(0, 1);
          j[2] = jacobian(0, 2);
          j[3] = jacobian(1, 0);
          j[4] = jacobian(1, 1);
          j[5] = jacobian(1, 2);
          imageJacobians.at<cv::Vec6f>(v, u) = j;
        }
      }
    }
    std::static_pointer_cast<cv::BriskDescriptorExtractor>(extractor_)->setCameraProperties(rays, imageJacobians, mappingCamFocalLength);
  }

  bool Frontend::loadMap(std::string path)
  {
    std::ifstream mapfile(path);
    if (!mapfile.good())
    {
      return false;
    }

    // read each line
    std::string line;
    std::set<uint64_t> lmIds;
    uint64_t poseId = 0;
    LandmarkVec landmarks;
    while (std::getline(mapfile, line))
    {

      // Convert to stringstream
      std::stringstream ss(line);

      if (0 == line.compare(0, 7, "frame: "))
      {
        // store previous set into map
        landmarks_[poseId] = landmarks;
        // get pose id:
        std::stringstream frameSs(line.substr(7, line.size() - 1));
        frameSs >> poseId;
        if (!frameSs.eof())
        {
          std::string covisStr;
          frameSs >> covisStr; // comma
          frameSs >> covisStr;
          if (0 == covisStr.compare("covisibilities:"))
          {
            while (!frameSs.eof())
            {
              uint64_t covisId;
              frameSs >> covisId;
              covisibilities_[poseId].insert(covisId);
            }
          }
        }
        // move to filling next set of landmarks
        landmarks.clear();
      }
      else
      {
        if (poseId > 0)
        {
          Landmark landmark;

          // get keypoint idx
          size_t keypointIdx;
          std::string keypointIdxString;
          std::getline(ss, keypointIdxString, ',');
          std::stringstream(keypointIdxString) >> keypointIdx;

          // get landmark id
          uint64_t landmarkId;
          std::string landmarkIdString;
          std::getline(ss, landmarkIdString, ',');
          std::stringstream(landmarkIdString) >> landmarkId;
          landmark.landmarkId = landmarkId;

          // read 3d position
          for (int i = 0; i < 3; ++i)
          {
            std::string coordString;
            std::getline(ss, coordString, ',');
            double coord;
            std::stringstream(coordString) >> coord;
            landmark.point[i] = coord;
          }

          // Get descriptor
          std::string descriptorstring;
          std::getline(ss, descriptorstring);
          landmark.descriptor = cv::Mat(1, 48, CV_8UC1);
          for (int col = 0; col < 48; ++col)
          {
            uint32_t byte;
            std::stringstream(descriptorstring.substr(2 * col, 2)) >> std::hex >> byte;
            landmark.descriptor.at<uchar>(0, col) = byte;
          }
          lmIds.insert(landmarkId);
          landmarks.push_back(landmark);
        }
      }
    }
    if (poseId > 0)
    {
      // store into map
      landmarks_[poseId] = landmarks;
    }
    std::cout << "loaded " << lmIds.size() << " landmarks from " << landmarks_.size() << " poses." << std::endl;
    return lmIds.size() > 0;
  }

  bool Frontend::loadDBoW2Voc(std::string path)
  {
    std::cout << "Loading DBoW2 vocabulary from " << path << std::endl;
    dBowVocabulary_.load(path);
    // Hand over vocabulary to dataset. false = do not use direct index:
    dBowDatabase_.setVocabulary(dBowVocabulary_, false, 0);
    std::cout << "loaded DBoW2 vocabulary with " << dBowVocabulary_.size() << " words." << std::endl;
    return true;
  }

  int Frontend::detectAndDescribe(
      const cv::Mat &grayscaleImage, const Eigen::Vector3d &extractionDirection,
      std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) const
  {

    // run BRISK detector
    detector_->detect(grayscaleImage, keypoints);

    // run BRISK descriptor extractor
    // orient the keypoints according to the extraction direction:
    Eigen::Vector3d ep;
    Eigen::Vector2d reprojection;
    Eigen::Matrix<double, 2, 3> Jacobian;
    Eigen::Vector2d eg_projected;
    for (size_t k = 0; k < keypoints.size(); ++k)
    {
      cv::KeyPoint &ckp = keypoints[k];
      const Eigen::Vector2d kp(ckp.pt.x, ckp.pt.y);
      // project ray
      camera_.backProject(kp, &ep);
      // obtain image Jacobian
      camera_.project(ep + extractionDirection.normalized() * 0.001, &reprojection);
      // multiply with gravity direction
      eg_projected = reprojection - kp;
      double angle = atan2(eg_projected[1], eg_projected[0]);
      // set
      ckp.angle = angle / M_PI * 180.0;
    }
    extractor_->compute(grayscaleImage, keypoints, descriptors);

    return keypoints.size();
  }

  bool Frontend::ransac(const std::vector<cv::Point3d> &worldPoints,
                        const std::vector<cv::Point2d> &imagePoints,
                        kinematics::Transformation &T_CW, std::vector<int> &inliers,
                        bool requireRansac) const
  {
    if (worldPoints.size() != imagePoints.size())
    {
      return false;
    }

    inliers.clear();

    // For re-initialization, we need at least 5 points for reliable RANSAC
    if (requireRansac && worldPoints.size() < 5)
    {
      return false;
    }

    // For tracking, we need at least 2 points
    if (worldPoints.size() < 2)
    {
      return false;
    }

    cv::Mat rvec, tvec;
    bool ransacSuccess = false;

    // Do RANSAC if we have enough points
    if (worldPoints.size() >= 5)
    {
      ransacSuccess = cv::solvePnPRansac(
          worldPoints, imagePoints, cameraMatrix_, distCoeffs_,
          rvec, tvec, false, 500, 15.0, 0.80, inliers, cv::SOLVEPNP_EPNP);
    }
    else if (!requireRansac)
    {
      // Mark all points as inliers
      inliers.resize(worldPoints.size());
      std::iota(inliers.begin(), inliers.end(), 0);
      return true;
    }

    if (!ransacSuccess)
    {
      return false;
    }

    // Set pose
    cv::Mat R = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Rodrigues(rvec, R);
    Eigen::Matrix4d T_CW_mat = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; i++)
    {
      T_CW_mat(i, 3) = tvec.at<double>(i);
      for (int j = 0; j < 3; j++)
      {
        T_CW_mat(i, j) = R.at<double>(i, j);
      }
    }
    T_CW = kinematics::Transformation(T_CW_mat);

    // For re-initialization, require at least 70% inliers
    if (requireRansac)
    {
      return (double(inliers.size()) / double(imagePoints.size()) > 0.7);
    }

    // For tracking, accept any valid pose
    return true;
  }

  bool Frontend::assignDBoW2Histograms()
  {
    unsigned i = 0;

    for (const auto&[poseID, landmarks] : landmarks_)
    {
      std::vector<std::vector<unsigned char>> features;
      features.reserve(landmarks.size());
      for(const auto& landmark : landmarks)
      {
        features.push_back(landmark.descriptor);
      }
      dBowDatabase_.add(features);
      dBowDatabaseIndicesToPoseID.push_back(poseID);
    }

    return true;
  }

  bool Frontend::detectAndMatch(const cv::Mat &image, const Eigen::Vector3d &extractionDirection,
                                DetectionVec &detections, kinematics::Transformation &T_CW,
                                cv::Mat &visualisationImage, bool needsReInitialisation)
  {
    constexpr float DIST_THRESHOLD = 60;
    constexpr float MAX_ACCEPTABLE_PIXEL_DISTANCE = 30;

    detections.clear(); // make sure empty

    // to gray:
    cv::Mat grayScale;
    cv::cvtColor(image, grayScale, CV_BGR2GRAY);

    // run BRISK detector and descriptor extractor:
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detectAndDescribe(grayScale, extractionDirection, keypoints, descriptors);

    if(needsReInitialisation)
    {
      std::vector<std::vector<unsigned char>> features;
      for (size_t k = 0; k < keypoints.size(); ++k)
      {                                                        // go through all keypoints in the frame
        uchar *keypointDescriptor = descriptors.data + k * 48; // descriptors are 48 bytes long
        features.push_back(std::vector<unsigned char>(keypointDescriptor, keypointDescriptor+48));
      }
      
      DBoW2::QueryResults dBoWResult;
      dBowDatabase_.query(features, dBoWResult, 80);

      std::vector<unsigned> similarPoses;
      similarPoses.reserve(dBoWResult.size());

      for(const auto& result : dBoWResult)
      {
        similarPoses.push_back(dBowDatabaseIndicesToPoseID[result.Id]);
      }

      //Task 2. Step 1. Identify Active Keyframe (Most Matches)

      int maxMatches = 0; //counts max amount of matches      
      for (const auto &poseID : similarPoses) {      //loop over each similar pose and count
        int matchCount = 0;  //variable  to count matches
          for (const auto &lm : landmarks_[poseID]) { // go through all landmarks seen from this pose
            for (size_t k = 0; k < keypoints.size(); ++k) { //go through all keypoints in the frame
              uchar *keypointDescriptor = descriptors.data + k * 48; // descriptors are 48 bytes long
              const float dist = brisk::Hamming::PopcntofXORed(  //compute hamming distance 
                keypointDescriptor, lm.descriptor.data, 3); // compute desc. distance: 3 for 3x128bit (=48 bytes)
              // If distance is within given threshold, increase match count
              if (dist < DIST_THRESHOLD) {
                  matchCount++;
              }
          }
        }

        if (matchCount > maxMatches) {  //poseID is new active keyframe if it has most matches
            maxMatches = matchCount;
            activeKeyframe = poseID;  
        }
      }


      candidateKeyframes.clear();  //   `candidateKeyframes = []` (List to store keyframes we will use for matching).
      candidateKeyframes.push_back(activeKeyframe); // Always include the best keyframe

      if (covisibilities_.count(activeKeyframe) > 0) { //if activekeyframe is in coVis
        candidateKeyframes.reserve(covisibilities_[activeKeyframe].size());
        for (const auto &covisKeyframe : covisibilities_[activeKeyframe]) {  //loop through each co-vis keyframe
            candidateKeyframes.push_back(covisKeyframe);   //Adds co_vis to canditateKeyframes
        }
      }
    }
  
    //Task 2. Step 2 "try matching map keyframes co-visible with it and you will likely be able to
    // find more matches"
    // match co-visible keyframes form coVisibilities against each keyframe


    // Vectors that store 2d-3d point correspondences
    std::vector<cv::Point2d> imagePoints;
    std::vector<cv::Point3d> worldPoints;
    std::vector<uint64_t> landmarkIds;

    // match to map:
    for (const auto &poseID : candidateKeyframes)
    { // go through all poses
      LandmarkVec lms = landmarks_[poseID];
      for (const auto &lm : lms)
      { // go through all landmarks seen from this pose
        for (size_t k = 0; k < keypoints.size(); ++k)
        {                                                        // go through all keypoints in the frame
          uchar *keypointDescriptor = descriptors.data + k * 48; // descriptors are 48 bytes long
          const float dist = brisk::Hamming::PopcntofXORed(
              keypointDescriptor, lm.descriptor.data, 3); // compute desc. distance: 3 for 3x128bit (=48 bytes)

          if (dist < DIST_THRESHOLD)
          {
            //Task 2. Step 3: Rule out potential matches
            //reason: not projecting in live frame; too far away

            // The following distinction does not make it better in our experiments...
            if (false)
            {
              // Obtain pose prior here
              Eigen::Vector2d projectedPoint;
              Eigen::Vector3d worldPoint = lm.point;
              Eigen::Vector2d imgPointEigen(keypoints.at(k).pt.x,keypoints.at(k).pt.y);
              arp::cameras::ProjectionStatus projectionStatus = camera_.project(T_CW.R() * worldPoint + T_CW.t(), &projectedPoint);

              if (projectionStatus == arp::cameras::ProjectionStatus::Successful && (imgPointEigen-projectedPoint).norm() < MAX_ACCEPTABLE_PIXEL_DISTANCE)
              {
                // Save corresponding 3d points in world frame along the landmarkIDs
                Eigen::Vector3d wP = lm.point;
                worldPoints.push_back(cv::Point3d(wP.x(), wP.y(), wP.z()));
                landmarkIds.push_back(lm.landmarkId);

                // Save 2d image points coordinates from keypoint
                imagePoints.push_back(keypoints.at(k).pt);
              }
            }
            else
            {
              // Save corresponding 3d points in world frame along the landmarkIDs
              Eigen::Vector3d wP = lm.point;
              worldPoints.push_back(cv::Point3d(wP.x(), wP.y(), wP.z()));
              landmarkIds.push_back(lm.landmarkId);

              // Save 2d image points coordinates from keypoint
              imagePoints.push_back(keypoints.at(k).pt);
            }
          }
        }
      }
    }

    // run RANSAC (to remove outliers and get pose T_CW estimate)
    std::vector<int> inliers;
    bool ransacSuccess = ransac(worldPoints, imagePoints, T_CW, inliers, needsReInitialisation);

    if (ransacSuccess)
    {
      // Reserve space to avoid re-allocations
      detections.reserve(inliers.size());

      // set detections
      for (const auto &idx : inliers)
      {
        detections.push_back(
            {Eigen::Vector2d(imagePoints.at(idx).x, imagePoints.at(idx).y),
             Eigen::Vector3d(worldPoints.at(idx).x, worldPoints.at(idx).y, worldPoints.at(idx).z),
             landmarkIds.at(idx)});
      }

      // visualise by painting stuff into visualisationImage

      for (const auto &point : keypoints)
      {
        cv::circle(visualisationImage, point.pt, 4, cv::Scalar(0, 0, 255));
      }

      for (const auto &detection : detections)
      {
        cv::circle(visualisationImage, cv::Point2d(detection.keypoint.x(), detection.keypoint.y()), 4, cv::Scalar(255, 0, 0));
      }
      
      for (const auto &cvPointW : worldPoints)
      {
        Eigen::Vector2d projectedPoint;
        Eigen::Vector3d worldPoint(cvPointW.x, cvPointW.y, cvPointW.z);
        arp::cameras::ProjectionStatus projectionStatus = camera_.project(T_CW.R() * worldPoint + T_CW.t(), &projectedPoint);

        if (projectionStatus == arp::cameras::ProjectionStatus::Successful)
        {
          cv::circle(visualisationImage, cv::Point2d(projectedPoint.x(), projectedPoint.y()), 4, cv::Scalar(0, 255, 0));
        }
      }
    }

    return ransacSuccess; // return true if successful...
  }

} // namespace arp