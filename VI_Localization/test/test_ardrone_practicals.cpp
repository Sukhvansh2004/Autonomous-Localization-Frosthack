// Bring in my package's API, which is what I'm testing
#include "arp/cameras/PinholeCamera.hpp"
#include "arp/cameras/RadialTangentialDistortion.hpp"

// Bring test utils
#include "utils/test_utils.hpp"

// Bring in gtest
#include <gtest/gtest.h>

#include <iostream>

// Test 1: Test the projection and unprojection
TEST(PinholeCamera, projectBackProject)
{
    // create an arbitrary camera model
    arp::cameras::PinholeCamera<arp::cameras::RadialTangentialDistortion> pinholeCamera =
        arp::cameras::PinholeCamera<arp::cameras::RadialTangentialDistortion>::testObject();

    // create a random visible point in the camera coordinate frame C
    auto point_C = pinholeCamera.createRandomVisiblePoint();

    // project
    Eigen::Vector2d imagePoint;
    pinholeCamera.project(point_C, &imagePoint);

    // backProject
    Eigen::Vector3d ray_C;
    pinholeCamera.backProject(imagePoint, &ray_C);

    // now they should align:
    EXPECT_TRUE(fabs(ray_C.normalized().transpose() * point_C.normalized() - 1.0) < 1.0e-10);
}

// Test 2: Projection/Backprojection of Jacobian based method
TEST(PinholeCamera, projectBackProjectJacobian)
{
    // create an arbitrary camera model
    arp::cameras::PinholeCamera<arp::cameras::RadialTangentialDistortion> pinholeCamera =
        arp::cameras::PinholeCamera<arp::cameras::RadialTangentialDistortion>::testObject();

    // create a random visible point in the camera coordinate frame C
    auto point_C = pinholeCamera.createRandomVisiblePoint();

    // project
    Eigen::Vector2d imagePoint;
    Eigen::Matrix<double, 2, 3> pointJacobian; // Matrix for Jacobian u,v wrt. x,y,z
    pinholeCamera.project(point_C, &imagePoint, &pointJacobian);

    // backProject
    Eigen::Vector3d ray_C;
    pinholeCamera.backProject(imagePoint, &ray_C);

    // now they should align:
    EXPECT_TRUE(fabs(ray_C.normalized().transpose() * point_C.normalized() - 1.0) < 1.0e-10);
}

// Test 3: Test Jacobian obtained by RadialTangentialDistortionModel
//  numeric difference versions: compare analytical Jacobians (derived by formulas) to numerical JAcobians (approximation)
TEST(RadialTangentialDistortion, DistortionJacobian)
{
    // Instantiate distortion model
    arp::cameras::RadialTangentialDistortion distortion_Model = arp::cameras::RadialTangentialDistortion::testObject();

    Eigen::Matrix<double, 2, 2> analyticJacobian;
    Eigen::Vector2d image_point;
    image_point << 0.42, 0.69;

    Eigen::Vector2d distorted_point;
    distortion_Model.distort(image_point, &distorted_point, &analyticJacobian);

    // Define a lambda function to compute numerical Jacobian
    auto projectionFunc = [&](const Eigen::Vector2d &x) -> Eigen::VectorXd
    {
        Eigen::Vector2d distorted;               // define a 2D vector (immage coordinates)
        distortion_Model.distort(x, &distorted); // call project function
        return distorted;                        // return 2d point
    };

    // Compute numerical Jacobian
    // 2x2 Matrix called numerical Jacobian
    Eigen::Matrix<double, 2, 2> numericalJacobian = testutils::calculateNumericalJacobian(projectionFunc, image_point);

    // Check if analytical and numerical Jacobians are close
    if ((analyticJacobian - numericalJacobian).norm() >= 1e-5)
    {
        std::cout << "Analytical Distortion Jacobian:\n"
                  << analyticJacobian << std::endl;
        std::cout << "Numerical Distortion Jacobian:\n"
                  << numericalJacobian << std::endl;
        std::cout << "Difference Norm: " << (analyticJacobian - numericalJacobian).norm() << std::endl;
    }

    // Check if analytical and numerical Jacobians are close
    EXPECT_TRUE((analyticJacobian - numericalJacobian).norm() < 1e-5);
}

// Test 4: Test Jacobian obtained by projection method
// select random 3D point, compute analytical Jacobian, compute numerical Jacobian, Compare Jacobians
// Tests the projection Jacobian against numerical Jacobian
TEST(PinholeCamera, ProjectionJacobian)
{
    // Instantiate a camera model
    // create an arbitrary pinholecamera model (with Ra.Ta. distortion)
    arp::cameras::PinholeCamera<arp::cameras::RadialTangentialDistortion> camera =
        arp::cameras::PinholeCamera<arp::cameras::RadialTangentialDistortion>::testObject();

    // Generate a random 3D point
    // create a random visible point in the camera coordinate frame C
    Eigen::Vector3d point_C = camera.createRandomVisiblePoint();

    // Project the point and obtain the analytical Jacobian
    Eigen::Vector2d imagePoint;                           // image point stores 2 coordinates as a Vector
    Eigen::Matrix<double, 2, 3> pointJacobian;            // Matrix for Jacobian u,v wrt. x,y,z
    camera.project(point_C, &imagePoint, &pointJacobian); // project function for class pinholecamera
    //& get adress of a variable: pointer, and pass the memory adress to pin.Ca..

    // Define a lambda function to compute numerical Jacobian
    auto projectionFunc = [&](const Eigen::VectorXd &x) -> Eigen::VectorXd
    {
        Eigen::Vector2d proj;     // define a 2D vector (immage coordinates)
        camera.project(x, &proj); // call project function
        return proj;              // return 2d point
    };

    // Compute numerical Jacobian
    // 2x3 Matrix called numerical Jacobian
    Eigen::Matrix<double, 2, 3> numericalJacobian = testutils::calculateNumericalJacobian(projectionFunc, point_C);

    // Check if analytical and numerical Jacobians are close
    if ((pointJacobian - numericalJacobian).norm() >= 1e-5)
    {
        std::cout << "Analytical Projection Jacobian:\n"
                  << pointJacobian << std::endl;
        std::cout << "Numerical Projection Jacobian:\n"
                  << numericalJacobian << std::endl;
        std::cout << "Difference Norm: " << (pointJacobian - numericalJacobian).norm() << std::endl;
    }

    // Check if analytical and numerical Jacobians are close
    EXPECT_TRUE((pointJacobian - numericalJacobian).norm() < 1e-5);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
