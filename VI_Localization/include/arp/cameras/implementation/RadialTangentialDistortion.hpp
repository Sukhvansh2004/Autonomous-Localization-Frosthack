/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Feb 3, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file implementation/RadialTangentialDistortion.hpp
 * @brief Header implementation file for the RadialTangentialDistortion class.
 * @author Stefan Leutenegger
 */

#include <Eigen/LU>
#include <iostream>
#include <stdexcept>
#include <cmath>

/// \brief arp Main namespace of this package.
namespace arp
{
  /// \brief cameras Namespace for camera-related functionality.
  namespace cameras
  {

    // The default constructor with all zero ki
    RadialTangentialDistortion::RadialTangentialDistortion()
        : k1_(0.0),
          k2_(0.0),
          p1_(0.0),
          p2_(0.0)
    {
    }

    // Constructor initialising ki
    RadialTangentialDistortion::RadialTangentialDistortion(double k1, double k2,
                                                           double p1, double p2)
    {
      k1_ = k1;
      k2_ = k2;
      p1_ = p1;
      p2_ = p2;
    }

    bool RadialTangentialDistortion::distort(
        const Eigen::Vector2d &pointUndistorted,
        Eigen::Vector2d *pointDistorted) const
    {
      const double rSquared = pointUndistorted.x() * pointUndistorted.x() + pointUndistorted.y() * pointUndistorted.y();
      const double factor = (1 + k1_ * rSquared + k2_ * rSquared * rSquared);

      *pointDistorted = factor * pointUndistorted + Eigen::Vector2d(2 * p1_ * pointUndistorted.x() * pointUndistorted.y() + p2_ * (rSquared + 2 * pointUndistorted.x() * pointUndistorted.x()), p1_ * (rSquared + 2 * pointUndistorted.y() * pointUndistorted.y()) + 2 * p2_ * pointUndistorted.x() * pointUndistorted.y());
      return true;
    }
    bool RadialTangentialDistortion::distort(
        const Eigen::Vector2d &pointUndistorted, Eigen::Vector2d *pointDistorted,
        Eigen::Matrix2d *pointJacobian) const
    {
      // Different powers of the x and y coordinates -- needed for jacobian
      const double y_pow2 = pointUndistorted.y() * pointUndistorted.y();
      const double x_pow2 = pointUndistorted.x() * pointUndistorted.x();
      const double y_pow3 = y_pow2 * pointUndistorted.y();
      const double x_pow3 = x_pow2 * pointUndistorted.x();
      const double y_pow4 = y_pow2 * y_pow2;
      const double x_pow4 = x_pow2 * x_pow2;

      const double rSquared = x_pow2 + y_pow2;
      const double factor = (1 + k1_ * rSquared + k2_ * rSquared * rSquared);

      *pointDistorted = factor * pointUndistorted + Eigen::Vector2d(2 * p1_ * pointUndistorted.x() * pointUndistorted.y() + p2_ * (rSquared + 2 * x_pow2), p1_ * (rSquared + 2 * y_pow2) + 2 * p2_ * pointUndistorted.x() * pointUndistorted.y());

      *pointJacobian = Eigen::Matrix2d();

      const Eigen::Vector2d v1(1 + k1_ * (3 * x_pow2 + y_pow2) + k2_ * (5 * x_pow4 + 6 * x_pow2 * y_pow2 + y_pow4),
                               2 * k1_ * pointUndistorted.x() * pointUndistorted.y() + k2_ * pointUndistorted.y() * (4 * x_pow3 + 4 * pointUndistorted.x() * y_pow2));

      const Eigen::Vector2d v2(k1_ * pointUndistorted.x() * 2 * pointUndistorted.y() + k2_ * pointUndistorted.x() * (4 * pointUndistorted.y() * x_pow2 + 4 * y_pow3),
                               1 + k1_ * (x_pow2 + 3 * y_pow2) + k2_ * (x_pow4 + 6 * x_pow2 * y_pow2 + 5 * y_pow4));

      pointJacobian->col(0) = v1 + Eigen::Vector2d(2 * p1_ * pointUndistorted.y() + 2 * p2_ * (pointUndistorted.x() + 2 * pointUndistorted.x()), 2 * p1_ * pointUndistorted.x() + 2 * p2_ * pointUndistorted.y());
      pointJacobian->col(1) = v2 + Eigen::Vector2d(2 * p1_ * pointUndistorted.x() + 2 * p2_ * pointUndistorted.y(), 2 * p1_ * pointUndistorted.y() + 4 * p1_ * pointUndistorted.y() + 2 * p2_ * pointUndistorted.x());

      return true;
    }

    bool RadialTangentialDistortion::undistort(
        const Eigen::Vector2d &pointDistorted,
        Eigen::Vector2d *pointUndistorted) const
    {
      // this is expensive: we solve with Gauss-Newton...
      Eigen::Vector2d x_bar = pointDistorted; // initialise at distorted point
      const int n = 5;                        // just 5 iterations max.
      Eigen::Matrix2d E;                      // error Jacobian

      bool success = false;
      for (int i = 0; i < n; i++)
      {

        Eigen::Vector2d x_tmp;

        distort(x_bar, &x_tmp, &E);

        Eigen::Vector2d e(pointDistorted - x_tmp);
        Eigen::Matrix2d E2 = (E.transpose() * E);
        Eigen::Vector2d du = E2.inverse() * E.transpose() * e;

        x_bar += du;

        const double chi2 = e.dot(e);
        if (chi2 < 1e-4)
        {
          success = true;
        }
        if (chi2 < 1e-15)
        {
          success = true;
          break;
        }
      }
      *pointUndistorted = x_bar;

      return success;
    }

  } // namespace cameras
} // namespace arp
