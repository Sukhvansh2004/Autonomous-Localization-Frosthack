/*
 * Imu.cpp
 *
 *  Created on: 8 Feb 2017
 *      Author: sleutene
 */

#include <arp/kinematics/Imu.hpp>
#include <iostream>

namespace arp
{
  namespace kinematics
  {
    // Function to calculate the skew-symmetric matrix
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
    {
      Eigen::Matrix3d skew;
      skew << 0, -v.z(), v.y(),
          v.z(), 0, -v.x(),
          -v.y(), v.x(), 0;
      return skew;
    }

    // Function to compute the continuous-time Jacobian Fc
    ImuKinematicsJacobian computeContinuousTimeJacobian(const RobotState &state, const Eigen::Vector3d &acc_S)
    {
      ImuKinematicsJacobian Fc = ImuKinematicsJacobian::Zero();

      Fc.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
      Fc.block<3, 3>(3, 9) = -state.q_WS.toRotationMatrix();
      Fc.block<3, 3>(6, 3) = -skewSymmetric(state.q_WS.toRotationMatrix() * (acc_S - state.b_a));
      Fc.block<3, 3>(6, 12) = -state.q_WS.toRotationMatrix();

      return Fc;
    }

    bool Imu::stateTransition(const RobotState &state_k_minus_1,
                              const ImuMeasurement &z_k_minus_1,
                              const ImuMeasurement &z_k,
                              RobotState &state_k,
                              ImuKinematicsJacobian *jacobian)
    {
      // Calculate time difference in seconds.
      const double dt = double(z_k.timestampMicroseconds - z_k_minus_1.timestampMicroseconds) * 1.0e-6;
      if (dt < 1.0e-12 || dt > 0.05)
      {
        state_k = state_k_minus_1;
        if (jacobian)
        {
          jacobian->setIdentity();
        }
        return false;
      }

      // Gravity vector in World frame
      const Eigen::Vector3d Wg(0, 0, -9.81);

      // Step 1: Compute Δχ₁ using fc(state_k_minus_1, z_k_minus_1)
      Eigen::Vector3d WtS = state_k_minus_1.v_W * dt;
      Eigen::Quaterniond q_delta1 = deltaQ((z_k_minus_1.omega_S - state_k_minus_1.b_g) * dt);
      Eigen::Vector3d Wv = (state_k_minus_1.q_WS.toRotationMatrix() * (z_k_minus_1.acc_S - state_k_minus_1.b_a) + Wg) * dt;

      // Predict intermediate state χ_k_minus_1 ⊞ Δχ₁
      RobotState predicted_state;
      predicted_state.t_WS = state_k_minus_1.t_WS + WtS;
      predicted_state.q_WS = state_k_minus_1.q_WS * q_delta1;
      predicted_state.v_W = state_k_minus_1.v_W + Wv;
      predicted_state.b_g = state_k_minus_1.b_g;
      predicted_state.b_a = state_k_minus_1.b_a;

      // Step 2: Compute Δχ₂ using fc(predicted_state, z_k)
      Eigen::Vector3d WtS2 = predicted_state.v_W * dt;
      Eigen::Quaterniond q_delta2 = deltaQ((z_k.omega_S - predicted_state.b_g) * dt);
      Eigen::Vector3d Wv2 = (predicted_state.q_WS.toRotationMatrix() * (z_k.acc_S - predicted_state.b_a) + Wg) * dt;

      // Update state using χ_k = χ_k_minus_1 ⊞ 0.5 * (Δχ₁ + Δχ₂)
      state_k.t_WS = state_k_minus_1.t_WS + 0.5 * (WtS + WtS2);
      state_k.q_WS = state_k_minus_1.q_WS * q_delta1.slerp(0.5, q_delta2);
      state_k.v_W = state_k_minus_1.v_W + 0.5 * (Wv + Wv2);
      state_k.b_g = state_k_minus_1.b_g;
      state_k.b_a = state_k_minus_1.b_a;

      // Step 3: Calculate the Jacobian (if requested)
      if (jacobian)
      {
        // Compute Fc for state_k_minus_1 and predicted_state
        ImuKinematicsJacobian Fc_k_minus_1 = computeContinuousTimeJacobian(state_k_minus_1, z_k_minus_1.acc_S);
        ImuKinematicsJacobian Fc_k = computeContinuousTimeJacobian(predicted_state, z_k.acc_S);

        // Trapezoidal integration to compute F
        jacobian->setIdentity();
        *jacobian += 0.5 * dt * Fc_k_minus_1;
        *jacobian += 0.5 * dt * Fc_k * (*jacobian + 0.5 * dt * Fc_k_minus_1);
      }

      return true;
    }

  }
} // namespace arp
