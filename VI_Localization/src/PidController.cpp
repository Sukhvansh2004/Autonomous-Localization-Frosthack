/*
 * PidController.cpp
 *
 *  Created on: 23 Feb 2017
 *      Author: sleutene
 */

#include <arp/PidController.hpp>
#include <stdexcept>

namespace arp {

// Set the controller parameters
void PidController::setParameters(const Parameters & parameters)
{
  parameters_ = parameters;
}

// Implements the controller as u(t)=c(e(t))
double PidController::control(uint64_t timestampMicroseconds, double e,
                              double e_dot)
{
  //calculate dt after we start
  double dt=0.0;
  if (lastTimestampMicroseconds_ != 0)
  {
    dt = (timestampMicroseconds - lastTimestampMicroseconds_) * 1.0e-6;
  }

  //Limit the maximum time delta allowed for integrating to something like 0.1s.
  if (dt > 0.1)
  {
    dt = 0.1;
  }

  //update timestamp
  lastTimestampMicroseconds_=timestampMicroseconds;

  //calculate u
  //stored in parameters_
  double u = parameters_.k_p * e + parameters_.k_i * integratedError_ + parameters_.k_d * e_dot;

  //saturation check (anti reset windup)

  if (u > maxOutput_)
  {
    u = maxOutput_;                     
  } 
  else if (u < minOutput_) 
  {
    u = minOutput_;                        
  }
  else
  {
    integratedError_ += e * dt;                     
  }
  return u;
}

void PidController::setOutputLimits(double minOutput, double maxOutput)
{
  minOutput_ = minOutput;
  maxOutput_ = maxOutput;
}

void PidController::resetIntegrator()
{
  integratedError_ = 0.0;
}

}  // namespace arp
