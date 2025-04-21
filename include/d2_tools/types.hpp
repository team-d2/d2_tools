#pragma once

#include <eigen3/Eigen/Dense>

namespace d2_tools {
namespace types {

struct IMUdata {
    double timestamp;
    Eigen::Vector3d acc, gyro;
    bool is_valid;
};
    
} // namespace d2_tools
} // namespace types
//