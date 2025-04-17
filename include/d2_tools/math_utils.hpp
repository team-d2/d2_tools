/*
    * @file math_utils.hpp
    * @brief This file is created with reference to the following repository: https://github.com/scomup/MathematicalRobotics.git
    * @brief licensed under the MIT License.
    * @author LogWat
*/

#include <eigen3/Eigen/Dense>
#include <cmath>

namespace d2_tools {

const double epsilon = 1e-5;

// ベクトルからskew対称行列を生成 (I+[ω]× のω)
Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d skew_matrix;
    skew_matrix << 0, -v(2), v(1),
                   v(2), 0, -v(0),
                   -v(1), v(0), 0;
    return skew_matrix;
}

// SO(3)の指数写像
Eigen::Matrix3d expSO3(const Eigen::Vector3d &omega) {
    double theta2 = omega.dot(omega);
    double theta = std::sqrt(theta2);
    bool near_zero = theta2 < epsilon;
    Eigen::Matrix3d W = skew(omega);

    if (near_zero) {
        return Eigen::Matrix3d::Identity() + W;
    } else {
        Eigen::Matrix3d K = W / theta;
        Eigen::Matrix3d K2 = K * K;
        return Eigen::Matrix3d::Identity() + K * std::sin(theta) + K2 * (1 - std::cos(theta)); // (I + [ω]× + [ω]×^2)
    }
}

} // namespace d2_tools
