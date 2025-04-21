/*
    * @file math_utils.hpp
    * @brief This file is created with reference to the following repository: https://github.com/scomup/MathematicalRobotics.git
    * @brief licensed under the MIT License.
    * @author LogWat
*/
#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>

namespace d2_tools {
namespace math {

const double epsilon = 1e-10;

// helpers --------------------------------------------------------------------------------------

// 2Dのposeベクトル(x, y, theta)を2D変換行列に変換
Eigen::Matrix3d v2m(const Eigen::Vector3d &v) {
    Eigen::Matrix3d T;
    T << std::cos(v(2)), -std::sin(v(2)), v(0),
         std::sin(v(2)),  std::cos(v(2)), v(1),
         0,               0,              1;
    return T;
}
// 2D変換行列をposeベクトル(x, y, theta)に変換
Eigen::Vector3d m2v(const Eigen::Matrix3d &T) {
    Eigen::Vector3d pose;
    pose(0) = T(0, 2);
    pose(1) = T(1, 2);
    pose(2) = std::atan2(T(1, 0), T(0, 0));
    return pose;
}

// 3Dのposeベクトル(x, y, z, roll, pitch, yaw)を3D変換行列に変換
Eigen::Matrix4d p2m(const Eigen::VectorXd &p) {
    if (p.size() != 6) throw std::invalid_argument("p must be a 6D vector");
    Eigen::Vector3d t = p.head<3>();
    Eigen::Matrix3d R = expSO3(p.tail<3>());
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

// 回転行列 R と 並進ベクトル t から 剛体変換行列 T を生成
Eigen::Matrix4d makeT(const Eigen::MatrixXd &R, const Eigen::VectorXd &t) {
    auto n = t.size();
    if (R.rows() != n || R.cols() != n) throw std::invalid_argument("R must be a square matrix");
    if (t.size() != n) throw std::invalid_argument("t must be a vector of the same size as R");
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}
// 剛体変換行列 T から 回転行列 R と 並進ベクトル t を生成
std::pair<Eigen::Matrix3d, Eigen::Vector3d> makeRt(const Eigen::Matrix4d &T) {
    return {T.block<3, 3>(0, 0), T.block<3, 1>(0, 3)};
}


Eigen::Matrix2d skewSO2(const double &theta) {
    Eigen::Matrix2d skew_matrix;
    skew_matrix << 0, -theta,
                   theta, 0;
    return skew_matrix;
}

Eigen::Matrix2d unSkewSO2(const Eigen::Matrix2d &skew_matrix) {
    Eigen::Vector2d v;
    v(0) = skew_matrix(1, 0);
    v(1) = skew_matrix(0, 1);
    return v;
}

// ベクトルからskew対称行列を生成 (I+[ω]× のω)
Eigen::Matrix3d skewSO3(const Eigen::Vector3d &v) {
    Eigen::Matrix3d skew_matrix;
    skew_matrix << 0, -v(2), v(1),
                   v(2), 0, -v(0),
                   -v(1), v(0), 0;
    return skew_matrix;
}

// skew対称行列からベクトルを生成
Eigen::Vector3d unSkewSO3(const Eigen::Matrix3d &skew_matrix) {
    Eigen::Vector3d v;
    v(0) = skew_matrix(2, 1);
    v(1) = skew_matrix(0, 2);
    v(2) = skew_matrix(1, 0);
    return v;
}

Eigen::Matrix2d leftJacobianSO2(const double &phi) {
    if (std::abs(phi) < epsilon) return Eigen::Matrix2d::Identity() + 0.5 * skewSO2(phi);
    auto s = std::sin(phi);
    auto c = std::cos(phi);
    return (s / phi) * Eigen::Matrix2d::Identity() + ((1 - c) / phi) * skewSO2(1.0);
}

Eigen::Matrix2d invLeftJacobianSO2(const double &phi) {
    if (std::abs(phi) < epsilon) return Eigen::Matrix2d::Identity() - 0.5 * skewSO2(phi);
    auto half_angle = 0.5 * phi;
    auto cot_half_angle = 1.0 / std::tan(half_angle);
    return half_angle * cot_half_angle * Eigen::Matrix2d::Identity() - half_angle * skewSO2(1.0);
}



// ----------------------------------------------------------------------------------------------

Eigen::Matrix2d expSO2(const double &theta) {
    return Eigen::Matrix2d(cos(theta), -sin(theta), sin(theta), cos(theta));
}

Eigen::Matrix2d logSO2(const Eigen::Matrix2d &R) {
    double theta = std::atan2(R(1, 0), R(0, 0));
    return Eigen::Matrix2d(theta);
}


// SO(3)の指数写像
Eigen::Matrix3d expSO3(const Eigen::Vector3d &omega) {
    double theta2 = omega.dot(omega);
    double theta = std::sqrt(theta2);
    bool near_zero = theta2 < epsilon;
    Eigen::Matrix3d W = skewSO3(omega);

    if (near_zero) {
        return Eigen::Matrix3d::Identity() + W;
    } else {
        Eigen::Matrix3d K = W / theta;
        Eigen::Matrix3d K2 = K * K;
        return Eigen::Matrix3d::Identity() + K * std::sin(theta) + K2 * (1 - std::cos(theta));
    }
}

// transformation (section 2.)
Eigen::Matrix3d expSE3(const Eigen::VectorXd &xi) {
    if (xi.size() != 6) throw std::invalid_argument("xi must be a 6D vector");
    Eigen::Matrix3d omega = xi.head<3>();
    Eigen::Matrix3d v = xi.tail<3>();
    Eigen::Matrix3d R = expSO3(omega);
    double theta2 = omega.dot(omega);   // 角速度normの2乗 (r?)
    if (theta2 < epsilon) return makeT(R, v);
}

} // namespace math
} // namespace d2_tools
