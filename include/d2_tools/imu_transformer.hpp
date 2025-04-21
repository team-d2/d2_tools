#pragma once

#include "d2_tools/math.hpp"
#include "d2_tools/types.hpp"

namespace d2_tools {

class IMUTransformer {

public:
    IMUTransformer(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const d2_tools::types::IMUdata &imu_data)
        : R_(R), t_(t), imu_data_(imu_data) {}

    IMUTransformer(const Eigen::Matrix4d &T, const d2_tools::types::IMUdata &imu_data) : imu_data_(imu_data) {
        auto [R, t] = d2_tools::math::makeRt(T);
        R_ = R;
        t_ = t;
    }
    IMUTransformer(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) : R_(R), t_(t) {}
    IMUTransformer(const Eigen::Matrix4d &T) {
        auto [R, t] = d2_tools::math::makeRt(T);
        R_ = R;
        t_ = t;
    }

    // transform実行
    d2_tools::types::IMUdata transform(const Eigen::Vector3d &domg, const d2_tools::types::IMUdata &imu_data) {
        imu_data_ = imu_data;
        d2_tools::types::IMUdata transformed_data;
        transformed_data.timestamp = imu_data_.timestamp;
        transformed_data.gyro = R_ * (imu_data_.gyro);
        Eigen::Matrix3d skew1 = d2_tools::math::skewSO3(transformed_data.gyro);
        Eigen::Matrix3d skew2 = d2_tools::math::skewSO3(t_);
        transformed_data.acc = R_ * imu_data_.acc - skew1 * skew1 * t_ - skew2 * R_ * domg;
        transformed_data.is_valid = imu_data_.is_valid;
        return transformed_data;
    }

private:
    Eigen::Matrix3d R_; // 回転行列
    Eigen::Vector3d t_; // 並進ベクトル
    d2_tools::types::IMUdata imu_data_; // IMUデータ
};


} // namespace d2_tools

