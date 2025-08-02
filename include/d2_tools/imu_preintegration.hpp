#pragma once

#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/PreintegrationBase.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/slam/BearingFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>

namespace d2_tools {
namespace imu_utils {
using namespace gtsam;
using symbol_shorthand::X; // Pose3 (x, y, z, qx, qy, qz, qw)
using symbol_shorthand::V; // Velocity3 (vx, vy, vz)
using symbol_shorthand::B; // Bias (acc_bias, gyro_bias)

class IMUPreintegration {
public:
    IMUPreintegration(const PreintegrationParams::shared_ptr& params)
        : preintegrated_(params) {}

    // IMUデータを追加
    void addIMUData(const imuBias::ConstantBias& bias, const Vector3& acc, const Vector3& gyro, double dt) {
        preintegrated_.integrateMeasurement(acc, gyro, dt);
        preintegrated_.bias = bias;
    }

    // 前処理されたIMUデータを取得
    PreintegratedImuMeasurements getPreintegratedMeasurements() const {
        return preintegrated_;
    }

private:
    PreintegratedImuMeasurements preintegrated_; // 前処理されたIMUデータ
    imuBias::ConstantBias bias_; // IMUバイアス
};

} // namespace imu_utils
} // namespace d2_tools
