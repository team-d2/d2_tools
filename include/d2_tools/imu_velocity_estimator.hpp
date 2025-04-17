#pragma once

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>

namespace d2_tools {

struct IMUdata {
    double timestamp;
    Eigen::Vector3d acc, gyro;
    bool is_valid;
};

class IMUVelocityEstimator {
public:
    IMUVelocityEstimator(int num_imus) : num_imus_(num_imus) {
        x_ = Eigen::VectorXd::Zero(STATE_SIZE);
        x_(6) = 1.0; // 初期クォータニオン

        P_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE); // 初期状態共分散

        Q_ = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE) * 1e-2; // プロセスノイズ共分散
        Q_.block<3, 3>(0, 0) *= 1e-2; // 位置のプロセスノイズ
        Q_.block<3, 3>(3, 3) *= 0.1; // 速度のプロセスノイズ
        Q_.block<4, 4>(6, 6) *= 1e-2; // クォータニオンのプロセスノイズ
        Q_.block<3, 3>(10, 10) *= 1e-3; // 加速度バイアスのプロセスノイズ
        Q_.block<3, 3>(13, 13) *= 1e-3; // ジャイロバイアスのプロセスノイズ

        R_= Eigen::MatrixXd::Identity(3, 3) * 0.1; // 観測ノイズ共分散

        imu_weights_.resize(num_imus_, 1.0 / num_imus_); // 各IMUの信頼度
        last_time_ = 0.0; // 前回の更新時刻
    }

    void setIMUWeights(const std::vector<double> &weights) {
        if (weights.size() == num_imus_) {
            imu_weights_ = weights;
            double sum = 0.0;
            for (double w : imu_weights_) sum += w;
            if (sum > 0.0) {
                for (double &w : imu_weights_) w /= sum; // 正規化
            }
        }
    }

    void predict(const std::vector<IMUdata> &imu_data) {
        auto validated_data = validateIMUData(imu_data);
        IMUdata fused_data = fuseIMUData(validated_data);
        if (!fused_data.is_valid) {
            std::cerr << "Invalid IMU data" << std::endl;
            return;
        }

        // 時間差分
        double dt = fused_data.timestamp - last_time_;
        if (last_time_ == 0.0 || dt <= 0.0) {
            last_time_ = fused_data.timestamp;
            return;
        }

        // 状態ベクトル取得
        Eigen::Vector3d pos = x_.segment<3>(0);
        Eigen::Vector3d vel = x_.segment<3>(3);
        Eigen::Vector4d q = x_.segment<4>(6);
        Eigen::Vector3d acc_bias = x_.segment<3>(10);
        Eigen::Vector3d gyro_bias = x_.segment<3>(13);

        // バイアス補正
        Eigen::Vector3d acc = fused_data.acc - acc_bias;
        Eigen::Vector3d gyro = fused_data.gyro - gyro_bias;

        Eigen::Matrix3d R = quaternion2RotationMatrix(q);
        
        Eigen::Vector3d gravity(0, 0, -9.81);

        // 予測
        pos += vel * dt + 0.5 * (R * acc + gravity) * dt * dt; // 位置
        vel += (R * acc + gravity) * dt; // 速度
        q += quaternionDerivative(q, gyro) * dt; // クォータニオン
        q.normalize();
        acc_bias += Eigen::Vector3d::Zero(); // 変化しないと仮定
        gyro_bias += Eigen::Vector3d::Zero(); // 変化しないと仮定

        // 状態ベクトル更新
        x_.segment<3>(0) = pos;
        x_.segment<3>(3) = vel;
        x_.segment<4>(6) = q;
        x_.segment<3>(10) = acc_bias;
        x_.segment<3>(13) = gyro_bias;

        // ヤコビアン行列計算
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
        F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt; // 位置のヤコビアン
        F.block<3, 3>(0, 6) = -0.5 * dt * quaternion2RotationMatrix(q) * Eigen::Matrix3d::Identity(); // クォータニオンのヤコビアン
        F.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt; // 速度のヤコビアン
        F.block<3, 3>(6, 10) = -0.5 * dt * quaternion2RotationMatrix(q) * Eigen::Matrix3d::Identity(); // 加速度バイアスのヤコビアン
        F.block<3, 3>(6, 13) = -0.5 * dt * quaternion2RotationMatrix(q) * Eigen::Matrix3d::Identity(); // ジャイロバイアスのヤコビアン
        F.block<3, 3>(10, 10) = Eigen::Matrix3d::Identity(); // 加速度バイアスのヤコビアン
        F.block<3, 3>(13, 13) = Eigen::Matrix3d::Identity(); // ジャイロバイアスのヤコビアン
        
        // 信念分布更新
        P_ = F * P_ * F.transpose() + Q_ * dt;

        last_time_ = fused_data.timestamp; // 更新時刻
    }

    // ゼロ速度更新 (ZUPT)
    void zeroVelocityUpdate(bool is_stationary) {
        if (!is_stationary) return;

        // ゼロ速度観測
        Eigen::Vector3d z = Eigen::Vector3d::Zero(); // 観測値 (速度0)
        Eigen::Vector3d h = x_.segment<3>(3); // 予測値 (速度)

        // 観測行列
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, STATE_SIZE);
        H.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity(); // 速度に関する観測

        // カルマンゲイン
        Eigen::MatrixXd K = P_ * H.transpose() * (H * P_ * H.transpose() + R_).inverse();

        // 状態更新
        Eigen::VectorXd y = z - h; // 観測残差
        x_ += K * y;

        normalizeQuaternion(x_); // クォータニオンの正規化

        // 誤差共分散更新 (信念分布更新)
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
        P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
    }

    // 静止状態検出
    bool isStationary(const std::vector<IMUdata> &imu_data, double threshold = 0.1, int window_size = 10) {
        static std::vector<Eigen::Vector3d> acc_buffer;

        auto validated_data = validateIMUData(imu_data);
        IMUdata fused_data = fuseIMUData(validated_data);
        if (!fused_data.is_valid) {
            std::cerr << "Invalid IMU data" << std::endl;
            return false;
        }
        acc_buffer.push_back(fused_data.acc);
        if (acc_buffer.size() > window_size) {
            acc_buffer.erase(acc_buffer.begin());
        }
        if (acc_buffer.size() < window_size) return false;

        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        for (const auto &acc : acc_buffer) mean += acc;
        mean /= acc_buffer.size();

        Eigen::Vector3d variance = Eigen::Vector3d::Zero();
        for (const auto &acc : acc_buffer) variance += (acc - mean).cwiseAbs2();
        variance /= acc_buffer.size();
    }

    Eigen::Vector3d getPosition() const {
        return x_.segment<3>(0);
    }
    Eigen::Vector3d getVelocity() const {
        return x_.segment<3>(3);
    }
    Eigen::Vector4d getQuaternion() const {
        return x_.segment<4>(6);
    }
    Eigen::Vector3d getAccelerationBias() const {
        return x_.segment<3>(10);
    }
    Eigen::Vector3d getGyroBias() const {
        return x_.segment<3>(13);
    }

private:
    // quaternion to rotation matrix
    Eigen::Matrix3d quaternion2RotationMatrix(const Eigen::Vector4d &q) {
        double qx = q(0), qy = q(1), qz = q(2), qw = q(3);
        Eigen::Matrix3d R;
        R << 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
             2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
             2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy);
        return R;
    }

    // quaternion normalization
    void normalizeQuaternion(Eigen::VectorXd &state) {
        double norm = std::sqrt(state(6) * state(6) + state(7) * state(7) + state(8) * state(8) + state(9) * state(9));
        if (norm > 0.0) {
            state(6) /= norm;
            state(7) /= norm;
            state(8) /= norm;
            state(9) /= norm;
        }
    }

    // 角速度からクォータニオンの微分を計算
    Eigen::Vector4d quaternionDerivative(const Eigen::Vector4d &q, const Eigen::Vector3d &w) {
        Eigen::Vector4d q_dot;
        q_dot(0) = 0.5 * (-q(1) * w(0) - q(2) * w(1) - q(3) * w(2));
        q_dot(1) = 0.5 * (q(0) * w(0) + q(2) * w(2) - q(3) * w(1));
        q_dot(2) = 0.5 * (q(0) * w(1) - q(1) * w(2) + q(3) * w(0));
        q_dot(3) = 0.5 * (q(0) * w(2) + q(1) * w(1) - q(2) * w(0));
        return q_dot;
    }

    // 複数IMUdataをセンサ融合
    IMUdata fuseIMUData(const std::vector<IMUdata> &imu_data) {
        IMUdata fused_data;
        fused_data.timestamp = imu_data[0].timestamp;
        fused_data.acc.setZero();
        fused_data.gyro.setZero();
        double total_weight = 0.0;

        // 加重平均
        for (size_t i = 0; i < imu_data.size(); i++) {
            if (imu_data[i].is_valid) {
                fused_data.acc += imu_weights_[i] * imu_data[i].acc;
                fused_data.gyro += imu_weights_[i] * imu_data[i].gyro;
                total_weight += imu_weights_[i];
            }
        }

        // 正規化
        if (total_weight > 0.0) {
            fused_data.acc /= total_weight;
            fused_data.gyro /= total_weight;
            fused_data.is_valid = true;
        } else {
            fused_data.is_valid = false;
        }

        return fused_data;
    }

    // 異常値検出
    std::vector<IMUdata> validateIMUData(const std::vector<IMUdata> &imu_data) {
        std::vector<IMUdata> validated_data = imu_data;

        // 閾値設定
        const double ACCEL_THRESHOLD = 30.0; // m/s^2 (例: 3G)
        const double GYRO_THRESHOLD = 90.0; // rad/s (例: 90deg/s)

        for (size_t i = 0; i < validated_data.size(); i++) {
            if (validated_data[i].is_valid) {
                if (validated_data[i].acc.norm() > ACCEL_THRESHOLD) {
                    validated_data[i].is_valid = false;
                }
                if (validated_data[i].gyro.norm() > GYRO_THRESHOLD) {
                    validated_data[i].is_valid = false;
                }
            }
        }
        return validated_data;
    }

    // variables & constants ---------------------------------------------------------------------------------------
    static const int STATE_SIZE = 16; // [pose(3), velocity(3), quaternion(4), acceleration_bias(3), gyro_bias(3)]

    Eigen::VectorXd x_; // 状態ベクトル
    Eigen::MatrixXd P_; // 状態共分散 
    Eigen::MatrixXd Q_; // プロセスノイズ共分散
    Eigen::MatrixXd R_; // 観測ノイズ共分散

    double last_time_; // 前回の更新時刻
    int num_imus_;     // 使用するIMUdata数
    std::vector<double> imu_weights_; // 各IMUの信頼度
};


class IMUVelocityEstimatorComponent : public rclcpp::Node {
public:
    IMUVelocityEstimatorComponent(const rclcpp::NodeOptions &options);
    IMUVelocityEstimatorComponent(const std::string &node_name, const rclcpp::NodeOptions &options);

private:
    IMUdata rosToIMUData(const sensor_msgs::msg::Imu::ConstPtr &msg) {
        IMUdata data;
        data.timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        data.acc = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        data.gyro = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        data.is_valid = true; // TODO: validate IMU data
        return data;
    }

    void IMURigidBodyTransform(const std::vector<IMUdata> &imu_data, const std::vector<Eigen::Vector3d> &transforms,
        const std::vector<Eigen::Matrix3d> &rotations, const std::vector<double> &weights) {
    }

    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &msg);

    // variables & constants ---------------------------------------------------------------------------------------
    IMUVelocityEstimator estimator_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr> imu_subs_;
    std::vector<IMUdata> imu_data_;
    std::vector<std::string> imu_topics_;
    std::vector<Eigen::Vector3d> imu_transforms_;
    std::vector<Eigen::Matrix3d> imu_rotations_;
    std::vector<double> imu_weights_;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr velocity_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    double last_time_;
    double stationary_threshold_;
    int stationary_window_size_;
    bool is_stationary_;
};

} // namespace d2_tools
