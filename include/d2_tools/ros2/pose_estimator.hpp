#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <d2_tools/filter/ekf.hpp>
#include <d2_tools/model/3d_pose_ekf.hpp>

namespace d2::tools {

namespace filter {
    template <typename T> class ExtendedKalmanFilter;
}

// class EKFPoseSystemModel;

namespace ros2 {

class PoseEstimator {
public:
    using PointT = pcl::PointXYZI;
    // enum class FilterType {
    //     EKF,
    //     UKF,
    //     ESKF
    // };

    explicit PoseEstimator(
        std::shared_ptr<pcl::Registration<PointT, PointT>> reg,
        const Eigen::Vector3d& init_pos,
        const Eigen::Quaterniond& init_rot,
        const Eigen::Vector3d& init_gravity,
        const Eigen::MatrixXd& process_noise,
        double cool_dt = 1.0
    ):
        cool_dt_(cool_dt),
        reg_(reg)
    {
        last_measurement_ = Eigen::Matrix4d::Identity();
        last_measurement_.block<3, 3>(0, 0) = init_rot.toRotationMatrix();
        last_measurement_.block<3, 1>(0, 3) = init_pos;

        process_noise_ = Eigen::MatrixXd::Zero(19, 19);
        process_noise_.block<3, 3>(0, 0) = process_noise.block<3, 3>(0, 0); // Position noise
        process_noise_.block<3, 3>(3, 3) = process_noise.block<3, 3>(3, 3); // Velocity noise
        process_noise_.block<4, 4>(6, 6) = process_noise.block<4, 4>(6, 6); // Quaternion noise
        process_noise_.block<3, 3>(10, 10) = process_noise.block<3, 3>(10, 10); // Bias acceleration noise
        process_noise_.block<3, 3>(13, 13) = process_noise.block<3, 3>(13, 13); // Bias gyro noise
        process_noise_.block<3, 3>(16, 16) = process_noise.block<3, 3>(16, 16); // Gravity noise

        Eigen::VectorXd initial_state(19);
        initial_state.setZero();
        initial_state.middleRows(0, 3) = init_pos;
        initial_state.middleRows(6, 4) = Eigen::Vector4d(init_rot.w(), init_rot.x(), init_rot.y(), init_rot.z()).normalized();
        initial_state.middleRows(16, 3) = init_gravity;

        model_ = std::make_unique<model::EKFPoseSystemModel>();
        filter_ = std::make_unique<filter::ExtendedKalmanFilter<model::EKFPoseSystemModel>>(
            *model_,
            19,
            initial_state,
            Eigen::MatrixXd::Identity(19, 19) * 0.01, // Initial covariance
            process_noise_
        );
    }


    ~PoseEstimator() = default;


    void predict(const rclcpp::Time& stamp, const Eigen::VectorXd& control) {
        if (init_stamp_ == rclcpp::Time()) init_stamp_ = stamp;
        if ((stamp - init_stamp_).seconds() < cool_dt_ || prev_stamp_ == rclcpp::Time() || (stamp - prev_stamp_).seconds() < cool_dt_) {
            prev_stamp_ = stamp;
            return;
        }
        double dt = (stamp - prev_stamp_).seconds();
        prev_stamp_ = stamp;
        filter_->setProcessNoise(process_noise_ * dt);
        filter_->predict(dt, control);
    }


    pcl::PointCloud<PointT>::Ptr correct(const rclcpp::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) {
        if (init_stamp_ == rclcpp::Time()) init_stamp_ = stamp;
        last_correct_stamp_ = stamp;

        Eigen::Matrix4d no_guess = last_measurement_;
        Eigen::Matrix4d init_guess = Eigen::Matrix4d::Identity(), imu_guess;
        init_guess = imu_guess = matrix();

        pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
        std::cout << "initial guess: " << init_guess.transpose() << std::endl;
        std::cout << "input cloud: " << cloud->size() << " points" << std::endl;
        reg_->setInputSource(cloud);
        reg_->align(*aligned, init_guess.cast<float>());

        Eigen::Matrix4d trans = reg_->getFinalTransformation().cast<double>();
        Eigen::Vector3d pos = trans.block<3, 1>(0, 3);
        Eigen::Quaterniond q(trans.block<3, 3>(0, 0));
        if (quat().coeffs().dot(q.coeffs()) < 0.0) q.coeffs() *= -1.0;

        Eigen::VectorXd measurement(7);
        measurement.head<3>() = pos;
        measurement.middleRows(3, 4) = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()).normalized();
        last_measurement_ = trans;

        world_pred_error_ = no_guess.inverse() * trans;
        Eigen::MatrixXd measurement_noise = Eigen::MatrixXd::Identity(7, 7);
        measurement_noise.middleRows(0, 3) *= 0.01; // Position noise
        measurement_noise.middleRows(3, 4) *= 0.001; // Quaternion noise
        filter_->correct(measurement, measurement_noise);
        std::cout << "Corrected state: " << filter_->getState().transpose() << std::endl;
        imu_pred_error_ = imu_guess.inverse() * trans;

        return aligned;
    }


    // -----------------------------------------------------------------------------
    Eigen::Vector3d pos() const {
        return Eigen::Vector3d(filter_->getState().head<3>());
    }
    Eigen::Quaterniond quat() const {
        return Eigen::Quaterniond(filter_->getState().segment<4>(6)).normalized();
    }
    Eigen::Matrix4d matrix() const {
        Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
        mat.block<3, 3>(0, 0) = quat().toRotationMatrix();
        mat.block<3, 1>(0, 3) = pos();
        return mat;
    }
    const std::optional<Eigen::Matrix4d>& worldPredictError() const {
        return world_pred_error_;
    }
    const std::optional<Eigen::Matrix4d>& imuPredictError() const {
        return imu_pred_error_;
    }

private:
    rclcpp::Time init_stamp_, prev_stamp_, last_correct_stamp_;
    double cool_dt_;

    std::shared_ptr<pcl::Registration<PointT, PointT>> reg_;

    std::unique_ptr<model::EKFPoseSystemModel> model_;
    std::unique_ptr<filter::ExtendedKalmanFilter<model::EKFPoseSystemModel>> filter_;

    Eigen::MatrixXd process_noise_;

    Eigen::Matrix4d last_measurement_;
    std::optional<Eigen::Matrix4d> world_pred_error_;
    std::optional<Eigen::Matrix4d> imu_pred_error_;
};

} // namespace ros2
} // namespace d2::tools
