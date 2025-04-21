#include <rclcpp/rclcpp.hpp>    
#include <sensor_msgs/msg/imu.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <memory>

#include "d2_tools/imu_transformer.hpp"

namespace d2_tools {

class IMUTransformComponent : public rclcpp::Node {
public:
    explicit IMUTransformComponent(const rclcpp::NodeOptions &options)
        : Node("imu_transform_component", options) {
            this->declare_parameter("input", "imu_input");
                input_ = this->get_parameter("input").as_string();
            this->declare_parameter("output", "imu_output");
                output_ = this->get_parameter("output").as_string();
            this->declare_parameter("output_frame", "base_link");
                output_frame_ = this->get_parameter("output_frame").as_string();
            
            // transform (tx, ty, tz, roll, pitch, yaw)
            this->declare_parameter("transform", std::vector<double>{0, 0, 0, 0, 0, 0});
            std::vector<double> transform = this->get_parameter("transform").as_double_array();
            if (transform.size() != 6) {
                RCLCPP_ERROR(this->get_logger(), "Transform parameter must be a 6D vector");
                return;
            }
                Eigen::Vector3d t(transform[0], transform[1], transform[2]);
                tf2::Quaternion q;
                q.setRPY(transform[3], transform[4], transform[5]);
                Eigen::Matrix3d R;
                tf2::Matrix3x3(q).getRPY(R(0, 0), R(1, 1), R(2, 2));
                imu_transformer_ = std::make_unique<IMUTransformer>(R, t);
            RCLCPP_INFO(this->get_logger(), "Transformation - Translation: [%f, %f, %f], Rotation: [%f, %f, %f]",
                transform[0], transform[1], transform[2], transform[3], transform[4], transform[5]);

            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                input_,
                rclcpp::SensorDataQoS(),
                std::bind(&IMUTransformComponent::imuCallback, this, std::placeholders::_1)
            );
            imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>(output_, rclcpp::SensorDataQoS());

        }

private:
    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr &imu_raw) {
        double sensor_now = imu_raw->header.stamp.sec + imu_raw->header.stamp.nanosec * 1e-9;
        if (prev_sensor_data_time_ == 0.0 || prev_imu_gyro_.norm() == 0.0) {
            prev_imu_gyro_ = Eigen::Vector3d(imu_raw->angular_velocity.x, imu_raw->angular_velocity.y, imu_raw->angular_velocity.z);
            prev_sensor_data_time_ = sensor_now;
            return;
        }
        double dt = sensor_now - prev_sensor_data_time_;
        if (dt <= 0.0) {
            RCLCPP_WARN(this->get_logger(), "Invalid IMU data time difference: %f", dt);
            return;
        }
        prev_sensor_data_time_ = sensor_now;
        d2_tools::types::IMUdata imu_data;
        imu_data.timestamp = sensor_now;
        imu_data.acc = Eigen::Vector3d(imu_raw->linear_acceleration.x, imu_raw->linear_acceleration.y, imu_raw->linear_acceleration.z);
        imu_data.gyro = Eigen::Vector3d(imu_raw->angular_velocity.x, imu_raw->angular_velocity.y, imu_raw->angular_velocity.z);
        imu_data.is_valid = true; // TODO: validate IMU data
        Eigen::Vector3d domg = (imu_data.gyro - prev_imu_gyro_) / dt;
        prev_imu_gyro_ = imu_data.gyro;
        auto transformed_data = imu_transformer_->transform(domg, imu_data);
        sensor_msgs::msg::Imu imu_transformed;
        imu_transformed.header = imu_raw->header;
        imu_transformed.header.frame_id = output_frame_;
        imu_transformed.linear_acceleration.x = transformed_data.acc(0);
        imu_transformed.linear_acceleration.y = transformed_data.acc(1);
        imu_transformed.linear_acceleration.z = transformed_data.acc(2);
        imu_transformed.angular_velocity.x = transformed_data.gyro(0);
        imu_transformed.angular_velocity.y = transformed_data.gyro(1);
        imu_transformed.angular_velocity.z = transformed_data.gyro(2);
        imu_transformed.orientation = imu_raw->orientation;
        imu_transformed.orientation_covariance = imu_raw->orientation_covariance;
        imu_transformed.angular_velocity_covariance = imu_raw->angular_velocity_covariance;
        imu_transformed.linear_acceleration_covariance = imu_raw->linear_acceleration_covariance;
        imu_pub_->publish(std::move(imu_transformed));
        // RCLCPP_DEBUG(this->get_logger(), "IMU data transformed and published");
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    std::unique_ptr<IMUTransformer> imu_transformer_;
    std::string input_, output_, output_frame_;
    double prev_sensor_data_time_;
    Eigen::Vector3d prev_imu_gyro_;
};

} // namespace d2_tools

