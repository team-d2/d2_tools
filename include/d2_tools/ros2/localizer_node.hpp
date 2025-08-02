#pragma once

#include <mutex>
#include <memory>
#include <iostream>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_eigen_kdl/tf2_eigen_kdl.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <pclomp/ndt_omp.h>
#include <multigrid_pclomp/multigrid_ndt_omp.h>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>

#include <d2_tools/ros2/pose_estimator.hpp>

namespace d2::tools::ros2 {

class LocalizerNode : public rclcpp::Node {
public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    LocalizerNode(const rclcpp::NodeOptions & options)
        : rclcpp::Node("tools_localizer", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        robot_odom_frame_ = this->declare_parameter<std::string>("robot_odom_frame_id", "odom");
        odom_child_frame_ = this->declare_parameter<std::string>("odom_child_frame_id", "base_link");
        invert_acc_ = this->declare_parameter<bool>("invert_acc", false);
        invert_gyro_ = this->declare_parameter<bool>("invert_gyro", false);
        downsample_leaf_size_ = this->declare_parameter<double>("downsample_leaf_size", 0.1);
        ndt_neighbor_search_radius_ = this->declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
        ndt_resolution_ = this->declare_parameter<double>("ndt_resolution", 1.0);
        num_threads_ = this->declare_parameter<int>("num_threads", 8);
        cool_time_duration_ = this->declare_parameter<double>("cool_time_duration", 0.5);
        pos_noise_scale_ = this->declare_parameter<double>("pos_noise_scale", 0.01);
        vel_noise_scale_ = this->declare_parameter<double>("vel_noise_scale", 0.01);
        quat_noise_scale_ = this->declare_parameter<double>("quat_noise_scale", 0.01);
        vel_bias_noise_scale_ = this->declare_parameter<double>("vel_bias_noise_scale", 0.01);
        gyro_bias_noise_scale_ = this->declare_parameter<double>("gyro_bias_noise_scale", 0.01);
        gravity_noise_scale_ = this->declare_parameter<double>("gravity_noise_scale", 0.01);
        specify_init_pose_ = this->declare_parameter<bool>("specify_init_pose", false);

        globalmap_pcd_path_ = this->declare_parameter<std::string>("globalmap_pcd_path", "");
        globalmap_.reset(new PointCloudT());
        pcl::io::loadPCDFile(globalmap_pcd_path_, *globalmap_);
        globalmap_->header.frame_id = "map";
        RCLCPP_INFO(this->get_logger(), "Loaded global map from: %s", globalmap_pcd_path_.c_str());
        if (globalmap_->empty()) {
            RCLCPP_ERROR(this->get_logger(), "Global map is empty, please check the PCD file.");
            return;
        }

        // downsampler
        downsampler_ = std::make_shared<pcl::VoxelGrid<PointT>>();
        downsampler_->setLeafSize(downsample_leaf_size_, downsample_leaf_size_, downsample_leaf_size_);

        // downsample global map
        PointCloudT::Ptr downsampled_globalmap(new PointCloudT());
        downsampler_->setInputCloud(globalmap_);
        downsampler_->filter(*downsampled_globalmap);
        globalmap_ = downsampled_globalmap;
        RCLCPP_INFO(this->get_logger(), "Downsampled global map to %zu points.", globalmap_->size());

        RCLCPP_INFO(this->get_logger(), "Downsampler initialized with leaf size: %f", downsample_leaf_size_);
        registration_ = createRegistration();
        RCLCPP_INFO(this->get_logger(), "Created registration with %d threads.", num_threads_);
        registration_->setInputTarget(globalmap_);
        RCLCPP_INFO(this->get_logger(), "Set global map as target for registration.");

        if (specify_init_pose_) {
            RCLCPP_INFO(this->get_logger(), "Specify initial pose is enabled.");
            auto init_pose = this->declare_parameter<std::vector<float>>("init_pose", {0.0f, 0.0f, 0.0f});
            auto init_quat = this->declare_parameter<std::vector<float>>("init_quat", {0.0, 0.0, 0.0, 1.0});
            if (init_pose.size() != 3 || init_quat.size() != 4) {
                RCLCPP_ERROR(this->get_logger(), "Invalid initial pose or quaternion size. Expected 3 and 4 elements respectively.");
                return;
            }
            Eigen::Vector3d pos(init_pose[0], init_pose[1], init_pose[2]);
            Eigen::Quaterniond quat(init_quat[0], init_quat[1], init_quat[2], init_quat[3]);
            Eigen::Vector3d gravity(0.0f, 0.0f, -9.81f); // Assuming gravity is downwards
            pose_estimator_.reset(new PoseEstimator(
                registration_,
                pos,
                quat,
                gravity,
                createInitCovariance(),
                cool_time_duration_
            ));
        } else {
            RCLCPP_INFO(this->get_logger(), "Specify initial pose is disabled, using default identity pose.");
            Eigen::Vector3d init_pos(0.0, 0.0, 0.0);
            Eigen::Quaterniond init_rot(1.0, 0.0, 0.0, 0.0);
            Eigen::Vector3d init_gravity(0.0, 0.0, -9.81); // Assuming gravity is downwards
            pose_estimator_.reset(new PoseEstimator(
                registration_,
                init_pos,
                init_rot,
                init_gravity,
                createInitCovariance(),
                cool_time_duration_
            ));
        }

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", rclcpp::SensorDataQoS(),
            std::bind(&LocalizerNode::imuCallback, this, std::placeholders::_1));
        points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "points_raw", rclcpp::SensorDataQoS(),
            std::bind(&LocalizerNode::pointsCallback, this, std::placeholders::_1));
        // globalmap_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        //     "global_map", rclcpp::SensorDataQoS(),
        //     std::bind(&LocalizerNode::globalMapCallback, this, std::placeholders::_1));
        initialpose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "initialpose", rclcpp::QoS(1),
            std::bind(&LocalizerNode::initialPoseCallback, this, std::placeholders::_1));

        // publisher
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        aligned_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "aligned_points", rclcpp::SensorDataQoS());
        pose_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "odom", rclcpp::QoS(5));
        globalmap_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("globalmap", rclcpp::QoS(1).transient_local());
        sensor_msgs::msg::PointCloud2::UniquePtr globalmap_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*globalmap_, *globalmap_msg);
        globalmap_msg->header.frame_id = "map";
        globalmap_pub_->publish(std::move(globalmap_msg));

        RCLCPP_INFO(this->get_logger(), "LocalizerNode initialized with robot_odom_frame: %s, odom_child_frame: %s",
                    robot_odom_frame_.c_str(), odom_child_frame_.c_str());
    }

    LocalizerNode(const rclcpp::NodeOptions & options, const std::string & node_name);

private:
    void pointsCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg) {
        RCLCPP_INFO(this->get_logger(), "Received point cloud with %zu points at time: %f", msg->width * msg->height, msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
        if (!globalmap_) {
            RCLCPP_WARN(this->get_logger(), "Global map is not set, skipping point cloud processing.");
            return;
        }
        
        auto stamp = rclcpp::Time(msg->header.stamp.sec, msg->header.stamp.nanosec);
        PointCloudT::Ptr pcl_cloud(new PointCloudT());
        pcl::fromROSMsg(*msg, *pcl_cloud);
        if (pcl_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty point cloud, skipping processing.");
            return;
        }

        // transform point cloud to odom_child_frame_
        PointCloudT::Ptr transformed_cloud(new PointCloudT());
        try {
            geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform(
                odom_child_frame_, msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(1.0));
            pcl_ros::transformPointCloud(*pcl_cloud, *transformed_cloud, transform);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
            return;
        }

        // downsample point cloud
        PointCloudT::Ptr downsampled_cloud(new PointCloudT());
        downsampler_->setInputCloud(transformed_cloud);
        downsampler_->filter(*downsampled_cloud);
        last_scan_ = downsampled_cloud;

        // pose estimation (filter)
        RCLCPP_INFO(this->get_logger(), "Processing point cloud at time: %f", stamp.seconds());
        std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
        if (!pose_estimator_) {
            RCLCPP_ERROR(this->get_logger(), "Pose estimator is not initialized, cannot process point cloud.");
            return;
        }
        std::lock_guard<std::mutex> imu_lock(imu_buffer_mutex_);
        auto imu_iter = imu_buffer_.begin();
        for (; imu_iter != imu_buffer_.end(); ++imu_iter) {
            const auto& imu_msg = *imu_iter;
            const auto& imu_stamp = rclcpp::Time(imu_msg->header.stamp.sec, imu_msg->header.stamp.nanosec);
            if (imu_stamp > stamp) break;
            Eigen::Vector3d acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
            Eigen::Vector3d gyro(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
            if (invert_acc_) acc *= -1.0;
            if (invert_gyro_) gyro *= -1.0;
            Eigen::VectorXd control(6);
            control.head<3>() = acc;
            control.tail<3>() = gyro;
            pose_estimator_->predict(imu_stamp, control);
        }
        imu_buffer_.erase(imu_buffer_.begin(), imu_iter);

        // correct pose using registration
        auto aligned = pose_estimator_->correct(stamp, downsampled_cloud);
        if (aligned_pub_->get_subscription_count() > 0) {
            sensor_msgs::msg::PointCloud2::UniquePtr aligned_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
            pcl::toROSMsg(*aligned, *aligned_msg);
            aligned_msg->header.frame_id = "map";
            aligned_msg->header.stamp = stamp;
            aligned_pub_->publish(std::move(aligned_msg));
        }

        publishOdometry(stamp, pose_estimator_->matrix());
    }


    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr & msg) {
        RCLCPP_INFO(this->get_logger(), "IMU data received at time: %f", msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
        std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
        imu_buffer_.push_back(msg);
    }


    void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr & msg) {
        RCLCPP_INFO(this->get_logger(), "Initial pose received!");
        std::lock_guard<std::mutex> lock(pose_estimator_mutex_);
        const auto& p = msg->pose.pose.position;
        const auto& q = msg->pose.pose.orientation;
        Eigen::Vector3d init_pos(p.x, p.y, p.z);
        Eigen::Quaterniond init_rot(q.w, q.x, q.y, q.z);
        Eigen::Vector3d init_gravity(0.0, 0.0, -9.81); // Assuming gravity is downwards
        
        pose_estimator_.reset(new PoseEstimator(    
            registration_,
            init_pos,
            init_rot,
            init_gravity,
            createInitCovariance(),
            cool_time_duration_
        ));
    }

    void publishOdometry(const rclcpp::Time& stamp, const Eigen::Matrix4d& pose) {
        geometry_msgs::msg::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
        map_wrt_frame.header.stamp = stamp;
        map_wrt_frame.header.frame_id = robot_odom_frame_;
        map_wrt_frame.child_frame_id = "map";
        try {
            geometry_msgs::msg::TransformStamped frame_wrt_odom = tf_buffer_.lookupTransform(
                robot_odom_frame_, odom_child_frame_, stamp, rclcpp::Duration::from_seconds(0.1));
            // Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();
            geometry_msgs::msg::TransformStamped map_wrt_odom;
            tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);
            tf2::Transform odom_wrt_map;
            tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
            odom_wrt_map = odom_wrt_map.inverse(); // convert to odom frame
            geometry_msgs::msg::TransformStamped odom_trans;
            odom_trans.transform = tf2::toMsg(odom_wrt_map);
            odom_trans.header.stamp = stamp;
            odom_trans.header.frame_id = "map";
            odom_trans.child_frame_id = odom_child_frame_;
            tf_broadcaster_->sendTransform(odom_trans);
        } catch (const tf2::TransformException & ex) {
            geometry_msgs::msg::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
            odom_trans.header.stamp = stamp;
            odom_trans.header.frame_id = "map";
            odom_trans.child_frame_id = odom_child_frame_;
            tf_broadcaster_->sendTransform(odom_trans);
        }
        nav_msgs::msg::Odometry::UniquePtr odom_msg(new nav_msgs::msg::Odometry);
        odom_msg->header.stamp = stamp;
        odom_msg->header.frame_id = "map";
        odom_msg->child_frame_id = odom_child_frame_;
        odom_msg->pose.pose = tf2::toMsg(Eigen::Isometry3d(pose.cast<double>()));
        odom_msg->twist.twist.linear.x = 0.0;
        odom_msg->twist.twist.linear.y = 0.0;
        odom_msg->twist.twist.linear.z = 0.0;

        pose_pub_->publish(std::move(odom_msg));
    }


    Eigen::Matrix<double, 19, 19> createInitCovariance() {
        Eigen::Matrix<double, 19, 19> init_cov = Eigen::MatrixXd::Identity(19, 19);
        init_cov.block<3, 3>(0, 0) *= pos_noise_scale_;
        init_cov.block<3, 3>(3, 3) *= vel_noise_scale_;
        init_cov.block<4, 4>(6, 6) *= quat_noise_scale_;
        init_cov.block<3, 3>(10, 10) *= vel_bias_noise_scale_;
        init_cov.block<3, 3>(13, 13) *= gyro_bias_noise_scale_;
        init_cov.block<3, 3>(16, 16) *= gravity_noise_scale_;
        return init_cov;
    }

    std::shared_ptr<pcl::Registration<PointT, PointT>> createRegistration() const {
        // std::shared_ptr<pclomp::NormalDistributionsTransform<PointT, PointT>> ndt_omp(new pclomp::NormalDistributionsTransform<PointT, PointT>());
        // ndt_omp->setTransformationEpsilon(0.01);
        // ndt_omp->setResolution(ndt_resolution_);
        // ndt_omp->setNumThreads(num_threads_);
        // ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);


        std::shared_ptr<small_gicp::RegistrationPCL<PointT, PointT>> gicp(new small_gicp::RegistrationPCL<PointT, PointT>());
        gicp->setNumThreads(num_threads_);
        gicp->setMaxCorrespondenceDistance(1.0);
        gicp->setCorrespondenceRandomness(20);
        gicp->setVoxelResolution(1.0);
        gicp->setRegistrationType("VGICP");
        gicp->setMaximumIterations(200);
        return gicp;
    }

    // variables -------------------------------------------------------------------------
    std::string robot_odom_frame_, odom_child_frame_;
    // std::string reg_method_, ndt_neighbor_search_method_;
    double ndt_neighbor_search_radius_;
    double ndt_resolution_;
    int num_threads_;
    bool invert_acc_, invert_gyro_;
    bool specify_init_pose_;

    double pos_noise_scale_, vel_noise_scale_, quat_noise_scale_, vel_bias_noise_scale_, gyro_bias_noise_scale_, gravity_noise_scale_;

    double cool_time_duration_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
    // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // 複数callbackで参照される可能性

    // imu input buffer
    std::mutex imu_buffer_mutex_;
    std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer_;

    // globalmap and registration method
    std::string globalmap_pcd_path_;
    pcl::PointCloud<PointT>::Ptr globalmap_;
    pcl::VoxelGrid<PointT>::Ptr downsampler_;
    std::shared_ptr<pcl::Registration<PointT, PointT>> registration_;
    double downsample_leaf_size_;

    // pose estimator
    std::mutex pose_estimator_mutex_;
    std::unique_ptr<PoseEstimator> pose_estimator_;

    pcl::PointCloud<PointT>::ConstPtr last_scan_;
};

} // namespace d2::tools::ros2
