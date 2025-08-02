#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace d2::tools::filter {

template <class SystemModel>
class ErrorStateKalmanFilter {
public:
    explicit ErrorStateKalmanFilter(
        const SystemModel& model,
        const int state_dim,
        const Eigen::VectorXd& initial_error_state,
        const Eigen::MatrixXd& initial_cov,
        const Eigen::MatrixXd& perturbation_cov)
        : model_(model), state_dim_(state_dim), Xe_(initial_error_state), P_(initial_cov), Qi_(perturbation_cov) {}

    /**
     * @brief Predict the next error state and update the error state covariance
     * @param dt Time step for prediction
     */
    void predict(const double dt) {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6); // Clamp dt to avoid instability
        
        // ノミナル状態の予測
        Eigen::VectorXd Xnominal_pred = model_.predictNominal(Xnominal_, dt_c);
        // 誤差状態の予測 (誤差平均は0)
        // Eigen::VectorXd Xe_pred = model_.predictErrorState(Xe_, dt_c);

        Eigen::MatrixXd Fx = model_.errorStateTransitionJacobian(Xe_, dt_c);
        Eigen::MatrixXd Fi = model_.perturbationJacobian(Xe_, dt_c);
        Eigen::MatrixXd P_pred = Fx * P_ * Fx.transpose() + Fi * Qi_ * Fi.transpose();

        P_ = P_pred;
    }

    /**
     * @brief Predict the next error state with control input
     * @param dt Time step for prediction
     * @param control Control input vector
     * */
    void predict(const double dt, const Eigen::VectorXd& control) {
        double dt_c = std::max(std::min(dt, 1.0), 1e-6); // Clamp dt to avoid instability
        
        // ノミナル状態の予測
        Eigen::VectorXd Xnominal_pred = model_.predictNominal(Xnominal_, control, dt_c);
        // 誤差状態の予測 (誤差平均は0)
        // Eigen::VectorXd Xe_pred = model_.predictErrorState(Xe_, control, dt_c);

        Eigen::MatrixXd Fx = model_.errorStateTransitionJacobian(Xe_, control, dt_c);
        Eigen::MatrixXd Fi = model_.perturbationJacobian(Xe_, control, dt_c);
        Eigen::MatrixXd P_pred = Fx * P_ * Fx.transpose() + Fi * Qi_ * Fi.transpose();

        P_ = P_pred;
    }

    /**
     * @brief Correct the error state using measurement
     * @param measurement Measurement vector
     * @param V Measurement covariance matrix
     */
    void correct(const Eigen::VectorXd& measurement, const Eigen::MatrixXd& V) {
        // Calculate Kalman gain
        Eigen::MatrixXd H = model_.measurementJacobian(Xe_);
        Eigen::MatrixXd K = P_ * H.transpose() * (H * P_ * H.transpose() + V).inverse();
        // Update error state
        Eigen::VectorXd y = measurement - model_.h(Xtrue_); // 観測が状態に依存するとする
        Xe_ = K * y;
        // Josephson correct for covariance
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * V * K.transpose();
        
        // Update true state
        Xtrue_ = model_.updateTrueState(Xnominal_, Xe_);
        
        // reinitialize
        Xe_ = Eigen::VectorXd::Zero(state_dim_);
        Eigen::MatrixXd G = model_

    }




private:
    Eigen::VectorXd Xtrue_; // True state vector
    Eigen::VectorXd Xnominal_; // Nominal state vector
    Eigen::VectorXd Xe_; // Error state vector
    Eigen::MatrixXd P_; // Error covariance matrix
    Eigen::MatrixXd Fi_; // Perturbation Jacobian matrix
    Eigen::MatrixXd Qi_; // Perturbation covariance matrix
    SystemModel model_; // System model

    int state_dim_; // Dimension of the error state vector
};

} // namespace d2::tools::filter
