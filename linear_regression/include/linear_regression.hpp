#ifndef __LINEAR_REGRESSION_HPP
#define __LINEAR_REGRESSION_HPP

#include <eigen3/Eigen/Dense>

class linear_regression
{

public:
    linear_regression();

    float OLS_Cost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd,std::vector<float>> GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iters);
    float RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif