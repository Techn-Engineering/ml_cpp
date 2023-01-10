#include "../include/DataHandler.hpp"
#include "../include/linear_regression.hpp"

#include <iostream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

int main(int argc, char *argv[])
{

    DataHandler data_hand(argv[1], argv[2], argv[3]);
    linear_regression lr; 

    std::vector<std::vector<std::string>> dataset = data_hand.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataMat = data_hand.CSVtoEigen(dataset,rows,cols);
    Eigen::MatrixXd norm = data_hand.Normalize(dataMat, true);

    std::cout << dataMat.rows() << std::endl;

    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> split_data = data_hand.TrainTestSplit(norm, 0.8);
    std::tie(X_train, y_train, X_test, y_test) = split_data;

    Eigen::VectorXd vec_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd vec_test = Eigen::VectorXd::Ones(X_test.rows());

    X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
    X_train.col(X_train.cols()-1) = vec_train;

    X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
    X_test.col(X_train.cols()-1) = vec_test;

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
    float alpha = 0.01;
    int iters = 1000;

    Eigen::VectorXd thetaOut;
    std::vector<float> cost;

    std::tuple<Eigen::VectorXd,std::vector<float>> gd = lr.GradientDescent(X_train, y_train, theta, alpha, iters);
    std::tie(thetaOut,cost) = gd;

    //data_hand.Vectortofile(cost,"datasets/cost.txt");
    //data_hand.EigentoFile(thetaOut,"datasets/thetaOut.txt");

    auto mu_data = data_hand.Mean(dataMat);
    auto mu_z = mu_data(0,11);

    auto scaled_data = dataMat.rowwise() - dataMat.colwise().mean();

    auto sigma_data = data_hand.Std(scaled_data);
    auto sigma_z = sigma_data(0,11);

    Eigen::MatrixXd y_train_hat = (X_train*thetaOut*sigma_z).array() + mu_z;
    Eigen::MatrixXd y = dataMat.col(11).topRows(1279);

    float R_Squared = lr.RSquared(y,y_train_hat);
    std::cout << "R-Squared: " << R_Squared << std::endl;

    //data_hand.EigentoFile(y_train_hat,"datasets/y_train_hat.txt");

    return EXIT_SUCCESS;
}