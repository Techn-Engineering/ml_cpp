#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <unordered_set>
#include <iostream>
#include "data.h"
#include <eigen3/Eigen/Dense>

class data_handler
{
    std::unique_ptr<std::vector<std::shared_ptr<data>>> data_array;
    std::unique_ptr<std::vector<std::shared_ptr<data>>> test_data;
    std::unique_ptr<std::vector<std::shared_ptr<data>>> training_data;
    std::unique_ptr<std::vector<std::shared_ptr<data>>> validation_data;

    int num_classes ;
    int feature_vector_size;
    constexpr static int HEADER_SIZE = 4;
    constexpr static int LABEL_HEADER_SIZE = 2;
    std::string dataset;
    std::string data_limiter;
    bool header;
    std::map<uint8_t, int> class_map;

    static constexpr double TRAIN_SET_PERCENT = 0.75;
    static constexpr double TEST_SET_PERCENT = 0.20;
    static constexpr double VALIDATION_SET_PERCENT = 0.05;

    public:
    data_handler();
    data_handler(std::string, std::string, bool);
    ~data_handler();

    std::vector<std::vector<std::string>> read_csv();
    Eigen::MatrixXd csv_to_eigen(std::vector<std::vector<std::string>> dataset, int rows, int cols);
    Eigen::MatrixXd normalize(Eigen::MatrixXd data, bool normalize_target);
    
    void read_feature_vector(std::string path);
    void read_feature_labels(std::string path);
    void vector_to_file(std::vector<float> vector, std::string filename);
    void eigen_to_file(Eigen::MatrixXd data, std::string filename);
    void split_data();
    void count_classes();
    auto average(Eigen::MatrixXd data) -> decltype(data.colwise().mean());
    auto standart_deviation(Eigen::MatrixXd data) -> decltype(((data.array().square().colwise().sum())/(data.rows()-1)).sqrt());
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> train_test_split(Eigen::MatrixXd data, float train_size);
    uint32_t convert_to_little_endian(const unsigned char *bytes);
    int get_class_counts();

    std::unique_ptr<std::vector<std::shared_ptr<data>>> get_training_data();
    std::unique_ptr<std::vector<std::shared_ptr<data>>> get_test_data();
    std::unique_ptr<std::vector<std::shared_ptr<data>>> get_validation_data();
};
#endif