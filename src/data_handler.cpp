///  Created by Tech Engineering, efe@lexpai.com on 6/11/2022.
///  Copyright © 2023 Tech Engineering. All rights reserved.
///  Copyright © 2023 LexpAI. All rights reserved.

#include "data_handler.h"
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <memory>
#include <unordered_set>
#include <random>
#include <algorithm>

data_handler::data_handler()
{
    std::cout << "The constructor of Data Handler without args, has been initialized!" << std::endl;

    data_array = std::make_unique<std::vector<std::shared_ptr<data>>>();
    test_data = std::make_unique<std::vector<std::shared_ptr<data>>>();
    training_data = std::make_unique<std::vector<std::shared_ptr<data>>>();
    validation_data = std::make_unique<std::vector<std::shared_ptr<data>>>();
}
data_handler::data_handler(std::string set, std::string data_lim, bool head)
{
    std::cout << "The constructor of Data Handler with args, has been initialized!" << std::endl;
    dataset = set;
    data_limiter = data_lim;
    header = head;
}
data_handler::~data_handler()
{
    // Free Dynamically Allocated stuff
    std::cout << std::endl << "The destructor of Data Handler has been called!" << std::endl;
}

std::vector<std::vector<std::string>> data_handler::read_csv() 
{
    std::ifstream file(dataset);

    std::vector<std::vector<std::string>> data_string; 

    std::string line = ""; 

    while(getline(file,line))
    {
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(data_limiter));
        data_string.push_back(vec);
    }

    file.close();

    return data_string;
}
Eigen::MatrixXd data_handler::csv_to_eigen(std::vector<std::vector<std::string>> dataset, int rows, int cols)
{
    if(header == true)
    {
        rows = rows - 1;
    }

    Eigen::MatrixXd mat(cols,rows); 
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; ++j)
        {
            mat(j,i) = atof(dataset[i][j].c_str());
        }
    }
    return mat.transpose();
}
Eigen::MatrixXd data_handler::normalize(Eigen::MatrixXd data, bool normalize_target)
{
    Eigen::MatrixXd dataNorm;
    if(normalize_target == true) 
    {
        dataNorm = data;
    } else 
    {
        dataNorm = data.leftCols(data.cols() - 1);
    }

    auto mean = average(dataNorm); 
    Eigen::MatrixXd scaled_data = dataNorm.rowwise() - mean;
    auto std = standart_deviation(scaled_data); 

    Eigen::MatrixXd norm = scaled_data.array().rowwise()/std; 

    if(normalize_target==false) 
    {
        norm.conservativeResize(norm.rows(), norm.cols() + 1);
        norm.col(norm.cols()-1) = data.rightCols(1);
    }

    return norm;  
}
int data_handler::get_class_counts()
{
    return num_classes;
}

void data_handler::read_feature_vector(std::string path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file at path: " << path << std::endl;
        exit(1);
    }

    uint32_t header[HEADER_SIZE];
    for (int i = 0; i < HEADER_SIZE; i++)
    {
        char buffer[sizeof(uint32_t)];
        file.read(buffer, sizeof(uint32_t));
        header[i] = convert_to_little_endian(reinterpret_cast<unsigned char*>(buffer));
    }

    std::cout << "\nReceived the headers of image file.\n";
        
    int image_size = header[2]*header[3];

    for (int i = 0; i < header[1]; i++)
    {
        auto d = std::make_shared<data>();
        uint8_t element;

        for (int j = 0; j < image_size; j++)
        {
            file.read(reinterpret_cast<char*>(&element), sizeof(uint8_t));
            if (file)
            {
                d->append_to_feature_vector(element);
            } 
            else
            {
                std::cerr << "Error! Reading the file: Check out the method of append_to_feature_vector " << j << std::endl;
                exit(1);
            }
        }
        data_array->push_back(d);
    }

    std::cout << "Successfully read and stored " << data_array->size() << " features into a vector.\n";  
    std::cout << "Reading the feature vector, has been done!\n\n";
}

void data_handler::read_feature_labels(std::string path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file at path: " << path << std::endl;
        exit(1);
    }

    uint32_t header[LABEL_HEADER_SIZE];
    for (int i = 0; i < LABEL_HEADER_SIZE; i++)
    {
        char buffer[sizeof(uint32_t)];
        file.read(buffer, sizeof(uint32_t));
        header[i] = convert_to_little_endian(reinterpret_cast<unsigned char*>(buffer));
    }

    std::cout << "Received the headers of the label file.\n";
        
    for (int i = 0; i < header[1]; i++)
    {
        uint8_t element;

        file.read(reinterpret_cast<char*>(&element), sizeof(uint8_t));
        if (file)
        {
            data_array->at(i)->set_label(element);
        } 
        else
        {
            std::cerr << "Error! Reading the file: Check out the method of read_feature_labels\n";
            exit(1);
        }
    }

    std::cout << "Successfully read and stored " << data_array->size() << " feature labels.\n";  
    std::cout << "Reading the feature vector, has been done!\n\n";
}

void data_handler::vector_to_file(std::vector<float> vector, std::string filename)
{
    std::ofstream output_file(filename);
    std::ostream_iterator<float> output_iterator(output_file, "\n");
    std::copy(vector.begin(), vector.end(), output_iterator);
}

void data_handler::eigen_to_file(Eigen::MatrixXd data, std::string filename)
{
    std::ofstream output_file(filename);
    if(output_file.is_open())
    {
        output_file << data << "\n";
    }
}

void data_handler::split_data()
{
    std::vector<int> indices(data_array->size());
    std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, ..., size-1.

    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::shuffle(indices.begin(), indices.end(), eng); // Shuffle the indices

    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;

    for (int i = 0; i < indices.size(); ++i)
    {
        if (i < train_size)
        {
            training_data->push_back(data_array->at(indices[i]));
        }
        else if (i < train_size + test_size)
        {
            test_data->push_back(data_array->at(indices[i]));
        }
        else
        {
            validation_data->push_back(data_array->at(indices[i]));
        }
    }

    std::cout << "Training Data Size: " << training_data->size() << std::endl;
    std::cout << "Test Data Size: " << test_data->size() << std::endl;
    std::cout << "Validation Data Size: " << validation_data->size() << std::endl;
    std::cout << "Splitting the whole data, has been done!\n\n";
}
void data_handler::count_classes()
{
    int count = 0;
    for(const auto & data_point : *data_array)
    {
        if(class_map.find(data_point->get_label()) == class_map.end())
        {
            class_map[data_point->get_label()] = count;
            data_point->set_enumerated_label(count);
            count++;
        }
    }
    num_classes = count;
    std::cout << num_classes << " Unique Classes have been successfully extracted!\n";
}

auto data_handler::average(Eigen::MatrixXd data) -> decltype(data.colwise().mean())
{
    return data.colwise().mean();
}

auto data_handler::standart_deviation(Eigen::MatrixXd data) -> decltype(((data.array().square().colwise().sum())/(data.rows()-1)).sqrt())
{
    return ((data.array().square().colwise().sum())/(data.rows()-1)).sqrt();
}

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> data_handler::train_test_split(Eigen::MatrixXd data, float train_size)
{
    int rows = data.rows();
    int train_rows = round(train_size*rows);
    int test_rows = rows - train_rows;
    Eigen::MatrixXd train = data.topRows(train_rows);
    Eigen::MatrixXd X_train = train.leftCols(data.cols()-1);
    Eigen::MatrixXd y_train = train.rightCols(1);
    Eigen::MatrixXd test = data.bottomRows(test_rows);
    Eigen::MatrixXd X_test = test.leftCols(data.cols()-1);
    Eigen::MatrixXd y_test = test.rightCols(1);

    return std::make_tuple(X_train, y_train, X_test, y_test); 
}
uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes)
{
    return (uint32_t) ((bytes[0] << 24) |
                       (bytes[1] << 16) |
                       (bytes[2] << 8) |
                       (bytes[3]));
}

std::unique_ptr<std::vector<std::shared_ptr<data>>> data_handler::get_training_data()
{
    return std::make_unique<std::vector<std::shared_ptr<data>>>(*training_data);
}
std::unique_ptr<std::vector<std::shared_ptr<data>>> data_handler::get_test_data()
{
    return std::make_unique<std::vector<std::shared_ptr<data>>>(*test_data);
}

std::unique_ptr<std::vector<std::shared_ptr<data>>> data_handler::get_validation_data()
{
    return std::make_unique<std::vector<std::shared_ptr<data>>>(*validation_data);
}
