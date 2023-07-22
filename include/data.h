#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include <iostream>
#include <memory> // needed for std::unique_ptr

class data
{
    std::unique_ptr<std::vector<uint8_t>> feature_vector;
    uint8_t label;
    int enum_label;
    double distance;

    public:
    data();
    ~data();

    void set_feature_vector(std::unique_ptr<std::vector<uint8_t>>);
    void append_to_feature_vector(uint8_t);
    void set_label(uint8_t);
    void set_enumerated_label(int);
    void set_distance(double);

    int get_feature_vector_size();
    uint16_t get_label();
    uint16_t get_enumerated_label();

    std::unique_ptr<std::vector<uint8_t>> get_feature_vector();
    double get_distance();
};

#endif