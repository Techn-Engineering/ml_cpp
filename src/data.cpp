///  Created by Tech Engineering, efe@lexpai.com on 6/11/2022.
///  Copyright © 2023 Tech Engineering. All rights reserved.
///  Copyright © 2023 LexpAI. All rights reserved.

#include "data.h"

data::data() : feature_vector(std::make_unique<std::vector<uint8_t>>())
{
    // When we run the constructor, we initialize the feature_vector
    // std::make_unique is used to create a unique_ptr that takes care of automatic deletion

}
data::~data()
{
   // No need to delete feature_vector as unique_ptr will handle this
}

void data::set_feature_vector(std::unique_ptr<std::vector<uint8_t>> vect)
{
    feature_vector = std::move(vect);
}
void data::append_to_feature_vector(uint8_t val)
{
    feature_vector->push_back(val);
}
void data::set_label(uint8_t val)
{
    label = val;
}
void data::set_enumerated_label(int val)
{
    enum_label = val;
}
void data::set_distance(double val)
{
    distance = val;
}

int data::get_feature_vector_size()
{
    return feature_vector->size();
}
uint16_t data::get_label()
{
    return label;
}
uint16_t data::get_enumerated_label()
{
    return enum_label;
}

std::unique_ptr<std::vector<uint8_t>> data::get_feature_vector()
{
    // Return a copy of feature_vector to preserve encapsulation
    return std::make_unique<std::vector<uint8_t>>(*feature_vector);
}
double data::get_distance()
{
    return distance;
}