#ifndef __COHEIR_H
#define __COHEIR_H

#include "data.h"
#include <vector>

class coheir
{

    protected:
    std::shared_ptr<std::vector<std::shared_ptr<data>>> training_data;
    std::shared_ptr<std::vector<std::shared_ptr<data>>> test_data;
    std::shared_ptr<std::vector<std::shared_ptr<data>>> validation_data;


    public:
    void set_training_data(std::shared_ptr<std::vector<std::shared_ptr<data>>> vect);
    void set_test_data(std::shared_ptr<std::vector<std::shared_ptr<data>>> vect);
    void set_validation_data(std::shared_ptr<std::vector<std::shared_ptr<data>>> vect);
};

#endif