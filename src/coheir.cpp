#include "../include/coheir.h"

void coheir::set_training_data(std::shared_ptr<std::vector<std::shared_ptr<data>>> vect)
{
    training_data = vect;
}
void coheir::set_test_data(std::shared_ptr<std::vector<std::shared_ptr<data>>> vect)
{
    test_data = vect;
}
void coheir::set_validation_data(std::shared_ptr<std::vector<std::shared_ptr<data>>> vect)
{
    validation_data = vect;
}