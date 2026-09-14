#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include "layer.hpp"

#include <vector>

class Dropout : public Layer {
private:
    float dropRate;
    std::vector<float> mask; 
    bool training = true; 

public:
    Dropout(float rate, bool isTraining) : dropRate(rate), training(isTraining) {}

    void setTraining(bool isTraining); 
    void forward(const Matrix& input, Matrix& output) override; 
    void backward(const Matrix& gradientOut, Matrix& gradIn, float) override;

    void save(std::ofstream&) const override;
    void load(std::ifstream&) override;

};

#endif
