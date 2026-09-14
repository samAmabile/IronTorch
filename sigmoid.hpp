#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "layer.hpp" 

class Sigmoid : public Layer { 
private:
    Matrix lastOutput; 

public: 
    void forward(const Matrix& input, Matrix& output) override; 
    void backward(const Matrix& gradientOut, Matrix& gradIn, float) override;
    void save(std::ofstream&) const override {}
    void load(std::ifstream&) override {}

}; 

#endif

