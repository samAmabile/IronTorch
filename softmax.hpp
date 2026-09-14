#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "layer.hpp"
#include "loss.hpp"

class Softmax : public Layer { 
private: 
    Matrix prevOut; 
public: 
    void forward(const Matrix& input, Matrix& output) override{
        output.copyFrom(input); 
        Loss::softmax(output); 
    }
    void backward(const Matrix& gradientOut, Matrix& gradIn, float) override {
        gradIn.copyFrom(gradientOut); 
    }
    void save(std::ofstream&) const override {}
    void load(std::ifstream&) override {}
};

#endif
