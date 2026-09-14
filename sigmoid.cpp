#include "sigmoid.hpp"
#include <cmath>

void Sigmoid::forward(const Matrix& input, Matrix& output) { 

    lastOutput.copyFrom(input); 

    for (size_t i{0}; i < input.size(); ++i){
        float x = input.getData(i); 
        lastOutput.setData(i, 1.0f / (1.0f + std::exp(-x))); 
    }

    output.copyFrom(lastOutput); 
}

void Sigmoid::backward(const Matrix& gradientOut, Matrix& gradIn, float /*lr*/) { 
    gradIn.copyFrom(gradientOut); 

}









