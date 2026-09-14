#include "relu.hpp"

void ReLU::forward(const Matrix& input, Matrix& output) { 
    output.copyFrom(input); 
    size_t matrixElements = input.getRows() * input.getCols();
    
    /*if (mask.size() != matrixElements){
        mask.resize(matrixElements);
    }*/

    if (leakyMask.size() != matrixElements){
       leakyMask.resize(matrixElements);
    }
    
    //output.setMask(mask); 
    output.setLeakyMask(leakyMask, alpha);
}

void ReLU::backward(const Matrix& gradientOut, Matrix& gradIn, float /*learnRate*/) {
    gradIn.copyFrom(gradientOut); 
    
    //gradIn.applyMask(mask); 
    gradIn.applyLeakyMask(leakyMask); 
}




void ReLU::save(std::ofstream&) const {}

void ReLU::load(std::ifstream&) {}
