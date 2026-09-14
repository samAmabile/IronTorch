#include "loss.hpp" 
#include <cmath> 
#include <algorithm>
#include <cassert>

float Loss::computeMSE(const Matrix& output, const Matrix& target) {
    const float* outData = output.dataPtr();
    const float* tarData = target.dataPtr();
    float sum = 0.0 ;
    size_t total = output.getRows() * output.getCols(); 

    for (size_t i{0}; i < total; ++i){
        float diff = outData[i] - tarData[i]; 
        sum += diff * diff;
    }

    return sum/static_cast<float>(total);
}

Matrix Loss::calcGrad(const Matrix& output, const Matrix& target) { 
    size_t rows = output.getRows();
    size_t cols = output.getCols();

    Matrix grad(rows, cols);
    float* gradData = grad.dataPtr();
    const float* outData = output.dataPtr();
    const float* tarData = target.dataPtr();
    size_t total = rows * cols; 
    
    float invTotal = 2.0 / static_cast<float>(total);
    bool hasOnes = false; 
    for (size_t i{0}; i < total; ++i) {
        if (tarData[i] == 1.0) hasOnes = true;
        gradData[i] = (outData[i] - tarData[i]) * invTotal; 
    }
    assert(hasOnes && "Only zeroes present in targets array"); 

    return grad; 
}

float Loss::crossEntropy(const Matrix& output, const Matrix& target){

    size_t rows = output.getRows(); 
    size_t cols = output.getCols(); 

    const float* outData = output.dataPtr(); 
    const float* tarData = target.dataPtr();

    float loss = 0.0; 
    const float epsilon = 1e-15f; 
    size_t numElements = rows * cols; 

    for (size_t i{0}; i < numElements; ++i){

        float p = std::clamp(outData[i], epsilon, 1.0f - epsilon);
        float y = tarData[i];
        assert((y == 0.0 ||  y == 1.0) && "Target aint 1.0 or 0.0 check it");
        loss -= (y * std::log(p) + (1.0f -y) * std::log(1.0f - p));

    }

    return loss / static_cast<float>(rows); 

}

void Loss::calcCrossEntropyGrad(const Matrix& output, const Matrix& target, Matrix& grad){   
    output.matSub(target, grad);

    float invRows = 1.0f / static_cast<float>(output.getRows());
 
    grad.scale(invRows); 
}


float Loss::crossEntropySM(const Matrix& output, const Matrix& target){
    //softmax(output);
    size_t rows = output.getRows(); 
    size_t cols = output.getCols(); 

    const float* outData = output.dataPtr(); 
    const float* tarData = target.dataPtr();

    float loss = 0.0; 
    const float epsilon = 1e-15f; 
    size_t numElements = rows * cols; 

    for (size_t i{0}; i < numElements; ++i){

        float p = std::clamp(outData[i], epsilon, 1.0f - epsilon);
        float y = tarData[i];
        assert((y == 0.0 ||  y == 1.0) && "Target aint 1.0 or 0.0 check it");
        loss -= (y * std::log(p) + (1.0f -y) * std::log(1.0f - p));

    }

    return loss / static_cast<float>(rows); 

}

void Loss::calcCrossEntropyGradSM(const Matrix& output, const Matrix& target, Matrix& grad){   
    //softmax(output);
    output.matSub(target, grad);

    float invRows = 1.0f / static_cast<float>(output.getRows());
 
    grad.scale(invRows); 
}

float Loss::categoricalCrossEntropy(const Matrix& output, const Matrix& target) {
    size_t rows = output.getRows();
    size_t cols = output.getCols(); 
    const float* outData = output.dataPtr(); 
    const float* tarData = target.dataPtr(); 

    float loss = 0.0f; 
    const float epsilon = 1e-15f;

    for (size_t r{0}; r < rows; ++r) {
        for (size_t c{0}; c < cols; ++c) {
            if (tarData[r * cols + c] == 1.0f) {
                float p = std::clamp(outData[r * cols + c], epsilon, 1.0f - epsilon); 
                loss -= std::log(p); 
                break; 
            }
        }
    }

    return loss / static_cast<float>(rows); 
}

void Loss::softmax(Matrix& output){
    size_t rows = output.getRows(); 
    size_t cols = output.getCols(); 

    for (size_t r{0}; r < rows; ++r){
        float maxVal = -1e18; 
        for (size_t c{0}; c < cols; ++c){
            maxVal = std::max(maxVal, output(r, c)); 
        }

        float sumTotal = 0.0; 

        for (size_t c{0}; c < cols; ++c){
            output(r, c) = std::exp(output(r, c) - maxVal);
            sumTotal += output(r, c);
        }
        for (size_t c{0}; c < cols; ++c){
            output(r, c) /= sumTotal; 
        }
    }
}
        


