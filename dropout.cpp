#include "dropout.hpp"

#include <random>
#include <algorithm> 


void Dropout::setTraining(bool isTraining) {
    training = isTraining; 
}

void Dropout::forward(const Matrix& input, Matrix& output) {
    output.copyFrom(input);

    if (!training) return; 

    size_t n = input.size(); 
    if (mask.size() != n) mask.resize(n); 

    std::random_device rd; 
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); 
    
    float scale = 1.0f / (1.0f - dropRate); 
    
    for (size_t i{0}; i < n; ++i) {
        if (dist(generator) < dropRate) {
            mask[i] = 0.0f; 
        } else {
            mask[i] = scale; 
        }
    }
    
    output.applyLeakyMask(mask);
}

void Dropout::backward(const Matrix& gradientOut, Matrix& gradIn, float) {
    gradIn.copyFrom(gradientOut); 
    
    if (training) gradIn.applyLeakyMask(mask); 

}

void Dropout::save(std::ofstream&) const {}
void Dropout::load(std::ifstream&) {}

