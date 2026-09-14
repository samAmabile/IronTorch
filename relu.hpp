#ifndef RELU_HPP
#define RELU_HPP

#include "layer.hpp"
#include <vector>

class ReLU : public Layer {
private:
    std::vector<bool> mask;
    std::vector<float> leakyMask;

    float alpha = 0.05f; 

public:
    ReLU(size_t maskSize): mask(maskSize, false) {}
    void forward(const Matrix& input, Matrix& output) override;
    void backward(const Matrix& gradientOut, Matrix& gradIn, float /*learnRate*/) override; 

    void save(std::ofstream&) const override;

    void load(std::ifstream&) override;

}; 

#endif

        

