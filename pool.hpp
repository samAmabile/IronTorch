#ifndef POOL_HPP
#define POOL_HPP

#include "layer.hpp"

class Pooling : public Layer {
private:
    size_t batchSize;
    size_t seqLen;
    size_t hiddenDim;

    Matrix maxIndices;
public:
    Pooling(size_t bSize, size_t sLen, size_t hidDim) : 
        batchSize(bSize), 
        seqLen(sLen),
        hiddenDim(hidDim){}
    void forward(const Matrix& input, Matrix& output) override; 
    void backward(const Matrix& gradientOut, Matrix& gradIn, float) override; 

    void save(std::ofstream&) const override {}
    void load(std::ifstream&) override {}
}; 

#endif
