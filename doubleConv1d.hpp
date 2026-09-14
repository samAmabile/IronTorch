#ifndef DOUBLECONV1D_HPP
#define DOUBLECONV1D_HPP

#include "layer.hpp"
#include "conv1d.hpp"

#include <memory>

class DoubleConv1d: public Layer {
private:
    std::shared_ptr<Conv1d> smallConv; 
    std::shared_ptr<Conv1d> largeConv;

    size_t smallWidth; 

    Matrix A, B; 
    Matrix gradA, gradB; 
    Matrix gradInA, gradInB;

public:
    
    DoubleConv1d(size_t inChans, size_t outChans, size_t kSizeA, size_t kSizeB, size_t batchSize, size_t seqLen); 

    size_t getOutDim() const override; 
    void forward(const Matrix& input, Matrix& output) override;
    void backward(const Matrix& gradientOut, Matrix& gradIn, float lr) override; 

    std::shared_ptr<ParamaterLayer> getSmallConv() { return smallConv; }
    std::shared_ptr<ParamaterLayer> getLargeConv() { return largeConv; }

    void save(std::ofstream&) const override; 
    void load(std::ifstream&) override; 

};

#endif

