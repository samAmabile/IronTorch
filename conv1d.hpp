#ifndef CONV1D_HPP
#define CONV1D_HPP

#include "paramaterlayer.hpp"
#include "matrix.hpp"

class Conv1d : public ParamaterLayer {
private:
    size_t inChannels; 
    size_t outChannels;
    size_t kernelSize; 
    size_t hiddenDim; 

    Matrix kernel; 
    Matrix bias; 
    Matrix gradKernel; 
    Matrix gradBias; 

    Matrix padded; 

    Matrix prevIn; 
    
    size_t batchSize; 
    size_t seqLen;


public:

    Conv1d(size_t inChans, size_t outChans, size_t kSize, size_t bSize, size_t sLen); 

    size_t getOutDim() const override;

    void forward(const Matrix& input, Matrix& output) override; 
    void backward(const Matrix& gradientOut, Matrix& gradIn, float /*learnRate*/) override; 
    Matrix& getWeights() override;
    Matrix& getGradient() override;
    Matrix& getBias() override;
    Matrix& getBiasGradient() override;

    void update(float lr) override;
    bool hasBias() const override { return true; }

    void save(std::ofstream& ofs) const override; 
    void load(std::ifstream& ifs) override; 

};

#endif
