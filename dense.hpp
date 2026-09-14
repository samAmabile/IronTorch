#ifndef DENSE_HPP
#define DENSE_HPP

#include "paramaterlayer.hpp" 
#include "matrix.hpp" 

#include <fstream>

class Dense: public ParamaterLayer { 
private: 

    Matrix weights; 
    Matrix weightsT;
    Matrix bias; 
    Matrix gradWeights;
    Matrix gradWeightsT;
    Matrix gradBias;
    Matrix gradBiasT;
    Matrix prevIn; 
    Matrix prevInT;

    Matrix scratch;
    Matrix scratchT; 

public:

    Dense(size_t inSize, size_t outSize); 

    void forward(const Matrix& input, Matrix& output) override; 
    void backward(const Matrix& gradientOut, Matrix& gradIn, float /*learnRate*/) override; 

    Matrix& getWeights() override;
    Matrix& getBias() override;
    Matrix& getGradient() override;
    Matrix& getBiasGradient() override;

    void update(float lr) override;
    bool hasBias() const override { return true; }

    void save(std::ofstream& ofs) const override; 
    void load(std::ifstream& ifs) override; 

}; 

#endif
