#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP
#include <cstdint>

#include "paramaterlayer.hpp"
#include <vector>

class Embedding : public ParamaterLayer {
private:
    Matrix weights; 
    Matrix gradWeights; 
    std::vector<uint32_t> indexCache;
    Matrix indexCacheMat;
    Matrix rowScratch; 
    size_t vocabSize, embedDim, batchSize, hiddenDim; 

public:
    Embedding(); 
    Embedding(size_t vSize, size_t emDim);

    //Matrix forward(const std::vector<uint32_t>& indices); 

    void forward(const Matrix& input, Matrix& output) override; 
    void backward(const Matrix& gradOut, Matrix& /*gradIn*/, float /*learnRate*/) override; 

    Matrix& getWeights() override;
    Matrix& getGradient() override;
    Matrix& getBias() override;
    Matrix& getBiasGradient() override;

    void update(float lr) override;
    bool hasBias() const override { return false; }

    void save(std::ofstream&) const override; 
    void load(std::ifstream&) override; 
};

#endif
