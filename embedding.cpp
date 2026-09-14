#include "embedding.hpp"
#include <cmath>
#include <iostream>
#include <cstring>


Embedding::Embedding(): 
    vocabSize(0), embedDim(0), 
    weights(1, 1), gradWeights(1, 1, 0.0)
{
}
Embedding::Embedding(size_t vSize, size_t emDim): 
    weights(vSize, emDim), 
    gradWeights(vSize, emDim, 0.0),
    vocabSize(vSize), 
    embedDim(emDim)
    //batchSize(bSize),
    //hiddenDim(hidDim)
{

    float limit = 1.0 / std::sqrt(static_cast<float>(emDim));
    //float limit = 0.1; 
    weights.randomize(-limit, limit);

}

void Embedding::forward(const Matrix& input, Matrix& output){
    indexCacheMat.copyFrom(input);

    batchSize = input.getRows(); 
    size_t seqLen = input.getCols(); 
    size_t totalTokens = batchSize * seqLen; 

    output.setDimensions(totalTokens, embedDim); 

    const float* weightData = weights.dataPtr(); 
    float* outData = output.dataPtr(); 
    const float* inData = input.dataPtr(); 

    #pragma omp simd
    for (size_t i = 0; i < totalTokens; i++){
        uint32_t tokenID = static_cast<uint32_t>(inData[i]); 
        if (tokenID >= vocabSize) tokenID = 0; 

        std::memcpy(outData + (i * embedDim), weightData + (tokenID * embedDim), 
                embedDim * sizeof(float)); 
    }
}

void Embedding::backward(const Matrix& gradOut, Matrix& /*gradIn*/, float /*learnRate*/){

    size_t rows = indexCacheMat.getRows();
    size_t seqLen = indexCacheMat.getCols();

    const float* cacheData = indexCacheMat.dataPtr();
    const float* gradOutData = gradOut.dataPtr(); 
    float* gradWeightData = gradWeights.dataPtr(); 

    for (size_t b = 0; b < rows; ++b){
        const size_t offset = b * seqLen; 

        for (size_t s = 0; s < seqLen; ++s){
            size_t tokenIndex = offset + s; 
            uint32_t tokenID = static_cast<uint32_t>(cacheData[tokenIndex]); 

            if (tokenID >= vocabSize){
                continue; 
            }

            float* targetGradRow = gradWeightData + (tokenID * embedDim); 
            const float* sourceGradRow = gradOutData + (tokenIndex * embedDim);
            
            #pragma omp simd
            for (size_t c = 0; c < embedDim; ++c){
                targetGradRow[c] += sourceGradRow[c];
            }
        }
    }

}

Matrix& Embedding::getWeights() { return weights; }
Matrix& Embedding::getGradient() { return gradWeights; }
Matrix& Embedding::getBias() { throw std::runtime_error("No bias to return from Embedding"); }
Matrix& Embedding::getBiasGradient() { throw std::runtime_error("No gradient bias to return from Embedding"); }

void Embedding::update(float lr){
    weights.subtractScaled(gradWeights, lr);

    gradWeights.fill(0.0f); 

}

void Embedding::save(std::ofstream& ofs) const {
    weights.save(ofs); 
} 
void Embedding::load(std::ifstream& ifs) {
    weights.load(ifs); 

    vocabSize = weights.getRows();
    embedDim = weights.getCols(); 

    gradWeights.setDimensions(vocabSize, embedDim); 
}
        


