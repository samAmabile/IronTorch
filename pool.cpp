#include "pool.hpp"
#include <iostream>

void Pooling::forward(const Matrix& input, Matrix& output) {


    output.setDimensions(batchSize, hiddenDim * 2); 
    output.fill(-1e9f);

    maxIndices.setDimensions(batchSize, hiddenDim);
    maxIndices.fill(0.0f);

    const float* inData = input.dataPtr(); 
    float* outData = output.dataPtr(); 
    float* maxIndicesData = maxIndices.dataPtr(); 

    for (size_t b{0}; b < batchSize; ++b){
        const size_t offset = b * seqLen; 
        float* outRow = outData + (b * hiddenDim * 2); 
        float* maxRow = outRow;
        float* avgRow = outRow + hiddenDim;

        float* indicesRow = maxIndicesData + (b * hiddenDim); 

        for (size_t c{0}; c < hiddenDim; ++c){
            maxRow[c] = -1e9f; 
            avgRow[c] = 0.0f; 
        }

        for (size_t r{0}; r < seqLen; ++r){
            const float* inRow = inData + ((offset + r) * hiddenDim); 
            

            for (size_t c{0}; c < hiddenDim; ++c){
                float val = inRow[c]; 
                if (val > maxRow[c]){
                    maxRow[c] = val; 
                    indicesRow[c] = static_cast<float>(r); 
                }

                avgRow[c] += val; 
            }

        }

        float invSeqLen = 1.0f  / static_cast<float>(seqLen);
        for(size_t c{0}; c < hiddenDim; ++c){
            avgRow[c] *= invSeqLen; 
        }
    }


}

void Pooling::backward(const Matrix& gradientOut, Matrix& gradIn, float /*lr*/) {

    //size_t curBatchSize = gradientOut.getRows(); 
    size_t curBatchSize = batchSize;
    size_t curSeqLen = seqLen;
    
    gradIn.setDimensions(curBatchSize * curSeqLen, hiddenDim); 
    gradIn.fill(0.0f);

    const float* gradOutData = gradientOut.dataPtr(); 
    const float* maxIndicesData = maxIndices.dataPtr(); 
    float* gradInData = gradIn.dataPtr(); 

    float invSeqLen = 1.0f / static_cast<float>(curSeqLen); 

    for (size_t b{0}; b < curBatchSize; ++b){

        const size_t offset = b * curSeqLen; 
        const float* gradOutRow = gradOutData + (b * hiddenDim * 2); 
        const float* gradMaxRow = gradOutRow; 
        const float* gradAvgRow = gradOutRow + hiddenDim; 

        const float* indicesRow = maxIndicesData + (b * hiddenDim);

    
        for (size_t c{0}; c < hiddenDim; ++c){

            size_t winRow = static_cast<size_t>(indicesRow[c]);

            size_t maxTargetIndex = offset + winRow; 

            gradInData[maxTargetIndex * hiddenDim + c] = gradMaxRow[c]; 

            float avgGradShare = gradAvgRow[c] * invSeqLen; 
            for (size_t r{0}; r < curSeqLen; ++r){
                size_t avgTargetIndex = offset + r; 
                gradInData[avgTargetIndex * hiddenDim + c] += avgGradShare; 
            }

        }
    }

}
