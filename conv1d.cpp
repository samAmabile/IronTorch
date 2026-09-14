#include "conv1d.hpp"
#include <cstring>
#include <iostream>


//int initLayers this gets initialized with (embDim, hidden, kerneSize, hidden) 
Conv1d::Conv1d(size_t inChans, size_t outChans, size_t kSize, size_t bSize, size_t sLen) :
    inChannels(inChans), outChannels(outChans), 
    kernelSize(kSize), batchSize(bSize), seqLen(sLen),
    kernel(outChans, inChans * kSize), bias(1, outChans, 0.0),
    gradKernel(outChans, inChans * kSize, 0.0),
    gradBias(1, outChans, 0.0),
    prevIn(1, 1), padded(1, 1) 
{
        kernel.randomize(-0.1, 0.1);
}

size_t Conv1d::getOutDim() const {
    return outChannels;
}

void Conv1d::forward(const Matrix& input, Matrix& output) { 
    prevIn.copyFrom(input); 

    //size_t inLen = input.getRows(); 
    size_t pad = kernelSize / 2;
    size_t paddedLen = seqLen + 2 * pad;

    padded.setDimensions(batchSize * paddedLen, inChannels);
    padded.fill(0.0f);

    const float* inData = input.dataPtr();
    float* paddedData = padded.dataPtr(); 

    #pragma omp simd
    for (size_t b = 0; b < batchSize; ++b){
        const float* inBlock = inData + (b * seqLen * inChannels);
        float* paddedBlock = paddedData + (b * paddedLen * inChannels); 
        std::memcpy(paddedBlock + pad * inChannels, inBlock, seqLen * inChannels * sizeof(float));
    }
 
    //size_t outLen = inLen; 
    output.setDimensions(batchSize * seqLen, outChannels);
    output.fill(0.0f); 

    float* outData = output.dataPtr(); 
    const float* kernelData = kernel.dataPtr(); 
    const float* biasData = bias.dataPtr(); 

    for (size_t b = 0; b < batchSize; ++b){
        const float* paddedBlock = paddedData + (b * paddedLen * inChannels);
        float* outBlock = outData + (b * seqLen * outChannels);

        #pragma omp parallel for
        for (size_t oc = 0; oc < outChannels; ++oc){
            //const float* kernelRow = kernelData + (oc * inChannels * kernelSize); 
            float biasVal = biasData[oc]; 

            for (size_t i = 0; i < seqLen; ++i){
                float sum = 0.0f; 

                for (size_t k = 0; k < kernelSize; ++k){
                    const float* paddedRow = paddedBlock + ((i + k) * inChannels); 
                    //const float* kSubRow = kernelRow + (k * inChannels);
                    
                    #pragma omp simd
                    for (size_t ic = 0; ic < inChannels; ++ic){
                        sum += paddedRow[ic] * kernelData[oc * (inChannels * kernelSize) + (ic * kernelSize + k)]; 
                    }
                }

                outBlock[i * outChannels + oc] = sum + biasVal; 

            }
        }
    }

    
}


void Conv1d::backward(const Matrix& gradientOut, Matrix& gradIn, float /*learnRate*/){ 
    //std::cout << "DEBUG: Conv1d backward input rows: " << gradientOut.getRows() 
    //          << " prevIn rows: " << prevIn.getRows() << std::endl;
    //gradKernel.fill(0.0); //in update now
    gradIn.setDimensions(prevIn.getRows(), prevIn.getCols());
    gradIn.fill(0.0f);

    //size_t outLen = gradientOut.getRows(); 
    size_t pad = kernelSize / 2;
    //size_t inRows = prevIn.getRows();

    const float* gradOutData = gradientOut.dataPtr(); 
    const float* prevInData = prevIn.dataPtr(); 
    const float* kernelData = kernel.dataPtr(); 

    float* gradInData = gradIn.dataPtr(); 
    float* gradKernelData = gradKernel.dataPtr(); 
    float* gradBiasData = gradBias.dataPtr(); 
   
    for (size_t b = 0; b < batchSize; ++b){
        const float* gradOutBlock = gradOutData + (b * seqLen * outChannels); 
        const float* prevInBlock = prevInData + (b * seqLen * inChannels); 
        float* gradInBlock = gradInData + (b * seqLen * inChannels);
        
        #pragma omp parallel for
        for (size_t oc = 0; oc < outChannels; ++oc){
            for(size_t i = 0; i < seqLen; ++i){
                float gradOutVal = gradOutBlock[i * outChannels + oc]; 

                gradBiasData[oc] += gradOutVal; 

                for (size_t k = 0; k < kernelSize; ++k){
                    int rowIndex = static_cast<int>(i + k) - static_cast<int>(pad); 

                    if (rowIndex >= 0 && rowIndex < static_cast<int>(seqLen)){
                        const float* prevInRow = prevInBlock + (rowIndex * inChannels); 
                        float* gradInRow = gradInBlock + (rowIndex * inChannels); 
                        float* gradKernelRow = gradKernelData + (oc * inChannels * kernelSize); 

                        #pragma omp simd
                        for (size_t ic = 0;  ic < inChannels; ++ic){
                            size_t kIdx = ic * kernelSize + k; 

                            gradKernelRow[kIdx] += gradOutVal * prevInRow[ic];
                            gradInRow[ic] += gradOutVal * kernelData[oc * (inChannels * kernelSize) + kIdx];
                        }
                    }
                }
            }
        }
    }

}

Matrix& Conv1d::getWeights() {
    return kernel;
}

Matrix& Conv1d::getGradient() { 
    return gradKernel; 
}

Matrix& Conv1d::getBias() {
    return bias; 
}

Matrix& Conv1d::getBiasGradient() {
    return gradBias; 
}

void Conv1d::update(float lr) {
    kernel.subtractScaled(gradKernel, lr);
    bias.subtractScaled(gradBias, lr);
    
    gradKernel.fill(0.0);
    gradBias.fill(0.0);

}

void Conv1d::save(std::ofstream& ofs) const {
    kernel.save(ofs); 
    bias.save(ofs);
}

void Conv1d::load(std::ifstream& ifs) {
    kernel.load(ifs); 
    bias.load(ifs);
}

 
