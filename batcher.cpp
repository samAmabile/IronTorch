#include "batcher.hpp" 

std::vector<std::vector<uint32_t>> Batcher::batchData(const std::vector<uint32_t>& input){
    std::vector<std::vector<uint32_t>> batches;

    size_t window = batchSize * seqLen; 
    size_t numBatches = input.size() / window; 
    batches.reserve(numBatches); 

    for (size_t i{0}; i < numBatches; ++i){
        auto start = input.begin() + (i * window); 
        auto end = start + window; 

        batches.emplace_back(start, end); 
    }

    return batches; 
}

void Batcher::fillBatchMatrix(const std::vector<Script>& dataset, size_t startIndex, size_t maxLen, Matrix& batch){
    
    if (startIndex + batchSize > dataset.size()){
        throw std::runtime_error("Attempted to access past end of data when creating Batch");
    }
    for (size_t r{0}; r < batchSize; ++r){
        const auto& sample = dataset[startIndex + r];

        for (size_t c{0}; c < seqLen; ++c) {

            float tokenVal = (c < sample.tokens.size()) ? static_cast<float>(sample.tokens[c]) : 0.0f;
            batch(r, c) = tokenVal;
        }
    }
}






