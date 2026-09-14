#ifndef BATCHER_HPP
#define BATCHER_HPP

#include <vector> 
#include <cstdint>
#include "matrix.hpp"
#include "tokenizer.hpp"

class Batcher {
private: 
    size_t batchSize; 
    size_t seqLen; 

public:
    Batcher(size_t b, size_t s): batchSize(b), seqLen(s) {}

    std::vector<std::vector<uint32_t>> batchData(const std::vector<uint32_t>& input);
    void fillBatchMatrix(const std::vector<Script>& dataset, size_t startIndex, size_t maxLen, Matrix& batch);

};

#endif


