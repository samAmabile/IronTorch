#ifndef ARCH_HPP
#define ARCH_HPP

#include "model.hpp" 
#include <memory>
#include <cstddef>

namespace Architecture{

    void initClassifierLayers(Model& m, size_t batchSize, size_t seqLen, size_t embDim, size_t hiddenDim, size_t kernelSize, bool isTraining);

    void initNTPLayers(Model& m, size_t vocabSize, size_t batchSize, size_t seqLen, size_t embDim, size_t hiddenDim, size_t kernelSize); 

}

#endif
